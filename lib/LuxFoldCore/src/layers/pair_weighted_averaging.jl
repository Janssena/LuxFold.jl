"""
    PairWeightedAveraging(chn_msa, chn_pair, head_dim, num_heads; kwargs...)

Updates the MSA representation `m` by performing a weighted average of its values, where 
the weights are derived from the pair representation `z`. To minimize allocations and memory
overhead, the gating operation is performed fully in-place on the output projection array.

# Arguments
- `chn_msa`: Channels in the MSA input `m`.
- `chn_pair`: Channels in the pair representation `z`.
- `head_dim`: Dimension of each attention head.
- `num_heads`: Number of attention heads.

# Keyword Arguments
- `use_bias`: NamedTuple or Bool specifying which internal layers should use bias.
- `eps`: Small constant for numerical stability in LayerNorm.

# Inputs
- `m`: MSA tensor. Expected shape: `[chn_msa, N, S, B]` where `N` is the residue
  sequence length (number of positions), `S` is the MSA sequence depth (number of sequences), and `B` is batch size.
- `z`: Pair representation tensor. Expected shape: `[chn_pair, N, N, B]`.
- `mask`: Optional attention mask. Expected shape: `[N, N, B]`.

# Returns
- `y`: Updated MSA tensor. Shape: `[chn_msa, N, S, B]`.
- `st`: Updated state containing states for `layer_norm_m`, `layer_norm_z`, `linear_v`, `linear_z`, `linear_g`, and `linear_out`.
"""
struct PairWeightedAveraging{LNM,LNZ,LV,LZ,LG,LO} <: Lux.AbstractLuxContainerLayer{(:layer_norm_m, :layer_norm_z, :linear_v, :linear_z, :linear_g, :linear_out)}
    layer_norm_m::LNM
    layer_norm_z::LNZ
    linear_v::LV
    linear_z::LZ
    linear_g::LG
    linear_out::LO
    num_heads::Int
    head_dim::Int
end

function PairWeightedAveraging(
    chn_msa::Int, chn_pair::Int, head_dim::Int, num_heads::Int;
    use_bias=true, eps=1e-5
)
    use_bias = resolve_defaults(use_bias, (:layer_norm_m, :layer_norm_z, :linear_v, :linear_z, :linear_g, :linear_out))

    layer_norm_m = use_bias.layer_norm_m ? Lux.LayerNorm((chn_msa, 1, 1); dims=1, epsilon=eps) : LayerNormNoBias((chn_msa, 1, 1); dims=1, epsilon=eps)
    layer_norm_z = use_bias.layer_norm_z ? Lux.LayerNorm((chn_pair, 1, 1); dims=1, epsilon=eps) : LayerNormNoBias((chn_pair, 1, 1); dims=1, epsilon=eps)

    return PairWeightedAveraging(
        layer_norm_m,
        layer_norm_z,
        Lux.Dense(chn_msa => head_dim * num_heads; use_bias=use_bias.linear_v),
        Lux.Dense(chn_pair => num_heads; use_bias=use_bias.linear_z),
        Lux.Dense(chn_msa => head_dim * num_heads, Lux.sigmoid; use_bias=use_bias.linear_g),
        Lux.Dense(head_dim * num_heads => chn_msa; use_bias=use_bias.linear_out),
        num_heads,
        head_dim
    )
end

(l::PairWeightedAveraging)(inputs::NamedTuple, ps, st) = l(
    inputs.m,
    inputs.z,
    get(inputs, :mask, nothing),
    ps, st
)

@inline apply_pwa_mask!(b, ::Nothing) = nothing

@inline function apply_pwa_mask!(b::AbstractArray{T}, mask::AbstractArray) where T
    neginf = -floatmax(T)
    mask_reshaped = reshape(mask, 1, size(mask)...) # [1, N, N, B]
    @. b = ifelse(mask_reshaped, b, neginf)
    return nothing
end

function (l::PairWeightedAveraging)(m::AbstractArray{T,4}, z::AbstractArray{T,4}, mask, ps, st) where T
    chn_msa, N, S, B = size(m)
    chn_pair, _, _, _ = size(z)
    H, D = l.num_heads, l.head_dim

    m_ln, st_m = l.layer_norm_m(m, ps.layer_norm_m, st.layer_norm_m)
    z_ln, st_z = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)

    v, st_v = l.linear_v(m_ln, ps.linear_v, st.linear_v) # [H*D, N, S, B]
    g, st_g = l.linear_g(m_ln, ps.linear_g, st.linear_g) # [H*D, N, S, B]

    b, st_lz = l.linear_z(z_ln, ps.linear_z, st.linear_z) # [H, N, N, B]

    apply_pwa_mask!(b, mask)
    w = Lux.softmax(b; dims=3)

    v = reshape(v, D, H, N, S, B)

    # Flat tensors for batched matrix multiplication contracting Nj (dim 3 of v / dim 2 of w)
    v_flat = reshape(permutedims(v, (1, 4, 3, 2, 5)), D * S, N, H * B)
    w_flat = reshape(permutedims(w, (2, 3, 1, 4)), N, N, H * B)

    o = Lux.batched_matmul(v_flat, w_flat; lhs_contracting_dim=2, rhs_contracting_dim=2)

    o = reshape(o, D, S, N, H, B)
    o = reshape(permutedims(o, (1, 4, 3, 2, 5)), D * H, N, S, B)

    @. o *= g
    y, st_out = l.linear_out(o, ps.linear_out, st.linear_out)

    return y, (layer_norm_m=st_m, layer_norm_z=st_z, linear_v=st_v, linear_z=st_lz, linear_g=st_g, linear_out=st_out)
end
