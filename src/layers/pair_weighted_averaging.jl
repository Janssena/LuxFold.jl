struct PairWeightedAveraging{LNM,LNZ,LV,LZ,LG,LO} <: Lux.AbstractLuxContainerLayer{(:layer_norm_m, :layer_norm_z, :linear_v, :linear_z, :linear_g, :linear_out)}
    layer_norm_m::LNM
    layer_norm_z::LNZ
    linear_v::LV
    linear_z::LZ
    linear_g::LG
    linear_out::LO
    num_heads::Int
    head_dim::Int
    inf::Float32
end

function PairWeightedAveraging(
    chn_msa::Int, chn_pair::Int, head_dim::Int, num_heads::Int;
    use_bias=true, eps=1e-5, inf=1e9
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
        head_dim,
        Float32(inf)
    )
end

(l::PairWeightedAveraging)(inputs::NamedTuple, ps, st) = l(
    inputs.m,
    inputs.z,
    get(inputs, :mask, nothing),
    ps, st
)

function (l::PairWeightedAveraging)(m::AbstractArray{T,4}, z::AbstractArray{T,4}, mask, ps, st) where T
    C_m, N_seq, N_res, B = size(m)
    C_z, _, _, _ = size(z)
    H, C_h = l.num_heads, l.head_dim

    # 1. Normalize
    m_ln, st_m = l.layer_norm_m(m, ps.layer_norm_m, st.layer_norm_m)
    z_ln, st_z = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)

    # 2. Values and Gating from m
    v, st_v = l.linear_v(m_ln, ps.linear_v, st.linear_v) # [H*C_h, N_seq, N_res, B]
    g, st_g = l.linear_g(m_ln, ps.linear_g, st.linear_g) # [H*C_h, N_seq, N_res, B]

    # 3. Bias from z
    b, st_lz = l.linear_z(z_ln, ps.linear_z, st.linear_z) # [H, N_res, N_res, B]

    # 4. Attention Weights (Softmax)
    # b: [H, Ni, Nj, B]. We want softmax over Nj (dim 3).
    if !isnothing(mask)
        # mask: [Ni, Nj, B]. Reshape to [1, Ni, Nj, B]
        # mask_bias = reshape(T(l.inf) .* (mask .- one(T)), 1, N_res, N_res, B)
        # b = b .+ mask_bias
        neginf = -floatmax(T)
        mask_reshaped = reshape(mask, 1, N_res, N_res, B)
        @. b = ifelse(mask_reshaped, b, neginf)
    end
    w = Lux.softmax(b; dims=3)

    # 5. Weighted Averaging
    # We want o[c, h, i, s, b] = sum_j w[h, i, j, b] * v[c, h, j, s, b]
    v = reshape(v, C_h, H, N_seq, N_res, B)
    
    # Align batch dimensions (H, B) to the end for broadcasting/batching
    # v_flat: [C_h * N_seq, N_res, H * B]
    v_flat = reshape(permutedims(v, (1, 3, 4, 2, 5)), C_h * N_seq, N_res, H * B)
    
    # w_flat: [N_res, N_res, H * B] (Ni, Nj, H*B)
    w_flat = reshape(permutedims(w, (2, 3, 1, 4)), N_res, N_res, H * B)

    # Use keywords to contract Nj (dim 2 for both v_flat and w_flat)
    # This avoids permuting w to put Nj at dim 1.
    o = Lux.batched_matmul(v_flat, w_flat; lhs_contracting_dim=2, rhs_contracting_dim=2)
    
    # Re-align and reshape back to [C_h*H, N_seq, N_res, B] to match gating layout
    o = reshape(o, C_h, N_seq, N_res, H, B)
    o = reshape(permutedims(o, (1, 4, 2, 3, 5)), C_h * H, N_seq, N_res, B)

    # 6. Gating and Output
    o_gated = @. o * g
    y, st_out = l.linear_out(o_gated, ps.linear_out, st.linear_out)

    return y, (layer_norm_m=st_m, layer_norm_z=st_z, linear_v=st_v, linear_z=st_lz, linear_g=st_g, linear_out=st_out)
end
