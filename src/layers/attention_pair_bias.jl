"""
    AttentionPairBias(chn_in, chn_z, head_dim, num_heads; kwargs...)

Attention layer that incorporates a bias derived from a pair representation `z`. This is a 
core component in many protein structure prediction models (e.g., AlphaFold).

# Arguments
- `chn_in`: Number of channels in the input sequence `x`.
- `chn_z`: Number of channels in the pair representation `z`.
- `head_dim`: Dimension of each attention head.
- `num_heads`: Number of attention heads.

# Keyword Arguments
- `rank`: The rank of the input sequence tensor (typically 3 or 4).
- `chn_cond`: If provided, the layer uses `AdaLN` for normalization, conditioned on a 
  signal with `chn_cond` channels.
- `use_gate`: Whether to apply gating to the attention output.
- `use_bias`: A `NamedTuple` or `Bool` specifying bias usage for internal layers.
- `affine`: A `NamedTuple` or `Bool` specifying affine transformation for LayerNorms.

# Inputs
- `x`: Input sequence tensor. Shape: `[chn_in, N, (S, ) B]`.
- `z`: Pair representation tensor. Shape: `[chn_z, N, N, B]`.
- `cond`: Optional conditioning signal for `AdaLN`. Shape: `[chn_cond, N, (S, ) B]`.
- `mask`: Optional attention mask. Shape: `[N, (S, ) B]`.

# Returns
- `(y, scores)`:
  - `y`: The output tensor. Shape matches `x`.
  - `scores`: The attention scores. Shape: `[num_heads, N, N, (S, ) B]`.
- `st`: Updated state.
"""
struct AttentionPairBias{LNI,LNZ,LZ,MHA,LO} <: Lux.AbstractLuxContainerLayer{(:layer_norm_in, :layer_norm_z, :linear_z, :mha, :linear_out)}
    layer_norm_in::LNI
    layer_norm_z::LNZ
    linear_z::LZ
    mha::MHA
    linear_out::LO
end

function AttentionPairBias(
    chn_in::Int,
    chn_z::Int,
    head_dim::Int,
    num_heads::Int;
    rank::Int=3,
    chn_cond::Union{Nothing,Int}=nothing, # if isInt, then use AdaLN 
    use_gate::Bool=true,
    use_bias=false,
    affine=true,
    use_layernorm_in=true,
    kwargs...
)
    @assert rank == 3 || rank == 4 "rank should be either 3 or 4."
    affine = resolve_defaults(affine, (:layer_norm_in, :layer_norm_z))
    use_bias = resolve_defaults(use_bias, (:layer_norm_in, :layer_norm_z, :linear_z, :mha, :linear_out))

    if isnothing(chn_cond)
        shape = (chn_in, ntuple(one, rank - 2)...)
        layer_norm_in = if !use_layernorm_in
            Lux.NoOpLayer()
        elseif !affine.layer_norm_in || use_bias.layer_norm_in
            Lux.LayerNorm(shape; dims=1, affine=affine.layer_norm_in)
        else
            LayerNormNoBias(shape; dims=1)
        end
        linear_out = Lux.NoOpLayer()
    else
        layer_norm_in = AdaLN(chn_in => chn_cond; rank, affine=affine.layer_norm_in, use_bias=use_bias.layer_norm_in)
        linear_out = Lux.Dense(chn_cond => chn_in, Lux.sigmoid; use_bias=use_bias.linear_out)
    end

    layer_norm_z = if !affine.layer_norm_z || use_bias.layer_norm_z
        Lux.LayerNorm((chn_z, 1, 1); dims=1, affine=affine.layer_norm_z)
    else
        LayerNormNoBias((chn_z, 1, 1); dims=1)
    end

    return AttentionPairBias(
        layer_norm_in,
        layer_norm_z,
        Lux.Dense(chn_z => num_heads; use_bias=use_bias.linear_z),
        Attention(chn_in, head_dim, num_heads; use_gate, use_bias=use_bias.mha, kwargs...),
        linear_out
    )
end

(l::AttentionPairBias)(inputs::NamedTuple, ps, st) = l(
    inputs.x,
    inputs.z,
    get(inputs, :cond, nothing),
    get(inputs, :mask, nothing),
    ps, st
)

(l::AttentionPairBias)(x, z, ps, st) = l(x, z, nothing, nothing, ps, st)
(l::AttentionPairBias)(x, z, ::Nothing, ps, st) = l(x, z, nothing, nothing, ps, st)

(l::AttentionPairBias)(x, z, mask::AbstractArray{Bool}, ps, st) =
    l(x, z, nothing, mask, ps, st)

(l::AttentionPairBias)(x, z, cond::AbstractArray{<:Real}, ps, st) =
    l(x, z, cond, nothing, ps, st)

function (l::AttentionPairBias)(x, z, ::Nothing, mask, ps, st)
    x, layer_norm_in = l.layer_norm_in(x, ps.layer_norm_in, st.layer_norm_in)

    z, layer_norm_z = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)
    bias, linear_z = l.linear_z(z, ps.linear_z, st.linear_z)

    attn, mha = l.mha(x, bias, mask, ps.mha, st.mha)

    return attn, merge(st, (; layer_norm_in, layer_norm_z, linear_z, mha))
end

function (l::AttentionPairBias)(x, z, cond::AbstractArray, mask, ps, st)
    x, layer_norm_in = l.layer_norm_in(x, cond, ps.layer_norm_in, st.layer_norm_in)

    z, layer_norm_z = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)
    bias, linear_z = l.linear_z(z, ps.linear_z, st.linear_z)

    attn, mha = l.mha(x, bias, mask, ps.mha, st.mha)

    g, linear_out = l.linear_out(cond, ps.linear_out, st.linear_out)

    y = @. g * attn

    return y, (; layer_norm_in, layer_norm_z, linear_z, mha, linear_out)
end

"""
    MSARowAttentionPairBias(chn_in, chn_z, head_dim, num_heads; kwargs...)

A specialized version of `AttentionPairBias` configured for MSA row attention. 
Sets `rank=4` and standard AlphaFold-style bias/normalization defaults.
"""
function MSARowAttentionPairBias(chn_in, chn_z, head_dim, num_heads; kwargs...)
    return AttentionPairBias(chn_in, chn_z, head_dim, num_heads;
        rank=4,
        chn_cond=nothing,
        use_bias=(linear_z=true, mha=false, layer_norm_in=true, layer_norm_z=true, linear_out=false),
        kwargs...)
end