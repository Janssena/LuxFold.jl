"""
    AttentionPairBias(layer_norm_in, layer_norm_z, linear_z, mha, linear_out)

Attention layer with pair bias support, specializing for global or local attention via static dispatch.

# Arguments
- `ln_a`: Input LayerNorm or AdaLN.
- `ln_z`: Pair representation LayerNorm.
- `mha`: MultiHeadAttention layer. Should be one of TriAttnCore or AttnCore.
- `head_dim`: Dimension of each attention head.
- `num_heads`: Number of attention heads.
- `block_size`: Size of spatial blocks for local attention.
"""
struct AttentionPairBias{LNI,LNZ,LZ,MHA,LO} <: Lux.AbstractLuxContainerLayer{(:layer_norm_in,:layer_norm_z,:linear_z,:mha,:linear_out)}
    layer_norm_in::LNI
    layer_norm_z::LNZ
    linear_z::LZ
    mha::MHA
    linear_out::LO
end

# c_q: int,         # CHN_IN
# c_k: int,         # CHN_IN
# c_v: int,         # CHN_IN
# c_s: int,         # CHN_COND
# c_z: int,         # CHN_PAIR
# c_hidden: int,    # HEAD_DIM
# no_heads: int,    # NUM_HEADS

function AttentionPairBias(
    chn_in::Int,
    chn_z::Int,
    head_dim::Int,
    num_heads::Int;
    chn_cond::Union{Nothing, Int} = nothing, # if isInt, then use AdaLN 
    use_gate::Bool = true,
    use_bias=false,
    affine=true,
    kwargs...
)
    affine = resolve_defaults(affine, (:layer_norm_in, :layer_norm_z))
    use_bias = resolve_defaults(use_bias, (:layer_norm_in, :layer_norm_z, :linear_z, :mha, :linear_out))

    if isnothing(chn_cond)
        shape = (chn_in, 1)
        layer_norm_in = if !affine.layer_norm_in || use_bias.layer_norm_in 
            Lux.LayerNorm(shape; dims=1, affine=affine.layer_norm_in)
        else
            LayerNormNoBias(shape; dims=1)
        end
        linear_out = Lux.NoOpLayer()
    else
        layer_norm_in = AdaLN(chn_in => chn_cond; affine=affine.layer_norm_in, rank=3, use_bias=use_bias.layer_norm_in)
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

(l::AttentionPairBias)(x, z, cond::AbstractArray{<:AbstractFloat}, ps, st) = 
    l(x, z, cond, nothing, ps, st)

function (l::AttentionPairBias)(x, z, ::Nothing, mask, ps, st)
    x, layer_norm_in = l.layer_norm_in(x, ps.layer_norm_in, st.layer_norm_in)
    
    z, layer_norm_z = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)
    bias, linear_z = l.linear_z(z, ps.linear_z, st.linear_z)

    attn, mha = l.mha(x, bias, mask, ps.mha, st.mha)

    return attn, merge(st, (; layer_norm_z, linear_z, layer_norm_in, mha))
end

function (l::AttentionPairBias)(x, z, cond::AbstractArray, mask, ps, st)
    x, layer_norm_in = l.layer_norm_in(x, cond, ps.layer_norm_in, st.layer_norm_in)

    z, layer_norm_z = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)
    bias, linear_z = l.linear_z(z, ps.linear_z, st.linear_z)

    (attn, scores), mha = l.mha(x, bias, mask, ps.mha, st.mha)

    g, linear_out = l.linear_out(cond, ps.linear_out, st.linear_out)
    
    y = @. g * attn
    
    return (y, scores), (; layer_norm_z, linear_z, layer_norm_in, mha, linear_out)
end