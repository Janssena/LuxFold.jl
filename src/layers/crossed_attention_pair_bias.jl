import LuxTriangleAttention: prep_bias

# Overload prep_bias for Tuple inputs to support cross-attention
prep_bias(bias::AbstractArray, x::Tuple, bias_layout) =
    prep_bias(bias, first(x), bias_layout)

"""
    CrossedAttentionPairBias(chn_q, chn_k, chn_v, chn_cond, chn_z, head_dim, num_heads; kwargs...)

Cross-attention layer that incorporates pair bias and optional conditioning. Supports 
both global and local (window-based) attention modes.

# Arguments
- `chn_q`: Channels for the query input `a`.
- `chn_k`: Channels for the key/value input.
- `chn_v`: Channels for the value input.
- `chn_cond`: Channels for the conditioning signal `cond`.
- `chn_z`: Channels for the pair representation `z`.
- `head_dim`: Dimension of each attention head.
- `num_heads`: Number of attention heads.

# Keyword Arguments
- `use_adaln`: If `true`, uses Adaptive Layer Normalization (`AdaLN`) conditioned on `cond`.
- `blocksize`: Block size for queries in local attention. If `nothing`, global attention is used.
- `windowsize`: Window size for keys/values in local attention.

# Inputs
- `a`: The query sequence tensor. Shape: `[chn_q, Nq, B]` where `Nq` is the query sequence length and `B` is batch size.
- `z`: The pair representation tensor. Shape: `[chn_z, Nq, Nk, B]` where `Nk` is the key sequence length.
- `cond`: Optional conditioning signal. Shape: `[chn_cond, B]` or `[chn_cond, Nq, B]`.
- `mask`: Optional attention mask. Shape: `[Nk, B]`.

# Returns
- `y`: Output cross-attention tensor. Shape: `[chn_q, Nq, B]`.
- `st`: Updated state.
"""
struct CrossedAttentionPairBias{LOCAL,LNAQ,LNAK,LZ,MHA,LO} <: Lux.AbstractLuxContainerLayer{(:layer_norm_a_q, :layer_norm_a_k, :linear_z, :mha, :linear_out)}
    local_mode::LOCAL
    layer_norm_a_q::LNAQ
    layer_norm_a_k::LNAK
    linear_z::LZ
    mha::MHA
    linear_out::LO
end

function CrossedAttentionPairBias(
    chn_q::Int, chn_k::Int, chn_v::Int, chn_cond::Int, chn_z::Int,
    head_dim::Int, num_heads::Int;
    use_adaln::Bool=true,
    blocksize::Union{Nothing,Int}=nothing,
    windowsize::Union{Nothing,Int}=nothing
)
    # 1. Normalization
    if use_adaln
        # AF3 AdaLN usually has no bias in its internal LayerNorms
        layer_norm_a_q = AdaLN(chn_q => chn_cond; rank=3, use_bias=(false, (gate=true, shift=true)))
        layer_norm_a_k = AdaLN(chn_k => chn_cond; rank=3, use_bias=(false, (gate=true, shift=true)))
        linear_out = Lux.Dense(chn_cond => chn_q, Lux.sigmoid; use_bias=true)
    else
        # AF3 CrossAttention LayerNorm has bias by default
        layer_norm_a_q = Lux.LayerNorm((chn_q, 1); dims=1)
        layer_norm_a_k = Lux.LayerNorm((chn_k, 1); dims=1)
        linear_out = Lux.NoOpLayer()
    end

    # 2. Pair Bias Projection
    linear_z = Lux.Dense(chn_z => num_heads; use_bias=true)

    # 3. Attention
    mha = Attention(
        chn_q, chn_k, chn_v, head_dim, num_heads;
        use_gate=true, fuse_qkv=false,
        use_bias=(false, (gate=false,))
    )

    local_mode = isnothing(blocksize) || isnothing(windowsize) ? nothing : (blocksize, windowsize)

    return CrossedAttentionPairBias(
        local_mode, layer_norm_a_q, layer_norm_a_k, linear_z, mha, linear_out
    )
end

# Top-level Dispatches
(l::CrossedAttentionPairBias)(inputs::NamedTuple, ps, st) = l(
    inputs.a, inputs.z,
    get(inputs, :cond, nothing),
    get(inputs, :mask, nothing),
    ps, st
)

(l::CrossedAttentionPairBias)(a, z, ps, st) = l(a, z, nothing, nothing, ps, st)
(l::CrossedAttentionPairBias)(a, z, cond::AbstractArray, ps, st) = l(a, z, cond, nothing, ps, st)
(l::CrossedAttentionPairBias)(a, z, ::Nothing, ps, st) = l(a, z, nothing, nothing, ps, st)
(l::CrossedAttentionPairBias)(a, z, mask::AbstractArray{Bool}, ps, st) = l(a, z, nothing, mask, ps, st)

# Global attention masking
@inline apply_cab_mask!(bias_z, ::Nothing) = nothing

@inline function apply_cab_mask!(bias_z::AbstractArray{T}, mask::AbstractArray; neginf=T == Float64 ? -1e9 : -floatmax(T)) where T
    Nk = size(mask, 1)
    mask_reshaped = reshape(mask, 1, 1, Nk, size(mask, 2)) # [1, 1, Nk, B]
    @. bias_z = ifelse(mask_reshaped, bias_z, neginf)
    return nothing
end

# Local attention masking (pure boolean broadcasting)
@inline apply_local_cab_mask!(mha_bias_p, ::Nothing) = nothing

@inline function apply_local_cab_mask!(mha_bias_p::AbstractArray{T}, mask::AbstractArray; neginf=T == Float64 ? -1e9 : -floatmax(T)) where T
    windowsize = size(mha_bias_p, 1) # Key block size (window size)
    blocksize = size(mha_bias_p, 2) # Query block size

    m_blocked_q = pad_and_block(mask, blocksize; dims=1) # [blocksize, nb, B]
    m_blocked_k = pad_and_block(mask, windowsize; dims=1) # [windowsize, nb, B]

    mask_k = reshape(m_blocked_k, windowsize, 1, 1, size(m_blocked_k, 2), size(m_blocked_k, 3))
    mask_q = reshape(m_blocked_q, 1, blocksize, 1, size(m_blocked_q, 2), size(m_blocked_q, 3))

    @. mha_bias_p = ifelse(mask_k & mask_q, mha_bias_p, mha_bias_p + neginf)
    return nothing
end

# Functor for unconditioned inputs (cond === Nothing)
function (l::CrossedAttentionPairBias)(a, z, ::Nothing, mask, ps, st)
    # 1. Normalize without cond
    a_q_ln, st_q = l.layer_norm_a_q(a, ps.layer_norm_a_q, st.layer_norm_a_q)
    a_k_ln, st_k = l.layer_norm_a_k(a, ps.layer_norm_a_k, st.layer_norm_a_k)

    # 2. Pair Bias Projection
    bias_z, st_lz = l.linear_z(z, ps.linear_z, st.linear_z)

    # 3. MHA (Global vs Local)
    y, st_mha = _run_mha(l, a_q_ln, a_k_ln, bias_z, mask, ps.mha, st.mha)

    st_final = (
        layer_norm_a_q=st_q,
        layer_norm_a_k=st_k,
        linear_z=st_lz,
        mha=st_mha,
        linear_out=st.linear_out
    )

    return y, st_final
end

# Functor for conditioned inputs (cond::AbstractArray)
function (l::CrossedAttentionPairBias)(a, z, cond::AbstractArray, mask, ps, st)
    # 1. Normalize with cond
    a_q_ln, st_q = l.layer_norm_a_q(a, cond, ps.layer_norm_a_q, st.layer_norm_a_q)
    a_k_ln, st_k = l.layer_norm_a_k(a, cond, ps.layer_norm_a_k, st.layer_norm_a_k)

    # 2. Pair Bias Projection
    bias_z, st_lz = l.linear_z(z, ps.linear_z, st.linear_z)

    # 3. MHA (Global vs Local)
    y, st_mha = _run_mha(l, a_q_ln, a_k_ln, bias_z, mask, ps.mha, st.mha)

    # 4. In-place Gating (AdaLN)
    g, st_lo = l.linear_out(cond, ps.linear_out, st.linear_out)
    @. y *= g

    st_final = (
        layer_norm_a_q=st_q,
        layer_norm_a_k=st_k,
        linear_z=st_lz,
        mha=st_mha,
        linear_out=st_lo
    )

    return y, st_final
end

function _run_mha(l::CrossedAttentionPairBias{Nothing}, a_q, a_k, bias_z, mask, ps, st)
    apply_cab_mask!(bias_z, mask)

    o, st_new = l.mha((x=(a_q, a_k, a_k), bias=bias_z, mask=nothing), ps, st)

    return o, st_new
end

function _run_mha(l::CrossedAttentionPairBias{<:NTuple{2,Int}}, a_q, a_k, bias_z, mask, ps, st)
    Nq = size(a_q, 2)
    T = eltype(a_q)
    blocksize, windowsize = l.local_mode

    # 1. Block inputs
    a_q_blocked = pad_and_block(a_q, blocksize; dims=2) # -> [chn_in, blocksize, nb, B]
    a_k_blocked = pad_and_block(a_k, windowsize; dims=2) # -> [chn_in, windowsize, nb, B]

    # 2. Block pair bias z: [num_heads, Nq, Nk, B] -> [num_heads, blocksize, nb, windowsize, nb, B]
    z_blocked = pad_and_block(bias_z, (blocksize, windowsize); dims=(2, 3))
    nb = size(z_blocked, 5)
    b_z = stack([view(z_blocked, :, :, i, :, i, :) for i in 1:nb]; dims=4) # [num_heads, blocksize, windowsize, nb, B]

    mha_bias_p = permutedims(b_z, (3, 2, 1, 4, 5)) # [windowsize, blocksize, num_heads, nb, B]
    apply_local_cab_mask!(mha_bias_p, mask)

    o_blocked, st_new = l.mha((x=(a_q_blocked, a_k_blocked, a_k_blocked), bias=mha_bias_p, mask=nothing), ps, st)
    o = unblock_and_slice(o_blocked, Nq; dims=2)::Array{T,3}

    return o, st_new
end

