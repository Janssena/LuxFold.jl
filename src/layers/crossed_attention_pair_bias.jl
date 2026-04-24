struct CrossedAttentionPairBias{LOCAL, LNAQ, LNAK, LZ, MHA, LO, NQ, NK} <: Lux.AbstractLuxContainerLayer{(:layer_norm_a_q, :layer_norm_a_k, :linear_z, :mha, :linear_out)}
    local_mode::LOCAL
    layer_norm_a_q::LNAQ
    layer_norm_a_k::LNAK
    linear_z::LZ
    mha::MHA
    linear_out::LO
    n_query::NQ
    n_key::NK
    inf::Float32
end

function CrossedAttentionPairBias(
    chn_q::Int, chn_k::Int, chn_v::Int, chn_cond::Int, chn_z::Int,
    head_dim::Int, num_heads::Int;
    use_ada_layer_norm::Bool=true,
    n_query::Union{Nothing,Int}=nothing,
    n_key::Union{Nothing,Int}=nothing,
    inf::Real=1e9
)
    # 1. Normalization
    if use_ada_layer_norm
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
    
    local_mode = static(!isnothing(n_query))

    return CrossedAttentionPairBias(
        local_mode, layer_norm_a_q, layer_norm_a_k, linear_z, mha, linear_out,
        n_query, n_key, Float32(inf)
    )
end

# Top-level Dispatches
(l::CrossedAttentionPairBias)(inputs::NamedTuple, ps, st) = l(
    inputs.a, inputs.z,
    get(inputs, :s, nothing),
    get(inputs, :mask, nothing),
    ps, st
)

(l::CrossedAttentionPairBias)(a, z, ps, st) = l(a, z, nothing, nothing, ps, st)
(l::CrossedAttentionPairBias)(a, z, cond::AbstractArray, ps, st) = l(a, z, cond, nothing, ps, st)

# Method 1: No Conditioning
function (l::CrossedAttentionPairBias)(a, z, ::Nothing, mask, ps, st)
    T = eltype(a)
    a_q_ln, st_q = l.layer_norm_a_q(a, ps.layer_norm_a_q, st.layer_norm_a_q)
    a_k_ln, st_k = l.layer_norm_a_k(a, ps.layer_norm_a_k, st.layer_norm_a_k)
    
    bias_z, st_lz = l.linear_z(z, ps.linear_z, st.linear_z)
    
    (res, st_mha) = _run_core_attention(l, a_q_ln, a_k_ln, bias_z, mask, ps.mha, st.mha)
    (o, scores) = res
    
    st_final = (layer_norm_a_q=st_q, layer_norm_a_k=st_k, linear_z=st_lz, mha=st_mha, linear_out=st.linear_out)
    return ((o::Array{T,3}, scores::Array{T,5}), st_final)
end

# Method 2: With Conditioning (AdaLN)
function (l::CrossedAttentionPairBias)(a, z, s::AbstractArray, mask, ps, st)
    T = eltype(a)
    a_q_ln, st_q = l.layer_norm_a_q((a=a, s=s), ps.layer_norm_a_q, st.layer_norm_a_q)
    a_k_ln, st_k = l.layer_norm_a_k((a=a, s=s), ps.layer_norm_a_k, st.layer_norm_a_k)
    
    bias_z, st_lz = l.linear_z(z, ps.linear_z, st.linear_z)
    
    (res, st_mha) = _run_core_attention(l, a_q_ln, a_k_ln, bias_z, mask, ps.mha, st.mha)
    (o, scores) = res
    
    g, st_lo = l.linear_out(s, ps.linear_out, st.linear_out)
    y = o .* g
    
    st_final = (layer_norm_a_q=st_q, layer_norm_a_k=st_k, linear_z=st_lz, mha=st_mha, linear_out=st_lo)
    return ((y::Array{T,3}, scores::Array{T,5}), st_final)
end

# Internal Core Attention Logic - Dispatch based on local_mode
@inline function _run_core_attention(l::CrossedAttentionPairBias{False}, a_q_ln, a_k_ln, bias_z, mask, ps, st)
    T = eltype(a_q_ln)
    N, B = size(a_q_ln, 2), size(a_q_ln, 3)
    
    # Global Branch
    mha_bias = if isnothing(mask)
        bias_z
    else
        b_mask = reshape((one(T) .- T.(mask)) .* T(-1e9), 1, 1, N, B)
        bias_z .+ b_mask
    end
    (res, st_new) = l.mha((x=(a_q_ln, a_k_ln, a_k_ln), bias=mha_bias, mask=nothing), ps, st)
    (o, scores_raw) = res
    scores = reshape(scores_raw, size(scores_raw, 1), size(scores_raw, 2), size(scores_raw, 3), 1, B)
    return (o, scores), st_new
end

@inline function _run_core_attention(l::CrossedAttentionPairBias{True}, a_q_ln, a_k_ln, bias_z, mask, ps, st)
    T = eltype(a_q_ln)
    N, B = size(a_q_ln, 2), size(a_q_ln, 3)
    nq, nk = l.n_query::Int, l.n_key::Int
    
    # Local Branch
    a_q_blocked = pad_and_block(a_q_ln, nq; dims=2)
    a_k_blocked = pad_and_block(a_k_ln, nk; dims=2)
    
    z_blocked = pad_and_block(bias_z, (nq, nk); dims=(2, 3))
    nb = size(z_blocked, 5)
    b_z = stack([view(z_blocked, :, :, i, :, i, :) for i in 1:nb]; dims=4)
    
    mha_bias_p = permutedims(b_z, (3, 2, 1, 4, 5))
    mha_bias = if isnothing(mask)
        mha_bias_p
    else
        m_blocked_q = pad_and_block(mask, nq; dims=1)
        m_blocked_k = pad_and_block(mask, nk; dims=1)
        # 2D Mask: mask[q, k] = mask[q] * mask[k]
        b_mask = (one(T) .- (T.(reshape(m_blocked_q, 1, nq, nb, B)) .* T.(reshape(m_blocked_k, nk, 1, nb, B)))) .* T(-1e9)
        mha_bias_p .+ reshape(b_mask, nk, nq, 1, nb, B)
    end
    
    (res, st_new) = l.mha((x=(a_q_blocked, a_k_blocked, a_k_blocked), bias=mha_bias, mask=nothing), ps, st)
    (o_blocked, scores_l) = res
    o = unblock_and_slice(o_blocked, N; dims=2)
    return (o, scores_l), st_new
end
