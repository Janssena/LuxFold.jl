struct Attention{SG,QKV,G,O} <: Lux.AbstractLuxContainerLayer{(:qkv, :gate, :out)}
    should_gate::SG
    qkv::QKV
    gate::G
    out::O
    head_dim::Int
    num_heads::Int # H
end

Attention(chn_in::Int, head_dim::Int, num_heads::Int; kwargs...) =
    Attention(chn_in, chn_in, chn_in, head_dim, num_heads; kwargs...)

"""
Ideally we combine with TriAttnCore at some point, as they do exactly the same aside 
from the triangle_attention vs Lux.scaled_dot_product_attention calls.
"""
function Attention(
    chn_q::Int, chn_k::Int, chn_v::Int, head_dim::Int, num_heads::Int;
    use_bias=false, fuse_qkv::Bool=true, use_gate=static(true)
)
    use_bias = resolve_defaults(use_bias, (:qkv, :gate, :out))

    use_gate_static = static(use_gate)

    gate = if known(use_gate_static)
        Lux.Dense(chn_q => num_heads * head_dim, Lux.sigmoid; use_bias=use_bias.gate)
    else
        Lux.NoOpLayer()
    end

    if fuse_qkv
        fuse_all = chn_q == chn_k == chn_v
        fuse_kv = (chn_q !== chn_k) && (chn_k == chn_v)

        if fuse_all
            qkv = Lux.Dense(chn_q => 3 * num_heads * head_dim)
        elseif fuse_kv
            qkv = Lux.Chain(
                q = Lux.Dense(chn_q => num_heads * head_dim; use_bias=use_bias.qkv),
                kv = Lux.Dense(chn_k => 2 * num_heads * head_dim; use_bias=use_bias.qkv)
            )
        else
            throw(ErrorException("When fuse_qkv = true, the inputs channels for q, k, and v should either all match, or k and v should be equal."))
        end
    else
        qkv = Lux.BranchLayer(
            q = Lux.Dense(chn_q => num_heads * head_dim; use_bias=use_bias.qkv),
            k = Lux.Dense(chn_k => num_heads * head_dim; use_bias=use_bias.qkv),
            v = Lux.Dense(chn_v => num_heads * head_dim; use_bias=use_bias.qkv)
        )
    end

    return Attention(
        use_gate_static,
        qkv,
        gate,
        Lux.Dense(num_heads * head_dim => chn_q; use_bias=use_bias.out),
        head_dim,
        num_heads
    )
end


(l::Attention)(inputs::NamedTuple, ps, st) = l(
    inputs.x, # Either a tuple or an AbstractArray
    get(inputs, :bias, nothing),
    get(inputs, :mask, nothing),
    ps, st
)

(l::Attention)(x::Union{<:Tuple,AbstractArray{T}}, ps, st) where T = l(x, nothing, nothing, ps, st)
(l::Attention)(x::Union{<:Tuple,AbstractArray{T}}, bias::AbstractArray{T}, ps, st) where T =
    l(x, bias, nothing, ps, st)

(l::Attention)(x::Union{<:Tuple,AbstractArray{T}}, mask::AbstractArray{Bool}, ps, st) where T =
    l(x, nothing, mask, ps, st)

# x is either [C, N, B] or [C, N, S, B]
function (l::Attention)(x, bias, mask, ps, st)
    (q, k, v), st_qkv = _prep_qkv(l.qkv, x, ps.qkv, st.qkv; head_dim=l.head_dim, num_heads=l.num_heads)
    mask, bias = _prep_mask(mask), _prep_bias(bias, q)
    
    # S is the MSA dimension
    attn, scores = Lux.scaled_dot_product_attention(
        q, k, v; # [head_dim, H, N, B] or [head_dim, H, N, S, B]
        head_dim=1, token_dim=3,
        mask, # [N, 1, 1, B] or [N, 1, 1, S, B]
        bias # [N, N, H, B] or [N, N, H, 1, B]
    ) # attn is [head_dim, H, N, B] or [head_dim, H, N, S, B]
        
    _dims = size(attn)[3:end] # [N, B] or [N, S, B] dims
    attn = reshape(attn, l.head_dim * l.num_heads, _dims...)
    attn, st_gate = _gate_maybe(l.gate, attn, x, ps.gate, st.gate)

    y, st_out = l.out(attn, ps.out, st.out)

    return (y, scores), (qkv=st_qkv, gate=st_gate, out=st_out)
end

# fused qkv
function _prep_qkv(qkv::Lux.Dense, x::AbstractArray, ps, st; head_dim, num_heads)
    _qkv, st_qkv = qkv(x, ps, st)# [3 * H * head_dim, N, B] or [3 * H * head_dim, N, N, B]

    _qkv_reshaped = reshape(_qkv, head_dim, num_heads, 3, size(_qkv)[2:end]...)
    q = view(_qkv_reshaped, :, :, 1, ntuple(_ -> Colon(), ndims(_qkv_reshaped)-3)...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    k = view(_qkv_reshaped, :, :, 2, ntuple(_ -> Colon(), ndims(_qkv_reshaped)-3)...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    v = view(_qkv_reshaped, :, :, 3, ntuple(_ -> Colon(), ndims(_qkv_reshaped)-3)...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    return (q, k, v), st_qkv
end

function _prep_qkv(qkv::Lux.Chain, x::Tuple, ps, st; head_dim, num_heads)
    x_q, x_kv = x
    q, st_q = qkv.q(x_q, ps.qkv.q, st.qkv.q) # [H * head_dim, N, B]
    _kv, st_kv = qkv.kv(x_kv, ps.qkv.kv, st.qkv.kv) # [2 * H * head_dim, N, B]

    _kv_reshaped = reshape(_kv, head_dim, num_heads, 2, size(_kv)[2:end]...)
    q = reshape(q, head_dim, num_heads, size(q)[2:end]...) # [H, head_dim, N, B]
    k = view(_kv_reshaped, :, :, 1, ntuple(_ -> Colon(), ndims(_kv_reshaped)-3)...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    v = view(_kv_reshaped, :, :, 2, ntuple(_ -> Colon(), ndims(_kv_reshaped)-3)...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    return (q, k, v), (q=st_q, kv=st_kv)
end

# nonfused qkv
function _prep_qkv(qkv::Lux.BranchLayer, x::AbstractArray, ps, st; head_dim, num_heads)
    (q, k, v), st_qkv = qkv(x, ps, st)

    q = reshape(q, head_dim, num_heads, size(q)[2:end]...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    k = reshape(k, head_dim, num_heads, size(k)[2:end]...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    v = reshape(v, head_dim, num_heads, size(v)[2:end]...) # [H, head_dim, N, B] or [H, head_dim, N, N, B]
    return (q, k, v), st_qkv
end

function _prep_qkv(qkv::Lux.BranchLayer, x::Tuple, ps, st; head_dim, num_heads)
    x_q, x_k, x_v = x
    q, st_q = qkv.q(x_q, ps.qkv.q, st.qkv.q)
    k, st_k = qkv.q(x_k, ps.qkv.k, st.qkv.k)
    v, st_v = qkv.q(x_v, ps.qkv.v, st.qkv.v)

    q = reshape(q, head_dim, num_heads, size(q)[2:end]...) # [head_dim, H, N, B]
    k = reshape(k, head_dim, num_heads, size(k)[2:end]...) # [head_dim, H, N, B]
    v = reshape(v, head_dim, num_heads, size(v)[2:end]...) # [head_dim, H, N, B]
    return (q, k, v), (q=st_q, k=st_k, v=st_v)
end

_prep_mask(::Nothing) = nothing
function _prep_mask(mask::AbstractArray{T,2}) where T
    N, B = size(mask)
    return reshape(mask, N, 1, 1, B)
end

function _prep_mask(mask::AbstractArray{T,3}) where T
    N, S, B = size(mask)
    return reshape(mask, N, 1, 1, S, B)
end

_prep_bias(::Nothing) = nothing
function _prep_bias(bias::AbstractArray{T,3}) where T
    H, N, B = size(bias)
    return reshape(permutedims(bias, (2, 1, 3)), 1, N, H, B)
end

function _prep_bias(bias::AbstractArray{T,4}, ::AbstractArray{T,4}) where T
    # bias: [H, Ni, Nj, B] -> q, k, v are 4D
    return permutedims(bias, (3, 2, 1, 4)) # -> [Nj, Ni, H, B]
end

function _prep_bias(bias::AbstractArray{T,4}, ::AbstractArray{T,5}) where T
    # bias: [H, Ni, Nj, B] -> q, k, v are 5D
    H, Ni, Nj, B = size(bias)
    return reshape(permutedims(bias, (3, 2, 1, 4)), Nj, Ni, H, 1, B) # -> [Nj, Ni, H, 1, B]
end

_gate_maybe(::Lux.NoOpLayer, x, g, ps, st) = x, st
function _gate_maybe(l, x, g, ps, st)
    g, st_gate = l(g, ps, st)
    y = @. x * g
    return y, st_gate
end
