struct Attention{SG,QKV,G,O} <: Lux.AbstractLuxContainerLayer{(:qkv,:gate,:out)}
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
        @assert chn_q == chn_k == chn_v "Input channels for q, k, and v should be equal when fuse_qkv=true."
        qkv = Lux.Dense(chn_q => 3 * num_heads * head_dim)
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
    inputs.x, 
    get(inputs, :bias, nothing), 
    get(inputs, :mask, nothing), 
    ps, st
)

(l::Attention)(x::AbstractArray, ps, st) = l(x, nothing, nothing, ps, st)
(l::Attention)(x::AbstractArray{T}, bias::AbstractArray{T}, ps, st) where T = 
    l(x, bias, nothing, ps, st)

(l::Attention)(x::AbstractArray{T}, mask::AbstractArray{Bool}, ps, st) where T = 
    l(x, nothing, mask, ps, st)

function (l::Attention)(x, bias, mask, ps, st)
    _, N, B = size(x)
    (q, k, v), st_qkv = _prep_qkv(l.qkv, x, ps.qkv, st.qkv; head_dim=l.head_dim, num_heads=l.num_heads)
    mask, bias = _prep_mask(mask), _prep_bias(bias)

    attn, scores = Lux.scaled_dot_product_attention(
        q, k, v; # [head_dim, H, N, B]
        head_dim=1, token_dim=3, 
        mask, # [N, 1, 1, B]
        bias # [N, N, H, B]
    )

    attn = reshape(attn, l.head_dim * l.num_heads, N, B)
    attn, st_gate = _gate_maybe(l.gate, attn, x, ps.gate, st.gate)

    y, st_out = l.out(attn, ps.out, st.out)

    return (y, scores), (qkv=st_qkv, gate=st_gate, out=st_out)
end

# fused qkv
function _prep_qkv(qkv::Lux.Dense, x::AbstractArray, ps, st; head_dim, num_heads)
    _qkv, st_qkv = qkv(x, ps, st)# [3 * H * head_dim, N, B]
    
    _qkv_reshaped = reshape(_qkv, head_dim, num_heads, 3, size(_qkv)[2:end]...)
    q = view(_qkv_reshaped, :, :, 1, :, :) # [H, head_dim, N, B]
    k = view(_qkv_reshaped, :, :, 2, :, :) # [H, head_dim, N, B]
    v = view(_qkv_reshaped, :, :, 3, :, :) # [H, head_dim, N, B]
    return (q, k, v), st_qkv
end

# nonfused qkv, the below doesn't work when input channels are different.
function _prep_qkv(qkv::Lux.BranchLayer, x::AbstractArray, ps, st; head_dim, num_heads)
    (q, k, v), st_qkv = qkv(x, ps, st)

    q = reshape(q, head_dim, num_heads, size(q)[2:end]...) # [head_dim, H, N, B]
    k = reshape(k, head_dim, num_heads, size(k)[2:end]...) # [head_dim, H, N, B]
    v = reshape(v, head_dim, num_heads, size(v)[2:end]...) # [head_dim, H, N, B]
    return (q, k, v), st_qkv
end

_prep_mask(::Nothing) = nothing
function _prep_mask(mask::AbstractArray{T,2}) where T 
    N, B = size(mask)
    return reshape(mask, N, 1, 1, B)
end

_prep_bias(::Nothing) = nothing
function _prep_bias(bias::AbstractArray{T,3}) where T 
    H, N, B = size(bias)
    return reshape(permutedims(bias, (2, 1, 3)), 1, N, H, B)
end

function _prep_bias(bias::AbstractArray{T,4}) where T 
    # bias: [H, Ni, Nj, B]
    return permutedims(bias, (3, 2, 1, 4)) # -> [Nj, Ni, H, B]
end

_gate_maybe(::Lux.NoOpLayer, x, g, ps, st) = x, st
function _gate_maybe(l, x, g, ps, st)
    g, st_gate = l(g, ps, st)
    y = @. x * g
    return y, st_gate
end
