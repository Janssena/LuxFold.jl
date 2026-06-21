"""
    MSAColumnAttention(chn_m, chn_hidden, no_heads; use_bias=true, epsilon=1f-5)

Column-wise MSA attention (Algorithm 8) — transposes the MSA and runs row-style
attention without pair bias. Uses a raw `LayerNorm + Attention` sub-layer (NOT
`AttentionPairBias`, which always creates dead `linear_z`/`layer_norm_z` params).

The transpose pattern handles the axis permutation internally; the caller always
provides `m` in the standard `[C, N_res, N_seq, B]` layout (N_res = residues, N_seq = sequences).
The internal transpose puts N_seq at the token dimension so `Attention` attends over sequences.

# Arguments
- `chn_m`: MSA channel dimension
- `chn_hidden`: Per-head hidden (attention head) dimension
- `no_heads`: Number of attention heads

# Keyword Arguments
- `use_bias`: `Bool` or `NamedTuple` with keys `(:layer_norm, :mha)` (default: `true`)
- `epsilon`: LayerNorm epsilon (default: `1f-5`)

# Inputs
- `m`: MSA tensor `[chn_m, N_res, N_seq, B]`
- `mask`: Optional Bool mask `[N_res, N_seq, B]`, or `nothing`

# Returns
- `m_out`: Updated MSA tensor `[chn_m, N_res, N_seq, B]` (attention update only, no residual)
- `st`: Updated state
"""
struct MSAColumnAttention{LN, MHA} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :mha)}
    layer_norm::LN
    mha::MHA
end

# Per-layer bias helper: for MSA attention, QKV has no bias (matching Python),
# but gate and output projections do.
_msa_att_use_bias(b::Bool) =
    b ? (qkv=false, gate=true, out=true) : false
_msa_att_use_bias(b) = b  # NamedTuple override → pass through

function MSAColumnAttention(chn_m::Int, chn_hidden::Int, no_heads::Int; use_bias=true, epsilon::Real=1f-5)
    use_bias = resolve_defaults(use_bias, (:layer_norm, :mha))

    shape = (chn_m, 1, 1)
    layer_norm = if use_bias.layer_norm
        Lux.LayerNorm(shape; dims=1, epsilon)
    else
        LayerNormNoBias(shape; dims=1, epsilon)
    end

    mha = Attention(chn_m, chn_hidden, no_heads;
        use_gate=static(true),
        use_bias=_msa_att_use_bias(use_bias.mha)
    )
    return MSAColumnAttention(layer_norm, mha)
end

(l::MSAColumnAttention)(m, ps, st) = l(m, nothing, ps, st)

(l::MSAColumnAttention)(inputs::NamedTuple, ps, st) = l(
    inputs.m,
    get(inputs, :mask, get(inputs, :msa_mask, nothing)),
    ps, st
)

# No-mask dispatch: Attention called without mask for type stability
function (l::MSAColumnAttention)(m, ::Nothing, ps, st)
    # Transpose: [C, N_res, N_seq, B] → [C, N_seq, N_res, B]
    m_t = permutedims(m, (1, 3, 2, 4))
    m_norm, st_ln  = l.layer_norm(m_t, ps.layer_norm, st.layer_norm)
    m_attn, st_mha = l.mha(m_norm, ps.mha, st.mha)
    m_out = permutedims(m_attn, (1, 3, 2, 4))
    return m_out, merge(st, (; layer_norm=st_ln, mha=st_mha))
end

# Masked dispatch: transpose mask [N_res,N_seq,B] → [N_seq,N_res,B] before Attention
function (l::MSAColumnAttention)(m, mask::AbstractArray{Bool}, ps, st)
    # Transpose: [C, N_res, N_seq, B] → [C, N_seq, N_res, B]
    # Puts N_seq at dim 2 so Attention attends over sequences (column attention).
    m_t    = permutedims(m, (1, 3, 2, 4))
    mask_t = permutedims(mask, (2, 1, 3))  # [N_res,N_seq,B] → [N_seq,N_res,B]

    # LayerNorm then self-attention along N_seq (column attention)
    m_norm, st_ln  = l.layer_norm(m_t, ps.layer_norm, st.layer_norm)
    m_attn, st_mha = l.mha(m_norm, mask_t, ps.mha, st.mha)

    # Transpose back: [C, N_seq, N_res, B] → [C, N_res, N_seq, B]
    m_out = permutedims(m_attn, (1, 3, 2, 4))

    return m_out, merge(st, (; layer_norm=st_ln, mha=st_mha))
end

# =============================================================================

"""
    MSAColumnGlobalAttention(chn_in, chn_hidden, no_heads; use_bias=true, epsilon=1f-5, eps=1f-8)

Global column-wise MSA attention — efficient large-MSA variant that replaces per-sequence
queries with a single global query (weighted average over sequences per residue).
Used in `ExtraMSABlock` instead of standard `MSAColumnAttention`.

K/V projections are shared across all heads (`chn_in → chn_hidden`, no head expansion),
while Q is projected to `chn_in → chn_hidden * no_heads`. The result is gated and projected
back to `chn_in`. The shared K/V heads are broadcast to all Q heads via grouped KV (GQA)
inside `Lux.scaled_dot_product_attention`.

The transpose pattern is the same as `MSAColumnAttention`: input `[C, N_res, N_seq, B]`
is transposed to `[C, N_seq, N_res, B]` internally so that N_seq is the attended dimension.

# Arguments
- `chn_in`: MSA channel dimension (ExtraMSA: 64)
- `chn_hidden`: Per-head hidden dimension (ExtraMSA: 8)
- `no_heads`: Number of attention heads (ExtraMSA: 8)

# Keyword Arguments
- `use_bias`: `Bool` or `NamedTuple` with keys `(:layer_norm, :linear_q, :linear_k, :linear_v, :linear_g, :linear_o)` (default: `true`, but Q/K/V default to no-bias matching Python)
- `epsilon`: LayerNorm epsilon (default: `1f-5`)
- `eps`: Small constant for numerical stability in global average (default: `1f-8`)

# Inputs
- `m`: MSA tensor `[chn_in, N_res, N_seq, B]`
- `mask`: Bool mask `[N_res, N_seq, B]`, or `nothing`

# Returns
- `m_out`: Updated MSA tensor `[chn_in, N_res, N_seq, B]` (attention update only, no residual)
- `st`: Updated state
"""
struct MSAColumnGlobalAttention{LN, LQ, LK, LV, LG, LO} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :linear_q, :linear_k, :linear_v, :linear_g, :linear_o)}
    layer_norm::LN
    linear_q::LQ
    linear_k::LK
    linear_v::LV
    linear_g::LG
    linear_o::LO
    chn_hidden::Int
    no_heads::Int
    eps::Float32
end

# Python convention: Q/K/V have no bias (init="glorot" with bias=False);
# gate has bias (init="gating"); output has bias (init="final").
_global_att_use_bias(b::Bool) = b ? (
    layer_norm=true,
    linear_q=false, linear_k=false, linear_v=false,
    linear_g=true, linear_o=true
) : (
    layer_norm=false,
    linear_q=false, linear_k=false, linear_v=false,
    linear_g=false, linear_o=false
)
_global_att_use_bias(b) = b

function MSAColumnGlobalAttention(
    chn_in::Int, chn_hidden::Int, no_heads::Int;
    use_bias=true, epsilon::Real=1f-5, eps=1f-8
)
    ub = _global_att_use_bias(use_bias)

    shape = (chn_in, 1, 1)
    layer_norm = ub.layer_norm ? Lux.LayerNorm(shape; dims=1, epsilon) :
                                 LayerNormNoBias(shape; dims=1, epsilon)

    linear_q = Lux.Dense(chn_in => chn_hidden * no_heads; use_bias=ub.linear_q)
    linear_k = Lux.Dense(chn_in => chn_hidden;            use_bias=ub.linear_k)
    linear_v = Lux.Dense(chn_in => chn_hidden;            use_bias=ub.linear_v)
    linear_g = Lux.Dense(chn_in => chn_hidden * no_heads, Lux.sigmoid; use_bias=ub.linear_g)
    linear_o = Lux.Dense(chn_hidden * no_heads => chn_in; use_bias=ub.linear_o)

    return MSAColumnGlobalAttention(
        layer_norm, linear_q, linear_k, linear_v, linear_g, linear_o,
        chn_hidden, no_heads, Float32(eps)
    )
end

# Global query: weighted mean of m [C, N_seq, N_res, B] over N_seq (dim 2).
# mask is [N_seq, N_res, B] (already transposed) or nothing.
# With mask: masked sum / count. Without mask: plain sum / N_seq.
_msa_global_query(m::AbstractArray{T}, ::Nothing, eps) where T =
    dropdims(sum(m; dims=2); dims=2) ./ (T(size(m, 2)) + T(eps))

function _msa_global_query(m::AbstractArray{T}, mask::AbstractArray{Bool}, eps) where T
    _zero = zero(T)
    mask_4d  = reshape(mask, 1, size(mask)...)                            # [1,N_seq,N_res,B]
    mask_sum = reshape(T.(sum(mask; dims=1)), 1, 1, size(mask, 2), size(mask, 3))  # [1,1,N_res,B]
    return dropdims(sum(ifelse.(mask_4d, m, _zero); dims=2) ./ (mask_sum .+ T(eps)); dims=2)
end

# SDPA mask: reshape [N_seq, N_res, B] → [N_seq, 1, 1, NB], or pass nothing.
_prep_msa_sdpa_mask(::Nothing, N_seq, NB) = nothing
_prep_msa_sdpa_mask(mask::AbstractArray{Bool}, N_seq, NB) = reshape(mask, N_seq, 1, 1, NB)

# Shared implementation. m and mask are already in [C, N_seq, N_res, B] layout.
function _msa_global_attention(l::MSAColumnGlobalAttention, m::AbstractArray{T}, mask, ps, st) where T
    C, N_seq, N_res, B_sz = size(m)
    NB = N_res * B_sz

    m, st_ln = l.layer_norm(m, ps.layer_norm, st.layer_norm)

    # --- Global query: (weighted) mean over N_seq ---
    q = _msa_global_query(m, mask, l.eps)                             # [C, N_res, B]

    # Q projection → reshape for SDPA: [C_h*H, N_res, B] → [C_h, H, 1, NB]
    # (1 global query token per residue-batch pair; NB = N_res * B as batch)
    q_proj, st_lq = l.linear_q(q, ps.linear_q, st.linear_q)          # [C_h*H, N_res, B]
    q_proj = reshape(q_proj, l.chn_hidden, l.no_heads, 1, NB)           # [C_h, H, 1, NB]

    # K/V projections (shared across heads): [C_h, N_seq, N_res, B] → [C_h, 1, N_seq, NB]
    # Memory layout of [C_h, N_seq, N_res, B] is identical to [C_h, 1, N_seq, NB] —
    # no permutation needed; GQA in SDPA broadcasts the singleton head to H.
    k, st_lk = l.linear_k(m, ps.linear_k, st.linear_k)               # [C_h, N_seq, N_res, B]
    v, st_lv = l.linear_v(m, ps.linear_v, st.linear_v)               # [C_h, N_seq, N_res, B]
    k = reshape(k, l.chn_hidden, 1, N_seq, NB)                          # [C_h, 1, N_seq, NB]
    v = reshape(v, l.chn_hidden, 1, N_seq, NB)                          # [C_h, 1, N_seq, NB]

    # SDPA: default scale = 1/√chn_hidden; GQA repeats K/V across H heads internally
    o, _ = Lux.scaled_dot_product_attention(q_proj, k, v;
        head_dim=1, token_dim=3, mask=_prep_msa_sdpa_mask(mask, N_seq, NB))
    # o: [C_h, H, 1, NB] → [C_h, H, 1, N_res, B] for gating broadcast over N_seq
    o = reshape(o, l.chn_hidden, l.no_heads, 1, N_res, B_sz)

    # --- Gating ---
    g, st_lg = l.linear_g(m, ps.linear_g, st.linear_g)               # [C_h*H, N_seq, N_res, B]
    g = reshape(g, l.chn_hidden, l.no_heads, N_seq, N_res, B_sz)        # [C_h, H, N_seq, N_res, B]
    o = reshape(o .* g, l.chn_hidden * l.no_heads, N_seq, N_res, B_sz)  # [C_h*H, N_seq, N_res, B]

    # Output projection, then transpose back to [C, N_res, N_seq, B]
    m_out, st_lo = l.linear_o(o, ps.linear_o, st.linear_o)            # [C, N_seq, N_res, B]
    m_out = permutedims(m_out, (1, 3, 2, 4))                           # [C, N_res, N_seq, B]

    st_new = merge(st, (;
        layer_norm=st_ln, linear_q=st_lq, linear_k=st_lk,
        linear_v=st_lv, linear_g=st_lg, linear_o=st_lo
    ))
    return m_out, st_new
end

(l::MSAColumnGlobalAttention)(m, ps, st) = l(m, nothing, ps, st)

(l::MSAColumnGlobalAttention)(inputs::NamedTuple, ps, st) = l(
    inputs.m,
    get(inputs, :mask, get(inputs, :msa_mask, nothing)),
    ps, st
)

function (l::MSAColumnGlobalAttention)(m, ::Nothing, ps, st)
    return _msa_global_attention(l, permutedims(m, (1, 3, 2, 4)), nothing, ps, st)
end

function (l::MSAColumnGlobalAttention)(m, mask::AbstractArray{Bool}, ps, st)
    m    = permutedims(m,    (1, 3, 2, 4))
    mask = permutedims(mask, (2, 1, 3))    # [N_res, N_seq, B] → [N_seq, N_res, B]
    return _msa_global_attention(l, m, mask, ps, st)
end
