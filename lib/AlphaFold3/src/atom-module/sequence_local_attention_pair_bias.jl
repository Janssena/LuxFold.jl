"""
    SequenceLocalAttentionPairBias(chn_q, chn_k, chn_v, chn_cond, chn_z, head_dim, num_heads,
                                   n_query, n_key; use_adaln=true, inf=1f9)

Sequence-local atom attention with pair bias (AF3 Algorithm 24, the cross/local variant).
A 1:1 port of `openfold3.core.model.layers.attention_pair_bias.CrossAttentionPairBias`.

Unlike LuxFoldCore's `AttentionPairBias` (global) and `CrossedAttentionPairBias`
(non-overlapping block-diagonal), this implements openfold's **centered-window** local
attention: each query block of `n_query` atoms attends to an `n_key`-wide key window
**centered** on it (pulling in neighbouring atoms). Query/key/conditioning are blocked
with the AF3 `convert_single_rep_to_blocks` (centered windows); the pair bias comes from
the **pre-blocked** `atom_pair [chn_z, n_query, n_key, N_blocks, B]`.

The field layout mirrors `CrossedAttentionPairBias` so the existing
`sync_af3_cross_attention_pair_bias!` test helper applies directly.

# Inputs
- `a [chn_q, N_atom, B]` — atom single representation
- `atom_pair [chn_z, n_query, n_key, N_blocks, B]` — pre-blocked atom pair (bias source)
- `cond [chn_cond, N_atom, B]` — single conditioning for AdaLN (when `use_adaln`)
- `mask [N_atom, B]` — `AbstractArray{Bool}` padding mask (optional)

# Returns
- `a_out [chn_q, N_atom, B]`, updated `st`.
"""
struct SequenceLocalAttentionPairBias{LNAQ,LNAK,LZ,MHA,LO} <: Lux.AbstractLuxContainerLayer{(:layer_norm_a_q, :layer_norm_a_k, :linear_z, :mha, :linear_out)}
    n_query::Int
    n_key::Int
    inf::Float32
    layer_norm_a_q::LNAQ
    layer_norm_a_k::LNAK
    linear_z::LZ
    mha::MHA
    linear_out::LO
end

function SequenceLocalAttentionPairBias(
    chn_q::Int, chn_k::Int, chn_v::Int, chn_cond::Int, chn_z::Int,
    head_dim::Int, num_heads::Int, n_query::Int, n_key::Int;
    use_adaln::Bool=true, inf=1f9,
)
    if use_adaln
        layer_norm_a_q = AdaLN(chn_q => chn_cond; rank=3, use_bias=(false, (gate=true, shift=false)))
        layer_norm_a_k = AdaLN(chn_k => chn_cond; rank=3, use_bias=(false, (gate=true, shift=false)))
        linear_out = Lux.Dense(chn_cond => chn_q, Lux.sigmoid; use_bias=true)
    else
        layer_norm_a_q = Lux.LayerNorm((chn_q, 1); dims=1)
        layer_norm_a_k = Lux.LayerNorm((chn_k, 1); dims=1)
        linear_out = Lux.NoOpLayer()
    end

    linear_z = Lux.Dense(chn_z => num_heads; use_bias=false)
    mha = Attention(
        chn_q, chn_k, chn_v, head_dim, num_heads;
        use_gate=true, fuse_qkv=false, use_bias=(false, (q=true,)),
    )

    return SequenceLocalAttentionPairBias(
        n_query, n_key, Float32(inf),
        layer_norm_a_q, layer_norm_a_k, linear_z, mha, linear_out,
    )
end

# AdaLN (use_adaln=true) vs plain-LayerNorm (false) is encoded directly in the
# `layer_norm_a_q` field type — dispatch the helpers on it; no flag field, no forward branch.

function _slapb_norm(l::SequenceLocalAttentionPairBias{<:AdaLN}, a, cond, ps, st)
    aq, st_q = l.layer_norm_a_q(a, cond, ps.layer_norm_a_q, st.layer_norm_a_q)
    ak, st_k = l.layer_norm_a_k(a, cond, ps.layer_norm_a_k, st.layer_norm_a_k)
    return aq, ak, st_q, st_k
end

function _slapb_norm(l::SequenceLocalAttentionPairBias{<:Lux.LayerNorm}, a, cond, ps, st)
    aq, st_q = l.layer_norm_a_q(a, ps.layer_norm_a_q, st.layer_norm_a_q)
    ak, st_k = l.layer_norm_a_k(a, ps.layer_norm_a_k, st.layer_norm_a_k)
    return aq, ak, st_q, st_k
end

# AdaLN-Zero output gate (AdaLN variant) vs identity (plain LayerNorm variant)
function _slapb_gate(l::SequenceLocalAttentionPairBias{<:AdaLN}, a_out, cond, ps, st)
    g, st_lo = l.linear_out(cond, ps.linear_out, st.linear_out)
    return a_out .* g, st_lo
end
_slapb_gate(::SequenceLocalAttentionPairBias{<:Lux.LayerNorm}, a_out, cond, ps, st) = (a_out, st.linear_out)

# NamedTuple dispatch — keys (`x`, `z`, `cond`, `mask`) match LuxFoldCore's
# `AttentionPairBias`, so `DiffusionTransformerBlock` can thread one NamedTuple to either.
(l::SequenceLocalAttentionPairBias)(inputs::NamedTuple, ps, st) = l(
    inputs.x, 
    inputs.z, 
    get(inputs, :cond, nothing), 
    get(inputs, :mask, nothing), 
    ps, st
)

function (l::SequenceLocalAttentionPairBias)(a::AbstractArray{T}, atom_pair, cond, mask, ps, st) where T
    Nq, Nk = l.n_query, l.n_key
    N_atom = size(a, 2)

    # --- normalise the full sequence (per-position; equivalent to block-then-norm),
    #     then block with the AF3 centered-window scheme ---
    a_q_norm, a_k_norm, st_q, st_k = _slapb_norm(l, a, cond, ps, st)

    a_q, _, pair_mask = convert_single_rep_to_blocks(a_q_norm, Nq, Nk, mask)   # a_q [chn, Nq, Nb, B]
    _, a_k, _         = convert_single_rep_to_blocks(a_k_norm, Nq, Nk, mask)   # a_k [chn, Nk, Nb, B]
    Nb, B = size(a_q, 3), size(a_q, 4)

    # --- pair bias from the pre-blocked atom_pair + additive mask bias ---
    pair_bias, st_lz = l.linear_z(atom_pair, ps.linear_z, st.linear_z)      # [heads, Nq, Nk, Nb, B]
    bias = permutedims(pair_bias, (3, 2, 1, 4, 5))                           # [Nk, Nq, heads, Nb, B]
    pair_mask_bias = reshape(permutedims(pair_mask, (2, 1, 3, 4)), Nk, Nq, 1, Nb, B)  # Bool [Nk, Nq, 1, Nb, B]
    # Additive mask bias: valid keys keep `bias`, masked keys get `bias - T(l.inf)` (→ softmax
    # weight ≈ 0). We subtract `inf` (1e9) rather than overwrite so a *fully*-masked query window
    # still softmaxes to `softmax(bias)` (shift-invariance cancels the constant) — matching the
    # Python reference `bias + inf*(mask-1)`. `inf` must be 1e9, not `floatmax`: floatmax is so
    # large it absorbs the O(1) `bias` term (`bias - floatmax == -floatmax`), flattening fully-
    # masked windows to uniform and breaking parity.
    # TODO(mask-float16): `T(l.inf)` (1f9) → `Inf16` at Float16, so `bias - Inf16 = -Inf16` and a
    # fully-masked window NaNs. The Python reference is equally broken there, so the Float16+mask
    # parity config is skipped; Float16 no-mask (no fully-masked windows) is fine.
    bias = @. ifelse(pair_mask_bias, bias, bias - T(l.inf))             # valid: bias, masked: ≈ -inf

    # --- per-block multi-head attention ---
    o, st_mha = l.mha((x=(a_q, a_k, a_k), bias=bias, mask=nothing), ps.mha, st.mha)  # [chn, Nq, Nb, B]
    a_out = unblock_and_slice(o, N_atom; dims=2)                        # [chn, N_atom, B]

    # --- AdaLN-Zero output gate (dispatched; identity when use_adaln=false) ---
    a_out, st_lo = _slapb_gate(l, a_out, cond, ps, st)

    st_final = merge(st, (;
        layer_norm_a_q=st_q, layer_norm_a_k=st_k,
        linear_z=st_lz, mha=st_mha, linear_out=st_lo,
    ))
    return a_out, st_final
end
