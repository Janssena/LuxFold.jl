_prep_template_mask(::Nothing, N_templ, B) = nothing
_prep_template_mask(mask::AbstractArray{Bool}, N_templ, B) =
    reshape(mask, N_templ, 1, 1, 1, B)

# =============================================================================

"""
    TemplatePointwiseAttention(chn_t, chn_z, chn_hidden, no_heads; use_gate=false, use_bias=(false, (out=true,)))

Pointwise cross-attention that fuses template pair embeddings into the pair representation
(Algorithm 17). The pair embedding `z` provides the query (one token per residue pair);
the stacked template embeddings `t` provide keys and values (one token per template).

Wraps a single `Attention` sub-layer. Since `chn_z ≠ chn_t` in general, `Attention` uses
the fused-KV path: `q = Dense(chn_z → H*d)`, `kv = Dense(chn_t → 2*H*d)`.

# Arguments
- `chn_t`: Template pair embedding channel dimension (e.g. 64)
- `chn_z`: Pair embedding channel dimension (e.g. 128)
- `chn_hidden`: Hidden dimension per attention head (e.g. 16)
- `no_heads`: Number of attention heads (e.g. 4)

# Keyword Arguments
- `use_gate`: Whether to use gating in the internal MHA (default: `false`)
- `use_bias`: Bias control forwarded directly to `Attention`, which resolves it via
  `resolve_defaults` internally. The default `(false, (out=true,))` matches openfold:
  no bias on Q/K/V projections, bias on the output projection.

# Inputs
- `t`: Template pair embedding tensor of shape `[chn_t, N_res, N_res, N_templ, B]`
- `z`: Pair embedding tensor of shape `[chn_z, N_res, N_res, B]`
- `template_mask`: Optional Bool mask of shape `[N_templ, B]`;
  `true` = valid template, `false` = invalid (suppressed in attention)

# Returns
- `z_update`: Pair embedding update of shape `[chn_z, N_res, N_res, B]`
- `st`: Updated state
"""
struct TemplatePointwiseAttention{MHA} <: Lux.AbstractLuxContainerLayer{(:mha,)}
    mha::MHA
end

function TemplatePointwiseAttention(
    chn_t::Int, chn_z::Int, chn_hidden::Int, no_heads::Int;
    use_gate=false,
    use_bias=(false, (out=true,))
)
    return TemplatePointwiseAttention(
        Attention(chn_z, chn_t, chn_t, chn_hidden, no_heads; use_gate, use_bias)
    )
end

(l::TemplatePointwiseAttention)(inputs::NamedTuple, ps, st) = l(
    inputs.t, 
    inputs.z, 
    get(inputs, :template_mask, nothing), 
    ps, st
)

function (l::TemplatePointwiseAttention)(t, z, template_mask, ps, st)
    C_z, N_res, _, B = size(z)
    N_templ = size(t, 4)
    N_pair  = N_res * N_res

    # Collapse N_res×N_res into N_pair; z gets a singleton template dim as the Q sequence
    q_x  = reshape(z, C_z, 1, N_pair, B)

    # Permute t: [C_t, Ni, Nj, N_templ, B] → [C_t, N_templ, Ni, Nj, B], then collapse spatial
    kv_x = reshape(permutedims(t, (1, 4, 2, 3, 5)), size(t, 1), N_templ, N_pair, B)

    # Reshape Bool mask [N_templ, B] → [N_templ, 1, 1, 1, B] for broadcast against logits
    # [N_templ, 1, H, N_pair, B]. prep_mask(::5D) returns as-is; nothing skips masking.
    mask = _prep_template_mask(template_mask, N_templ, B)

    y, st_mha = l.mha((q_x, kv_x), nothing, mask, ps.mha, st.mha)

    # Restore spatial layout: [C_z, 1, N_pair, B] → [C_z, N_res, N_res, B]
    z_update = reshape(y, C_z, N_res, N_res, B)

    return z_update, merge(st, (; mha=st_mha))
end
