# =============================================================================
# TemplateSingleEmbedder
# =============================================================================

"""
    TemplateSingleEmbedder(chn_in, chn_out; use_bias=true)

Embeds template torsion-angle features into the MSA channel dimension
(Algorithm 2, line 7): `Dense(chn_in → chn_out, relu) → Dense(chn_out → chn_out)`.

# Arguments
- `chn_in`: Input channel dimension (e.g. 57 for monomer angle features)
- `chn_out`: Output channel dimension (usually `c_m` = 256)

# Keyword Arguments
- `use_bias`: `Bool` or `NamedTuple` for per-layer bias control (default: `true`)

# Inputs
- `x`: Template angle features of shape `[chn_in, N_res, N_templ, B]`

# Returns
- `y`: Single template embedding of shape `[chn_out, N_res, N_templ, B]`
- `st`: Updated state
"""
struct TemplateSingleEmbedder{C} <: Lux.AbstractLuxContainerLayer{(:chain,)}
    chain::C
end

function TemplateSingleEmbedder(chn_in::Int, chn_out::Int; use_bias=true)
    use_bias = resolve_defaults(use_bias, (:layer_1, :layer_2))
    return TemplateSingleEmbedder(
        Lux.Chain(
            Lux.Dense(chn_in  => chn_out, Lux.relu; use_bias=use_bias.layer_1),
            Lux.Dense(chn_out => chn_out;            use_bias=use_bias.layer_2),
        )
    )
end

function (l::TemplateSingleEmbedder)(x, ps, st)
    y, st_chain = l.chain(x, ps.chain, st.chain)
    return y, merge(st, (; chain=st_chain))
end

# =============================================================================
# TemplatePairEmbedder
# =============================================================================

"""
    TemplatePairEmbedder(chn_in, chn_out; use_bias=true)

Projects raw template pair features to the template embedding channel dimension
(Algorithm 2, line 9): a single `Dense(chn_in → chn_out)`.

Operates natively on 5D tensors `[chn_in, N_res, N_res, N_templ, B]`; all templates
are projected in parallel.

# Arguments
- `chn_in`: Input channel dimension (e.g. 88 for monomer pair features)
- `chn_out`: Output channel dimension (usually `c_t` = 64)

# Keyword Arguments
- `use_bias`: Whether to include bias in the projection (default: `true`)

# Inputs
- `x`: Template pair features of shape `[chn_in, N_res, N_res, N_templ, B]`

# Returns
- `y`: Template pair embedding of shape `[chn_out, N_res, N_res, N_templ, B]`
- `st`: Updated state
"""
struct TemplatePairEmbedder{L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
end

function TemplatePairEmbedder(chn_in::Int, chn_out::Int; use_bias=true)
    return TemplatePairEmbedder(Lux.Dense(chn_in => chn_out; use_bias))
end

function (l::TemplatePairEmbedder)(x, ps, st)
    y, st_linear = l.linear(x, ps.linear, st.linear)
    return y, merge(st, (; linear=st_linear))
end

# =============================================================================
# TemplateEmbedder — top-level pipeline
# =============================================================================

# Angle embedding dispatch — type-stable: angle_feat::Nothing returns nothing.
_embed_single(::Any, ::Nothing, ps, st) = nothing, st
function _embed_single(embedder, angle_feat::AbstractArray, ps, st)
    return embedder(angle_feat, ps, st)
end

"""
    TemplateEmbedder(chn_templ_in_pair, chn_templ_in_angles,
                     chn_templ, chn_pair, chn_msa,
                     chn_hidden_tri_att, chn_hidden_tri_mul,
                     no_templ_blocks, no_heads_tri, pair_transition_n,
                     chn_hidden_pt_att, no_heads_pt_att; ...)

Complete template embedding pipeline (Algorithm 2, lines 7–11). Processes pre-built
template features into pair and (optionally) single representations.

All templates are processed in parallel through the pipeline — no loop over `N_templ`.

# Arguments
- `chn_templ_in_pair`: Input pair feature channels (e.g. 88 for monomer)
- `chn_templ_in_angles`: Input angle feature channels (e.g. 57 for monomer)
- `chn_templ`: Template embedding channel dimension (c_t, e.g. 64)
- `chn_pair`: Pair representation channel dimension (c_z, e.g. 128)
- `chn_msa`: MSA representation channel dimension for angle output (c_m, e.g. 256)
- `chn_hidden_tri_att`: Head dimension for triangular attention
- `chn_hidden_tri_mul`: Hidden dimension for triangular multiplication
- `no_templ_blocks`: Number of blocks in the template pair stack
- `no_heads_tri`: Number of attention heads in the pair stack
- `pair_transition_n`: Expansion factor for the pair transition MLP
- `chn_hidden_pt_att`: Hidden dimension per head in pointwise attention
- `no_heads_pt_att`: Number of heads in pointwise attention

# Keyword Arguments
- `tri_mul_first`: Triangle operation order in each stack block (default: `false`)
- `use_bias`: `Bool` or `NamedTuple` for bias control across sub-layers (default: `true`)
- `epsilon`: LayerNorm epsilon (default: `1f-5`)

# Inputs (NamedTuple)
- `pair_feat`: Pre-built pair features `[chn_templ_in_pair, N_res, N_res, N_templ, B]`
- `pair_mask`: Pair mask `[N_res, N_res, B]` Bool (broadcast to all templates internally)
- `template_mask`: Valid-template mask `[N_templ, B]` Bool
- `z`: Pair embedding `[chn_pair, N_res, N_res, B]`
- `angle_feat`: (optional) Pre-built angle features `[chn_templ_in_angles, N_res, N_templ, B]`;
  absent or `nothing` disables angle embedding and returns `template_single=nothing`

# Returns
- `(template_pair, template_single)`: NamedTuple where
  - `template_pair`: Pair embedding update `[chn_pair, N_res, N_res, B]`
  - `template_single`: Angle embedding `[chn_msa, N_res, N_templ, B]`, or `nothing`
- `st`: Updated state
"""
struct TemplateEmbedder{TSE, TPE, TPS, TPA} <:
    Lux.AbstractLuxContainerLayer{(:template_single_embedder, :template_pair_embedder,
                                    :template_pair_stack, :template_pointwise_att)}
    template_single_embedder::TSE
    template_pair_embedder::TPE
    template_pair_stack::TPS
    template_pointwise_att::TPA
end

function TemplateEmbedder(
    chn_templ_in_pair::Int,
    chn_templ_in_angles::Int,
    chn_templ::Int,
    chn_pair::Int,
    chn_msa::Int,
    chn_hidden_tri_att::Int,
    chn_hidden_tri_mul::Int,
    no_templ_blocks::Int,
    no_heads_tri::Int,
    pair_transition_n::Int,
    chn_hidden_pt_att::Int,
    no_heads_pt_att::Int;
    tri_mul_first=false,
    use_bias=true,
    epsilon=1f-5
)
    use_bias = resolve_defaults(
        use_bias,
        (:template_single_embedder, :template_pair_embedder,
         :template_pair_stack, :template_pointwise_att)
    )

    return TemplateEmbedder(
        TemplateSingleEmbedder(chn_templ_in_angles, chn_msa;
            use_bias=use_bias.template_single_embedder),
        TemplatePairEmbedder(chn_templ_in_pair, chn_templ;
            use_bias=use_bias.template_pair_embedder),
        TemplatePairStack(
            chn_templ, chn_hidden_tri_att, chn_hidden_tri_mul,
            no_templ_blocks, no_heads_tri, pair_transition_n;
            tri_mul_first, use_bias=use_bias.template_pair_stack, epsilon
        ),
        TemplatePointwiseAttention(chn_templ, chn_pair, chn_hidden_pt_att, no_heads_pt_att),
    )
end

# NamedTuple dispatch — angle_feat absent → nothing (disables angle embedding).
(l::TemplateEmbedder)(inputs::NamedTuple, ps, st) = l(
    inputs.pair_feat,
    inputs.pair_mask,
    inputs.template_mask,
    inputs.z,
    get(inputs, :angle_feat, nothing),
    ps, st
)

function (l::TemplateEmbedder)(pair_feat, pair_mask, template_mask, z, angle_feat, ps, st)
    N_res = size(pair_mask, 1)
    B     = size(pair_mask, ndims(pair_mask))

    # --- 1. Project pair features: [chn_templ_in_pair, N, N, N_templ, B] → [chn_templ, ...]
    t, st_tpe = l.template_pair_embedder(
        pair_feat, ps.template_pair_embedder, st.template_pair_embedder
    )

    # --- 2. Refine via TemplatePairStack.
    # pair_mask [N, N, B] is shared across all templates; TemplatePairStack expands it
    # internally to [N, N, N_templ*B] when merging the template and batch dimensions.
    t, st_tps = l.template_pair_stack(t, pair_mask, ps.template_pair_stack, st.template_pair_stack)

    # --- 3. Fuse templates into pair via pointwise cross-attention.
    # template_mask [N_templ, B] Bool — passed as mask (not bias) inside TPA.
    t_att, st_tpa = l.template_pointwise_att(
        t, z, template_mask, ps.template_pointwise_att, st.template_pointwise_att
    )

    # --- 4. Zero out pair update when no valid templates exist for a batch element.
    # sum(template_mask; dims=1) > 0 → [1, B]; reshape broadcasts over [chn_pair, N, N, B].
    t_valid = reshape(sum(template_mask; dims=1) .> 0, 1, 1, 1, B)
    t_att   = t_att .* t_valid

    # --- 5. Angle embedding (optional — dispatches on angle_feat type).
    template_single, st_tse = _embed_single(
        l.template_single_embedder, angle_feat,
        ps.template_single_embedder, st.template_single_embedder
    )

    st_new = merge(st, (;
        template_pair_embedder   = st_tpe,
        template_pair_stack      = st_tps,
        template_pointwise_att   = st_tpa,
        template_single_embedder = st_tse,
    ))

    return (; template_pair=t_att, template_single), st_new
end
