"""
    StructureModuleTransitionLayer(chn_s; use_bias=true)

A single residual block inside `StructureModuleTransition`. Matches
`openfold.model.structure_module.StructureModuleTransitionLayer`:
3 channel-preserving `Dense(chn_s => chn_s)` layers (the first two with relu),
followed by a residual addition.

# Arguments
- `chn_s`: Single representation channel dimension.

# Keyword Arguments
- `use_bias`: bias config; resolved over `:linear_1`, `:linear_2`, `:linear_3`.

# Inputs
- `s`: `[chn_s, N, B]`

# Returns
- `s_out`: `[chn_s, N, B]` — `s_initial + linear_3(relu(linear_2(relu(linear_1(s_initial)))))`
- `st`: Updated state.
"""
struct StructureModuleTransitionLayer{C} <: Lux.AbstractLuxContainerLayer{(:chain,)}
    chain::C
end

function StructureModuleTransitionLayer(chn_s::Int; use_bias=true)
    use_bias = resolve_defaults(use_bias, (:linear_1, :linear_2, :linear_3))
    return StructureModuleTransitionLayer(
        Lux.Chain(
            Lux.Dense(chn_s => chn_s, Lux.relu; use_bias=use_bias.linear_1),
            Lux.Dense(chn_s => chn_s, Lux.relu; use_bias=use_bias.linear_2),
            Lux.Dense(chn_s => chn_s; use_bias=use_bias.linear_3),
        )
    )
end

function (l::StructureModuleTransitionLayer)(s::AbstractArray, ps, st)
    s_update, chain = l.chain(s, ps.chain, st.chain)
    return s_update .+ s, merge(st, (; chain, ))
end


"""
    StructureModuleTransition(chn_s; num_layers=1, use_bias=true)

Stack of `num_layers` `StructureModuleTransitionLayer` blocks followed by a final
`LayerNorm`. Matches `openfold.model.structure_module.StructureModuleTransition`
*minus* the dropout layer (inference-only design — we don't include dropout).

The `LayerNorm` lives at the **end**, not the start. This is intentional and
differs from the existing `Transition` layer used by MSA/Pair transitions — the
two cannot be reused for each other.

# Arguments
- `chn_s`: Single representation channel dimension.

# Keyword Arguments
- `num_layers`: Number of `StructureModuleTransitionLayer` blocks (default 1, matches openfold).
- `use_bias`: bias config; resolved over `:layers`, `:layer_norm`.
- `epsilon`: LayerNorm epsilon (default: `1f-5`).

# Inputs
- `s`: `[chn_s, N, B]`

# Returns
- `s_out`: `[chn_s, N, B]`
- `st`: Updated state.
"""
struct StructureModuleTransition{L,LN} <: Lux.AbstractLuxContainerLayer{(:layers, :layer_norm)}
    layers::L
    layer_norm::LN
end

function StructureModuleTransition(chn_s::Int; num_layers::Int=1, use_bias=true, epsilon::Real=1f-5)
    use_bias = resolve_defaults(use_bias, (:layers, :layer_norm))
    layer_use_bias = use_bias.layers
    layers = Lux.Chain(
        ntuple(_ -> StructureModuleTransitionLayer(chn_s; use_bias=layer_use_bias), num_layers)...
    )
    # Trailing 1 lets the LayerNorm broadcast over [chn_s, N, B].
    ln_shape = (chn_s, 1)
    layer_norm = if use_bias.layer_norm
        Lux.LayerNorm(ln_shape; dims=1, epsilon)
    else
        LayerNormNoBias(ln_shape; dims=1, epsilon)
    end
    return StructureModuleTransition(layers, layer_norm)
end

function (l::StructureModuleTransition)(s::AbstractArray, ps, st)
    s, st_layers = l.layers(s, ps.layers, st.layers)
    s, st_ln = l.layer_norm(s, ps.layer_norm, st.layer_norm)
    return s, merge(st, (; layers=st_layers, layer_norm=st_ln))
end

# NamedTuple-friendly dispatch
(l::StructureModuleTransition)(inputs::NamedTuple, ps, st) = l(inputs.s, ps, st)
