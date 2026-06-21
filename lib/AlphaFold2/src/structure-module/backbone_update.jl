"""
    BackboneUpdate(chn_s; use_bias=true)

Predicts a 6-vector backbone update from the single representation: 3 entries
form the imaginary part of a unit-ish quaternion (`(1, x, y, z)`) and 3 are a
translation. Matches `openfold.model.structure_module.BackboneUpdate`
(`Linear(chn_s, 6, init="final")` — no activation).

# Arguments
- `chn_s`: Single representation channel dimension.

# Keyword Arguments
- `use_bias`: `Bool` or `NamedTuple` — passed through to the internal `Dense`.

# Inputs
- `s`: `[chn_s, N, B]` — single representation.

# Returns
- `update`: `[6, N, B]` — raw update vector (first 3 entries are quaternion
  imaginary parts, last 3 are translation).
- `st`: Updated state.
"""
struct BackboneUpdate{L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
end

function BackboneUpdate(chn_s::Int; use_bias=true)
    use_bias = resolve_defaults(use_bias, (:linear,))
    return BackboneUpdate(Lux.Dense(chn_s => 6; use_bias=use_bias.linear))
end

function (l::BackboneUpdate)(s::AbstractArray, ps, st)
    update, linear_st = l.linear(s, ps.linear, st.linear)
    return update, merge(st, (; linear=linear_st))
end

# NamedTuple-friendly dispatch
(l::BackboneUpdate)(inputs::NamedTuple, ps, st) = l(inputs.s, ps, st)
