# ── NamedTuple plumbing ─────────────────────────────────────────────────────────────────────────

# Walk a path of field names into a nested NamedTuple: `_getpath(ps, (:blocks,))` → `ps.blocks`.
_getpath(x, ::Tuple{}) = x
_getpath(x, p::Tuple) = _getpath(getproperty(x, first(p)), Base.tail(p))

# Inverse of `_getpath`: wrap `v` so that `_getpath(_nest(p, v), p) === v`.
_nest(::Tuple{}, v) = v
_nest(p::Tuple, v) = NamedTuple{(first(p),)}((_nest(Base.tail(p), v),))

# Replace the value at `path` inside `x`, keeping every other field. Used to scatter a chunk's
# updated state back into the whole stack's state.
_setpath(_, ::Tuple{}, v) = v
function _setpath(x::NamedTuple{K}, p::Tuple{Symbol,Vararg{Symbol}}, v) where {K}
    h = first(p)
    return NamedTuple{K}(map(k -> k === h ? _setpath(getfield(x, k), Base.tail(p), v) :
                                  getfield(x, k), K))
end

# Keys of `nt` that a gradient CAN flow through. Bool masks, integer index tensors and anything
# non-array are excluded — Enzyme cannot differentiate them, and the alternative to inferring this
# is making every caller declare it. Note this is only the *upper bound*: whether a gradient
# SHOULD flow is a modelling choice the caller makes with `active`, because nothing about a tensor
# says it is meant to be held fixed.
_differentiable(nt::NamedTuple{K}) where {K} =
    Tuple(k for k in K if getproperty(nt, k) isa AbstractArray &&
                          eltype(getproperty(nt, k)) <: AbstractFloat)

function _resolve_active(inputs::NamedTuple, active)
    can = _differentiable(inputs)
    active === nothing && return can
    for k in active
        k in keys(inputs) || throw(ArgumentError("active key $(repr(k)) is not an input"))
        k in can || throw(ArgumentError(
            "active key $(repr(k)) is a $(typeof(getproperty(inputs, k))); only float-valued " *
            "arrays can carry a gradient"))
    end
    # Preserve `inputs` order rather than the caller's, so the traced signature is a function of
    # the model, not of how the keyword happened to be written.
    return Tuple(k for k in keys(inputs) if k in active)
end

_split(nt::NamedTuple, dk::Tuple) =
    (NamedTuple{dk}(nt), NamedTuple{Tuple(k for k in keys(nt) if !(k in dk))}(nt))

# ⟨out, ct⟩ over the seeded output keys. `ntuple(…, Val(N))` unrolls, so `K[i]` is a literal in each
# body and the whole fold stays inferable — which matters because this runs inside the trace.
function _seed(out, ct::NamedTuple{K}) where {K}
    return sum(ntuple(i -> sum(getproperty(out, K[i]) .* getproperty(ct, K[i])), Val(length(K))))
end

_accumulate(::Nothing, new) = new
_accumulate(acc, new) = acc .+ new

# ── state reconciliation ────────────────────────────────────────────────────────────────────────

"""
    _reconcile_state(st_in, st_out)

Coerce a returned state back to the layout of the state that was passed *in*.

A compiled thunk's `st` argument has its type baked in, but a Lux layer's returned state is not
always structurally identical to what it was handed: nested sub-states drop empty entries
(`swish_gate` loses `:gate`) or reorder them (`adaln`), and the thunk then rejects the returned
state on type grounds. Keeping the input's key structure and taking the output's *value* wherever
the key exists fixes this generally, without touching each layer's plumbing.

Verified on a layer reproducing that exact asymmetry: threading the raw returned state fails,
threading the reconciled one succeeds with the rng correctly advanced.
"""
_reconcile_state(a::NamedTuple{K}, b::NamedTuple) where {K} =
    NamedTuple{K}(map(k -> haskey(b, k) ? _reconcile_state(getfield(a, k), getfield(b, k)) :
                           getfield(a, k), K))
_reconcile_state(_, b) = b

# ── rng protection ──────────────────────────────────────────────────────────────────────────────

"""
    _protect_rngs(st)

Replicate every rng reachable in `st`, so running a thunk cannot disturb the caller's state.

XLA may **donate** an input buffer. A layer that draws straight from `st.rng` instead of a
`Lux.replicate` of it therefore does not merely advance the caller's rng in place — it can leave the
caller's buffer deleted, and reading it afterwards fails with "Buffer has been deleted or donated".
Two calls with the same `st` then disagree, which looks like a seeding bug and is an aliasing one.

Replicating on the way in makes `forward`/`backward` total with respect to the state they are
handed: the input is never modified, and the *returned* state carries the advanced rng. The cost is
a copy of a 2-element `UInt64` device buffer per rng.
"""
_protect_rngs(x::AbstractRNG) = Lux.replicate(x)
_protect_rngs(nt::NamedTuple{K}) where {K} =
    NamedTuple{K}(map(k -> _protect_rngs(getfield(nt, k)), K))
_protect_rngs(t::Tuple) = map(_protect_rngs, t)
_protect_rngs(x) = x

# ── array construction ──────────────────────────────────────────────────────────────────────────

# A zero array of the given shape, on whatever device `x` lives on. `similar` inherits the array
# TYPE, so this produces a `ConcretePJRTArray` from a device prototype and an `Array` from a host
# one — which is what lets an architecture's glue build a recycling carry's initial value without
# knowing which side of the boundary it is on.
_zeros_like(x::AbstractArray, dims::Integer...) = fill!(similar(x, dims...), zero(eltype(x)))
