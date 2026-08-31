# ── the functions that actually get compiled ────────────────────────────────────────────────────

# `merge(c, x)` puts the differentiable half back in place. Field order may differ from the
# caller's NamedTuple; nothing downstream cares, access is by name. Returns `(out, st)` so the
# caller can thread state.
_fwd_fn(layer, x, c, ps, st) = layer(merge(c, x), ps, st)

_loss_fn(layer, x, c, ps, st, ct) = _seed(first(_fwd_fn(layer, x, c, ps, st)), ct)

# Two thunks rather than one branch: `diff_params` is a compile-time configuration flag, so it
# selects the traced program by dispatch (repo convention — never `if` on a StaticBool in a
# forward). Enzyme.gradient returns one slot per argument, `nothing` where the argument is Const,
# so `g[2]` is always dx and `g[4]` is dps or nothing.
_vjp_fn(::True, layer, x, c, ps, st, ct) =
    Enzyme.gradient(Reverse, _loss_fn, Const(layer), x, Const(c), ps, Const(st), Const(ct))
_vjp_fn(::False, layer, x, c, ps, st, ct) =
    Enzyme.gradient(Reverse, _loss_fn, Const(layer), x, Const(c), Const(ps), Const(st), Const(ct))

# Total seconds spent in the one prototype execution each constructor performs to learn its output
# keys and shapes. Instrumentation only: it answers "is that run worth eliminating?" with a number.
const PROBE_SECONDS = Ref(0.0)

"""
    COMPILE_TIMES :: Vector{Pair{Symbol,Float64}}

Per-entry compile seconds from the most recent architecture build, in build order. The factories
that fill it (`AlphaFold2HLO` and its counterparts) live in the model packages' StableHLOModels
extensions.

The factories' `verbose` logging already prints these, but printing is not measurement: a sweep
over configurations needs the numbers as data. Every factory's `_timed` appends here, so the same
collector serves all three models.

Append-only and process-global — call [`reset_compile_times!`](@ref) before a build you intend to
read back, or a sweep will accumulate every model it has ever built.
"""
const COMPILE_TIMES = Pair{Symbol,Float64}[]

"""
    reset_compile_times!()

Clear [`COMPILE_TIMES`](@ref) and `PROBE_SECONDS`, so the next build's numbers stand alone.
"""
reset_compile_times!() = (empty!(COMPILE_TIMES); PROBE_SECONDS[] = 0.0; nothing)

record_compile_time!(name::Symbol, seconds::Real) = push!(COMPILE_TIMES, name => Float64(seconds))

# ── CompiledHLOModule ───────────────────────────────────────────────────────────────────────────

"""
    CompiledHLOModule(layer, inputs, ps, st, ct; kwargs...)

One Lux layer compiled to StableHLO in both directions, run **once** per call.

Both thunks are compiled eagerly, in the constructor, for the shapes and element types of the
prototype arguments — the prototypes *are* the input-dimension declaration, since `@compile` traces
on real arguments. Every one must already live on the Reactant device (`dev = Lux.reactant_device()`);
nothing here moves data.

For a deep stack of one repeated block, use [`CompiledHLOStack`](@ref). To run several compiled
modules in sequence, collect them in a [`CompiledHLOModel`](@ref).

# Arguments
- `layer`: the Lux layer to compile.
- `inputs`: prototype input NamedTuple, e.g. `(; s, z, mask, pair_mask)`.
- `ps`, `st`: the layer's parameters and state.
- `ct`: prototype output cotangent, a NamedTuple over the output keys to seed, e.g. `(; s, z)`.

# Keyword Arguments
- `active`: which input keys carry a gradient — the `Enzyme.Active`/`Enzyme.Const` split, fixed at
  construction because the activity signature is baked into the traced program. The default
  (`nothing`) takes every float-valued array, which is the right *upper bound* but cannot know that
  a float input is meant to be held fixed (a frozen template, `s_trunk` during a diffusion-only
  gradient); name the ones you want when that matters. (default: `nothing`)
- `diff_params`: `StaticBool`. `true` also returns ∂/∂ps; `false` marks `ps` constant, giving a
  cheaper thunk when only input sensitivity is wanted. (default: `true`)
- `compile_options`: passed to `@compile`. Pass `LuxTriangleAttention.chunked_compile_options()`
  when the layer uses chunked triangle attention, or the rolled `stablehlo.while` is optimized back
  out. (default: `Reactant.CompileOptions()`)
- `compile_backward`: build the VJP thunk. `false` gives a forward-only module — useful for a layer
  used only at inference. Calling `backward` on such a module raises rather than returning something
  wrong. (default: `true`)
- `outputs`: what `forward` produces, as a NamedTuple mapping each output key to its size tuple.
  The default (`nothing`) discovers this by executing the compiled forward once, which is correct
  for any layer but costs one real execution per module. Declare it when the caller already knows
  the layer — an assembled model usually does — and that execution is skipped. (default: `nothing`)
- `verify_outputs`: run the discovery probe even when `outputs` was declared, and raise if the two
  disagree. The way to re-validate a declaration after a layer changes. (default: `false`)

# Returns
- `forward` → `(outputs, st_updated)`. The returned state is reconciled to the layout of the `st`
  passed in (see `_reconcile_state`), so it can be threaded straight back into the next call.
- `backward` → `(; dinputs, dps)`, `dps` being `nothing` when `diff_params = false`.
"""
struct CompiledHLOModule{L,DK,OK,CK,DP<:StaticBool,F,V,Z,S} <: AbstractHLOModule
    layer::L
    diff_params::DP
    fwd::F
    vjp::V
    ct_zeros::Z
    out_shapes::S
end

function CompiledHLOModule(layer, inputs::NamedTuple, ps, st, ct::NamedTuple;
    active::Union{Nothing,Tuple{Vararg{Symbol}}}=nothing,
    diff_params::Union{Bool,StaticBool}=true,
    compile_backward::Bool=true,
    compile_options=Reactant.CompileOptions(),
    outputs::Union{Nothing,NamedTuple}=nothing,
    verify_outputs::Bool=false)

    dk = _resolve_active(inputs, active)
    isempty(dk) && throw(ArgumentError(
        "no float-valued arrays in `inputs`; nothing to differentiate with respect to"))
    x, c = _split(inputs, dk)
    dp = diff_params isa StaticBool ? diff_params : static(diff_params)

    fwd = @compile compile_options = compile_options _fwd_fn(layer, x, c, ps, st)
    vjp = compile_backward ?
          (@compile compile_options = compile_options _vjp_fn(dp, layer, x, c, ps, st, ct)) :
          nothing

    # What the layer produces. `CompiledHLOModel`'s wiring validation needs the keys, and a caller
    # that declares prototype shapes rather than discovering them wants the sizes to check against.
    #
    # `outputs === nothing` (the default) discovers both by executing the compiled forward once —
    # correct for any layer, at the cost of one real execution per entry. A caller that already
    # KNOWS its layer's outputs declares them instead and skips that execution entirely; pass
    # `verify_outputs=true` to run the probe anyway and assert the declaration, which is how a
    # declaration should be (re)validated after a layer changes.
    if outputs === nothing || verify_outputs
        _t_probe = @elapsed o1 = first(fwd(layer, x, c, ps, st))
        PROBE_SECONDS[] += _t_probe
        ok, osz = keys(o1), map(size, o1)
        if outputs !== nothing
            Set(keys(outputs)) == Set(ok) && all(getproperty(outputs, k) == getproperty(osz, k)
                                                 for k in keys(outputs)) || throw(ArgumentError(
                "declared outputs do not match what $(nameof(typeof(layer))) produces:\n" *
                "  declared $(outputs)\n  actual   $(osz)"))
        end
    else
        ok, osz = keys(outputs), outputs
    end
    zs = map(zero, ct)

    return CompiledHLOModule{typeof(layer),dk,ok,keys(ct),typeof(dp),typeof(fwd),typeof(vjp),
        typeof(zs),typeof(osz)}(layer, dp, fwd, vjp, zs, osz)
end

"""
    output_shapes(m::AbstractHLOModule) -> NamedTuple

Sizes of what `forward` produces, recorded at construction from the one prototype run the
constructor performs anyway. `nothing` for module types that do not record them.

The point is cheap verification: a caller that *declares* prototype shapes from layer config —
rather than paying a full forward to discover them — can assert its declaration against this and
get a loud failure at build time instead of a thunk argument rejection several entries later.
"""
output_shapes(::AbstractHLOModule) = nothing
output_shapes(m::CompiledHLOModule) = m.out_shapes

active_keys(::CompiledHLOModule{L,DK}) where {L,DK} = DK
output_keys(::CompiledHLOModule{L,DK,OK}) where {L,DK,OK} = OK
ct_keys(::CompiledHLOModule{L,DK,OK,CK}) where {L,DK,OK,CK} = CK
ct_zeros(m::CompiledHLOModule) = m.ct_zeros

Base.show(io::IO, m::CompiledHLOModule) = print(io,
    "CompiledHLOModule(", nameof(typeof(m.layer)), "; diff_params=", m.diff_params isa True, ")")

function forward(m::CompiledHLOModule, inputs::NamedTuple, ps, st)
    x, c = _split(inputs, active_keys(m))
    stp = _protect_rngs(st)
    out, st_new = m.fwd(m.layer, x, c, ps, stp)
    return out, _reconcile_state(st, st_new)
end

"""
    compile_backward(m::CompiledHLOModule, inputs, ps, st, ct; compile_options=Reactant.CompileOptions())

See the [`AbstractHLOModule`](@ref) docstring of the same name for the contract. `inputs`/`ps`/`st`/
`ct` must be the same prototypes `m` was originally constructed with.
"""
function compile_backward(m::CompiledHLOModule, inputs::NamedTuple, ps, st, ct::NamedTuple;
    compile_options=Reactant.CompileOptions())

    m.vjp === nothing || return m
    x, c = _split(inputs, active_keys(m))
    vjp = @compile compile_options = compile_options _vjp_fn(m.diff_params, m.layer, x, c, ps, st, ct)
    return CompiledHLOModule{typeof(m.layer),active_keys(m),output_keys(m),ct_keys(m),
        typeof(m.diff_params),typeof(m.fwd),typeof(vjp),typeof(m.ct_zeros),typeof(m.out_shapes)}(
        m.layer, m.diff_params, m.fwd, vjp, m.ct_zeros, m.out_shapes)
end

function backward(m::CompiledHLOModule, inputs::NamedTuple, ps, st, ct::NamedTuple)
    m.vjp === nothing && throw(ArgumentError(
        "this CompiledHLOModule was built with compile_backward = false; it has no VJP thunk. " *
        "Call `compile_backward(m, inputs, ps, st, ct)` to build one."))
    x, c = _split(inputs, active_keys(m))
    g = m.vjp(m.diff_params, m.layer, x, c, ps, _protect_rngs(st), ct)
    return (; dinputs=g[2], dps=g[4])
end
