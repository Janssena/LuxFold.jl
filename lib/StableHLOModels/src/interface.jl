# ── the abstract interface ──────────────────────────────────────────────────────────────────────

"""
    AbstractHLOModule

Supertype for everything this package can run. The contract, in full:

  * `forward(m, inputs, ps, st) -> (outputs::NamedTuple, st_updated)` — Lux-style. `outputs`
    carries only the fields the module *produced*; the caller merges them into its own carry.
  * `backward(m, inputs, ps, st, ct) -> (; dinputs, dps)` — the VJP of `⟨forward(inputs), ct⟩`.
  * `active_keys(m)` — input keys a gradient flows through, fixed at construction.
  * `output_keys(m)` — keys `forward` produces, recorded at construction.
  * `ct_keys(m)` — output keys the compiled VJP seeds. Baked into the traced program, so a caller
    must supply exactly these, every time; [`ct_zeros`](@ref) provides the fill for any that a
    downstream module had nothing to say about.

`(m::AbstractHLOModule)(inputs, ps, st)` is `forward`.
"""
abstract type AbstractHLOModule end

"""
    forward(m::AbstractHLOModule, inputs::NamedTuple, ps, st) -> (outputs, st_updated)
"""
function forward end

"""
    backward(m::AbstractHLOModule, inputs::NamedTuple, ps, st, ct::NamedTuple) -> (; dinputs, dps)
"""
function backward end

"""
    value_and_gradient(model, inputs, ps, st, ct; kwargs...) -> (; outputs, dbatch, dps)

A model's outputs AND its VJP, from ONE forward pass.

The pairing is the point. A gradient always needs a forward — the loss is defined against what the
model predicts, so the predictions have to exist before anything can be differentiated — and a
separate `gradient` entry point would run that forward anyway, just without handing it back. So this
is the primitive, and a caller wanting only the gradient takes `.dbatch` from the result.

Implemented per architecture, dispatching on the model's name tag
(`CompiledHLOModel{StaticSymbol{:alphafold3}}`, …), because what has to be cached between the
forward and the reverse walk is specific to how that architecture's entries are wired. It takes the
same runtime keywords as that architecture's `forward`, and must be given the same values — a
gradient is only the gradient of the forward that was actually run.

!!! note "What one forward does and does not buy"
    It removes the duplicate glue-level forward. It cannot remove the forward sweep Enzyme runs
    inside each entry's VJP thunk: a compiled thunk is opaque to its neighbours, so an entry's
    reverse pass recomputes its own intermediates rather than reading them off a tape. The floor for
    value+gradient is one forward plus one (forward-sweep + reverse-sweep) per entry on the gradient
    path — the standard checkpointing trade.
"""
function value_and_gradient end

"""
    active_keys(m::AbstractHLOModule) -> Tuple{Vararg{Symbol}}

Input keys that carry a gradient. Fixed at construction: the activity split is baked into the
traced program.
"""
function active_keys end

"""
    output_keys(m::AbstractHLOModule) -> Tuple{Vararg{Symbol}}

Keys `forward` produces. Recorded at construction by running the compiled forward once on the
prototypes — the only way to know them without asking the caller to declare them.
"""
function output_keys end

"""
    ct_keys(m::AbstractHLOModule) -> Tuple{Vararg{Symbol}}

Output keys the compiled VJP seeds.
"""
function ct_keys end

"""
    ct_zeros(m::AbstractHLOModule) -> NamedTuple

Device-resident zero cotangents over [`ct_keys`](@ref), used to fill a seed slot that nothing
downstream contributed to.
"""
function ct_zeros end

(m::AbstractHLOModule)(inputs::NamedTuple, ps, st) = forward(m, inputs, ps, st)

"""
    compile_backward(m::AbstractHLOModule, inputs, ps, st, ct; compile_options=Reactant.CompileOptions())

Compile `m`'s VJP thunk if it does not have one, returning a module `backward` can be called on.
**Idempotent**: if `m` already has a VJP (it was built with `compile_backward = true`, the
constructor default), returns `m` unchanged — no recompile, no new object.

This is the "upgrade a forward-only build" operation, for exactly the case a forward-only build
exists for: compiling both directions can be prohibitively expensive, or even hang outright for
some layers (chunked triangle attention's reverse pass is one — see
`memory/lta_chunked_reverse_hangs.md`), so a caller builds forward-only first and calls this only
once it actually needs a gradient.

Not a true in-place upgrade: neither `CompiledHLOModule` nor `CompiledHLOStack` retains its
prototype `inputs`/`ps`/`st`/`ct` past construction (a compiled thunk doesn't need them, so keeping
them would only cost memory), so this needs them handed back in. **They must be the same
prototypes** — same shapes, same active/differentiable split — the module was originally built
with; anything else traces a VJP for a different program than the one `forward` runs.

Only implemented for concrete `AbstractHLOModule` types that have a `compile_backward` CONSTRUCTOR
kwarg to begin with (`CompiledHLOModule`, `CompiledHLOStack`) — not a required part of the
interface, so a type that does not support it raises the ordinary `MethodError` rather than
something bespoke.
"""
function compile_backward end
