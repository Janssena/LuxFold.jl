# ── saving and loading ──────────────────────────────────────────────────────────────────────────
#
# What CANNOT be saved, and why the format is shaped the way it is.
#
# The obvious thing to want here is to write the XLA executables out and read them back, skipping
# the compile entirely. That is not reachable through Reactant's Julia API as of 0.2.280, and not
# for want of a wrapper:
#
#   * PJRT does expose `PJRT_Executable_Serialize` / `PJRT_Executable_DeserializeAndLoad`, but
#     Reactant binds neither above the raw CAPI layer — `XLA.LoadedExecutable` has no serialize
#     method at all.
#   * Even with the executable bytes in hand, a `Reactant.Compiler.Thunk` is not just an executable.
#     Its argument-unwrapping and result-wrapping code is a RuntimeGeneratedFunction body stored in
#     a process-global `Dict` under a gensym tag baked into the thunk's TYPE
#     (`compiler/Thunk.jl:__thunk_fwd_body_cache`). Nothing in a serialized thunk regenerates that
#     body; it only exists as a side effect of having run `compile_xla` in this process.
#
# So a load necessarily recompiles. What a checkpoint can still do — and what this one does — is
# make the recompile take **no arguments**: everything the build needed is in the file, so nothing
# depends on the caller remembering which `evo_chunk` or which `N` the weights were paired with.
#
# The layout, a directory:
#
#   manifest.toml       plain text, human- and tool-readable without Julia. Read FIRST on load, so
#                       a version mismatch is a sentence rather than a deserialization stacktrace.
#   spec.jls            Julia `Serialization` of the `HLOBuildSpec`: the Lux layer object, the input
#                       shape signature, the build keywords.
#   state.jls           Julia `Serialization` of the host-resident `st`.
#   params.safetensors  `ps`, flattened to dot-joined leaf paths.
#
# Two mechanisms rather than one, deliberately. `ps` is the large numeric payload and goes to
# safetensors because that is already this repo's weight interchange format
# (`LuxFoldCore.read_state_dict`, `AlphaFold2.load_alphafold2_weights!`) — a checkpoint's parameters
# are the SAME parameters, so they stay loadable by anything that reads weights, including a native
# non-compiled model. The layer object and `st` cannot go there: they are structs, rngs, StaticBools
# and `nothing`s, not arrays, and `Serialization` is the only thing that round-trips them.
#
# The cost of `Serialization` is that it records concrete types: a checkpoint does not survive a
# change to the layer structs it holds. That is why the manifest carries package versions and is
# read before anything else.

const HLO_FORMAT_VERSION = "1"

# The packages whose versions decide whether a `.jls` still deserializes, for ANY checkpoint: Lux
# defines the layer structs every model is built from, and these two define the format and the
# compiler it targets. The package that defines a particular architecture's OWN structs is not
# known here — this package depends on no model — so each architecture adds itself through
# [`versioned_packages`](@ref).
const _BASE_VERSIONED_PACKAGES = (StableHLOModels, Lux, Reactant)

# `nothing` when the module was `include`d rather than loaded as a package — which is how
# `scripts/af2_xla_parity.jl` and friends use this one. Not an error: an unversioned module simply
# contributes no version to check against.
_pkgversion(m::Module) = pkgversion(m)

# ── the build spec ──────────────────────────────────────────────────────────────────────────────

"""
    ArraySpec(a::AbstractArray)

Shape and element type of one prototype array, with the contents dropped.

Dropping them is not a space optimisation, it is the whole point: Reactant traces on shape, dtype
and device, and traced code cannot branch on values, so a prototype's contents cannot reach the
compiled program. Two prototypes of the same `ArraySpec` compile to the same executable. Storing a
few hundred MB of prototype features in a checkpoint would therefore buy exactly nothing.
"""
struct ArraySpec{T,N}
    size::NTuple{N,Int}
end

ArraySpec(a::AbstractArray) = ArraySpec{eltype(a),ndims(a)}(size(a))

Base.size(s::ArraySpec) = s.size
Base.eltype(::ArraySpec{T}) where {T} = T

Base.show(io::IO, s::ArraySpec{T}) where {T} = print(io, T, "[", join(s.size, "×"), "]")

# Contents are irrelevant to the compile (see above) but NOT to the one prototype execution
# `CompiledHLOModule` performs to record its output shapes. So they are chosen to be harmless in
# that run rather than merely cheap: `one` for integers, because an index tensor of zeros is out of
# range for a 1-based gather; `true` for masks, because an all-false attention mask makes the probe
# softmax a row of `-Inf` and fill the output with NaN. Neither would raise — XLA gathers clamp and
# NaNs propagate silently — and neither would change a recorded SHAPE. They are avoided anyway,
# because a probe that quietly computes garbage is a bad thing to have made routine.
_materialize(s::ArraySpec{Bool,N}) where {N} = fill(true, s.size)
_materialize(s::ArraySpec{T,N}) where {T<:Integer,N} = ones(T, s.size)
_materialize(s::ArraySpec{T,N}) where {T,N} = zeros(T, s.size)

_spec_of(x::AbstractArray) = ArraySpec(x)
_spec_of(x) = x   # scalars, symbols, `nothing` — kept verbatim, they are the value itself

_materialize(x) = x

"""
    HLOBuildSpec(factory, layer, inputs; kwargs...)

Everything needed to rebuild a compiled model except the weights: *which* factory to call, the Lux
layer to call it on, the shape signature of the prototype inputs, and the keywords.

A spec is the call you would have written, recorded rather than made. Constructed by naming the
factory itself, so the two lines read the same:

```julia
hlo  = AlphaFold2HLO(model, feats, ps, st; evo_chunk = 4)
spec = HLOBuildSpec(AlphaFold2HLO, model, feats;  evo_chunk = 4)
```

Only keywords the BUILD depends on belong here. How the rebuilt model is then driven —
`num_recycles`, `early_stop_threshold`, a diffusion `seed` — is chosen per call at `forward`, so a
checkpoint neither records nor constrains it.

`inputs` is reduced to an [`ArraySpec`](@ref) per array and kept verbatim otherwise — a checkpoint
records the shapes a model was compiled FOR, never the features it was compiled from.

`KIND` is a `Symbol` type parameter naming the factory, which is what [`build_hlo`](@ref) dispatches
on. Which kinds exist is not fixed here: each architecture's extension defines its own
`HLOBuildSpec` constructor and `build_hlo` method (`:alphafold2` for `AlphaFold2HLO`, and so on),
so a kind is loadable exactly when the package that defines it is loaded.

# Fields
- `layer`: the Lux model. Weight-free — a Lux layer is a descriptor — so this is config only.
- `inputs`: `NamedTuple` of `ArraySpec`s (and verbatim non-array values).
- `build_kwargs`: `NamedTuple` of the factory keywords, minus `dev`, `verbose` and
  `compile_options`, which describe *this* build rather than *the* model and are supplied fresh at
  load time. Passing one of those here is an error rather than a silent drop.
"""
struct HLOBuildSpec{KIND,L,I<:NamedTuple,K<:NamedTuple}
    layer::L
    inputs::I
    build_kwargs::K
end

# `dev` is a live device handle and `verbose` is a logging choice — neither describes the model, and
# a checkpoint that pinned them would fight the loading process rather than serve it.
const _TRANSIENT_KWARGS = (:dev, :verbose, :compile_options)

function HLOBuildSpec{KIND}(layer, inputs::NamedTuple; kwargs...) where {KIND}
    kw = NamedTuple(kwargs)
    bad = Tuple(k for k in keys(kw) if k in _TRANSIENT_KWARGS)
    isempty(bad) || throw(ArgumentError(
        "$(join(map(repr, bad), ", ")) describe a particular build, not the model, and are " *
        "supplied at load time instead — pass them to `build_hlo`/`load_hlo`, not to the spec.\n" *
        "(`compile_options` in particular holds Reactant configuration objects this deliberately " *
        "does not serialize; see the `load_hlo` docstring for the chunked-attention case.)"))
    ispec = NamedTuple{keys(inputs)}(map(_spec_of, values(inputs)))
    return HLOBuildSpec{KIND,typeof(layer),typeof(ispec),typeof(kw)}(layer, ispec, kw)
end

"""
    spec_kind(spec::HLOBuildSpec) -> Symbol

Which factory the spec names — the tag [`build_hlo`](@ref) dispatches on.
"""
spec_kind(::HLOBuildSpec{KIND}) where {KIND} = KIND

"""
    versioned_packages(spec::HLOBuildSpec) -> Tuple{Vararg{Module}}

Modules whose versions are recorded in the manifest and checked on load, for this spec's kind.

`spec.jls` and `state.jls` are written with `Serialization`, which records concrete types, so a
checkpoint does not survive a change to the structs it holds. Those structs belong to the model
package — and this package deliberately does not depend on one, so it cannot name it. An
architecture's extension therefore declares its own:

```julia
StableHLOModels.versioned_packages(s::HLOBuildSpec{:alphafold2}) =
    (_BASE_VERSIONED_PACKAGES..., AlphaFold2)
```

Omitting the method is not an error, only a silent loss of the most useful warning a stale
checkpoint could have produced.
"""
versioned_packages(::HLOBuildSpec) = _BASE_VERSIONED_PACKAGES

function Base.show(io::IO, s::HLOBuildSpec{KIND}) where {KIND}
    println(io, "HLOBuildSpec{", KIND, "}(", nameof(typeof(s.layer)), ",")
    for k in keys(s.inputs)
        println(io, "    ", rpad(string(k), 22), " = ", sprint(show, getproperty(s.inputs, k)), ",")
    end
    for k in keys(s.build_kwargs)
        println(io, "    ", rpad(string(k), 22), " = ", getproperty(s.build_kwargs, k), ",")
    end
    return print(io, ")")
end

"""
    build_hlo(spec::HLOBuildSpec, ps, st; dev, verbose, compile_options) -> model

Run the compile the spec describes. **This is the expensive call** — minutes, the same minutes the
original build cost — because the executables themselves cannot be persisted (see the notes at the
top of `save.jl`).

`ps` and `st` are host-resident, as the factories want them; only their shapes and types matter
here, their contents reach the compiled program no more than the prototype inputs do.

A new factory joins by adding one method:

```julia
build_hlo(s::HLOBuildSpec{:my_model}, ps, st; dev, verbose, compile_options) =
    MyModelHLO(s.layer, materialize_inputs(s), ps, st; dev, verbose, compile_options,
               s.build_kwargs...)
```
"""
function build_hlo end

build_hlo(s::HLOBuildSpec{KIND}, ps, st; kwargs...) where {KIND} = throw(ArgumentError(
    "no `build_hlo` method for a spec of kind $(repr(KIND)). Either the checkpoint was written " *
    "by a version of this package that knows a factory this one does not, or the factory's " *
    "`build_hlo` method has not been defined."))

"""
    materialize_inputs(spec) -> NamedTuple

Rebuild a prototype input NamedTuple from the spec's shape signature. See [`ArraySpec`](@ref) for
why the fill values are what they are.
"""
materialize_inputs(s::HLOBuildSpec) =
    NamedTuple{keys(s.inputs)}(map(_materialize, values(s.inputs)))

# Scratch that a *run* leaves in `st` and a fresh model would not have. Saved as-is it would make a
# reloaded model resume mid-recycle. Generic default: nothing is scratch.
_reset_scratch(::HLOBuildSpec, st) = st

# ── saving ──────────────────────────────────────────────────────────────────────────────────────

"""
    save_hlo(path, spec::HLOBuildSpec, ps, st; force = false) -> path

Write a compiled model to `path` as a directory, as `spec` + `ps` + `st`.

The XLA executables are **not** written — they cannot be, through any API Reactant exposes today
(the reasoning is at the top of `save.jl`). What is written is everything needed for
[`load_hlo`](@ref) to reproduce them with no arguments: the exact layer, the exact input shapes, the
exact build keywords, and the weights. A load recompiles, and costs what the original build cost.

`ps` and `st` may live on any device; both are brought back to the host here. `st` additionally has
its per-run scratch reset (for AlphaFold2, `st.recycling_state`), so a loaded model starts where a
freshly built one would rather than mid-recycle.

# Arguments
- `path`: directory to create. `.hlomodel` is the suggested extension; nothing enforces it.
- `spec`: what to rebuild. See [`HLOBuildSpec`](@ref).
- `ps`, `st`: the parameters and state to pair with it.

# Keyword Arguments
- `force`: overwrite `path` if it already exists and is non-empty. Without it an existing directory
  is an error, because the alternative is silently destroying a checkpoint that took minutes of
  compile and possibly hours of training to produce. (default: `false`)

# Example
```julia
model      = AlphaFold(:model_3)
ps, st     = Lux.setup(Random.Xoshiro(42), model)
AlphaFold2.load_alphafold2_weights!(ps, :model_3)

spec = HLOBuildSpec(AlphaFold2HLO, model, feats; evo_chunk = 4)
hlo  = build_hlo(spec, ps, st)                    # the expensive part
save_hlo("af2_model3_N256.hlomodel", spec, ps, st)
```
"""
function save_hlo(path::AbstractString, spec::HLOBuildSpec, ps, st; force::Bool=false)
    if isdir(path) && !isempty(readdir(path))
        force || throw(ArgumentError(
            "$path already exists and is not empty. Pass `force = true` to overwrite it."))
        rm(path; recursive=true)
    elseif ispath(path) && !isdir(path)
        throw(ArgumentError("$path exists and is not a directory"))
    end
    mkpath(path)

    cpu = Lux.cpu_device()
    ps_h = cpu(ps)
    st_h = _reset_scratch(spec, cpu(st))

    flat = LuxFoldCore.flatten_params(ps_h)
    LuxFoldCore.write_state_dict(joinpath(path, "params.safetensors"), flat)
    Serialization.serialize(joinpath(path, "spec.jls"), spec)
    Serialization.serialize(joinpath(path, "state.jls"), st_h)

    open(io -> TOML.print(io, _manifest(spec, flat)), joinpath(path, "manifest.toml"), "w")
    return path
end

"""
    save_hlo(path, factory, layer, inputs, ps, st; force = false, kwargs...) -> path

Convenience form that builds the [`HLOBuildSpec`](@ref) from the same arguments the factory took,
for the common case of saving a model that was just constructed:

```julia
hlo = AlphaFold2HLO(model, feats, ps, st; evo_chunk = 4)
save_hlo("af2.hlomodel", AlphaFold2HLO, model, feats, ps, st; evo_chunk = 4)
```

The keywords have to be repeated because a built model does not retain them — a `CompiledHLOModel`
holds compiled thunks, not the recipe that produced them. Construct the spec ONCE and pass
it to both the factory and `save_hlo` (as in the [`save_hlo`](@ref) example above) if that
duplication matters.
"""
save_hlo(path::AbstractString, factory, layer, inputs::NamedTuple, ps, st;
    force::Bool=false, kwargs...) =
    save_hlo(path, HLOBuildSpec(factory, layer, inputs; kwargs...), ps, st; force)

# The manifest is documentation, not data — `spec.jls` is what a load actually reads. So a keyword
# TOML has no type for (a `Symbol`, a `StaticBool`, a config struct) is rendered as its string form
# rather than being allowed to abort the save over a field nothing will parse back.
_tomlable(v::Union{Integer,AbstractFloat,Bool,AbstractString}) = v
_tomlable(v) = string(v)

function _manifest(spec::HLOBuildSpec, flat::AbstractDict)
    nparam = sum(length, values(flat); init=0)
    nbytes = sum(a -> length(a) * sizeof(eltype(a)), values(flat); init=0)

    inputs = Dict{String,Any}()
    for k in keys(spec.inputs)
        v = getproperty(spec.inputs, k)
        inputs[string(k)] = v isa ArraySpec ?
                            Dict("size" => collect(size(v)), "eltype" => string(eltype(v))) :
                            Dict("value" => string(v))
    end

    return Dict(
        "format_version" => HLO_FORMAT_VERSION,
        "kind" => string(spec_kind(spec)),
        "created" => Libc.strftime("%Y-%m-%dT%H:%M:%S", time()),
        "layer" => string(nameof(typeof(spec.layer))),
        "julia_version" => string(VERSION),
        "packages" => Dict(string(nameof(m)) => string(_pkgversion(m))
                           for m in versioned_packages(spec) if _pkgversion(m) !== nothing),
        "inputs" => inputs,
        "build_kwargs" => Dict(string(k) => _tomlable(getproperty(spec.build_kwargs, k))
                               for k in keys(spec.build_kwargs)),
        "params" => Dict("tensors" => length(flat), "count" => nparam, "bytes" => nbytes),
    )
end

# ── loading ─────────────────────────────────────────────────────────────────────────────────────

"""
    load_hlo_manifest(path) -> Dict

Read a checkpoint's `manifest.toml` without loading anything else. Plain TOML, so this answers "what
is in this directory, and will it load here?" without deserializing a single struct — which is the
point, since deserializing is the step that can fail on a code change.
"""
function load_hlo_manifest(path::AbstractString)
    f = joinpath(path, "manifest.toml")
    isfile(f) || throw(ArgumentError(
        "$path is not a checkpoint directory: no manifest.toml. " *
        (isdir(path) ? "It holds: " * join(readdir(path), ", ") * "." : "It does not exist.")))
    return TOML.parsefile(f)
end

"""
    load_hlo_spec(path) -> HLOBuildSpec

Load only the build spec — the layer, the input shapes and the keywords. Compiles nothing, reads no
weights. What to call to inspect a checkpoint, or to rebuild it against different parameters.
"""
function load_hlo_spec(path::AbstractString)
    _check_format(load_hlo_manifest(path), path)
    try
        return Serialization.deserialize(joinpath(path, "spec.jls"))::HLOBuildSpec
    catch e
        _rethrow_stale(e, path, "spec.jls")
    end
end

"""
    load_hlo(path; build = true, dev = Lux.reactant_device(), verbose = true,
             compile_options = Reactant.CompileOptions()) -> (; model, ps, st, spec)

Load a checkpoint written by [`save_hlo`](@ref) and recompile it.

**This recompiles from scratch**, and takes as long as the original build did. The XLA executables
are not in the file because Reactant cannot serialize them; the checkpoint's job is to make the
rebuild need no arguments, not to make it fast. Pass `build = false` to get the weights and the spec
back without paying for it.

The returned `ps`/`st` are on `dev`, ready to hand straight to `AlphaFold2.predict` /
[`forward`](@ref). The spec is returned too, so a caller that wants a second model at other settings
can rebuild from it without re-reading the directory.

# Keyword Arguments
- `build`: run the compile. `false` returns `model = nothing` and skips it entirely. (default: `true`)
- `dev`: device to compile for and to move `ps`/`st` onto. (default: `Lux.reactant_device()`)
- `verbose`: per-entry compile progress. (default: `true`)
- `compile_options`: passed to every entry's `@compile`. **Not stored in the checkpoint** — it holds
  live Reactant configuration objects, and a stale serialized copy of a compiler setting is a
  particularly bad thing to resurrect silently. This matters for exactly one case: a model whose
  triangle attention was built with `tri_att_chunk_size` MUST be rebuilt with
  `LuxTriangleAttention.chunked_compile_options()`, or the rolled chunk loop is optimised away and
  the memory bound is lost without any error. (default: `Reactant.CompileOptions()`)

# Example
```julia
model, ps, st = load_hlo("af2_model3_N256.hlomodel")
out, st = AlphaFold2.predict(model, dev(feats), ps, st)
```

# Returns
- `model`: the rebuilt compiled model, or `nothing` when `build = false`.
- `ps`, `st`: parameters and state, on `dev`.
- `spec`: the [`HLOBuildSpec`](@ref) that was used.
"""
function load_hlo(path::AbstractString; build::Bool=true, dev=Lux.reactant_device(),
    verbose::Bool=true, compile_options=Reactant.CompileOptions())

    manifest = load_hlo_manifest(path)
    _check_format(manifest, path)

    # The spec is read BEFORE the version check, not after: which packages are worth comparing is a
    # property of the spec's kind (see `versioned_packages`), so there is nothing to warn about
    # until it is in hand. A spec that fails to deserialize raises through `_rethrow_stale`, which
    # points at `manifest.toml`'s `[packages]` anyway.
    spec = load_hlo_spec(path)
    _warn_versions(manifest, path, spec)

    st_h = try
        Serialization.deserialize(joinpath(path, "state.jls"))
    catch e
        _rethrow_stale(e, path, "state.jls")
    end

    # `ps` is rebuilt from the layer rather than from the file's key set: `Lux.setup` is what defines
    # the container's structure, and copying INTO it makes a missing or misnamed tensor an error
    # here instead of a shape mismatch several compiles later. The rng is irrelevant — every leaf is
    # overwritten — but it must be there, so it is fixed rather than arbitrary.
    ps_h, _ = Lux.setup(Random.Xoshiro(0), spec.layer)
    LuxFoldCore.load_flat_weights!(ps_h,
        LuxFoldCore.read_state_dict(joinpath(path, "params.safetensors")))

    model = build ? build_hlo(spec, ps_h, st_h; dev, verbose, compile_options) : nothing
    return (; model, ps=dev(ps_h), st=dev(st_h), spec)
end

function _check_format(manifest::AbstractDict, path)
    v = get(manifest, "format_version", nothing)
    v == HLO_FORMAT_VERSION || throw(ArgumentError(
        "$path was written in checkpoint format $(repr(v)); this StableHLOModels reads " *
        "$(repr(HLO_FORMAT_VERSION)). Rebuild and re-save it with the version that wrote it."))
    return nothing
end

# A version skew is a WARNING, not an error: most package updates leave the serialized structs
# untouched, and refusing to load on any drift would make checkpoints useless. When it does matter,
# `_rethrow_stale` turns the resulting deserialization failure into an explanation.
function _warn_versions(manifest::AbstractDict, path, spec::HLOBuildSpec)
    stored = get(manifest, "packages", Dict{String,Any}())
    for m in versioned_packages(spec)
        v = _pkgversion(m)
        v === nothing && continue
        was, now = get(stored, string(nameof(m)), nothing), string(v)
        was === nothing || was == now ||
            @warn "checkpoint was written against a different $(nameof(m))" path was now
    end
    return nothing
end

function _rethrow_stale(e, path, file)
    throw(ErrorException("""
        failed to deserialize $(joinpath(path, file)).

        `$file` is written with Julia's `Serialization`, which records concrete types, so a
        checkpoint does not survive a change to the layer or state structs it holds. Compare
        $(joinpath(path, "manifest.toml"))'s `[packages]` against the versions loaded here; if they
        differ, the checkpoint has to be rebuilt from weights by the code that wrote it.

        The underlying failure was:
        $(sprint(showerror, e))"""))
end
