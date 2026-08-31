# ── CompiledHLOModel ────────────────────────────────────────────────────────────────────────────

"""
    CompiledHLOModel(; name = static(:chain), inputs = nothing, entry = module, ...)
    CompiledHLOModel(pairs::Pair{Symbol,<:AbstractHLOModule}...; name = static(:chain), inputs = nothing)

An ordered, named collection of already-compiled [`AbstractHLOModule`](@ref)s, run in sequence —
the `Lux.Chain` of this package.

`CompiledHLOModel` compiles **nothing**. Entries arrive already compiled, because an entry's
prototype input shapes depend on the previous entry's outputs, so a compiling constructor would
have to either run each entry forward to discover them or be handed prototypes anyway. Keeping the
container inert also means entries are independent to build, which is what would make compiling
them concurrently straightforward later.

`forward(model, inputs, ps, st)` threads `inputs = merge(inputs, out)` across the entries, passing
`ps.<name>` and `st.<name>` to the entry named `<name>`, and returns the union of what the entries
produced together with the updated state.

# Keyword Arguments
- `inputs`: optional prototype input NamedTuple. When given, the constructor additionally checks
  that every entry's active inputs are actually available at its position — either a model input or
  something an earlier entry produced. Cheap insurance against a wiring mistake that would otherwise
  surface as a shape error deep inside a long compile. (default: `nothing`)
- `name`: a `Symbol` or `StaticSymbol` identifying *which* model this is, carried in the first type
  parameter so `forward`/`backward` can dispatch on it. The default, `static(:chain)`, selects the
  generic sequential walk documented above. A real architecture — whose glue between entries is not
  a plain `merge` — declares its own name and its own methods; see
  `AlphaFold2StableHLOModelsExt` (in `lib/AlphaFold2/ext/`) for the worked example. Per the repo's
  StaticBool/StaticSymbol convention, this is a compile-time tag, never a runtime field to branch
  on. (default: `static(:chain)`)

# Gradient wiring
`backward` walks the entries in reverse, maintaining one pending cotangent per tensor name:

  * a key that the entry **produces** has its pending cotangent consumed as that entry's seed, and
    is then **replaced** by the entry's own input-cotangent for that key;
  * a key the entry merely **reads** (a pass-through, or an external model input) **accumulates**,
    because several entries can read the same tensor and every path contributes.

A compiled VJP's `ct` key set is baked into its traced program, so an entry is always handed exactly
`ct_keys(entry)`; any slot nothing contributed to is filled from `ct_zeros(entry)`.
"""
struct CompiledHLOModel{NAME<:StaticSymbol,NAMES,E<:Tuple} <: AbstractHLOModule
    entries::E
end

function CompiledHLOModel(pairs::Pair{Symbol,<:AbstractHLOModule}...;
    inputs=nothing, name::Union{Symbol,StaticSymbol}=static(:chain))

    isempty(pairs) && throw(ArgumentError("a CompiledHLOModel needs at least one entry"))
    names = map(first, pairs)
    length(unique(names)) == length(names) || throw(ArgumentError(
        "duplicate entry names: $(join(unique(n for n in names if count(==(n), names) > 1), ", "))"))
    entries = map(last, pairs)
    nm = name isa StaticSymbol ? name : static(name)
    # Only the generic walk can be checked here. A named model supplies its own glue, so "available
    # at this position" is whatever its `forward` chooses to hand the entry, which this cannot know.
    nm === static(:chain) && _validate_wiring(names, entries, inputs)
    return CompiledHLOModel{typeof(nm),names,typeof(entries)}(entries)
end

CompiledHLOModel(; inputs=nothing, name::Union{Symbol,StaticSymbol}=static(:chain), kw...) =
    CompiledHLOModel((k => v for (k, v) in pairs(kw))...; inputs, name)

"""
    model_name(m::CompiledHLOModel) -> StaticSymbol

The architecture tag `forward`/`backward` dispatch on. `static(:chain)` for the generic walk.
"""
model_name(::CompiledHLOModel{NAME}) where {NAME} = NAME()

entry_names(::CompiledHLOModel{NAME,NAMES}) where {NAME,NAMES} = NAMES
entries(m::CompiledHLOModel) = m.entries

"""
    entry(m::CompiledHLOModel, name::Symbol) -> AbstractHLOModule

Look one entry up by name. What a model-specific `forward`/`backward` uses to reach a particular
compiled piece, since it drives the entries by hand rather than walking them in order.
"""
function entry(m::CompiledHLOModel, name::Symbol)
    i = findfirst(==(name), entry_names(m))
    i === nothing && throw(ArgumentError(
        "no entry named $(repr(name)); this model has $(entry_names(m))"))
    return m.entries[i]
end

# The model's own interface keys: what it reads that nobody upstream produces, and what it produces.
active_keys(m::CompiledHLOModel) = _model_active(entry_names(m), m.entries)
output_keys(m::CompiledHLOModel) = _model_outputs(m.entries)
ct_keys(m::CompiledHLOModel) = ct_keys(last(m.entries))
ct_zeros(m::CompiledHLOModel) = ct_zeros(last(m.entries))

function _model_outputs(entries::Tuple)
    ok = ()
    for e in entries, k in output_keys(e)
        k in ok || (ok = (ok..., k))
    end
    return ok
end

function _model_active(names::Tuple, entries::Tuple)
    produced, need = (), ()
    for e in entries
        for k in active_keys(e)
            (k in produced || k in need) || (need = (need..., k))
        end
        produced = (produced..., output_keys(e)...)
    end
    return need
end

function _validate_wiring(names::Tuple, entries::Tuple, inputs)
    produced = ()
    for (n, e) in zip(names, entries)
        # An entry's seed must actually be something it produces, or the cotangent is meaningless.
        bad = Tuple(k for k in ct_keys(e) if !(k in output_keys(e)))
        isempty(bad) || throw(ArgumentError(
            "entry $(repr(n)) was compiled to seed $(bad), which it does not produce " *
            "(it produces $(output_keys(e))). Recompile it with a `ct` over its own outputs."))
        if inputs !== nothing
            avail = (keys(inputs)..., produced...)
            missing_ = Tuple(k for k in active_keys(e) if !(k in avail))
            isempty(missing_) || throw(ArgumentError(
                "entry $(repr(n)) differentiates w.r.t. $(missing_), which is neither a model " *
                "input nor produced by an earlier entry. Available here: $(avail)."))
        end
        produced = (produced..., output_keys(e)...)
    end
    return nothing
end

function Base.show(io::IO, m::CompiledHLOModel)
    println(io, "CompiledHLOModel{", Static.known(model_name(m)), "}(")
    for (n, e) in zip(entry_names(m), m.entries)
        println(io, "    ", n, " = ", sprint(show, e), ",")
    end
    print(io, ")")
end

# ── forward ─────────────────────────────────────────────────────────────────────────────────────

# Left unconstrained rather than pinned to `{StaticSymbol{:chain}}`: a name is additive, so a model
# that needs only its own *factory* keeps this walk for free, and one that needs its own glue
# defines a more specific method that wins on dispatch.
function forward(m::CompiledHLOModel, inputs::NamedTuple, ps, st)
    names = entry_names(m)
    cur, produced = inputs, nothing
    st_new = st
    for (i, e) in enumerate(m.entries)
        n = names[i]
        out, sn = forward(e, cur, getproperty(ps, n), getproperty(st, n))
        st_new = merge(st_new, NamedTuple{(n,)}((sn,)))
        produced = produced === nothing ? out : merge(produced, out)
        cur = merge(cur, out)
    end
    return produced, st_new
end

# ── reverse ─────────────────────────────────────────────────────────────────────────────────────

"""
    backward(m::CompiledHLOModel, inputs, ps, st, ct)

Reverse walk across the entries. The forward is replayed once first, storing each entry's input and
the state it ran with, so every entry's backward sees exactly the arguments its forward saw.

Returns `(; dinputs, dps)`, `dps` being a NamedTuple over the entry names.
"""
function backward(m::CompiledHLOModel, inputs::NamedTuple, ps, st, ct::NamedTuple)
    names = entry_names(m)
    n_e = length(m.entries)

    # Forward, recording the (input, state) each entry ran with.
    saved = Vector{Any}(undef, n_e)
    cur, st_cur = inputs, st
    for (i, e) in enumerate(m.entries)
        n = names[i]
        sn_in = getproperty(st_cur, n)
        saved[i] = (cur, sn_in)
        out, sn = forward(e, cur, getproperty(ps, n), sn_in)
        st_cur = merge(st_cur, NamedTuple{(n,)}((sn,)))
        cur = merge(cur, out)
    end

    # Pending cotangent per tensor name. A Dict rather than a NamedTuple: the key set changes as the
    # walk proceeds, and with a handful of entries none of this is hot.
    pending = Dict{Symbol,Any}(k => v for (k, v) in pairs(ct))
    dps = Vector{Any}(undef, n_e)

    for i in n_e:-1:1
        e, n = m.entries[i], names[i]
        zs = ct_zeros(e)
        # Exactly the key set this entry's thunk was compiled with; zero-fill anything unseeded.
        cti = NamedTuple{ct_keys(e)}(map(k -> get(pending, k, getproperty(zs, k)), ct_keys(e)))
        # Consume: these slots are about to be superseded by this entry's own input-cotangents.
        for k in output_keys(e)
            delete!(pending, k)
        end
        ci, si = saved[i]
        g = backward(e, ci, getproperty(ps, n), si, cti)
        dps[i] = g.dps
        for (k, v) in pairs(g.dinputs)
            pending[k] = haskey(pending, k) ? _accumulate(pending[k], v) : v
        end
    end

    dk = Tuple(k for k in keys(pending))
    dinputs = NamedTuple{dk}(map(k -> pending[k], dk))
    return (; dinputs, dps=NamedTuple{names}(Tuple(dps)))
end
