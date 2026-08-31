# ── CompiledHLOStack ────────────────────────────────────────────────────────────────────────────

"""
    CompiledHLOStack(layer, inputs, ps, st, ct; depth, chunk_size=1, kwargs...)

An `M`-block chunk compiled once and replayed `K` times, giving a stack of depth `M * K`.

This is the package's reason to exist. Two measured facts about Reactant motivate it:

  * **Compile time, not run time, is the binding constraint.** A single-pass compile of a whole
    fold model does not finish; a compile of a few blocks costs seconds. Since a deep stack is the
    *same* block repeated, one compiled chunk serves every layer of it.
  * **Reverse mode is memory-bound.** A single-pass VJP over a deep stack keeps every block's
    activations live. Crossing the chunk boundary by hand — replaying a per-chunk VJP thunk
    backwards, feeding each chunk's input-cotangent to its predecessor — bounds the live set to one
    chunk, at the cost of storing (or recomputing) the chunk inputs.

# Arguments
- `layer`: **one `M`-block chunk**, already built — e.g. `PairformerModule(M)`, not the full stack
  and not a single block. `M` must equal `chunk_size`.
- `inputs`, `ps`, `st`, `ct`: prototypes as in [`CompiledHLOModule`](@ref), except that `ps` and
  `st` are the **whole stack's**, from which each chunk's slice is re-keyed (see below).

# Keyword Arguments
- `depth`: total number of blocks, `M * K`. Must be divisible by `chunk_size`.
- `chunk_size`: blocks compiled together as one thunk (`M`). `1` reproduces block-at-a-time
  replay. (default: `1`)
- `ps_path`: field path from `ps` to the block container. (default: `(:blocks,)`)
- `ps_key`: block index → field name in that container. (default: `i -> Symbol("block_", i)`)
- `shared_ps`: `StaticBool`. `true` means the blocks are **weight-tied** — every replay is handed
  the same `ps`, `layer` is the plain block (not a container), and `ps_path`/`ps_key` are unused.
  AlphaFold2's structure module is the motivating case: all eight of its blocks read one `ps.fold`.
  Two consequences follow, both handled here: `st` **threads** from replay to replay (the SM threads
  `st_fold`), and `dps` is the **sum** over replays rather than one gradient per block. (default:
  `false`)
- `checkpoint`: store every `checkpoint`-th **chunk** input and recompute the rest during the
  backward walk. `1` stores all of them. (default: `1`)
- `compile_backward`: build the VJP thunk. `false` gives a forward-only stack — useful for a stack
  used only at inference, and the only way to skip a chunk's VJP compile when that compile itself
  is prohibitively expensive (a chunked, rolled attention layer's reverse pass can hang rather than
  just being slow — see `memory/lta_chunked_reverse_hangs.md`). Calling `backward` on such a stack
  raises rather than returning something wrong. (default: `true`)
- `active`, `diff_params`, `compile_options`: as in [`CompiledHLOModule`](@ref).

# Parameter and state layout
With `shared_ps = false` (the default), `ps` and `st` must share one key structure: the container at
`ps_path` is keyed `ps_key(1)` … `ps_key(depth)` in **both**. For replay `k`, global blocks
`(k-1)M+1 … kM` are re-keyed onto the chunk's own local `ps_key(1) … ps_key(M)` — a pure NamedTuple
remap, no data movement. With `shared_ps = true` there is no remap: `ps` and `st` go through whole.

# Returns
- `forward` → `(outputs, st_updated)`; `outputs` is the union of fields the chunks produced (the
  carried tensors), and `st_updated` has each chunk's advanced state scattered back onto the global
  keys.
- `backward` → `(; dinputs, dps)`, `dps` nested at `ps_path` and keyed over all `depth` blocks.

!!! note "State threading depends on `shared_ps`"
    Untied (`shared_ps = false`), every chunk reads its own key slice of `ps`/`st` and no slot is
    ever reused, so chunk `k`'s input state is always `_chunk_slice(st, k)` of the *original* `st`,
    independent of what ran before it; updates go to a separate accumulator at the same global keys
    and the backward walk saves no state. Tied (`shared_ps = true`), state genuinely threads, so the
    backward walk saves it alongside each checkpointed input and replays each chunk with the state
    its forward actually used.
"""
struct CompiledHLOStack{L,DK,OK,CK,M,SH<:StaticBool,DP<:StaticBool,PP,PK,F,V,Z} <: AbstractHLOModule
    layer::L
    depth::Int
    num_chunks::Int
    ps_path::PP
    ps_key::PK
    shared_ps::SH
    diff_params::DP
    checkpoint::Int
    fwd::F
    vjp::V
    ct_zeros::Z
end

function CompiledHLOStack(layer, inputs::NamedTuple, ps, st, ct::NamedTuple;
    depth::Int,
    chunk_size::Int=1,
    active::Union{Nothing,Tuple{Vararg{Symbol}}}=nothing,
    ps_path::Tuple{Vararg{Symbol}}=(:blocks,),
    ps_key=i -> Symbol("block_", i),
    shared_ps::Union{Bool,StaticBool}=false,
    diff_params::Union{Bool,StaticBool}=true,
    checkpoint::Int=1,
    compile_backward::Bool=true,
    compile_options=Reactant.CompileOptions())

    depth >= 1 || throw(ArgumentError("depth must be >= 1, got $depth"))
    chunk_size >= 1 || throw(ArgumentError("chunk_size must be >= 1, got $chunk_size"))
    depth % chunk_size == 0 || throw(ArgumentError(
        "depth ($depth) must be divisible by chunk_size ($chunk_size)"))
    checkpoint >= 1 || throw(ArgumentError("checkpoint must be >= 1, got $checkpoint"))

    dk = _resolve_active(inputs, active)
    isempty(dk) && throw(ArgumentError(
        "no float-valued arrays in `inputs`; nothing to differentiate with respect to"))
    # A stacked reverse seeds only the *final* output; every intermediate chunk is seeded purely by
    # the cotangent flowing back from its successor. That is only well defined if each seeded output
    # is also a carried input, which is exactly what the Stack convention gives (`s`, `z`, `m`, …).
    loose = Tuple(k for k in keys(ct) if !(k in dk))
    isempty(loose) || throw(ArgumentError(
        "every key of `ct` must also be a differentiable input key so the cotangent can be " *
        "carried between chunks; $(loose) is not. Use a CompiledHLOModule, or seed only the " *
        "carried outputs."))

    x, c = _split(inputs, dk)
    dp = diff_params isa StaticBool ? diff_params : static(diff_params)
    K = depth ÷ chunk_size

    sh = shared_ps isa StaticBool ? shared_ps : static(shared_ps)

    # With `shared_ps`, `ps` is handed to every replay untouched, so there is no key layout to check
    # and `layer` is the plain (weight-tied) block. Otherwise `layer` must be the M-block CONTAINER
    # (e.g. `PairformerModule(M)`), not a bare block — even at M = 1: each chunk's slice arrives
    # nested at `ps_path`, which a bare block rejects with a confusing FieldError inside the trace.
    sh isa False && let g = _getpath(ps, ps_path)
        for i in 1:depth
            haskey(g, ps_key(i)) || throw(ArgumentError(
                "ps$(isempty(ps_path) ? "" : "." * join(ps_path, ".")) has no field " *
                "$(repr(ps_key(i))); expected blocks keyed $(repr(ps_key(1))) … " *
                "$(repr(ps_key(depth))). Available: $(keys(g))."))
        end
        gs = _getpath(st, ps_path)
        keys(gs) == keys(g) || throw(ArgumentError(
            "ps and st must share one key structure at $(ps_path); ps has $(keys(g)), " *
            "st has $(keys(gs))."))
    end

    # Chunk 1's slice serves as the compile prototype; every other chunk has the same shapes.
    ps1 = _slice_proto(sh, ps_path, ps_key, ps, chunk_size)
    st1 = _slice_proto(sh, ps_path, ps_key, st, chunk_size)

    fwd = @compile compile_options = compile_options _fwd_fn(layer, x, c, ps1, st1)
    vjp = compile_backward ?
          (@compile compile_options = compile_options _vjp_fn(dp, layer, x, c, ps1, st1, ct)) :
          nothing

    ok = keys(first(fwd(layer, x, c, ps1, st1)))
    zs = map(zero, ct)

    return CompiledHLOStack{typeof(layer),dk,ok,keys(ct),chunk_size,typeof(sh),typeof(dp),
        typeof(ps_path),typeof(ps_key),typeof(fwd),typeof(vjp),typeof(zs)}(
        layer, depth, K, ps_path, ps_key, sh, dp, checkpoint, fwd, vjp, zs)
end

_slice_proto(::True, _, _, whole, _) = whole
_slice_proto(::False, path, key, whole, M) = _chunk_slice(path, key, whole, 1, M)

active_keys(::CompiledHLOStack{L,DK}) where {L,DK} = DK
output_keys(::CompiledHLOStack{L,DK,OK}) where {L,DK,OK} = OK
ct_keys(::CompiledHLOStack{L,DK,OK,CK}) where {L,DK,OK,CK} = CK
ct_zeros(s::CompiledHLOStack) = s.ct_zeros
chunk_size(::CompiledHLOStack{L,DK,OK,CK,M}) where {L,DK,OK,CK,M} = M

Base.show(io::IO, s::CompiledHLOStack) = print(io,
    "CompiledHLOStack(", nameof(typeof(s.layer)), "; depth=", s.depth,
    ", chunk_size=", chunk_size(s), ", shared_ps=", s.shared_ps isa True,
    ", diff_params=", s.diff_params isa True, ", checkpoint=", s.checkpoint, ")")

# ── chunk ↔ global key remapping ────────────────────────────────────────────────────────────────

# Global blocks (k-1)M+1 … kM, re-keyed onto the chunk's own local block_1 … block_M and nested
# back at `path` so the chunk layer sees the layout it was built with.
function _chunk_slice(path, key, whole, k::Int, M::Int)
    g = _getpath(whole, path)
    lo = (k - 1) * M
    return _nest(path, NamedTuple{ntuple(j -> key(j), M)}(ntuple(j -> getproperty(g, key(lo + j)), M)))
end
_chunk_slice(s::CompiledHLOStack, whole, k::Int) =
    _chunk_slice(s.ps_path, s.ps_key, whole, k, chunk_size(s))

# Which parameters chunk `k` runs with. Dispatched on the flag rather than branched on, per the
# repo's StaticBool convention: shared weights hand every replay the same `ps`.
_slice_ps(s::CompiledHLOStack, ps, k::Int) = _slice_ps(s.shared_ps, s, ps, k)
_slice_ps(::True, _, ps, _) = ps
_slice_ps(::False, s, ps, k) = _chunk_slice(s, ps, k)

# Which state chunk `k` runs with. Tied weights thread state forward (AF2's structure module threads
# `st_fold` across its blocks); independent blocks re-slice from the original state.
_slice_st(s::CompiledHLOStack, st, st_thread, k::Int) = _slice_st(s.shared_ps, s, st, st_thread, k)
_slice_st(::True, _, _, st_thread, _) = st_thread
_slice_st(::False, s, st, _, k) = _chunk_slice(s, st, k)

# How chunk `k`'s returned state combines. Tied: it IS the next chunk's state. Untied: scatter onto
# this chunk's global keys, leaving every other chunk's slice alone.
_merge_st(s::CompiledHLOStack, st_acc, stk_new, k::Int) =
    _merge_st(s.shared_ps, s, st_acc, stk_new, k)
_merge_st(::True, _, _, stk_new, _) = stk_new
_merge_st(::False, s, st_acc, stk_new, k) = _chunk_scatter(s, st_acc, stk_new, k)

# Inverse: write a chunk's (local-keyed) container back onto the global keys it came from.
function _chunk_scatter(s::CompiledHLOStack, whole, chunk, k::Int)
    M = chunk_size(s)
    g = _getpath(whole, s.ps_path)
    l = _getpath(chunk, s.ps_path)
    lo = (k - 1) * M
    upd = NamedTuple{ntuple(j -> s.ps_key(lo + j), M)}(ntuple(j -> getproperty(l, s.ps_key(j)), M))
    return _setpath(whole, s.ps_path, merge(g, upd))
end

# ── forward ─────────────────────────────────────────────────────────────────────────────────────

function forward(s::CompiledHLOStack, inputs::NamedTuple, ps, st)
    cur, produced, st_acc = inputs, nothing, st
    for k in 1:s.num_chunks
        out, stk_new = _chunk_step(s, cur, ps, st, st_acc, k)
        st_acc = _merge_st(s, st_acc, stk_new, k)
        produced = produced === nothing ? out : merge(produced, out)
        cur = merge(cur, out)
    end
    return produced, st_acc
end

# One chunk forward. Returns the produced fields and this chunk's updated state — locally keyed
# (`block_1 … block_M`) when untied, the whole threaded state when tied. Combining it is the
# caller's job, via `_merge_st`.
function _chunk_step(s::CompiledHLOStack, cur::NamedTuple, ps, st, st_acc, k::Int)
    x, c = _split(cur, active_keys(s))
    stk = _protect_rngs(_slice_st(s, st, st_acc, k))
    out, stk_new = s.fwd(s.layer, x, c, _slice_ps(s, ps, k), stk)
    # Reconciling keeps the state this forward RETURNS structurally identical to the state it was
    # given — required when it threads (tied weights), and worth it untied so the whole forward
    # stays re-callable with its own output, which a recycling loop wants.
    return out, _reconcile_state(stk, stk_new)
end

# ── reverse ─────────────────────────────────────────────────────────────────────────────────────

"""
    backward(s::CompiledHLOStack, inputs, ps, st, ct)

The hand-crossed chunk boundary: a forward pass storing every `checkpoint`-th chunk input, then a
backward walk that recomputes the inputs inside each segment and replays the VJP thunk chunk by
chunk, feeding each chunk's input-cotangent to its predecessor. Only *inputs* are saved — chunk `k`'s
state is re-sliced from the original `st`, since chunks share no state.

Returns `(; dinputs, dps)`.
"""
function backward(s::CompiledHLOStack, inputs::NamedTuple, ps, st, ct::NamedTuple)
    s.vjp === nothing && throw(ArgumentError(
        "this CompiledHLOStack was built with compile_backward = false; it has no VJP thunk. " *
        "Call `compile_backward(s, inputs, ps, st, ct)` to build one."))
    K, cp = s.num_chunks, s.checkpoint
    dk = active_keys(s)
    # Keys the cotangent carries between chunks (checked to be a subset of `dk` at construction),
    # vs. differentiable inputs that are *not* chunk outputs. The former is replaced at each step;
    # the latter accumulates across chunks, because every chunk reads the same external tensor.
    carry = ct_keys(s)
    extra = Tuple(k for k in dk if !(k in carry))

    # Forward, keeping one (input, state) pair per segment. Untied, the state entry is inert —
    # chunk k always re-slices the original `st`. Tied, state THREADS across replays, so the saved
    # entry is what makes a recomputed chunk see the state its forward actually saw; without it a
    # recomputed chunk would draw different noise and the gradient would be silently wrong.
    nseg = cld(K, cp)
    saved = Vector{Any}(undef, nseg)
    cur, st_acc = inputs, st
    for k in 1:K
        ((k - 1) % cp == 0) && (saved[(k-1)÷cp+1] = (cur, st_acc))
        out, stk_new = _chunk_step(s, cur, ps, st, st_acc, k)
        st_acc = _merge_st(s, st_acc, stk_new, k)
        cur = merge(cur, out)
        _collect_device(k)
        _observe(:forward, k)
    end

    dcarry, dextra = ct, nothing
    dps = s.diff_params isa True ? Vector{Any}(undef, K) : nothing

    for seg in nseg:-1:1
        lo = (seg - 1) * cp + 1
        hi = min(K, lo + cp - 1)
        # Recompute this segment's chunk inputs from its checkpoint. Chunk `hi`'s *output* is never
        # needed, so the replay stops one short — which makes `cp == 1` a genuine no-op.
        locals = Vector{Any}(undef, hi - lo + 1)
        cur, st_acc = saved[seg]
        for k in lo:hi
            locals[k-lo+1] = (cur, st_acc)
            if k < hi
                out, stk_new = _chunk_step(s, cur, ps, st, st_acc, k)
                st_acc = _merge_st(s, st_acc, stk_new, k)
                cur = merge(cur, out)
                _observe(:recompute, k)
            end
        end
        for k in hi:-1:lo
            ck, sk = locals[k-lo+1]
            xk, cck = _split(ck, dk)
            g = s.vjp(s.diff_params, s.layer, xk, cck, _slice_ps(s, ps, k),
                _protect_rngs(_slice_st(s, st, sk, k)), dcarry)
            dcarry = NamedTuple{carry}(g[2])
            isempty(extra) || (dextra = _accumulate_nt(dextra, NamedTuple{extra}(g[2])))
            dps === nothing || (dps[k] = g[4])
            _collect_device(k)
            _observe(:vjp, k)
        end
        # This segment's recomputed inputs die here; the next segment's are about to be built.
        locals = nothing
        GC.gc()
    end

    dinputs = isempty(extra) ? dcarry : merge(dextra, dcarry)
    dps === nothing && return (; dinputs, dps=nothing)
    return (; dinputs, dps=_finalize_dps(s.shared_ps, s, dps))
end

"""
    compile_backward(s::CompiledHLOStack, inputs, ps, st, ct; compile_options=Reactant.CompileOptions())

See the [`AbstractHLOModule`](@ref) docstring of the same name for the contract. `inputs`/`ps`/`st`/
`ct` must be the same prototypes `s` was originally constructed with — `ps`/`st` the WHOLE stack's,
not chunk 1's slice; the chunk-1 slice is taken here exactly as the constructor takes it.
"""
function compile_backward(s::CompiledHLOStack, inputs::NamedTuple, ps, st, ct::NamedTuple;
    compile_options=Reactant.CompileOptions())

    s.vjp === nothing || return s
    x, c = _split(inputs, active_keys(s))
    ps1 = _slice_proto(s.shared_ps, s.ps_path, s.ps_key, ps, chunk_size(s))
    st1 = _slice_proto(s.shared_ps, s.ps_path, s.ps_key, st, chunk_size(s))
    vjp = @compile compile_options = compile_options _vjp_fn(
        s.diff_params, s.layer, x, c, ps1, st1, ct)
    return CompiledHLOStack{typeof(s.layer),active_keys(s),output_keys(s),ct_keys(s),chunk_size(s),
        typeof(s.shared_ps),typeof(s.diff_params),typeof(s.ps_path),typeof(s.ps_key),
        typeof(s.fwd),typeof(vjp),typeof(s.ct_zeros)}(
        s.layer, s.depth, s.num_chunks, s.ps_path, s.ps_key, s.shared_ps, s.diff_params,
        s.checkpoint, s.fwd, vjp, s.ct_zeros)
end

# Tied weights: the SAME parameter is read by every replay, so its gradient is the SUM over replays
# — not one gradient per block. Getting this wrong is silent: the shapes are identical either way.
_finalize_dps(::True, _, dps) = reduce(_add_nt, dps)

# Untied: scatter each chunk's local-keyed dps back onto the global block keys.
function _finalize_dps(::False, s::CompiledHLOStack, dps)
    M = chunk_size(s)
    gkeys = Tuple(s.ps_key(i) for i in 1:s.depth)
    gvals = Tuple(getproperty(_getpath(dps[(i-1)÷M+1], s.ps_path), s.ps_key((i - 1) % M + 1))
                  for i in 1:s.depth)
    return _nest(s.ps_path, NamedTuple{gkeys}(gvals))
end

# Recursive elementwise add over the parameter tree.
_add_nt(a::NamedTuple{K}, b::NamedTuple{K}) where {K} =
    NamedTuple{K}(map(k -> _add_nt(getfield(a, k), getfield(b, k)), K))
_add_nt(a::Tuple, b::Tuple) = map(_add_nt, a, b)
_add_nt(a, b) = a .+ b

_accumulate_nt(::Nothing, new::NamedTuple) = new
_accumulate_nt(acc::NamedTuple{K}, new::NamedTuple{K}) where {K} =
    NamedTuple{K}(map(+, values(acc), values(new)))

# Device buffers are freed by finalizers, but the host heap barely grows when an 82 MiB pair tensor
# is dropped — so Julia sees no reason to collect and a deep walk accumulates gigabytes of dead
# device memory. Nothing in the runtime signals this; the GC has to be prodded by hand, and it has
# to be a FULL collection; an incremental one only sweeps the young generation.
#
# This does NOT fully solve it, and the residue is worth knowing about before you tune `checkpoint`
# against it. Measured on Metal at N=400, `--diff-params false`, min-available RAM:
#   depth 4  (ckpt 2): 8.7 GB     depth 16 (ckpt 4): 6.8 GB  incremental GC every 4 blocks
#                                 depth 16 (ckpt 4): 7.2 GB  full GC every block (+33% wall time)
# i.e. ~100-158 MB per block still accumulates — about one pair tensor — and only ~330 MB of the
# gap is the checkpoint tape. Something below Julia's GC (the PJRT/MLX allocator is the suspect)
# retains it, so `checkpoint` cannot buy back what is being lost per block, and depth 64 at N=400
# still does not fit on a 32 GB machine. Every 4 chunks is the compromise: most of the benefit,
# a fraction of the cost.
const GC_EVERY = Ref(4)
_collect_device(i::Int) = (GC_EVERY[] > 0 && i % GC_EVERY[] == 0 && GC.gc(); nothing)

# Per-chunk observation hook, called as `ON_BLOCK[](phase::Symbol, k::Int)` with `phase` one of
# `:forward`, `:recompute`, `:vjp`. Run-to-run memory comparisons at this scale are worthless — the
# noise floor is ±0.5 GiB — so the only way to see growth is to watch it *within* one walk.
const ON_BLOCK = Ref{Any}(nothing)
_observe(phase::Symbol, i::Int) = (ON_BLOCK[] === nothing || ON_BLOCK[](phase, i); nothing)
