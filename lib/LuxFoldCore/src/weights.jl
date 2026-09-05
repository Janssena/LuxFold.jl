# weights.jl — Shared weight download, caching, and parameter-copy machinery.
#
# Model-agnostic foundation for loading released weights into Lux parameter NamedTuples, shared by
# AlphaFold2 / AlphaFold3 / Boltz2. See docs/UNIFICATION_DESIGN.md §4–§5.
#
# ## Distribution & cache (design §5 DECISIONS)
#
# Each model's official weights are converted **once** to `.safetensors` (raw framework layout
# preserved) and hosted on the `luxfold/weights` HuggingFace repo. At load time a **sha256-pinned
# lazy fetch** (`fetch_weights`) downloads the file into a per-family cache directory and reuses it
# thereafter. The cache dir is env-var overridable (`WeightsCache`): AlphaFold2 + AlphaFold3 share
# `AF_CACHE` (default `~/.alphafold`); Boltz2 uses `BOLTZ_CACHE` (default `~/.boltz`). A hand-placed
# file in the cache (e.g. a gated checkpoint the user obtained separately) is used as-is.
#
# ## Layout correction (design §5.3/§5.5)
#
# Conversion is deliberately dumb — arrays are stored in their raw framework layout — so the
# framework→Julia layout correction happens **here at load time**. The primitive copiers are
# generalised over a `get(prefix, leaf)` closure: each backend's closure knows its own key syntax,
# `Float32` conversion, stacked-block slicing, and dimension convention. This is what makes one
# copier serve NPZ (JAX/AF2) and torch state-dicts (AF3/Boltz2) alike; the copier bodies encode the
# per-layer-type reshapes (JAX MHA/OPM), reused verbatim from AF2's original loader.

import Downloads, SafeTensors, SHA

# ============================================================================
# Cache directories
# ============================================================================

"""
    WeightsCache(env, default)

A resolvable weights-cache directory. `cache_dir` resolves it in the order (mirroring
`LuxFold`'s `ccd_dir`): a session `set_cache_dir!` override → the `env` environment variable →
`default`.

```julia
const AF_CACHE = WeightsCache("AF_CACHE", joinpath(homedir(), ".alphafold"))
cache_dir(AF_CACHE)                 # ~/.alphafold, or \$AF_CACHE if set
set_cache_dir!(AF_CACHE, "/data")   # session override
```
"""
struct WeightsCache
    override::Base.RefValue{String}
    env::String
    default::String
end
WeightsCache(env::AbstractString, default::AbstractString) =
    WeightsCache(Ref(""), String(env), String(default))

"""
    cache_dir(c::WeightsCache) -> String

Resolve the cache directory: session override → `ENV[c.env]` → `c.default`.
"""
function cache_dir(c::WeightsCache)
    isempty(c.override[]) || return c.override[]
    (haskey(ENV, c.env) && !isempty(ENV[c.env])) && return ENV[c.env]
    return c.default
end

"""
    set_cache_dir!(c::WeightsCache, path)

Override the cache directory for this Julia session (takes precedence over the env var and the
default). Pass an empty string to clear the override.
"""
set_cache_dir!(c::WeightsCache, path::AbstractString) = (c.override[] = String(path); nothing)

# ============================================================================
# Model registry schema
# ============================================================================

"""
    ModelRegistryEntry

One released-weight variant's metadata, shared across models (design §4.1). `format` selects the
on-disk reader backend (`:safetensors` under the hosted-conversion strategy; `:npz` for AF2's raw
DeepMind files); `layout` (`:jax` | `:torch`) selects the load-time layout correction; `options`
carries per-variant construction/loading flags.
"""
const ModelRegistryEntry = @NamedTuple begin
    architecture :: Symbol
    files        :: Vector{String}
    url          :: String
    sha256       :: String
    format       :: Symbol
    layout       :: Symbol
    options      :: NamedTuple
end

"""
    ModelRegistry

`Dict{Symbol, ModelRegistryEntry}` — a per-model registry mapping released-weight symbols to their
[`ModelRegistryEntry`](@ref). AF2/AF3/Boltz2 each populate one.
"""
const ModelRegistry = Dict{Symbol,ModelRegistryEntry}

# ============================================================================
# Download + lazy fetch
# ============================================================================

# Simple progress callback for Downloads.download.
function _download_progress(total::Integer, now::Integer)
    total > 0 || return
    pct = round(Int, 100 * now / total)
    print("\r  $(pct)%  $(round(now / 1e6; digits=1)) / $(round(total / 1e6; digits=1)) MB   ")
    now == total && println()
end

"""
    download_file(url, dest; force=false) -> dest

Download `url` to `dest` (creating parent directories), showing a progress bar. Skips the download
if `dest` already exists unless `force=true`.
"""
function download_file(url::AbstractString, dest::AbstractString; force::Bool=false)
    (isfile(dest) && !force) && return dest
    mkpath(dirname(dest))
    @info "Downloading weights…" url dest
    Downloads.download(url, dest; progress=_download_progress)
    return dest
end

_sha256(path::AbstractString) = open(io -> bytes2hex(SHA.sha256(io)), path)

"""
    fetch_weights(entry, cache; force=false) -> path

Lazily fetch `entry`'s hosted file into `cache_dir(cache)`, verifying its `sha256`. A file already
present with a matching hash (or a hand-placed file when `entry.sha256` is empty) is reused; a
present-but-corrupt file is re-downloaded. Returns the local path.
"""
function fetch_weights(entry::ModelRegistryEntry, cache::WeightsCache; force::Bool=false)
    path = joinpath(cache_dir(cache), only(entry.files))
    if isfile(path) && !force
        (isempty(entry.sha256) || _sha256(path) == entry.sha256) && return path
        @warn "Cached weight file failed its sha256 check; re-downloading." path
    end
    download_file(entry.url, path; force=true)
    if !isempty(entry.sha256)
        got = _sha256(path)
        got == entry.sha256 ||
            error("sha256 mismatch for $path:\n  got      $got\n  expected $(entry.sha256)")
    end
    return path
end

# ============================================================================
# State-dict reading
# ============================================================================

"""
    read_state_dict(path) -> AbstractDict{String}

Read a `.safetensors` file into a `key → tensor` mapping. Tensors are returned in their stored
(raw framework) layout; the load-time `get` closure applies `Float32` conversion and any layout
correction (see the file header).
"""
read_state_dict(path::AbstractString) = SafeTensors.load_safetensors(path)

"""
    write_state_dict(path, sd::AbstractDict{String, <:AbstractArray}) -> path

Write `sd` to a `.safetensors` file (creating parent directories). Arrays are stored in their given
Julia layout and round-trip **losslessly** through [`read_state_dict`](@ref) (verified). This is what
lets the AF2 conversion run in pure Julia: `NPZ.npzread` yields Julia-layout arrays, and a state-dict
written here loads back through the *same* copiers as the original NPZ. Used by the maintainer
conversion scripts under `scripts/`.
"""
function write_state_dict(path::AbstractString, sd::AbstractDict{String,<:AbstractArray})
    mkpath(dirname(path))
    SafeTensors.serialize(path, sd)
    return path
end

# ============================================================================
# Flat parameter (de)serialisation — "smart convert, trivial load"
#
# A maintainer converter applies any framework→Lux layout transforms ONCE and writes Lux-native
# arrays under dot-joined `ps` leaf paths (e.g. `evoformer.blocks.block_1.msa_att_row.linear_z.weight`).
# The runtime loader then copies them straight back with zero layout knowledge — model-agnostic, so
# the same two functions serve every model whose weights are hosted in this flat form.
# ============================================================================

"""
    flatten_params(ps) -> Dict{String,Array}

Flatten a nested Lux parameter `NamedTuple`/`Tuple` `ps` into a flat `Dict` keyed by dot-joined leaf
paths. Empty `NamedTuple` leaves (e.g. `NoOpLayer` params) contribute nothing. Inverse of
[`load_flat_weights!`](@ref); pass the result to [`write_state_dict`](@ref) to produce a converted
`.safetensors`.
"""
function flatten_params(ps)
    d = Dict{String,Array}()
    _flatten_params!(d, ps, "")
    return d
end
_join(prefix, k) = isempty(prefix) ? string(k) : string(prefix, ".", k)
_flatten_params!(d, x::NamedTuple, prefix) =
    (for k in keys(x); _flatten_params!(d, x[k], _join(prefix, k)); end; d)
_flatten_params!(d, x::Tuple, prefix) =
    (for (i, v) in enumerate(x); _flatten_params!(d, v, _join(prefix, i)); end; d)
_flatten_params!(d, x::AbstractArray, prefix) = (d[prefix] = collect(x); d)

"""
    load_flat_weights!(ps, sd::AbstractDict; prefix="") -> ps

Copy arrays from a flat state dict `sd` (dot-joined leaf paths, e.g. [`read_state_dict`](@ref) of a
converted `.safetensors`) into the nested parameter container `ps`, in place. Every array leaf in `ps`
must have a matching key in `sd` (extra `sd` keys are ignored). Inverse of [`flatten_params`](@ref).
"""
load_flat_weights!(ps, sd::AbstractDict; prefix::AbstractString="") = (_load_flat!(ps, sd, prefix); ps)
_load_flat!(x::NamedTuple, sd, prefix) =
    (for k in keys(x); _load_flat!(x[k], sd, _join(prefix, k)); end)
_load_flat!(x::Tuple, sd, prefix) =
    (for (i, v) in enumerate(x); _load_flat!(v, sd, _join(prefix, i)); end)
function _load_flat!(x::AbstractArray, sd, prefix)
    haskey(sd, prefix) || throw(ArgumentError("load_flat_weights!: missing key $(repr(prefix))"))
    x .= sd[prefix]
    return
end

# ============================================================================
# Primitive parameter copiers (generalised over a `get(prefix, leaf)` closure)
#
# `get(prefix, leaf)` returns the `Float32` array for one parameter leaf (already stacked-block
# sliced and in the raw framework layout), or `nothing` when the key is absent (optional biases).
# The stacked/non-stacked distinction lives entirely in the closure, so there is a single copier
# for both — no `_s!` duplication.
# ============================================================================

# JAX MHA Q/K/V: Julia reads [C_hd, C_in, H] → need [H*C_hd, C_in].
function _mha_qkv_w(w::AbstractArray{T,3}) where {T}
    C_hd, C_in, H = size(w)
    return reshape(permutedims(w, (3, 1, 2)), H * C_hd, C_in)
end

# JAX MHA output: Julia reads [C_out, C_hd, H] → need [C_out, H*C_hd].
function _mha_out_w(w::AbstractArray{T,3}) where {T}
    C_out, C_hd, H = size(w)
    return reshape(permutedims(w, (1, 3, 2)), C_out, H * C_hd)
end

# JAX OPM output: Julia reads [C_z, C_hid, C_hid] → need [C_z, C_hid^2].
function _opm_out_w(w::AbstractArray{T,3}) where {T}
    C_z, a, b = size(w)
    return reshape(w, C_z, a * b)
end

function _load_dense!(ps, get, prefix::String)
    ps.weight .= get(prefix, "weights")
    b = get(prefix, "bias")
    (hasproperty(ps, :bias) && !isnothing(b)) && (ps.bias .= b)
    return ps
end

# LayerNorm: reshape the 1D vector to the actual parameter shape in ps
# (Lux.LayerNorm((C,)) → [C,1]; ((C,1,1)) → [C,1,1]; etc.).
function _load_layernorm!(ps, get, prefix::String)
    ps.scale .= reshape(get(prefix, "scale"), size(ps.scale))
    b = get(prefix, "offset")
    (hasproperty(ps, :bias) && !isnothing(b)) && (ps.bias .= reshape(b, size(ps.bias)))
    return ps
end

# MHA self-attention (fused QKV): qkv.weight = vcat(Q, K, V).
# MHA cross-attention (fused KV): qkv.q.weight + qkv.kv.weight = vcat(K, V).
function _load_mha!(ps_mha, get, prefix::String; gated::Bool=true)
    qw = _mha_qkv_w(get(prefix, "query_w"))
    kw = _mha_qkv_w(get(prefix, "key_w"))
    vw = _mha_qkv_w(get(prefix, "value_w"))

    if hasproperty(ps_mha.qkv, :weight)
        ps_mha.qkv.weight .= vcat(qw, kw, vw)          # fused self-attention
    else
        ps_mha.qkv.q.weight  .= qw                       # fused-KV cross-attention
        ps_mha.qkv.kv.weight .= vcat(kw, vw)
    end

    ps_mha.out.weight .= _mha_out_w(get(prefix, "output_w"))
    ps_mha.out.bias   .= get(prefix, "output_b")

    if gated
        ps_mha.gate.weight .= _mha_qkv_w(get(prefix, "gating_w"))
        # gating_b: JAX stores [H, C_head] → Julia reads [C_head, H]; permute+flatten to H-first.
        gb = get(prefix, "gating_b")
        ps_mha.gate.bias .= ndims(gb) == 2 ? vec(permutedims(gb, (2, 1))) : vec(gb)
    end
    return ps_mha
end

# ============================================================================
# Generic loader entry points (methods added per package; design §5.2 API)
# ============================================================================

"""
    load_weights!(ps, model, variant::Symbol; kwargs...)

Load released weights for `variant` into `ps` (created from `model`). Dispatches on the model type;
each package (`AlphaFold`, `AlphaFold3Model`, `Boltz2`) adds a method delegating to its
`load_<model>_weights!`. Returns `ps`.
"""
function load_weights! end

"""
    download_weights(model; kwargs...)

Download (and, where applicable, convert/cache) the released weights for `model`'s family. Each
package adds a method delegating to its `download_<model>_weights`.
"""
function download_weights end
