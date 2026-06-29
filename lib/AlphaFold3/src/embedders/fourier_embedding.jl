
"""
    FourierEmbedding(chn_fourier; seed=42)

Fourier feature embedding of a scalar (the diffusion noise level), AF3 Algorithm 22 —
1:1 port of `openfold3...input_embedders.FourierEmbedding`.

Frequencies `w` (standard normal) and phases `b` (uniform `[0,1)`) are sampled once from
`seed` and held as **non-trainable state** (Python `register_buffer`). The Julia init RNG
does not reproduce torch's generator; parity is via syncing the `w`/`b` buffers.

# Inputs
- `x`: `[1, B]` (broadcastable) — the noise level σ.

# Returns
- `[chn_fourier, B]` = `cos(2π (x·w + b))`. Output dim is `chn_fourier` (no `sin` term).
- `st`: unchanged.
"""
struct FourierEmbedding <: Lux.AbstractLuxLayer
    chn_fourier::Int
    seed::Int
end

FourierEmbedding(chn_fourier::Int; seed::Int=42) = FourierEmbedding(chn_fourier, seed)

Lux.initialparameters(::Random.AbstractRNG, ::FourierEmbedding) = NamedTuple()

function Lux.initialstates(::Random.AbstractRNG, l::FourierEmbedding)
    local_rng = Random.Xoshiro(l.seed)
    w = randn(local_rng, Float32, l.chn_fourier)   # frequencies (synced from Python in tests)
    b = rand(local_rng, Float32, l.chn_fourier)    # phases
    return (; w, b)
end

function (l::FourierEmbedding)(x::AbstractArray, ps, st)
    T = eltype(x)
    out = @. cos(T(2π) * (x * st.w + st.b))     # [chn_fourier, B]
    return out, st
end
