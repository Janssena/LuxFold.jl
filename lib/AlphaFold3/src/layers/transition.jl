# NOTE: the local `SwiGLU`/GLU implementation was removed in favour of the shared
# `SwiGLU` from LuxTriangleAttention (re-exported via LuxFoldCore). `SwiGLUTransition`
# is a thin AF3 wrapper (LayerNorm → SwiGLU → Dense_out). The shared SwiGLU is used
# **unfused** so its two Denses (`linear`=value, `gate`=swish) line up 1:1 with openfold's
# `linear_b`/`linear_a` for weight sync.

"""
    SwiGLUTransition(chn_in, n; is_4d=false, kwargs...)

SwiGLU transition: `LayerNorm → SwiGLU(chn_in → n·chn_in) → Dense(n·chn_in → chn_in)`.

# Inputs
- `x`: `[chn_in, N, B]` (single) or `[chn_in, N, N, B]` (pair, `is_4d=true`)
- `mask`: optional `AbstractArray{Bool}` (`[N, B]` or `[N, N, B]`); `nothing` ⇒ no masking

# Returns
- `x_out`: same shape as `x`, with masked positions zeroed via `ifelse`
"""
struct SwiGLUTransition{LN,SG,LO} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :swiglu, :linear_out)}
    layer_norm::LN
    swiglu::SG
    linear_out::LO
end

function SwiGLUTransition(chn_in::Int, n::Int; is_4d::Bool=false)
    chn_hidden = n * chn_in
    shape = is_4d ? (chn_in, 1, 1) : (chn_in, 1)
    return SwiGLUTransition(
        Lux.LayerNorm(shape; dims=1),                          # openfold: LayerNorm WITH offset
        SwiGLU(chn_in => chn_hidden; fused=false, use_bias=false), # openfold swiglu_init: linear_a/linear_b bias=false
        Lux.Dense(chn_hidden => chn_in; use_bias=false),           # openfold: linear_out bias=false
    )
end

"""
    apply_transition_mask!(x, mask)

Zero out masked positions of `x [C, dims..., B]` in place. Dispatches on `mask::Nothing`
(no-op) vs `mask::AbstractArray{Bool}` (`[dims..., B]`) via `ifelse` (never multiplication).
Handles both single (`[N, B]`) and pair (`[N, N, B]`) masks.
"""
apply_transition_mask!(x, ::Nothing) = nothing
function apply_transition_mask!(x::AbstractArray{T}, mask::AbstractArray{Bool}) where {T}
    mask_reshaped = reshape(mask, 1, size(mask)...)
    @. x = ifelse(mask_reshaped, x, zero(T))
    return nothing
end

(l::SwiGLUTransition)(inputs::NamedTuple, ps, st) = l(
    inputs.x,
    get(inputs, :mask, nothing),
    ps, st
)

(l::SwiGLUTransition)(x::AbstractArray, ps, st) = l(x, nothing, ps, st)

function (l::SwiGLUTransition)(x, mask, ps, st)
    x_ln, st_ln = l.layer_norm(x, ps.layer_norm, st.layer_norm)
    x_swiglu, st_swiglu = l.swiglu(x_ln, ps.swiglu, st.swiglu)
    x_out, st_out = l.linear_out(x_swiglu, ps.linear_out, st.linear_out)

    apply_transition_mask!(x_out, mask)

    return x_out, merge(st, (; layer_norm=st_ln, swiglu=st_swiglu, linear_out=st_out))
end

# =============================================================================

"""
    ResidualSwiGLUBlock(chn, n; is_4d=false)

Two sequential residual `SwiGLUTransition`s: `x += transition_1(x, mask)`, then
`x += transition_2(x, mask)`. Replaces the `Lux.Chain(SkipConnection(...), SkipConnection(...))`
pattern. Parameters are keyed `transition_1`, `transition_2` (integer-indexable for
sync helpers).

# Inputs
- `x`: `[chn, N, B]` or `[chn, N, N, B]` (when `is_4d=true`)
- `mask`: `AbstractArray{Bool}` or `nothing`
"""
struct ResidualSwiGLUBlock{T1,T2} <: Lux.AbstractLuxContainerLayer{(:transition_1, :transition_2)}
    transition_1::T1
    transition_2::T2
end

function ResidualSwiGLUBlock(chn::Int, n::Int; is_4d::Bool=false)
    return ResidualSwiGLUBlock(
        SwiGLUTransition(chn, n; is_4d),
        SwiGLUTransition(chn, n; is_4d),
    )
end

function (l::ResidualSwiGLUBlock)(x, mask, ps, st)
    y1, st1 = l.transition_1(x, mask, ps.transition_1, st.transition_1)
    x = x .+ y1
    y2, st2 = l.transition_2(x, mask, ps.transition_2, st.transition_2)
    x = x .+ y2
    return x, merge(st, (; transition_1=st1, transition_2=st2))
end
