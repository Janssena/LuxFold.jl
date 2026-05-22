"""
    Transition(c_in; n=4, rank=4, use_bias=true)

A generic transition layer that applies a two-layer MLP with LayerNorm normalization.
This is the core building block for both `MSATransition` and `PairTransition`.

Implements the pattern: LayerNorm → Linear(c_in → n·c_in, relu) → Linear(n·c_in → c_in)

# Arguments
- `c_in`: Input channel dimension

# Keyword Arguments
- `n`: Expansion factor for the hidden dimension (default: 4)
- `rank`: Rank of the input tensor, 3 or 4 (default: 4)
- `use_bias`: NamedTuple or Bool specifying bias usage for linear layers

# Inputs
- `x`: Input tensor of shape `[c_in, N, B]` (rank 3) or `[c_in, N1, N2, B]` (rank 4)
- `mask`: Optional boolean mask of shape `[N, B]` (rank 3) or `[N1, N2, B]` (rank 4)

# Returns
- `y`: Output tensor of same shape as `x`
- `st`: Updated state containing states for `layer_norm`, `linear_1`, `linear_2`
"""
struct Transition{LN,L1,L2} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :linear_1, :linear_2)}
    layer_norm::LN
    linear_1::L1
    linear_2::L2
end

function Transition(
    chn_in::Int;
    n::Int=4,
    rank::Int=4,
    use_bias=true,
)
    @assert rank == 3 || rank == 4 "rank should be either 3 or 4."

    use_bias = resolve_defaults(use_bias, (:linear_1, :linear_2, :layer_norm))

    shape = rank == 3 ? (chn_in, 1) : (chn_in, 1, 1)
    layer_norm = if use_bias.layer_norm
        Lux.LayerNorm(shape; dims=1)
    else
        LayerNormNoBias(shape; dims=1)
    end

    linear_1 = Lux.Dense(chn_in => n * chn_in, Lux.relu; use_bias=use_bias.linear_1)
    linear_2 = Lux.Dense(n * chn_in => chn_in; use_bias=use_bias.linear_2)

    return Transition(layer_norm, linear_1, linear_2)
end

"""
    MSATransition(chn_msa; n=4)

Factory function that creates a `Transition` layer for MSA channels.
Default expansion factor `n=4` matches openfold's EvoformerStack and ExtraMSAStack.
Hardcodes `use_bias=true` and `rank=4` per the Python openfold defaults.

# Arguments
- `chn_msa`: MSA channel dimension

# Keyword Arguments
- `n`: Expansion factor (default: 4)

# Inputs
- `x`: MSA data tensor of shape `[c_m, N_res, N_seq, B]`
- `mask`: Optional boolean mask of shape `[N_res, N_seq, B]`
"""
MSATransition(chn_msa::Int; n::Int=4) = Transition(chn_msa; n, rank=4, use_bias=true)

"""
    PairTransition(c_z; n=2)

Factory function that creates a `Transition` layer for pair channels.
Default expansion factor `n=2` matches openfold's TemplatePairStack.
Hardcodes `use_bias=true` and `rank=4` per the Python openfold defaults.

# Arguments
- `c_z`: Pair channel dimension

# Keyword Arguments
- `n`: Expansion factor (default: 2)

# Inputs
- `x`: Pair representation tensor of shape `[c_z, N, N, B]`
- `mask`: Optional boolean mask of shape `[N, N, B]`
"""
PairTransition(c_z::Int; n::Int=2) = Transition(c_z; n, rank=4, use_bias=true)

"""
    apply_transition_mask!(x, mask)

Apply a boolean mask to the input tensor in-place.
Positions where `mask` is `false` are set to `zero(eltype(x))`.

# Arguments
- `x`: Input tensor (modified in-place)
- `mask`: Boolean mask array, or `nothing`
"""
apply_transition_mask!(x, ::Nothing) = nothing

function apply_transition_mask!(x::AbstractArray{T}, mask::AbstractArray{Bool}) where T
    _zero = zero(T)
    mask_reshaped = reshape(mask, 1, size(mask)...)
    @. x = ifelse(mask_reshaped, x, _zero)
    return nothing
end

# Dispatch methods

(l::Transition)(x, ps, st) = l(x, nothing, ps, st)

(l::Transition)(inputs::NamedTuple, ps, st) = l(
    inputs.x,
    get(inputs, :mask, nothing),
    ps, st
)

function (l::Transition)(x, mask, ps, st)
    x, layer_norm_st = l.layer_norm(x, ps.layer_norm, st.layer_norm)
    x, linear_1_st = l.linear_1(x, ps.linear_1, st.linear_1)
    x, linear_2_st = l.linear_2(x, ps.linear_2, st.linear_2)

    apply_transition_mask!(x, mask)

    return x, merge(st, (; layer_norm=layer_norm_st, linear_1=linear_1_st, linear_2=linear_2_st))
end
