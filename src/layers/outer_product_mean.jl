"""
    OuterProductMean(chn_in, chn_z, chn_hidden; kwargs...)

Computes an outer product mean from an MSA representation to update a pair representation. 
This layer captures correlations between columns in the MSA.

# Arguments
- `chn_in`: Number of channels in the MSA input `m`.
- `chn_z`: Number of channels in the output pair representation `y`.
- `chn_hidden`: The hidden dimension for the internal projections.

# Keyword Arguments
- `eps`: A small constant for numerical stability during normalization.
- `use_bias`: Whether to use bias in the initial projections.
- `use_clamp`: If `true`, clamps the normalization factor (mask sum) to at least 1.0.
- `project_first`: If `true` (AlphaFold3 style), applies the output projection before 
  dividing by the normalization factor. If `false` (Boltz2 style), divides first.

# Inputs
- `m`: MSA tensor. Expected shape: `[chn_in, N, S, B]` where `chn_in` is channels,
  `N` is the residue sequence length (number of positions), `S` is the MSA sequence depth (number of sequences), and `B` is batch size.
- `mask`: Optional MSA mask. Expected shape: `[N, S, B]`.

# Returns
- `y`: Pair update tensor. Shape: `[chn_z, N, N, B]`.
- `st`: Updated state containing states for `layer_norm`, `linear1`, `linear2`, and `linear_out`.
"""
struct OuterProductMean{LN,L1,L2,LO,UC,PF} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :linear1, :linear2, :linear_out)}
    layer_norm::LN
    linear1::L1
    linear2::L2
    linear_out::LO
    eps::Float32
    use_clamp::UC
    project_first::PF
end

function OuterProductMean(
    chn_in::Int, chn_z::Int, chn_hidden::Int;
    eps=1e-3, use_bias=true, use_clamp=false, project_first=false
)
    return OuterProductMean(
        Lux.LayerNorm((chn_in, 1, 1); dims=1),
        Lux.Dense(chn_in => chn_hidden; use_bias=use_bias),
        Lux.Dense(chn_in => chn_hidden; use_bias=use_bias),
        Lux.Dense(chn_hidden^2 => chn_z),
        Float32(eps),
        static(use_clamp),
        static(project_first)
    )
end

(l::OuterProductMean)(inputs::NamedTuple, ps, st) = l(
    inputs.m,
    get(inputs, :mask, nothing),
    ps, st
)

(l::OuterProductMean)(m, ps, st) = l(m, nothing, ps, st)

@inline _apply_normalization(norm, ::True, eps, T) = max.(norm, one(T))
@inline _apply_normalization(norm, ::False, eps, T) = norm .+ T(eps)

@inline function _project_and_normalize(outer, norm, linear_out, ps, st, ::True)
    y, st_out = linear_out(outer, ps, st)
    return y ./ norm, st_out
end

@inline function _project_and_normalize(outer, norm, linear_out, ps, st, ::False)
    outer_div = outer ./ norm
    y, st_out = linear_out(outer_div, ps, st)
    return y, st_out
end

@inline apply_opm_mask!(a, b, ::Nothing) = nothing

@inline function apply_opm_mask!(a::AbstractArray{T}, b::AbstractArray{T}, mask::AbstractArray{<:Any,3}) where T
    _zero = zero(T)
    mask_expanded = reshape(mask, 1, size(mask)...) # [1, N, S, B]
    @. a = ifelse(mask_expanded, a, _zero)
    @. b = ifelse(mask_expanded, b, _zero)
    return nothing
end

@inline _compute_norm(m, ::Nothing, N, S, B, T) = fill!(similar(m, 1, N, N, B), T(S))

@inline function _compute_norm(m, mask::AbstractArray{<:Any,3}, N, S, B, T)
    norm = Lux.batched_matmul(T.(mask), T.(mask); lhs_contracting_dim=2, rhs_contracting_dim=2)
    return reshape(norm, 1, N, N, B)
end

function (l::OuterProductMean)(m::AbstractArray{T,4}, mask, ps, st) where T
    chn_in, N, S, B = size(m)
    chn_hidden = size(ps.linear1.weight, 1)

    m_ln, st_ln = l.layer_norm(m, ps.layer_norm, st.layer_norm)

    a, st_l1 = l.linear1(m_ln, ps.linear1, st.linear1) # [H, N, S, B]
    b, st_l2 = l.linear2(m_ln, ps.linear2, st.linear2) # [H, N, S, B]

    apply_opm_mask!(a, b, mask)

    a_flat = reshape(a, chn_hidden * N, S, B)
    b_flat = reshape(b, chn_hidden * N, S, B)

    outer = Lux.batched_matmul(a_flat, b_flat; lhs_contracting_dim=2, rhs_contracting_dim=2)
    outer = reshape(outer, chn_hidden, N, chn_hidden, N, B)
    outer = permutedims(outer, (3, 1, 2, 4, 5))
    outer = reshape(outer, chn_hidden * chn_hidden, N, N, B)

    norm = _compute_norm(m, mask, N, S, B, T)
    norm_clamped = _apply_normalization(norm, l.use_clamp, l.eps, T)

    y, st_out = _project_and_normalize(outer, norm_clamped, l.linear_out, ps.linear_out, st.linear_out, l.project_first)

    return y, (layer_norm=st_ln, linear1=st_l1, linear2=st_l2, linear_out=st_out)
end
