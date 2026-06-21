"""
    compute_plddt(logits)

Compute per-residue pLDDT scores from pLDDT logits.

The computation is: softmax over channel dim → weighted sum by bin bounds → scale by 100.

# Arguments
- `logits`: pLDDT logits `[no_bins, N, B]`

# Returns
- `scores`: Per-residue pLDDT scores `[N, B]` in range [0, 100]
"""
function compute_plddt(logits::AbstractArray{T, 3}) where T
    no_bins = size(logits, 1)
    bin_width = one(T) / no_bins
    bounds = T.(range(0.5 * bin_width; step=bin_width, length=no_bins))
    probs = Lux.softmax(logits; dims=1)
    pred = sum(probs .* reshape(bounds, no_bins, 1, 1); dims=1)
    return dropdims(pred; dims=1) .* 100
end

"""
    _calculate_bin_centers(boundaries)

Compute bin center positions from bin boundary array.
Adds an extra bin center beyond the last boundary.

# Arguments
- `boundaries`: Vector of bin boundaries (length `no_bins - 1`)

# Returns
- `bin_centers`: Vector of bin centers (length `no_bins`)
"""
function _calculate_bin_centers(boundaries::AbstractVector{T}) where T
    step = boundaries[2] - boundaries[1]
    bin_centers = boundaries .+ step / 2
    return vcat(bin_centers, boundaries[end] + step + step / 2)
end

"""
    compute_predicted_aligned_error(logits; max_bin, no_bins)

Compute predicted aligned error (PAE) from TM-score logits.

# Arguments
- `logits`: TM-score logits `[no_bins, N, N, B]`

# Keyword Arguments
- `max_bin`: Maximum distance bin value in Å (default: 31)
- `no_bins`: Number of bins (default: 64)

# Returns
- NamedTuple with:
  - `predicted_aligned_error`: PAE `[N, N, B]`
  - `max_predicted_aligned_error`: Scalar max PAE value
"""
function compute_predicted_aligned_error(
    logits::AbstractArray{T, 4}; max_bin=31, no_bins=64
) where T
    boundaries = T.(range(0, max_bin; length=no_bins - 1))
    bin_centers = _calculate_bin_centers(boundaries)
    probs = Lux.softmax(logits; dims=1)
    pae = sum(probs .* reshape(bin_centers, no_bins, 1, 1, 1); dims=1)
    return (;
        predicted_aligned_error=dropdims(pae; dims=1),
        max_predicted_aligned_error=bin_centers[end],
    )
end

"""
    compute_tm(logits; residue_weights, asym_id, interface, max_bin, no_bins, eps)

Compute predicted TM-score (pTM) from TM-score logits.
When `interface=true` and `asym_id` is provided, computes interface TM-score (iPTM).

# Arguments
- `logits`: TM-score logits `[no_bins, N, N, B]`

# Keyword Arguments
- `residue_weights`: Per-residue weights `[N, B]` (default: ones)
- `asym_id`: Asymmetric unit IDs `[N, B]` for interface masking (default: nothing)
- `interface`: If true, apply interface mask based on `asym_id` (default: false)
- `max_bin`: Maximum distance bin value in Å (default: 31)
- `no_bins`: Number of bins (default: 64)
- `eps`: Small constant for numerical stability (default: 1f-8)

# Returns
- `score`: pTM or iPTM score (scalar per batch element)
"""
function compute_tm(
    logits::AbstractArray{T, 4};
    residue_weights=nothing,
    asym_id=nothing,
    interface=false,
    max_bin=31,
    no_bins=64,
    eps=1f-8,
) where T
    N = size(logits, 2)
    B = size(logits, 4)

    if residue_weights === nothing
        residue_weights = ones(T, N, B)
    end

    boundaries = T.(range(0, max_bin; length=no_bins - 1))
    bin_centers = _calculate_bin_centers(boundaries)

    clipped_n = max(sum(residue_weights), 19)
    d0 = 1.24 * (clipped_n - 15)^(1 / 3) - 1.8

    probs = Lux.softmax(logits; dims=1)

    bc = reshape(bin_centers, no_bins, 1, 1, 1)
    tm_per_bin = @. one(T) / (one(T) + bc ^ 2 / d0^2)
    predicted_tm_term = dropdims(sum(probs .* tm_per_bin; dims=1); dims=1)

    rw = reshape(residue_weights, 1, N, B)
    rw_outer = rw .* permutedims(rw, (2, 1, 3))    # [N, N, B]

    if interface && asym_id !== nothing
        interface_mask = (reshape(asym_id, N, 1, B) .!= reshape(asym_id, 1, N, B))  # [N,N,B] Bool
        @. predicted_tm_term = ifelse(interface_mask, predicted_tm_term, zero(T))
        pair_residue_weights = ifelse.(interface_mask, rw_outer, zero(T))
    else
        pair_residue_weights = rw_outer
    end
    denom = sum(pair_residue_weights; dims=2) .+ eps
    normed_residue_mask = pair_residue_weights ./ denom

    per_alignment = dropdims(sum(predicted_tm_term .* normed_residue_mask; dims=2); dims=2)

    weighted = per_alignment .* residue_weights

    max_val = maximum(weighted)
    argmax_idx = findfirst(weighted .== max_val)
    argmax_idx === nothing && return zero(T)
    return per_alignment[argmax_idx]
end
