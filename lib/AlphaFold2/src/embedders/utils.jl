"""
    dgram_from_positions(pos; min_bin=3.25, max_bin=50.75, no_bins=39, inf=1e8)

Compute a pairwise distance histogram from coordinates.  Returns `[no_bins, N, N, B]`.
"""
function dgram_from_positions(pos::AbstractArray{T};
    min_bin::Real=3.25, max_bin::Real=50.75, no_bins::Int=39, inf::Real=1e8) where T
    N, B = size(pos, 2), size(pos, 3)

    # Compute ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2(x_i · x_j)
    # pos is [3, N, B]
    norm_sq = sum(abs2, pos; dims=1)  # [1, N, B]

    # Reshape for broadcasting: [1, N, 1, B] + [1, 1, N, B]
    norm_i = reshape(norm_sq, 1, N, 1, B)
    norm_j = reshape(norm_sq, 1, 1, N, B)

    # Lazy adjoint avoids a [N, 3, B] copy — batched_matmul passes the transpose
    # flag directly to BLAS.
    dp = reshape(Lux.batched_matmul(Lux.batched_adjoint(pos), pos), 1, N, N, B)
    d_sq = @. norm_i + norm_j - 2 * dp

    # Bin boundaries: square linearly-spaced edges; last upper bound stays as inf
    # (d_sq is non-negative, so T(inf) is a valid sentinel without squaring).
    bins_sq = range(T(min_bin), T(max_bin); length=no_bins) .^ 2  # [no_bins]
    lo = reshape(bins_sq, no_bins, 1, 1, 1)
    hi = reshape([bins_sq[2:end]; T(inf)], no_bins, 1, 1, 1)
    return @. T((d_sq > lo) & (d_sq < hi))
end

"""
    build_template_angle_feat(aatype, torsion_angles_sin_cos,
        alt_torsion_angles_sin_cos, torsion_angles_mask)

Concatenate template angle features into `[57, T, N, B]` (or `[57, N, B]`).
"""
function build_template_angle_feat(
    aatype,
    torsion_angles_sin_cos::AbstractArray{T},
    alt_torsion_angles_sin_cos,
    torsion_angles_mask) where T
    aatype_onehot = _batched_onehot(aatype, 22, T)
    tail_sz = size(torsion_angles_sin_cos)[3:end]
    tsc  = reshape(torsion_angles_sin_cos, 14, tail_sz...)
    atsc = reshape(alt_torsion_angles_sin_cos, 14, tail_sz...)
    # Convert mask to T so cat sees a uniform element type and the return type is
    # fully determined at compile time (no Bool→T promotion in cat's internals).
    return cat(aatype_onehot, tsc, atsc, T.(torsion_angles_mask); dims=1)
end

apply_mask!(x, ::Nothing) = nothing

function apply_mask!(x::AbstractArray{T}, mask::AbstractArray{Bool}) where T
    mask_r = reshape(mask, 1, size(mask)...)
    @. x = ifelse(mask_r, x, zero(T))
    return nothing
end

"""
    build_template_pair_feat(pseudo_beta, pseudo_beta_mask,
        aatype, all_atom_positions, all_atom_mask; ...)

Build `[88, N, N, B]` template pair features.

Dimension conventions:
- `pseudo_beta`: `[3, N, B]`
- `pseudo_beta_mask`: `[N, B]` Bool
- `aatype`: `[N, B]`
- `all_atom_positions`: `[37, 3, N, B]`  (atom_type × xyz × N × B)
- `all_atom_mask`: `[37, N, B]` Bool  (atom_type × N × B)
"""
function build_template_pair_feat(
    pseudo_beta::AbstractArray{T}, pseudo_beta_mask::AbstractArray{Bool}, aatype,
    all_atom_positions, all_atom_mask::AbstractArray{Bool};
    min_bin::Real=3.25, max_bin::Real=50.75, no_bins::Int=39,
    use_unit_vector::Bool=false, eps::Real=T(1e-20), inf::Real=1e8,
) where T
    N, B = size(pseudo_beta, 2), size(pseudo_beta, 3)

    dgram = dgram_from_positions(pseudo_beta; min_bin, max_bin, no_bins, inf)
    template_mask_2d = reshape(pseudo_beta_mask, 1, N, B) .& reshape(pseudo_beta_mask, N, 1, B)

    aatype_onehot = _batched_onehot(aatype, 22, T)  # [22, N, B]

    n_idx, ca_idx, c_idx = 1, 2, 3
    n_xyz  = selectdim(all_atom_positions, 1, n_idx)
    ca_xyz = selectdim(all_atom_positions, 1, ca_idx)
    c_xyz  = selectdim(all_atom_positions, 1, c_idx)
    rot, trans = make_transform_from_reference(n_xyz, ca_xyz, c_xyz; eps)

    points    = reshape(trans, 3, 1, N, B)      # j-residue CA positions [3, 1, N, B]
    rots      = reshape(rot,   3, 3, N, 1, B)   # i-residue frames       [3, 3, N, 1, B]
    trans_exp = reshape(trans, 3, N, 1, B)      # i-residue translations [3, N, 1, B]
    rigid_vec = invert_apply(rots, trans_exp, points)  # [3, N, N, B]
    rigid_norms_sq = sum(abs2, rigid_vec; dims=1)
    inv_distance_scalar = one(T) ./ sqrt.(T(eps) .+ rigid_norms_sq)

    n_mask = selectdim(all_atom_mask, 1, n_idx)
    ca_mask = selectdim(all_atom_mask, 1, ca_idx)
    c_mask  = selectdim(all_atom_mask, 1, c_idx)
    template_mask_atomic    = n_mask .& ca_mask .& c_mask  # [N, B]
    template_mask_2d_atomic = reshape(template_mask_atomic, 1, N, B) .& reshape(template_mask_atomic, N, 1, B)  # [N, N, B]

    inv_distance_scalar = inv_distance_scalar .* reshape(template_mask_2d_atomic, 1, N, N, B)
    unit_vector = use_unit_vector ? rigid_vec .* inv_distance_scalar : zero(rigid_vec)

    # Pre-allocate output and fill each channel slice via broadcasting.
    # This avoids materialising two full [22, N, N, B] copies of aatype_onehot
    # and the dynamic-dispatch overhead of cat(to_concat...).
    act = zeros(T, 88, N, N, B)

    view(act,  1:39, :, :, :) .= dgram                                       # pairwise distance histogram
    view(act,    40, :, :, :) .= template_mask_2d                             # Bool → T via .=
    view(act, 41:62, :, :, :) .= reshape(aatype_onehot, 22, 1, N, B)         # aatype_j: j varies on dim 3
    view(act, 63:84, :, :, :) .= reshape(aatype_onehot, 22, N, 1, B)         # aatype_i: i varies on dim 2
    view(act, 85:87, :, :, :) .= unit_vector                                  # [3, N, N, B] unit vector
    view(act,    88, :, :, :) .= template_mask_2d_atomic                      # Bool → T via .=

    apply_mask!(act, template_mask_2d_atomic)

    return act
end

function _batched_onehot(x, n_classes, ::Type{T}) where T
    # Vectorized one-hot encoding using broadcasting
    # x is arbitrary shape, x contains class indices (0-indexed)
    # Returns one-hot tensor with shape [n_classes, ...]

    # Flatten x to 1D and create class indices
    x_flat = reshape(x, :)
    n_total = length(x_flat)

    # Create one-hot by comparing each element with class range.
    # @. fuses the == comparison and T() conversion into a single pass.
    class_indices = reshape(0:n_classes-1, n_classes, 1)
    x_indices = reshape(x_flat, 1, n_total)
    one_hot_flat = @. T(class_indices == x_indices)

    # Reshape back to [n_classes, size(x)...]
    return reshape(one_hot_flat, n_classes, size(x)...)
end
