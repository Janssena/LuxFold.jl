"""
    dgram_from_positions(pos; min_bin=3.25, max_bin=50.75, no_bins=39, inf=1e8)

Compute a pairwise distance histogram from coordinates.  Returns `[no_bins, N, N, B]`.
"""
function dgram_from_positions(pos::AbstractArray{T};
    min_bin::Real=3.25, max_bin::Real=50.75, no_bins::Int=39, inf::Real=1e8) where T
    N, B = size(pos, 2), size(pos, 3)
    out = similar(pos, no_bins, N, N, B)
    _fill_dgram!(out, pos; min_bin, max_bin, no_bins)
    return out
end

# Internal helper: writes dgram directly into `out` avoiding an intermediate allocation.
# Called by both dgram_from_positions and build_template_pair_feat (which passes a
# view of the pre-allocated output tensor so no copy is needed).
function _fill_dgram!(out::AbstractArray{T}, pos::AbstractArray{T};
    min_bin::Real, max_bin::Real, no_bins::Int) where T
    N, B = size(pos, 2), size(pos, 3)

    norm_sq = sum(abs2, pos; dims=1)         # [1, N, B]
    norm_i  = reshape(norm_sq, 1, N, 1, B)
    norm_j  = reshape(norm_sq, 1, 1, N, B)

    # Lazy adjoint: passes the transpose flag to BLAS, avoids [N, 3, B] copy.
    dp = reshape(Lux.batched_matmul(Lux.batched_adjoint(pos), pos), 1, N, N, B)

    # Bins are linearly spaced in distance space → bin index from sqrt(d_sq) analytically.
    # One sqrt per pair instead of no_bins float comparisons.  Multiply by reciprocal
    # (division ~5× slower).  Clamp in float space to guard Int32 overflow.
    # Sentinel -1 for d < min_bin → no class matches → all-zero row (matches strict-≥ semantics).
    min_bin_T     = T(min_bin)
    inv_bin_width = one(T) / T((max_bin - min_bin) / (no_bins - 1))
    bin_idx = @. floor(Int32, clamp(
        (sqrt(max(zero(T), norm_i + norm_j - 2 * dp)) - min_bin_T) * inv_bin_width,
        T(-1), T(no_bins - 1)))              # [1, N, N, B]

    class_idx = reshape(Int32.(0:no_bins-1), no_bins, 1, 1, 1)
    # Broadcast-assignment: writes directly into out, no [no_bins, N, N, B] intermediate.
    @. out = T(bin_idx == class_idx)
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

    act = zeros(T, 88, N, N, B)

    # Compute dgram directly into act[1:39], avoiding the 10 MB intermediate array
    # and the subsequent copy that the two-step approach required.
    _fill_dgram!(view(act, 1:39, :, :, :), pseudo_beta; min_bin, max_bin, no_bins)

    # template_mask_2d: computed inline, no [N, N, B] Bool intermediate.
    view(act, 40, :, :, :) .= reshape(pseudo_beta_mask, 1, N, B) .& reshape(pseudo_beta_mask, N, 1, B)

    aatype_onehot = _batched_onehot(aatype, 22, T)                         # [22, N, B]
    view(act, 41:62, :, :, :) .= reshape(aatype_onehot, 22, 1, N, B)      # aatype_j: j varies on dim 3
    view(act, 63:84, :, :, :) .= reshape(aatype_onehot, 22, N, 1, B)      # aatype_i: i varies on dim 2

    # Atomic backbone mask — always needed for act[88] and apply_mask!.
    n_mask  = selectdim(all_atom_mask, 1, 1)
    ca_mask = selectdim(all_atom_mask, 1, 2)
    c_mask  = selectdim(all_atom_mask, 1, 3)
    template_mask_atomic    = n_mask .& ca_mask .& c_mask                   # [N, B]
    template_mask_2d_atomic = reshape(template_mask_atomic, 1, N, B) .&    # [N, N, B]
                              reshape(template_mask_atomic, N, 1, B)
    view(act, 88, :, :, :) .= template_mask_2d_atomic

    # Unit vector — only computed when requested.  The default path (false) skips
    # make_transform_from_reference + invert_apply + all the pairwise distance work,
    # saving ~6 MB of allocations and the corresponding computation.
    # When false, act[85:87] stays 0 from the zeros() initialisation above.
    if use_unit_vector
        n_xyz  = selectdim(all_atom_positions, 1, 1)
        ca_xyz = selectdim(all_atom_positions, 1, 2)
        c_xyz  = selectdim(all_atom_positions, 1, 3)
        rot, trans = make_transform_from_reference(n_xyz, ca_xyz, c_xyz; eps)

        points    = reshape(trans, 3, 1, N, B)    # j-residue CA positions [3, 1, N, B]
        rots      = reshape(rot,   3, 3, N, 1, B) # i-residue frames       [3, 3, N, 1, B]
        trans_exp = reshape(trans, 3, N, 1, B)    # i-residue translations [3, N, 1, B]
        rigid_vec = invert_apply(rots, trans_exp, points)  # [3, N, N, B]

        # Fuse norms + masking + inversion into a single [1, N, N, B] array updated in-place
        # (was three separate allocations of [1, N, N, B]).
        inv_dist = sum(abs2, rigid_vec; dims=1)
        atomic_4d = reshape(template_mask_2d_atomic, 1, N, N, B)
        epsT = T(eps)
        @. inv_dist = ifelse(atomic_4d, one(T) / sqrt(epsT + inv_dist), zero(T))

        # Write directly into act[85:87], no separate unit_vector array.
        # Pre-name the view: @. would incorrectly dot view() itself when a range is used.
        uv_view = view(act, 85:87, :, :, :)
        @. uv_view = rigid_vec * inv_dist
    end

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
