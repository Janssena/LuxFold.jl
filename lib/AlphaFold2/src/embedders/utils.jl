"""
    dgram_from_positions(pos; min_bin=3.25, max_bin=50.75, no_bins=39, inf=1e8)

Compute a pairwise distance histogram from coordinates.  Returns `[no_bins, N, N, B]`.
"""
function dgram_from_positions(pos; min_bin=3.25f0, max_bin=50.75f0, no_bins=39, inf=1e8)
    T = eltype(pos)
    N, B = size(pos, 2), size(pos, 3)

    d_sq = zeros(T, 1, N, N, B)
    for b in 1:B, i in 1:N, j in 1:N
        d = pos[:, i, b] .- pos[:, j, b]
        d_sq[1, i, j, b] = sum(d .^ 2)
    end

    lower = T.(range(min_bin, max_bin; length=no_bins)) .^ 2
    upper = vcat(lower[2:end], T[inf])

    return T.((d_sq .> reshape(lower, no_bins, 1, 1, 1)) .*
              (d_sq .< reshape(upper, no_bins, 1, 1, 1)))
end

"""
    build_template_angle_feat(aatype, torsion_angles_sin_cos,
        alt_torsion_angles_sin_cos, torsion_angles_mask)

Concatenate template angle features into `[57, T, N, B]` (or `[57, N, B]`).
"""
function build_template_angle_feat(aatype, torsion_angles_sin_cos,
        alt_torsion_angles_sin_cos, torsion_angles_mask)
    T = eltype(torsion_angles_sin_cos)
    aatype_onehot = _batched_onehot(aatype, 22, T)
    tail_sz = size(torsion_angles_sin_cos)[3:end]
    tsc = reshape(torsion_angles_sin_cos, 14, tail_sz...)
    atsc = reshape(alt_torsion_angles_sin_cos, 14, tail_sz...)
    return cat(aatype_onehot, tsc, atsc, torsion_angles_mask; dims=1)
end

"""
    build_template_pair_feat(pseudo_beta, pseudo_beta_mask,
        aatype, all_atom_positions, all_atom_mask; ...)

Build `[88, N, N, B]` template pair features.

Dimension conventions:
- `pseudo_beta`: `[3, N, B]`
- `pseudo_beta_mask`: `[N, B]`
- `aatype`: `[N, B]`
- `all_atom_positions`: `[3, 37, N, B]`  (xyz × atom_type × N × B)
- `all_atom_mask`: `[37, N, B]`  (atom_type × N × B)
"""
function build_template_pair_feat(
    pseudo_beta, pseudo_beta_mask, aatype,
    all_atom_positions, all_atom_mask;
    min_bin=3.25f0, max_bin=50.75f0, no_bins=39,
    use_unit_vector=false, eps=Float32(1e-20), inf=1e8,
)
    T = eltype(pseudo_beta)
    N, B = size(pseudo_beta, 2), size(pseudo_beta, 3)

    dgram = dgram_from_positions(pseudo_beta; min_bin, max_bin, no_bins, inf)
    template_mask_2d = reshape(pseudo_beta_mask, 1, N, B) .* reshape(pseudo_beta_mask, N, 1, B)

    aatype_onehot = _batched_onehot(aatype, 22, T)
    aatype_i = reshape(aatype_onehot, 22, N, 1, B) .+ zeros(T, 1, 1, N, 1)
    aatype_j = reshape(aatype_onehot, 22, 1, N, B) .+ zeros(T, 1, N, 1, 1)

    n_idx, ca_idx, c_idx = 1, 2, 3
    n_xyz = selectdim(all_atom_positions, 2, n_idx)
    ca_xyz = selectdim(all_atom_positions, 2, ca_idx)
    c_xyz = selectdim(all_atom_positions, 2, c_idx)
    rot, trans = make_transform_from_reference(n_xyz, ca_xyz, c_xyz; eps)

    points = reshape(trans, 3, 1, N, B)
    rots = reshape(rot, 3, 3, 1, N, B)
    trans_exp = reshape(trans, 3, 1, N, B)
    rigid_vec = invert_apply(rots, trans_exp, points)
    inv_distance_scalar = T(1) ./ sqrt.(T(eps) .+ sum(rigid_vec .^ 2; dims=1))

    n_mask = selectdim(all_atom_mask, 1, n_idx)
    ca_mask = selectdim(all_atom_mask, 1, ca_idx)
    c_mask = selectdim(all_atom_mask, 1, c_idx)
    template_mask_atomic = n_mask .* ca_mask .* c_mask
    template_mask_2d_atomic = T.(reshape(template_mask_atomic, 1, N, B) .*
                                  reshape(template_mask_atomic, N, 1, B))

    inv_distance_scalar = inv_distance_scalar .* reshape(template_mask_2d_atomic, 1, N, N, B)
    unit_vector = rigid_vec .* reshape(inv_distance_scalar, 1, N, N, B)

    if !use_unit_vector
        unit_vector = zero(unit_vector)
    end

    to_concat = [
        dgram,
        reshape(template_mask_2d, 1, N, N, B),
        aatype_j,
        aatype_i,
        reshape(unit_vector[1, :, :, :], 1, N, N, B),
        reshape(unit_vector[2, :, :, :], 1, N, N, B),
        reshape(unit_vector[3, :, :, :], 1, N, N, B),
        reshape(template_mask_2d_atomic, 1, N, N, B),
    ]

    act = cat(to_concat...; dims=1)
    act = act .* reshape(template_mask_2d_atomic, 1, N, N, B)

    return act
end

function _batched_onehot(x, n_classes, T)
    sz = size(x)
    result = zeros(T, n_classes, sz...)
    for I in CartesianIndices(sz)
        result[x[I] + 1, I] = 1
    end
    return result
end
