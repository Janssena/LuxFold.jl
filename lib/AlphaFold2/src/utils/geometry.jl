"""
    make_transform_from_reference(n_xyz, ca_xyz, c_xyz; eps=1e-20)

Build the orthonormal frame that maps a canonical reference backbone onto
the given N, CA, C coordinates.  Returns `(rotation, translation)` where

    rotation :: Array{T, 3 + D}   shape [3, 3, <batch>]
    translation :: Array{T, 2 + D} shape [3, <batch>]

such that for a reference point `r` at (0, 0, 0):

    X = rotation[1:3, 1:3, ...] ⋅ r + translation[1:3, ...]

approximately yields the input N, CA, C atoms.
"""
function make_transform_from_reference(n_xyz::AbstractArray{T}, ca_xyz, c_xyz; eps=T(1e-20)) where T
    # Convert eps once to T so all arithmetic stays in T (avoids Float64 promotion
    # when eps is the default Float64 literal and the arrays are Float32/Float16).
    epsT = T(eps)
    translation = -ca_xyz
    n = n_xyz .+ translation
    c = c_xyz .+ translation

    cx = view(c, 1, :, :)
    cy = view(c, 2, :, :)
    cz = view(c, 3, :, :)

    norm_c1 = @. sqrt(epsT + cx^2 + cy^2)
    sin_c1  = @. -cy / norm_c1
    cos_c1  = @. cx / norm_c1

    tails = size(c)[2:end]
    c1_rots = zeros(T, 3, 3, tails...)
    selectdim(selectdim(c1_rots, 1, 1), 1, 1) .= cos_c1
    selectdim(selectdim(c1_rots, 1, 1), 1, 2) .= -sin_c1
    selectdim(selectdim(c1_rots, 1, 2), 1, 1) .= sin_c1
    selectdim(selectdim(c1_rots, 1, 2), 1, 2) .= cos_c1
    selectdim(selectdim(c1_rots, 1, 3), 1, 3) .= one(T)

    norm_c2 = @. sqrt(epsT + cx^2 + cy^2 + cz^2)
    sin_c2  = @. cz / norm_c2
    cos_c2  = @. sqrt(cx^2 + cy^2) / norm_c2

    c2_rots = zeros(T, 3, 3, tails...)
    selectdim(selectdim(c2_rots, 1, 1), 1, 1) .= cos_c2
    selectdim(selectdim(c2_rots, 1, 1), 1, 3) .= sin_c2
    selectdim(selectdim(c2_rots, 1, 2), 1, 2) .= one(T)
    selectdim(selectdim(c2_rots, 1, 3), 1, 1) .= -sin_c2
    selectdim(selectdim(c2_rots, 1, 3), 1, 3) .= cos_c2

    c_rots = Lux.batched_matmul(c2_rots, c1_rots)
    n = _batched_matvecmul(c_rots, n)

    ny = view(n, 2, :, :)
    nz = view(n, 3, :, :)

    norm_n = @. sqrt(epsT + ny^2 + nz^2)
    sin_n  = @. -nz / norm_n
    cos_n  = @. ny / norm_n

    n_rots = zeros(T, 3, 3, tails...)
    selectdim(selectdim(n_rots, 1, 1), 1, 1) .= one(T)
    selectdim(selectdim(n_rots, 1, 2), 1, 2) .= cos_n
    selectdim(selectdim(n_rots, 1, 2), 1, 3) .= -sin_n
    selectdim(selectdim(n_rots, 1, 3), 1, 2) .= sin_n
    selectdim(selectdim(n_rots, 1, 3), 1, 3) .= cos_n

    rots = Lux.batched_matmul(n_rots, c_rots)
    rots = _batched_transpose(rots)

    return rots, -translation
end

# --- helpers ---

function _batched_matvecmul(A, x)
    xr = reshape(x, 3, 1, size(x)[2:end]...)
    yr = Lux.batched_matmul(A, xr)
    return dropdims(yr; dims=2)
end

function _batched_transpose(A)
    return permutedims(A, (2, 1, 3:ndims(A)...))
end

"""
    invert_apply(rot, trans, points)

For each frame `(rot[:,:,k], trans[:,k])`, compute `R_kᵀ · (pts - t_k)`.

- rot:   [3, 3, N, ...]
- trans: [3, N, ...]
- points:[3, M, ...]   (M may differ from N; broadcasting on non-3 dims)
"""
function invert_apply(rot, trans, points)
    new_points = points .- trans
    return _batched_matvecmul(_batched_transpose(rot), new_points)
end
