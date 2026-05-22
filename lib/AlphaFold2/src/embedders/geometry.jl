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
function make_transform_from_reference(n_xyz, ca_xyz, c_xyz; eps=1e-20)
    T = eltype(n_xyz)

    translation = -ca_xyz
    n = n_xyz .+ translation
    c = c_xyz .+ translation

    cx = _vec_get(c, Val(1))
    cy = _vec_get(c, Val(2))
    cz = _vec_get(c, Val(3))

    norm_c1 = sqrt.(eps .+ cx .^ 2 .+ cy .^ 2)
    sin_c1 = -cy ./ norm_c1
    cos_c1 = cx ./ norm_c1

    c1_rots = _eye_3x3_batched(T, size(c))
    for I in CartesianIndices(size(c)[2:end])
        c1_rots[1, 1, I] = cos_c1[I]
        c1_rots[1, 2, I] = -sin_c1[I]
        c1_rots[2, 1, I] = sin_c1[I]
        c1_rots[2, 2, I] = cos_c1[I]
    end

    norm_c2 = sqrt.(eps .+ cx .^ 2 .+ cy .^ 2 .+ cz .^ 2)
    sin_c2 = cz ./ norm_c2
    cos_c2 = sqrt.(cx .^ 2 .+ cy .^ 2) ./ norm_c2

    c2_rots = _eye_3x3_batched(T, size(c))
    for I in CartesianIndices(size(c)[2:end])
        c2_rots[1, 1, I] = cos_c2[I]
        c2_rots[1, 3, I] = sin_c2[I]
        c2_rots[2, 2, I] = 1
        c2_rots[3, 1, I] = -sin_c2[I]
        c2_rots[3, 3, I] = cos_c2[I]
    end

    c_rots = Lux.batched_matmul(c2_rots, c1_rots)
    n = _batched_matvecmul(c_rots, n)

    ny = _vec_get(n, Val(2))
    nz = _vec_get(n, Val(3))

    norm_n = sqrt.(eps .+ ny .^ 2 .+ nz .^ 2)
    sin_n = -nz ./ norm_n
    cos_n = ny ./ norm_n

    n_rots = _eye_3x3_batched(T, size(n))
    for I in CartesianIndices(size(n)[2:end])
        n_rots[1, 1, I] = 1
        n_rots[2, 2, I] = cos_n[I]
        n_rots[2, 3, I] = -sin_n[I]
        n_rots[3, 2, I] = sin_n[I]
        n_rots[3, 3, I] = cos_n[I]
    end

    rots = Lux.batched_matmul(n_rots, c_rots)
    rots = _batched_transpose(rots)

    return rots, -translation
end

# --- helpers ---

_vec_get(x, ::Val{1}) = selectdim(x, 1, 1)
_vec_get(x, ::Val{2}) = selectdim(x, 1, 2)
_vec_get(x, ::Val{3}) = selectdim(x, 1, 3)

function _eye_3x3_batched(T, sz)
    tails = Base.tail(sz)
    x = zeros(T, 3, 3, tails...)
    for I in CartesianIndices(tails)
        x[1, 1, I] = 1
        x[2, 2, I] = 1
        x[3, 3, I] = 1
    end
    return x
end

function _batched_matvecmul(A, x)
    ts = size(x)[2:end]
    xr = reshape(x, 3, 1, ts...)
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
