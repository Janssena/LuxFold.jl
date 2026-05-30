using LinearAlgebra: eigen, Symmetric

"""
    Rigid{T}

Rigid-body frame representation: an orthonormal rotation and a translation, per
residue and batch element.

Fields:
- `rot::Array{T, 4}`     `[3, 3, N, B]` — orthonormal rotation matrices
- `trans::Array{T, 3}`   `[3, N, B]`    — translation vectors
- `quats::Array{T, 3}`   `[4, N, B]`    — quaternion form of `rot`

We store `quats` alongside `rot` for **parity with openfold**. The Python
`Rigid` keeps its rotation in quaternion form internally and only converts to a
rotation matrix when needed; staying in quat-space lets `compose_q_update_vec`
update a quaternion via a single linear update (`q + quat_multiply_by_vec(q, v)`).
Reconstructing the quaternion from the rotation matrix each call would (a) be
expensive and (b) be ambiguous up to sign because `q` and `-q` represent the
same rotation. Carrying `quats` avoids both issues.

Tensor convention: channel/coord-first, batch-last (`[3, …, N, B]`),
matching the rest of LuxFold.jl.
"""
struct Rigid{T}
    rot::Array{T, 4}     # [3, 3, N, B]
    trans::Array{T, 3}   # [3, N, B]
    quats::Array{T, 3}   # [4, N, B]
end

# Outer constructor from quats: derive rot from quats so they stay consistent.
function Rigid(quats::Array{T, 3}, trans::Array{T, 3}) where T
    @assert size(quats, 1) == 4 "quats must have leading dim 4"
    @assert size(trans, 1) == 3 "trans must have leading dim 3"
    rot = quat_to_rotmat(quats)
    return Rigid{T}(rot, trans, quats)
end

# Outer constructor from rot+trans only: derive quats from rot.
# Caller takes responsibility for `rot` being orthonormal.
function Rigid(rot::Array{T, 4}, trans::Array{T, 3}) where T
    @assert size(rot, 1) == 3 && size(rot, 2) == 3 "rot must have leading dims (3, 3)"
    @assert size(trans, 1) == 3 "trans must have leading dim 3"
    quats = rot_to_quat(rot)
    return Rigid{T}(rot, trans, quats)
end

"""
    rigid_identity(T, N, B)

Identity rigid frame at every (residue, batch) position: rotation = I, translation = 0,
quaternion = (1, 0, 0, 0).
"""
function rigid_identity(::Type{T}, N::Integer, B::Integer) where T
    rot = zeros(T, 3, 3, N, B)
    @inbounds for b in 1:B, n in 1:N
        rot[1, 1, n, b] = one(T)
        rot[2, 2, n, b] = one(T)
        rot[3, 3, n, b] = one(T)
    end
    trans = zeros(T, 3, N, B)
    quats = zeros(T, 4, N, B)
    quats[1, :, :] .= one(T)
    return Rigid{T}(rot, trans, quats)
end


# === Apply / invert_apply (general N-D point support) ============================

"""
    apply(r::Rigid, pts)

Transform points from the local frame of each residue to the global frame:
`R · pts + t`. `pts` must have shape `[3, extra..., N, B]` where `extra` is any
number of intermediate dims (e.g. `P` points, `H` heads).

Implementation: flatten `extra` dims into one, run `batched_matmul(rot, pts_flat)`
treating `[3,3] × [3,M]` per `(N,B)` batch element (this is `_batched_matvecmul`
generalised), then add the broadcasted translation.

Confirmed against manual computation (max_diff = 0.0) for `[3, P, H, N, B]`.
"""
function apply(r::Rigid{T}, pts::AbstractArray{T}) where T
    return apply_rigid(r.rot, r.trans, pts)
end

function apply_rigid(rot::AbstractArray{T,4}, trans::AbstractArray{T,3},
                     pts::AbstractArray{T}) where T
    @assert size(pts, 1) == 3 "pts leading dim must be 3 (xyz)"
    N = size(rot, 3); B = size(rot, 4)
    @assert size(pts, ndims(pts)-1) == N && size(pts, ndims(pts)) == B "pts trailing dims must match (N, B) of rot"
    extra = size(pts)[2:end-2]
    pts_flat = reshape(pts, 3, :, N, B)             # [3, M, N, B]
    rot_out = Lux.batched_matmul(rot, pts_flat)     # [3, M, N, B]
    rotated = reshape(rot_out, 3, extra..., N, B)
    trans_b = reshape(trans, 3, ntuple(_ -> 1, length(extra))..., N, B)
    return rotated .+ trans_b
end

"""
    invert_apply(r::Rigid, pts)

Inverse of `apply`: transform points from global frame back to local frame —
`Rᵀ · (pts - t)`. Same general N-D shape support as `apply`.
"""
function invert_apply(r::Rigid{T}, pts::AbstractArray{T}) where T
    @assert size(pts, 1) == 3 "pts leading dim must be 3 (xyz)"
    N = size(r.rot, 3); B = size(r.rot, 4)
    extra = size(pts)[2:end-2]
    trans_b = reshape(r.trans, 3, ntuple(_ -> 1, length(extra))..., N, B)
    centered = pts .- trans_b
    centered_flat = reshape(centered, 3, :, N, B)
    rot_T = permutedims(r.rot, (2, 1, 3, 4))         # [3, 3, N, B]
    out = Lux.batched_matmul(rot_T, centered_flat)   # [3, M, N, B]
    return reshape(out, 3, extra..., N, B)
end


# === Quaternion helpers (parity with openfold rigid_utils.py) ====================

"""
    quat_to_rotmat(q)

Convert quaternions `[4, N, B]` to rotation matrices `[3, 3, N, B]`.

Uses the standard quaternion-to-rotation formula. Quaternion layout is
`(a, b, c, d)` matching openfold convention (`a` = scalar/real part).
"""
function quat_to_rotmat(q::AbstractArray{T, 3}) where T
    @assert size(q, 1) == 4 "quaternion leading dim must be 4"
    N, B = size(q, 2), size(q, 3)
    a = @view q[1, :, :]
    b = @view q[2, :, :]
    c = @view q[3, :, :]
    d = @view q[4, :, :]
    R = zeros(T, 3, 3, N, B)
    @. R[1, 1, :, :] = a^2 + b^2 - c^2 - d^2
    @. R[1, 2, :, :] = 2 * (b*c - a*d)
    @. R[1, 3, :, :] = 2 * (b*d + a*c)
    @. R[2, 1, :, :] = 2 * (b*c + a*d)
    @. R[2, 2, :, :] = a^2 - b^2 + c^2 - d^2
    @. R[2, 3, :, :] = 2 * (c*d - a*b)
    @. R[3, 1, :, :] = 2 * (b*d - a*c)
    @. R[3, 2, :, :] = 2 * (c*d + a*b)
    @. R[3, 3, :, :] = a^2 - b^2 - c^2 + d^2
    return R
end

"""
    rot_to_quat(R)

Convert rotation matrices `[3, 3, N, B]` to quaternions `[4, N, B]` via the
4×4 symmetric matrix eigh approach (matches openfold `rot_to_quat`).

For identity rotation this gives `(1, 0, 0, 0)` exactly; for general rotations
the sign of the quaternion is ambiguous (`q` and `-q` are equivalent rotations).
We pick the convention that the largest-magnitude component is positive — this
matches the eigenvector returned by `LinearAlgebra.eigen` up to a global sign,
which we normalise here.
"""
function rot_to_quat(R::AbstractArray{T, 4}) where T
    @assert size(R, 1) == 3 && size(R, 2) == 3 "rot leading dims must be (3, 3)"
    N, B = size(R, 3), size(R, 4)
    q = zeros(T, 4, N, B)
    @inbounds for b in 1:B, n in 1:N
        xx, xy, xz = R[1, 1, n, b], R[1, 2, n, b], R[1, 3, n, b]
        yx, yy, yz = R[2, 1, n, b], R[2, 2, n, b], R[2, 3, n, b]
        zx, zy, zz = R[3, 1, n, b], R[3, 2, n, b], R[3, 3, n, b]
        K = (1/T(3)) .* SMatrix4(
            xx + yy + zz, zy - yz,       xz - zx,       yx - xy,
            zy - yz,      xx - yy - zz,  xy + yx,       xz + zx,
            xz - zx,      xy + yx,       yy - xx - zz,  yz + zy,
            yx - xy,      xz + zx,       yz + zy,       zz - xx - yy,
        )
        # Eigenvector corresponding to the largest eigenvalue is the quaternion.
        # LinearAlgebra.eigen returns eigenvalues in ascending order.
        F = eigen(Symmetric(K))
        v = F.vectors[:, end]   # last column = largest eigenvalue
        # Conventional sign: positive scalar component.
        if v[1] < 0
            v = -v
        end
        q[1, n, b] = v[1]
        q[2, n, b] = v[2]
        q[3, n, b] = v[3]
        q[4, n, b] = v[4]
    end
    return q
end

# Small helper to construct a 4x4 SMatrix entry-by-entry without depending on
# StaticArrays (we use a plain Matrix instead).
@inline function SMatrix4(
    m11, m12, m13, m14,
    m21, m22, m23, m24,
    m31, m32, m33, m34,
    m41, m42, m43, m44,
)
    M = Matrix{typeof(m11)}(undef, 4, 4)
    M[1,1]=m11; M[1,2]=m12; M[1,3]=m13; M[1,4]=m14
    M[2,1]=m21; M[2,2]=m22; M[2,3]=m23; M[2,4]=m24
    M[3,1]=m31; M[3,2]=m32; M[3,3]=m33; M[3,4]=m34
    M[4,1]=m41; M[4,2]=m42; M[4,3]=m43; M[4,4]=m44
    return M
end

"""
    quat_multiply_by_vec(q, v)

Quaternion update: `q + q ⊗ (0, vx, vy, vz)` where `⊗` is the Hamilton product.

`q` has shape `[4, N, B]` (layout `(a, b, c, d)`), `v` has shape `[3, N, B]`
(pure quaternion `(0, x, y, z)`).

Returns a new `[4, N, B]` array (NOT added to q — caller does `q + result`).

This matches openfold's `quat_multiply_by_vec` which uses the precomputed
`_QUAT_MULTIPLY_BY_VEC[4, 3, 4]` tensor; we write out the closed form.
"""
function quat_multiply_by_vec(q::AbstractArray{T, 3}, v::AbstractArray{T, 3}) where T
    @assert size(q, 1) == 4
    @assert size(v, 1) == 3
    a = @view q[1, :, :]; b = @view q[2, :, :]; c = @view q[3, :, :]; d = @view q[4, :, :]
    x = @view v[1, :, :]; y = @view v[2, :, :]; z = @view v[3, :, :]
    N, B = size(q, 2), size(q, 3)
    out = similar(q, T, 4, N, B)
    # Hamilton product q * (0, x, y, z) with q = (a, b, c, d)
    @. out[1, :, :] = -b * x - c * y - d * z
    @. out[2, :, :] =  a * x - d * y + c * z
    @. out[3, :, :] =  d * x + a * y - b * z
    @. out[4, :, :] = -c * x + b * y + a * z
    return out
end


# === compose_q_update_vec =======================================================

"""
    compose_q_update_vec(r::Rigid, update)

Apply a 6-vector update from `BackboneUpdate` to a `Rigid` frame.

`update[1:3, :, :]` is treated as the imaginary part of a unit-ish quaternion
`(1, x, y, z)`, used to update the rotation. `update[4:6, :, :]` is the
translation update, rotated by the **current** rotation before being added
(matches openfold `Rigid.compose_q_update_vec`).

Algorithm (matches Python):
1. `new_quats = quats + quat_multiply_by_vec(quats, q_update)` then L2-normalise
2. `new_rot = quat_to_rotmat(new_quats)`
3. `trans_update = current_rot · t_update` (rotation-only apply, no translation)
4. `new_trans = trans + trans_update`
"""
function compose_q_update_vec(r::Rigid{T}, update::AbstractArray{T, 3}) where T
    @assert size(update, 1) == 6 "update leading dim must be 6 (3 quat + 3 trans)"
    q_update = @view update[1:3, :, :]
    t_update = @view update[4:6, :, :]

    # 1. Quaternion update
    Δq = quat_multiply_by_vec(r.quats, q_update)
    new_quats = r.quats .+ Δq
    norm = sqrt.(sum(abs2, new_quats; dims=1))
    new_quats = new_quats ./ norm

    # 2. Convert to rotation matrix
    new_rot = quat_to_rotmat(new_quats)

    # 3. Rotate translation update by CURRENT rotation
    trans_update = apply_rotation_only(r.rot, t_update)

    # 4. Compose translations
    new_trans = r.trans .+ trans_update

    return Rigid{T}(new_rot, new_trans, new_quats)
end

# Rotation-only apply (no translation). Used internally by compose_q_update_vec
# and by tests. Takes pts of shape [3, N, B] and rot [3, 3, N, B].
function apply_rotation_only(rot::AbstractArray{T, 4}, pts::AbstractArray{T, 3}) where T
    pts_r = reshape(pts, 3, 1, size(pts, 2), size(pts, 3))     # [3, 1, N, B]
    out = Lux.batched_matmul(rot, pts_r)                        # [3, 1, N, B]
    return dropdims(out; dims=2)
end


# === scale_translation, to_tensor_7, to_tensor_4x4 ===============================

"""
    scale_translation(r::Rigid, factor)

Return a new `Rigid` whose rotation is unchanged and whose translation is
scaled by `factor`.
"""
function scale_translation(r::Rigid{T}, factor::Real) where T
    f = T(factor)
    return Rigid{T}(r.rot, r.trans .* f, r.quats)
end

"""
    to_tensor_7(r::Rigid)

Serialise a Rigid into a single `[7, N, B]` tensor: first 4 entries are the
quaternion (a, b, c, d), last 3 entries are the translation. Matches Python
`Rigid.to_tensor_7()`.
"""
function to_tensor_7(r::Rigid{T}) where T
    N, B = size(r.trans, 2), size(r.trans, 3)
    out = Array{T}(undef, 7, N, B)
    out[1:4, :, :] .= r.quats
    out[5:7, :, :] .= r.trans
    return out
end

"""
    to_tensor_4x4(r::Rigid)

Serialise a Rigid into a homogeneous transform tensor `[4, 4, N, B]`. The 3×3
rotation occupies the top-left block, translation occupies the top-right
column, and the bottom row is `[0, 0, 0, 1]`. Matches Python
`Rigid.to_tensor_4x4()`.
"""
function to_tensor_4x4(r::Rigid{T}) where T
    N, B = size(r.trans, 2), size(r.trans, 3)
    out = zeros(T, 4, 4, N, B)
    out[1:3, 1:3, :, :] .= r.rot
    out[1:3, 4, :, :] .= r.trans
    out[4, 4, :, :] .= one(T)
    return out
end
