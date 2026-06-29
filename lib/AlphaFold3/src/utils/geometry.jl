"""
    quat_to_rot(q)

Convert (normalised) quaternions to rotation matrices, a 1:1 port of
`openfold3.core.utils.rigid_utils.quat_to_rot` (channel-first layout).

# Inputs
- `q`: `[4, batch...]` quaternions `(a, b, c, d)` along dim 1.

# Returns
- `[3, 3, batch...]` rotation matrices.
"""
function quat_to_rot(q::AbstractArray{T}) where {T}
    a = selectdim(q, 1, 1); b = selectdim(q, 1, 2)
    c = selectdim(q, 1, 3); d = selectdim(q, 1, 4)
    aa = a .* a; bb = b .* b; cc = c .* c; dd = d .* d
    ab = a .* b; ac = a .* c; ad = a .* d
    bc = b .* c; bd = b .* d; cd = c .* d

    R = similar(q, 3, 3, size(q)[2:end]...)
    _setrot!(R, 1, 1, aa .+ bb .- cc .- dd); _setrot!(R, 1, 2, 2 .* (bc .- ad)); _setrot!(R, 1, 3, 2 .* (bd .+ ac))
    _setrot!(R, 2, 1, 2 .* (bc .+ ad)); _setrot!(R, 2, 2, aa .- bb .+ cc .- dd); _setrot!(R, 2, 3, 2 .* (cd .- ab))
    _setrot!(R, 3, 1, 2 .* (bd .- ac)); _setrot!(R, 3, 2, 2 .* (cd .+ ab)); _setrot!(R, 3, 3, aa .- bb .- cc .+ dd)
    return R
end

# R[i, j, batch...] .= v
@inline _setrot!(R, i, j, v) = (selectdim(selectdim(R, 1, i), 1, j) .= v; nothing)

"""
    sample_rotations(rng::AbstractRNG, shape; T=Float32)

Sample uniformly distributed rotation matrices by normalising random quaternions.
Port of `openfold3...diffusion_module.sample_rotations` + `quat_to_rot`.

`rng` is the mandatory first positional argument (convention: rng-consuming functions take
the RNG first).

# Returns
- `[3, 3, shape...]` rotation matrices (Julia channel-first; Python is `[shape..., 3, 3]`).
"""
function sample_rotations(rng::Random.AbstractRNG, shape::Tuple; T=Float32)
    q = randn(rng, T, 4, shape...)
    q = q ./ sqrt.(sum(abs2, q; dims=1))
    return quat_to_rot(q)
end

# Batched `R * x` over trailing batch dims: R [3,3,batch...], x [3,N,batch...] → [3,N,batch...]
function _apply_rotation(R::AbstractArray, x::AbstractArray)
    bs = size(R)[3:end]
    Bp = prod(bs)
    Nn = size(x, 2)
    y = Lux.batched_matmul(reshape(R, 3, 3, Bp), reshape(x, 3, Nn, Bp))
    return reshape(y, 3, Nn, bs...)
end

"""
    centre_random_augmentation(rng::AbstractRNG, xl, atom_mask; scale_trans=1f0,
                               rots=nothing, trans=nothing)

Random global rigid-body augmentation of atom coordinates (AF3 Algorithm 19), a 1:1
port of `openfold3...diffusion_module.centre_random_augmentation`. Centres on the
masked centroid, then applies a sampled rotation + translation.

`rng` is the mandatory first positional argument (convention: rng-consuming functions take
the RNG first).

# Inputs
- `rng`: RNG for the rotation/translation sampling
- `xl`: `[3, N_atom, batch...]` atom positions (dim 1 = xyz)
- `atom_mask`: `[N_atom, batch...]` `AbstractArray{Bool}` — masks centroid + output
- `scale_trans`: translation scale (default 1)
- `rots`, `trans`: optional pre-sampled rotation `[3,3,batch...]` / translation
  `[3,1,batch...]` (for deterministic testing); sampled when `nothing`

# Returns
- `[3, N_atom, batch...]` — same shape as `xl`.
"""
function centre_random_augmentation(
    rng::Random.AbstractRNG, xl::AbstractArray{T}, atom_mask::AbstractArray{Bool};
    scale_trans=one(T), rots=nothing, trans=nothing,
) where {T}
    batch_shape = size(xl)[3:end]
    mask3 = reshape(atom_mask, 1, size(atom_mask)...)               # [1, N_atom, batch...]

    # Masked centroid: ifelse-gate the coords before summing (exact zeros at masked atoms).
    xl_masked = @. ifelse(mask3, xl, zero(T))
    mean_xl   = sum(xl_masked; dims=2) ./ sum(mask3; dims=2)        # [3, 1, batch...]
    xl_c = xl .- mean_xl

    # Python samples rotations FIRST, then the translation.
    R = isnothing(rots)  ? sample_rotations(rng, batch_shape; T)              : rots
    t = isnothing(trans) ? T(scale_trans) .* randn(rng, T, 3, 1, batch_shape...) : trans

    xl_out = _apply_rotation(R, xl_c) .+ t                          # [3, N_atom, batch...]
    @. xl_out = ifelse(mask3, xl_out, zero(T))
    return xl_out
end
