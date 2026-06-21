# atom_utils.jl — Atom-position utilities for AlphaFold2
#
# Reference implementations: openfold/utils/feats.py
#   `torsion_angles_to_frames`, `frames_and_literature_positions_to_atom14_pos`
# See also: openspec/AF2-CONTEXT.md §8 for tensor layout conventions.
#
# Monomer-only: multimer indexing differs and is not supported here.
# All residue constant arrays are passed explicitly so callers can use
# device-resident versions from `st.residue_constants.*` (via `Lux.setup`).


"""
    torsion_angles_to_frames(r, angles, aatype, default_frames)

Convert predicted torsion angles into 8 rigid frames per residue (backbone +
pre-omega + phi + psi + chi1–chi4), expressed as rotation and translation arrays.

# Arguments
- `r::Rigid`: per-residue backbone frame. `rot [3,3,N,B]`, `trans [3,N,B]`.
- `angles`: `[2, 7, N, B]` — (sin, cos) pairs for 7 torsion angles per residue (index 1=sin, index 2=cos).
- `aatype`: `[N, B]` — 1-based residue indices (1=ALA … 21=Unk).
- `default_frames`: `[21, 8, 4, 4]` — from `st.residue_constants.default_frames`
  (device-resident after `Lux.setup`). Pass the global CPU constant only for testing.

# Returns
Named tuple `(; rot [3, 3, 8, N, B], trans [3, 8, N, B])` — 8 frames per residue
in the global coordinate frame. Element type and device follow from `r`.

The 8-frame output is consumed directly by `frames_and_literature_positions_to_atom14_pos`.
Monomer-only.
"""
function torsion_angles_to_frames(r::Rigid{<:AbstractArray{T}}, angles::AbstractArray, aatype::AbstractArray,
                                   default_frames::AbstractArray) where T
    N, B = size(aatype, 1), size(aatype, 2)

    # --- Step 1: gather default frames per residue (vectorised) ------------------
    # default_frames: [21, 8, 4, 4]  →  permute  →  [4, 4, 8, 21]
    # Index dim 4 with aatype [N, B]  →  [4, 4, 8, N, B]
    df_perm = permutedims(default_frames, (3, 4, 2, 1))          # [4, 4, 8, 21]
    df_sel  = T.(df_perm[:, :, :, aatype])                       # [4, 4, 8, N, B]

    default_rot = @view df_sel[1:3, 1:3, :, :, :]                   # [3, 3, 8, N, B]
    default_trans = @view df_sel[1:3, 4, :, :, :]                   # [3, 8, N, B]

    # --- Step 2: prepend backbone angle [sin=0, cos=1] (identity rotation) ------
    # angles layout: [1]=sin, [2]=cos — matches AngleResnet output (alpha[0]=sin, alpha[1]=cos
    # in OpenFold). Backbone prepend is the identity angle: sin=0, cos=1.
    bb_angle = similar(r.trans, T, 2, 1, N, B)
    bb_angle[1, 1, :, :] .= zero(T)  # sin=0 for identity
    bb_angle[2, 1, :, :] .= one(T)   # cos=1 for identity
    angles_full = cat(bb_angle, T.(angles); dims=2)              # [2, 8, N, B]

    sin_a = @view angles_full[1, :, :, :]                        # [8, N, B]  (index 1 = sin)
    cos_a = @view angles_full[2, :, :, :]                        # [8, N, B]  (index 2 = cos)

    # --- Step 3: build rotation matrices [3, 3, 8, N, B] ------------------------
    # Each rotation is around the X-axis: [[1,0,0],[0,cos,-sin],[0,sin,cos]]
    all_rots = similar(default_rot, T, 3, 3, 8, N, B)
    fill!(all_rots, zero(T))
    all_rots[1, 1, :, :, :] .= one(T)
    all_rots[2, 2, :, :, :] .=  cos_a
    all_rots[2, 3, :, :, :] .= -sin_a
    all_rots[3, 2, :, :, :] .=  sin_a
    all_rots[3, 3, :, :, :] .=  cos_a

    # --- Step 4: compose default_rot @ all_rots (batch over 8 groups) -----------
    dr_flat = reshape(default_rot, 3, 3, 8*N*B)
    ar_flat = reshape(all_rots,    3, 3, 8*N*B)
    comp_rot_flat = Lux.batched_matmul(dr_flat, ar_flat)         # [3, 3, 8*N*B]
    comp_rot  = reshape(comp_rot_flat, 3, 3, 8, N, B)           # [3, 3, 8, N, B]
    comp_trans = copy(default_trans)                             # [3, 8, N, B]

    # --- Step 5: chi-frame chain composition (chi2←chi1∘chi2, etc.) -------------
    # Mutable copies; we update chi2 first, then chi3 uses the updated chi2, etc.
    new_rot   = copy(comp_rot)
    new_trans = copy(comp_trans)

    for chi_idx in 2:4                      # chi2, chi3, chi4
        g_prev = chi_idx + 3               # 5 (chi1), 6 (chi2-updated), 7 (chi3-updated)
        g_cur  = chi_idx + 4               # 6 (chi2), 7 (chi3), 8 (chi4)
        r_new, t_new = _compose_frames(
            view(new_rot,   :, :, g_prev, :, :),  # [3, 3, N, B]
            view(new_trans, :,    g_prev, :, :),  # [3, N, B]
            view(comp_rot,  :, :, g_cur,  :, :),  # [3, 3, N, B]  (local, non-chained)
            view(comp_trans,:,    g_cur,  :, :),  # [3, N, B]
        )
        view(new_rot,   :, :, g_cur, :, :) .= r_new
        view(new_trans, :,    g_cur, :, :) .= t_new
    end

    # --- Step 6: apply backbone frame r to transform local → global --------------
    # global_rot[:,:,g] = r.rot @ new_rot[:,:,g]   ∀g ∈ 1:8
    r_rot_tiled = repeat(reshape(r.rot, 3, 3, 1, N, B); outer=(1, 1, 8, 1, 1))
    global_rot  = reshape(
        Lux.batched_matmul(reshape(r_rot_tiled, 3, 3, 8*N*B),
                           reshape(new_rot, 3, 3, 8*N*B)),
        3, 3, 8, N, B,
    )

    # global_trans[:,g] = r.rot @ new_trans[:,g] + r.trans
    new_trans_flat  = reshape(new_trans, 3, 1, 8*N*B)
    r_rot_for_t     = reshape(r_rot_tiled, 3, 3, 8*N*B)
    rotated_trans   = reshape(
        Lux.batched_matmul(r_rot_for_t, new_trans_flat),
        3, 8, N, B,
    )
    global_trans = rotated_trans .+ reshape(r.trans, 3, 1, N, B)  # [3, 8, N, B]

    return (; rot=global_rot, trans=global_trans)
end

# Helper: compose rigid frames f1=(rot1, trans1) and f2=(rot2, trans2).
# Result: rot = rot1 @ rot2;  trans = rot1 @ trans2 + trans1
# All shapes: rot [3,3,N,B], trans [3,N,B].
function _compose_frames(rot1::AbstractArray, trans1::AbstractArray,
                          rot2::AbstractArray, trans2::AbstractArray)
    N, B = size(trans1, 2), size(trans1, 3)
    new_rot   = Lux.batched_matmul(rot1, rot2)                    # [3, 3, N, B]
    t2_col    = reshape(trans2, 3, 1, N, B)
    new_trans = dropdims(Lux.batched_matmul(rot1, t2_col); dims=2) .+ trans1
    return new_rot, new_trans
end


"""
    frames_and_literature_positions_to_atom14_pos(frames, aatype,
                                                    group_idx, lit_positions, atom_mask)

Place canonical atom positions into 3D space using predicted per-residue frames.

# Arguments
- `frames`: `(; rot [3,3,8,N,B], trans [3,8,N,B])` — from `torsion_angles_to_frames`.
- `aatype`: `[N, B]` — 1-based residue indices.
- `group_idx`: `[21, 14]` (Int) — 0-based rigid-group index per atom from
  `st.residue_constants.atom14_group_idx`.
- `lit_positions`: `[21, 14, 3]` — canonical atom coords in rigid-group frame from
  `st.residue_constants.atom14_lit_pos`.
- `atom_mask`: `[21, 14]` (Bool) — true where an atom exists for this residue, from
  `st.residue_constants.atom14_mask`.

# Returns
`[3, 14, N, B]` — 3D Cartesian coordinates for up to 14 atoms per residue
(channel-first, batch-last). Non-existent atoms are zeroed.

Monomer-only. All constant arrays passed explicitly (device-agnostic).
"""
function frames_and_literature_positions_to_atom14_pos(
    frames, aatype::AbstractArray,
    group_idx::AbstractArray, lit_positions::AbstractArray, atom_mask::AbstractArray{Bool},
)
    T       = eltype(frames.rot)
    N, B    = size(aatype, 1), size(aatype, 2)
    n_atoms = 14
    n_groups = 8

    # --- Step 1: gather group indices per atom (vectorised) ----------------------
    # group_idx [21, 14] (0-based) → permute → [14, 21] → index with aatype → [14, N, B]
    gi_perm           = permutedims(group_idx, (2, 1))            # [14, 21]
    per_atom_group    = gi_perm[:, aatype]                        # [14, N, B]  0-based
    per_atom_group_1b = Int.(per_atom_group) .+ 1                 # [14, N, B]  1-based (1:8)

    # --- Step 2: one-hot encode group indices → [8, 14, N, B] ------------------
    one_hot = _atom14_onehot(per_atom_group_1b, n_groups, T)      # [8, 14, N, B]

    # --- Steps 3+4: gather per-atom frames via one-hot selection ----------------
    # frames.rot   [3, 3, 8, N, B]; one_hot [8, 14, N, B]
    # per_atom_rot [3, 3, 14, N, B] = Σ_g frames.rot[:,:,g] * one_hot[g,:,:]
    rot_e  = reshape(frames.rot,   3, 3, n_groups, 1, N, B)     # [3, 3, 8, 1, N, B]
    oh_e   = reshape(one_hot,      1, 1, n_groups, n_atoms, N, B) # [1, 1, 8, 14, N, B]
    per_atom_rot = dropdims(sum(rot_e .* oh_e; dims=3); dims=3)  # [3, 3, 14, N, B]

    trans_e        = reshape(frames.trans, 3, n_groups, 1, N, B) # [3, 8, 1, N, B]
    oh_e_trans     = reshape(one_hot,      1, n_groups, n_atoms, N, B)
    per_atom_trans = dropdims(sum(trans_e .* oh_e_trans; dims=2); dims=2)  # [3, 14, N, B]

    # --- Step 5: gather literature positions (vectorised) ----------------------
    # lit_positions [21, 14, 3] → permute → [14, 3, 21] → index → [14, 3, N, B]
    lit_perm = permutedims(T.(lit_positions), (2, 3, 1))         # [14, 3, 21]
    lit_sel  = permutedims(lit_perm[:, :, aatype], (2, 1, 3, 4)) # [3, 14, N, B]

    # --- Step 6: apply per-atom frame to literature positions -------------------
    # pred_pos = per_atom_rot @ lit_pos + per_atom_trans
    # Flatten (14, N, B) into batch dim for batched_matmul
    M_total  = n_atoms * N * B
    rot_flat = reshape(per_atom_rot, 3, 3, M_total)              # [3, 3, M]
    lit_flat = reshape(lit_sel,      3, 1, M_total)              # [3, 1, M]
    pred_pos = reshape(Lux.batched_matmul(rot_flat, lit_flat), 3, n_atoms, N, B)
    pred_pos = pred_pos .+ per_atom_trans                        # [3, 14, N, B]

    # --- Step 7: apply atom mask to zero non-existent atoms --------------------
    # atom_mask [21, 14] (Bool) → permute → [14, 21] → index → [1, 14, N, B]
    am_perm = permutedims(atom_mask, (2, 1))                     # [14, 21]  Bool
    am_sel  = reshape(am_perm[:, aatype], 1, n_atoms, N, B)     # [1, 14, N, B]

    _zero = zero(T)
    @. pred_pos = ifelse(am_sel, pred_pos, _zero)                # [3, 14, N, B]

    return pred_pos
end

# Local one-hot encoding: x (1-based integers), num_classes, element type T.
# Returns [num_classes, size(x)...].
# Vectorised — no scalar loops; GPU-safe via broadcasting.
function _atom14_onehot(x::AbstractArray{<:Integer}, num_classes::Int, ::Type{T}) where T
    x_flat = reshape(x, :)
    n_total = length(x_flat)
    # 1-based class indices broadcast against 1-based input values
    class_idx = reshape(Int32.(1:num_classes), num_classes, 1)
    x_idx     = reshape(x_flat, 1, n_total)
    one_hot_flat = @. T(class_idx == x_idx)
    return reshape(one_hot_flat, num_classes, size(x)...)
end
