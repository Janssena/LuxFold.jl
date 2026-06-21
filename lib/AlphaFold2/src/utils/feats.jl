# feats.jl — Position utility functions for AlphaFold2
#
# Reference implementation: openfold/utils/feats.py
#   `pseudo_beta_fn`, `atom14_to_atom37`, `build_extra_msa_feat`
#
# All functions are pure (no learned weights) and stateless.
# Julia layout: [C, spatial..., B] (channel-first, batch-last).
# See openspec/AF2-CONTEXT.md §8 for tensor layout conventions.


"""
    pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks=nothing)

Select the pseudo-beta atom position per residue: Cβ for all residues except Glycine,
Cα for Glycine.

# Arguments
- `aatype`: `[N, B]` — 1-based residue type indices (`restype_order["G"]` = Gly, no offset needed)
- `all_atom_positions`: `[3, 37, N, B]` — all-atom coordinates in atom37 format
- `all_atom_masks`: `[37, N, B]` Bool/Float or `nothing`

# Returns
- Without mask: `pseudo_beta` of shape `[3, N, B]`
- With mask: `(; pseudo_beta [3, N, B], pseudo_beta_mask [N, B])`
"""
function pseudo_beta_fn(aatype::AbstractArray, all_atom_positions::AbstractArray,
                        all_atom_masks=nothing)
    gly_idx = restype_order["G"]          # 1-based, no offset
    ca_idx  = atom_order["CA"] + 1        # atom_order is 0-based → Julia 1-based
    cb_idx  = atom_order["CB"] + 1

    is_gly = (aatype .== gly_idx)         # [N, B] Bool

    pseudo_beta = ifelse.(
        reshape(is_gly, 1, size(is_gly)...),
        all_atom_positions[:, ca_idx, :, :],
        all_atom_positions[:, cb_idx, :, :],
    )                                     # [3, N, B]

    if all_atom_masks === nothing
        return pseudo_beta
    else
        pseudo_beta_mask = ifelse.(
            is_gly,
            all_atom_masks[ca_idx, :, :],
            all_atom_masks[cb_idx, :, :],
        )                                 # [N, B]
        return (; pseudo_beta, pseudo_beta_mask)
    end
end


"""
    atom14_to_atom37(atom14_positions, residx_atom37_to_atom14, atom37_atom_exists)
    atom14_to_atom37(atom14_positions, aatype)

Convert dense atom14 coordinates to sparse atom37 format.

# Primary form
- `atom14_positions`: `[3, 14, N, B]`
- `residx_atom37_to_atom14`: `[37, N, B]` — 1-based indices into atom14 dim per residue
- `atom37_atom_exists`: `AbstractArray{Bool}` `[37, N, B]` — atom37 existence mask

# Convenience form
- `atom14_positions`: `[3, 14, N, B]`
- `aatype`: `[N, B]` — 1-based residue type indices; indices built from `restype_atom37_to_atom14`

# Returns
- `[3, 37, N, B]` — atom37 coordinates; non-existent atoms are exactly zero
"""
function atom14_to_atom37(atom14_positions::AbstractArray,
                          residx_atom37_to_atom14::AbstractArray,
                          atom37_atom_exists::AbstractArray{Bool})
    N14 = 14
    N, B = size(residx_atom37_to_atom14, 2), size(residx_atom37_to_atom14, 3)

    # Vectorised flat-index gather — no scalar loops, GPU-safe
    # residx_atom37_to_atom14 [37, N, B] is 1-based index into atom14 dim
    i_lin = reshape(0:(N-1), 1, N, 1)
    b_lin = reshape(0:(B-1), 1, 1, B)
    flat_idx = (Int.(residx_atom37_to_atom14) .- 1) .+ i_lin .* N14 .+ b_lin .* (N14 * N) .+ 1
    # flat_idx [37, N, B] — 1-based linear index into atom14_positions reshaped to [3, N14*N*B]

    flat_atom14      = reshape(atom14_positions, 3, N14 * N * B)
    atom37_positions = flat_atom14[:, flat_idx]   # [3, 37, N, B]

    # Mask with ifelse — never multiply (NaN * 0 = NaN)
    # Pre-compute reshape outside @. to avoid broadcast-fusion of reshape/size calls
    am_sel = reshape(atom37_atom_exists, 1, size(atom37_atom_exists)...)  # [1, 37, N, B]
    _zero  = zero(eltype(atom14_positions))
    @. atom37_positions = ifelse(am_sel, atom37_positions, _zero)
    return atom37_positions
end

function atom14_to_atom37(atom14_positions::AbstractArray, aatype::AbstractArray)
    # Build residx_atom37_to_atom14 [37, N, B] from constants + aatype [N, B]
    # restype_atom37_to_atom14 [21, 37] (1-based) → permute → [37, 21] → gather → [37, N, B]
    idx_perm  = permutedims(restype_atom37_to_atom14, (2, 1))   # [37, 21]
    residx    = idx_perm[:, aatype]                              # [37, N, B]

    # restype_atom37_mask [21, 37] Bool → [37, 21] → [37, N, B]
    mask_perm = permutedims(restype_atom37_mask, (2, 1))   # [37, 21] Bool
    exists    = mask_perm[:, aatype]                        # [37, N, B] Bool

    return atom14_to_atom37(atom14_positions, residx, exists)
end


"""
    build_extra_msa_feat(extra_msa, extra_msa_deletion_value, extra_msa_has_deletion)

Construct the feature tensor fed to `ExtraMSAEmbedder`.

# Arguments
- `extra_msa`: `[N, S_extra, B]` — integer token indices (0–22: 20 AA + gap + mask + unknown)
- `extra_msa_deletion_value`: `[N, S_extra, B]` — deletion values in [0, 1]
- `extra_msa_has_deletion`: `[N, S_extra, B]` — Bool/Float, whether a deletion is present

# Returns
- `[25, N, S_extra, B]` — 23 one-hot channels + has_deletion + deletion_value
"""
function build_extra_msa_feat(extra_msa::AbstractArray,
                               extra_msa_deletion_value::AbstractArray,
                               extra_msa_has_deletion::AbstractArray)
    T = eltype(extra_msa_deletion_value)
    N, S_extra, B = size(extra_msa)

    # One-hot: fixed 23 classes (0:22) — do NOT use dynamic minimum:maximum
    one_hot = stack([T.(extra_msa .== k) for k in 0:22]; dims=1)  # [23, N, S_extra, B]

    return cat(
        one_hot,
        reshape(T.(extra_msa_has_deletion),    1, N, S_extra, B),
        reshape(extra_msa_deletion_value,       1, N, S_extra, B);
        dims=1,
    )  # [25, N, S_extra, B]
end
