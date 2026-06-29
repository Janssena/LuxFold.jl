# Shared test data helpers for AtomAttentionEncoder and AtomAttentionDecoder tests.

const _N_token, _N_atom, _B = 4, 16, 2
const _nq, _nk = 4, 8
const _N_blocks = cld(_N_atom, _nq)
const _ca, _cap, _ct, _ch = 8, 4, 12, 4
const _no_heads, _no_blocks, _n_trans = 2, 2, 2
const _cre, _crnc = 128, 256
const _apt = _N_atom ÷ _N_token

# Random atom/token masks with ≥1 valid atom per token (defined aggregation) and the
# first atom of each query block valid (no all-masked local-attention window). Masks Bool.
function _rand_masks(rng)
    atom_mask  = rand(rng, Bool, _N_atom, _B)
    token_mask = rand(rng, Bool, _N_token, _B)
    for b in 1:_B, t in 1:_N_token
        atom_mask[(t - 1) * _apt + 1, b] = true
    end
    for b in 1:_B, blk in 0:(_N_blocks - 1)
        atom_mask[blk * _nq + 1, b] = true
    end
    token_mask[1, :] .= true
    return (atom_mask, token_mask)
end

# Build the (Julia batch, Python batch) pair with consistent random features.
# `masks` is `nothing` (≈ no mask → all-true) or a `(atom_mask, token_mask)` tuple.
function _make_batch(rng, T, masks=nothing)
    ref_pos       = randn(rng, T, 3, _N_atom, _B)
    ref_charge    = randn(rng, T, 1, _N_atom, _B)
    ref_mask      = rand(rng, Bool, 1, _N_atom, _B)
    ref_element   = randn(rng, T, _cre, _N_atom, _B)
    ref_chars     = randn(rng, T, _crnc, _N_atom, _B)
    ref_space_uid = rand(rng, 1:5, 1, _N_atom, _B)
    atom_mask, token_mask = isnothing(masks) ? (trues(_N_atom, _B), trues(_N_token, _B)) : masks
    num_atoms_per_token = fill(_apt, _N_token, _B)
    atom_to_token_index = repeat(reduce(vcat, [fill(t, _apt) for t in 1:_N_token]), 1, _B)

    batch_jl = (; ref_pos, ref_charge, ref_mask, ref_element,
                ref_atom_name_chars=ref_chars, ref_space_uid, atom_mask,
                token_mask, num_atoms_per_token, atom_to_token_index)

    batch_py = Dict(
        "ref_pos"             => to_py(ref_pos; swap_batch_dim=true),
        "ref_charge"          => to_py(ref_charge; swap_batch_dim=true).squeeze(-1),
        "ref_mask"            => to_py(ref_mask; swap_batch_dim=true).squeeze(-1),
        "ref_element"         => to_py(ref_element; swap_batch_dim=true),
        "ref_atom_name_chars" => to_py(ref_chars; swap_batch_dim=true).reshape(_B, _N_atom, 4, 64),
        "ref_space_uid"       => to_py(ref_space_uid; swap_batch_dim=true).squeeze(-1),
        "atom_mask"           => to_py(T.(atom_mask); swap_batch_dim=true),
        "token_mask"          => to_py(T.(token_mask); swap_batch_dim=true),
        "num_atoms_per_token" => to_py(num_atoms_per_token; swap_batch_dim=true),
        "atom_to_token_index" => to_py(atom_to_token_index .- 1; swap_batch_dim=true),
    )
    return batch_jl, batch_py
end
