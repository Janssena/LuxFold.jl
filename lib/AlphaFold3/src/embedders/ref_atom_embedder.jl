"""
    RefAtomFeatureEmbedder(chn_atom, chn_atom_pair; chn_ref_element=128, chn_ref_name_chars=256, use_bias=true)

Embeds per-atom reference conformer features into an atom single conditioning `atom_cond`
and a blocked atom pair conditioning `atom_pair` (AF3 Algorithm 5, lines 1–6).

This is a pure feed-forward layer (no attention); all sub-layers are `Lux.Dense`.

# Arguments
- `chn_atom`: Atom single channel dimension.
- `chn_atom_pair`: Atom pair channel dimension.

# Keyword Arguments
- `chn_ref_element`: Width of the one-hot element feature (default: 128).
- `chn_ref_name_chars`: Width of the flattened one-hot atom-name-chars feature (default: 256 = 4×64).
- `use_bias`: `Bool`/`NamedTuple` controlling per-`Dense` bias (default: true).

# Inputs
- `batch`: `NamedTuple` with `ref_pos [3, N_atom, B]`, `ref_charge [1, N_atom, B]`,
  `ref_mask [1, N_atom, B]`, `ref_element [chn_ref_element, N_atom, B]`,
  `ref_atom_name_chars [chn_ref_name_chars, N_atom, B]`, `ref_space_uid [1, N_atom, B]`,
  `atom_mask [N_atom, B]` (`AbstractArray{Bool}`).
- `n_query`, `n_key`: block height / width.

# Returns
- `(; atom_cond, atom_pair)` with `atom_cond [chn_atom, N_atom, B]`, `atom_pair [chn_atom_pair, n_query, n_key, N_blocks, B]`.
- `st`: Updated state.
"""
struct RefAtomFeatureEmbedder{P,C,M,E,A,O,I,V} <: Lux.AbstractLuxContainerLayer{(:linear_ref_pos, :linear_ref_charge, :linear_ref_mask, :linear_ref_element, :linear_ref_atom_chars, :linear_ref_offset, :linear_inv_sq_dists, :linear_valid_mask)}
    linear_ref_pos::P
    linear_ref_charge::C
    linear_ref_mask::M
    linear_ref_element::E
    linear_ref_atom_chars::A
    linear_ref_offset::O
    linear_inv_sq_dists::I
    linear_valid_mask::V
end

function RefAtomFeatureEmbedder(
    chn_atom::Int, chn_atom_pair::Int;
    chn_ref_element::Int=128, chn_ref_name_chars::Int=256, use_bias=false,
)
    use_bias = resolve_defaults(use_bias, (
        :linear_ref_pos, :linear_ref_charge, :linear_ref_mask, :linear_ref_element,
        :linear_ref_atom_chars, :linear_ref_offset, :linear_inv_sq_dists, :linear_valid_mask,
    ))

    return RefAtomFeatureEmbedder(
        Lux.Dense(3 => chn_atom; use_bias=use_bias.linear_ref_pos),
        Lux.Dense(1 => chn_atom; use_bias=use_bias.linear_ref_charge),
        Lux.Dense(1 => chn_atom; use_bias=use_bias.linear_ref_mask),
        Lux.Dense(chn_ref_element => chn_atom; use_bias=use_bias.linear_ref_element),
        Lux.Dense(chn_ref_name_chars => chn_atom; use_bias=use_bias.linear_ref_atom_chars),
        Lux.Dense(3 => chn_atom_pair; use_bias=use_bias.linear_ref_offset),
        Lux.Dense(1 => chn_atom_pair; use_bias=use_bias.linear_inv_sq_dists),
        Lux.Dense(1 => chn_atom_pair; use_bias=use_bias.linear_valid_mask),
    )
end

function (l::RefAtomFeatureEmbedder)(batch::NamedTuple, n_query::Int, n_key::Int, ps, st)
    ref_pos = batch.ref_pos                       # [3, N_atom, B]
    T = eltype(ref_pos)

    # ---- Single conditioning atom_cond [chn_atom, N_atom, B] ----
    cond_pos, st_pos       = l.linear_ref_pos(ref_pos, ps.linear_ref_pos, st.linear_ref_pos)
    cond_charge, st_charge = l.linear_ref_charge(asinh.(T.(batch.ref_charge)), ps.linear_ref_charge, st.linear_ref_charge)
    cond_mask, st_mask     = l.linear_ref_mask(T.(batch.ref_mask), ps.linear_ref_mask, st.linear_ref_mask)
    cond_elem, st_elem     = l.linear_ref_element(T.(batch.ref_element), ps.linear_ref_element, st.linear_ref_element)
    cond_chars, st_chars   = l.linear_ref_atom_chars(T.(batch.ref_atom_name_chars), ps.linear_ref_atom_chars, st.linear_ref_atom_chars)

    atom_cond = @. cond_pos + cond_charge + cond_mask + cond_elem + cond_chars

    # ---- Blocked pair conditioning atom_pair [chn_atom_pair, n_query, n_key, N_blocks, B] ----
    d_q, d_k, blk_mask = convert_single_rep_to_blocks(ref_pos, n_query, n_key, batch.atom_mask)
    v_q, v_k, _        = convert_single_rep_to_blocks(batch.ref_space_uid, n_query, n_key, batch.atom_mask)

    nb, B = size(d_q, 3), size(d_q, 4)

    # offset gated by the padding mask (Bool → ifelse)
    d_q_e = reshape(d_q, 3, n_query, 1, nb, B)
    d_k_e = reshape(d_k, 3, 1, n_key, nb, B)
    mask_e = reshape(blk_mask, 1, n_query, n_key, nb, B)             # Bool
    dlm = d_q_e .- d_k_e
    dlm = @. ifelse(mask_e, dlm, zero(T))                            # [3, n_query, n_key, nb, B]

    # same-residue indicator (float feature), also gated by the padding mask
    v_q_e = reshape(v_q, 1, n_query, 1, nb, B)
    v_k_e = reshape(v_k, 1, 1, n_key, nb, B)
    vlm = @. ifelse(mask_e & (v_q_e == v_k_e), one(T), zero(T))      # [1, n_query, n_key, nb, B]

    # project (Dense over channel dim 1; spatial dims flattened as batch)
    flat_dlm = reshape(dlm, 3, :)
    flat_vlm = reshape(vlm, 1, :)
    inv_sq   = one(T) ./ (one(T) .+ sum(abs2, flat_dlm; dims=1))     # [1, :]

    p_off, st_off   = l.linear_ref_offset(flat_dlm, ps.linear_ref_offset, st.linear_ref_offset)
    p_inv, st_inv   = l.linear_inv_sq_dists(inv_sq, ps.linear_inv_sq_dists, st.linear_inv_sq_dists)
    p_val, st_val   = l.linear_valid_mask(flat_vlm, ps.linear_valid_mask, st.linear_valid_mask)

    # each term is gated by vlm: (a+b+c) .* vlm  ==  a*vlm + b*vlm + c*vlm
    chn_atom_pair = l.linear_ref_offset.out_dims
    atom_pair = (p_off .+ p_inv .+ p_val) .* flat_vlm
    atom_pair = reshape(atom_pair, chn_atom_pair, n_query, n_key, nb, B)

    st_out = merge(st, (;
        linear_ref_pos=st_pos, linear_ref_charge=st_charge, linear_ref_mask=st_mask,
        linear_ref_element=st_elem, linear_ref_atom_chars=st_chars,
        linear_ref_offset=st_off, linear_inv_sq_dists=st_inv, linear_valid_mask=st_val,
    ))

    return (; atom_cond, atom_pair), st_out
end
