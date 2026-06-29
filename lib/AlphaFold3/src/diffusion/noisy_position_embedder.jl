"""
    NoisyPositionEmbedder(chn_single, chn_pair, chn_atom, chn_atom_pair; use_bias=true)

Folds the trunk single/pair representations and noisy atom coordinates into the atom
conditioning (AF3 Algorithm 5, lines 8–12). Used on the diffusion path of
`AtomAttentionEncoder`.

The two LayerNorms are offset-free (`LayerNormNoBias`), matching the Python
`create_offset=False`.

# Inputs
- `batch`: `NamedTuple` with `token_mask`, `num_atoms_per_token`, `atom_to_token_index`, `atom_mask`.
- `atom_cond [chn_atom, N_atom, B]`, `atom_pair [chn_atom_pair, n_query, n_key, N_blocks, B]`.
- `s_trunk [chn_single, N_token, B]`, `z_trunk [chn_pair, N_token, N_token, B]`.
- `atom_pos [3, N_atom, B]`: noisy atom positions.

# Returns
- `(; atom_cond, atom_pair, atom_single)` and updated `st`.
"""
struct NoisyPositionEmbedder{LS,S,LZ,Z,R} <: Lux.AbstractLuxContainerLayer{(:layer_norm_s, :linear_s, :layer_norm_z, :linear_z, :linear_r)}
    layer_norm_s::LS
    linear_s::S
    layer_norm_z::LZ
    linear_z::Z
    linear_r::R
end

function NoisyPositionEmbedder(
    chn_single::Int, chn_pair::Int, chn_atom::Int, chn_atom_pair::Int; use_bias=false,
)
    use_bias = resolve_defaults(use_bias, (:linear_s, :linear_z, :linear_r))
    return NoisyPositionEmbedder(
        LayerNormNoBias((chn_single, 1); dims=1),
        Lux.Dense(chn_single => chn_atom; use_bias=use_bias.linear_s),
        LayerNormNoBias((chn_pair, 1, 1); dims=1),
        Lux.Dense(chn_pair => chn_atom_pair; use_bias=use_bias.linear_z),
        Lux.Dense(3 => chn_atom; use_bias=use_bias.linear_r),
    )
end

function (l::NoisyPositionEmbedder)(batch::NamedTuple, atom_cond, atom_pair, s_trunk, z_trunk, atom_pos, n_query::Int, n_key::Int, ps, st)
    # 1. trunk single → atom single conditioning
    s_trunk_norm, st_lns = l.layer_norm_s(s_trunk, ps.layer_norm_s, st.layer_norm_s)
    s_trunk_proj, st_ls  = l.linear_s(s_trunk_norm, ps.linear_s, st.linear_s)
    s_trunk_atom = broadcast_token_feat_to_atoms(batch.token_mask, batch.num_atoms_per_token, s_trunk_proj)
    atom_cond = atom_cond .+ s_trunk_atom

    # 2. trunk pair → blocked atom pair conditioning
    z_trunk_norm, st_lnz = l.layer_norm_z(z_trunk, ps.layer_norm_z, st.layer_norm_z)
    z_trunk_proj, st_lz  = l.linear_z(z_trunk_norm, ps.linear_z, st.linear_z)
    z_trunk_atom = convert_pair_rep_to_blocks(batch, z_trunk_proj, n_query, n_key)
    atom_pair = atom_pair .+ z_trunk_atom

    # 3. noisy coordinate projection
    pos_proj, st_lr = l.linear_r(atom_pos, ps.linear_r, st.linear_r)
    atom_single = atom_cond .+ pos_proj

    st_out = merge(st, (;
        layer_norm_s=st_lns, linear_s=st_ls,
        layer_norm_z=st_lnz, linear_z=st_lz, linear_r=st_lr,
    ))
    return (; atom_cond, atom_pair, atom_single), st_out
end
