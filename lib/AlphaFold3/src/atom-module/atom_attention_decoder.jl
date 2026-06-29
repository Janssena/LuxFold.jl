"""
    AtomAttentionDecoder(; chn_atom, chn_atom_pair, chn_token, chn_hidden, no_heads,
                         no_blocks, n_transition, n_query, n_key,
                         use_ada_layer_norm=true)

Broadcasts the per-token representation back to atoms, runs the cross-attention atom
transformer, and projects to **3D atom position updates**. 1:1 port of
`openfold3.core.model.atom_module.AtomAttentionDecoder` (AF3 Algorithm 6).

# Keyword Arguments
- `chn_atom`: Channel dim for atom single representation
- `chn_atom_pair`: Channel dim for atom pair representation
- `chn_token`: Input channel dim for the per-token rep (broadcast back to atoms)
- `chn_hidden`: Head dim for the atom transformer attention
- `no_heads`: Number of attention heads
- `no_blocks`: Number of transformer blocks
- `n_transition`: Width multiplier for SwiGLU hidden dim
- `n_query`, `n_key`: Query/key block sizes for sequence-local attention
- `use_ada_layer_norm`: Use AdaLN in the atom transformer (default: `true`)

# Inputs
- `batch::NamedTuple`: atom features (`atom_mask`, `num_atoms_per_token`, `token_mask`)
- `token_agg`: Per-token activations `[chn_token, N_token, B]`
- `atom_single`: Atom single rep `[chn_atom, N_atom, B]`
- `atom_cond`: Atom single conditioning `[chn_atom, N_atom, B]`
- `atom_pair`: Blocked atom pair `[chn_atom_pair, n_query, n_key, N_blocks, B]`

# Returns
- `rl_update`: 3D position update `[3, N_atom, B]`
- `st`: Updated state
"""
struct AtomAttentionDecoder{QI,AT,LN,QO} <: Lux.AbstractLuxContainerLayer{(:linear_q_in, :atom_transformer, :layer_norm, :linear_q_out)}
    linear_q_in::QI
    atom_transformer::AT
    layer_norm::LN
    linear_q_out::QO
end

function AtomAttentionDecoder(;
    chn_atom::Int, chn_atom_pair::Int, chn_token::Int, chn_hidden::Int, no_heads::Int,
    no_blocks::Int, n_transition::Int, n_query::Int, n_key::Int,
    use_ada_layer_norm::Bool=true,
)
    return AtomAttentionDecoder(
        Lux.Dense(chn_token => chn_atom; use_bias=false),
        AtomTransformer(chn_atom, chn_atom_pair, chn_hidden, no_heads, no_blocks, n_transition, n_query, n_key; use_ada_layer_norm),
        LayerNormNoBias((chn_atom, 1); dims=1),
        Lux.Dense(chn_atom => 3; use_bias=false),
    )
end

function AtomAttentionDecoder(config::NamedTuple)
    return AtomAttentionDecoder(;
        chn_atom=config.c_atom, chn_atom_pair=config.c_atom_pair, chn_token=config.c_token,
        chn_hidden=config.c_hidden, no_heads=config.no_heads, no_blocks=config.no_blocks,
        n_transition=config.n_transition, n_query=config.n_query, n_key=config.n_key,
    )
end

function (l::AtomAttentionDecoder)(batch::NamedTuple, token_agg, atom_single, atom_cond, atom_pair, ps, st)
    # broadcast per-token activations to atoms
    token_agg_proj, st_qin = l.linear_q_in(token_agg, ps.linear_q_in, st.linear_q_in)
    atom_single = atom_single .+ broadcast_token_feat_to_atoms(batch.token_mask, batch.num_atoms_per_token, token_agg_proj)

    # atom transformer
    atom_single, st_at = l.atom_transformer((; a=atom_single, s=atom_cond, z=atom_pair, mask=batch.atom_mask), ps.atom_transformer, st.atom_transformer)

    # project to 3D position updates
    atom_single_ln, st_ln = l.layer_norm(atom_single, ps.layer_norm, st.layer_norm)
    pos_update, st_qout = l.linear_q_out(atom_single_ln, ps.linear_q_out, st.linear_q_out)

    st_out = merge(st, (;
        linear_q_in=st_qin, atom_transformer=st_at, layer_norm=st_ln, linear_q_out=st_qout,
    ))
    return pos_update, st_out
end
