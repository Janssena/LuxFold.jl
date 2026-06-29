"""
    AtomTransformer(chn_atom, chn_atom_pair, chn_hidden, no_heads, no_blocks, n_transition, n_query, n_key;
                    use_ada_layer_norm=true)

Factory returning a `DiffusionTransformer` configured for sequence-local cross-attention
with single conditioning (`chn_cond = chn_atom`). Used by both `AtomAttentionEncoder` and
`AtomAttentionDecoder`. The returned object is a `DiffusionTransformer`; there is no distinct type.

# Arguments
- `chn_atom`: Channel dim for atom single rep (`chn_a` and `chn_cond`)
- `chn_atom_pair`: Channel dim for the blocked atom pair (`chn_pair`)
- `chn_hidden`: Head dim for attention
- `no_heads`: Number of attention heads
- `no_blocks`: Number of transformer blocks
- `n_transition`: Width multiplier for the SwiGLU hidden dim
- `n_query`, `n_key`: Query/key block sizes for sequence-local attention

# Keyword Arguments
- `use_ada_layer_norm`: Use AdaLN in sequence-local attention (default: `true`)
"""
function AtomTransformer(
    chn_atom::Int, chn_atom_pair::Int, chn_hidden::Int, no_heads::Int, no_blocks::Int,
    n_transition::Int, n_query::Int, n_key::Int;
    use_ada_layer_norm::Bool=true,
)
    return DiffusionTransformer(;
        chn_a=chn_atom, chn_cond=chn_atom, chn_pair=chn_atom_pair, chn_hidden,
        no_heads, no_blocks, n_transition, n_query, n_key, use_ada_layer_norm,
    )
end

# =============================================================================

"""
    AtomAttentionEncoder(; chn_atom, chn_atom_pair, chn_token, chn_hidden, no_heads,
                         no_blocks, n_transition, n_query, n_key, add_noisy_pos=false,
                         use_ada_layer_norm=true, chn_single=nothing, chn_pair=nothing,
                         chn_ref_element=128, chn_ref_name_chars=256)

Builds atom single/pair conditioning, runs a sequence-local cross-attention atom
transformer, and aggregates per-atom activations into a per-token representation.
1:1 port of `openfold3.core.model.atom_module.AtomAttentionEncoder` (AF3 Algorithm 5).

Two modes (selected by `add_noisy_pos`): the input-embedder path (no noisy/trunk
inputs) and the diffusion path (folds `atom_pos`, `s_trunk`, `z_trunk` via
`NoisyPositionEmbedder`). Mode is encoded in the `noisy_position_embedder` field type
(`NoisyPositionEmbedder` vs `Lux.NoOpLayer`) and dispatched via `_apply_noisy_positions`.

# Keyword Arguments
- `chn_atom`: Channel dim for atom single representation
- `chn_atom_pair`: Channel dim for atom pair representation
- `chn_token`: Output channel dim for the per-token aggregation
- `chn_hidden`: Head dim for the atom transformer attention
- `no_heads`: Number of attention heads
- `no_blocks`: Number of transformer blocks
- `n_transition`: Width multiplier for SwiGLU hidden dim
- `n_query`, `n_key`: Query/key block sizes for sequence-local attention
- `add_noisy_pos`: Enable diffusion path with noisy positions (default: `false`)
- `use_ada_layer_norm`: Use AdaLN in the atom transformer (default: `true`)
- `chn_single`, `chn_pair`: Trunk single/pair dims (required when `add_noisy_pos=true`)
- `chn_ref_element`, `chn_ref_name_chars`: Ref atom feature embedding dims

# Inputs (input-embedder path, `add_noisy_pos=false`)
- `batch::NamedTuple`: atom features (`ref_*`, `atom_mask`, `num_atoms_per_token`,
  `atom_to_token_index`, `token_mask`)

# Inputs (diffusion path, `add_noisy_pos=true`)
- `batch::NamedTuple`: same as above
- `atom_pos`: Noisy atom positions `[3, N_atom, B]`
- `s_trunk`: Trunk single rep `[chn_single, N_token, B]`
- `z_trunk`: Trunk pair rep `[chn_pair, N_token, N_token, B]`

# Returns
- `(; token_agg, atom_single, atom_cond, atom_pair)`: per-token `token_agg [chn_token, N_token, B]`,
  per-atom single `atom_single [chn_atom, N_atom, B]` and conditioning `atom_cond [chn_atom, N_atom, B]`,
  blocked pair `atom_pair [chn_atom_pair, n_query, n_key, N_blocks, B]`
- `st`: Updated state
"""
struct AtomAttentionEncoder{R,NP,L,M,PM,AT,Q} <: Lux.AbstractLuxContainerLayer{(:ref_atom_feature_embedder, :noisy_position_embedder, :linear_l, :linear_m, :pair_mlp, :atom_transformer, :linear_q)}
    n_query::Int
    n_key::Int
    ref_atom_feature_embedder::R
    noisy_position_embedder::NP
    linear_l::L
    linear_m::M
    pair_mlp::PM
    atom_transformer::AT
    linear_q::Q
end

function AtomAttentionEncoder(;
    chn_atom::Int, chn_atom_pair::Int, chn_token::Int, chn_hidden::Int, no_heads::Int,
    no_blocks::Int, n_transition::Int, n_query::Int, n_key::Int,
    add_noisy_pos::Bool=false, use_ada_layer_norm::Bool=true,
    chn_single=nothing, chn_pair=nothing,
    chn_ref_element::Int=128, chn_ref_name_chars::Int=256,
)
    noisy = add_noisy_pos ?
        NoisyPositionEmbedder(chn_single, chn_pair, chn_atom, chn_atom_pair) :
        Lux.NoOpLayer()

    pair_mlp = Lux.Chain(
        Lux.WrappedFunction(Base.Fix1(broadcast, Lux.relu)),
        Lux.Dense(chn_atom_pair => chn_atom_pair, Lux.relu; use_bias=false),
        Lux.Dense(chn_atom_pair => chn_atom_pair, Lux.relu; use_bias=false),
        Lux.Dense(chn_atom_pair => chn_atom_pair; use_bias=false),
    )

    return AtomAttentionEncoder(
        n_query, n_key,
        RefAtomFeatureEmbedder(chn_atom, chn_atom_pair; chn_ref_element, chn_ref_name_chars),
        noisy,
        Lux.Dense(chn_atom => chn_atom_pair; use_bias=false),
        Lux.Dense(chn_atom => chn_atom_pair; use_bias=false),
        pair_mlp,
        AtomTransformer(chn_atom, chn_atom_pair, chn_hidden, no_heads, no_blocks, n_transition, n_query, n_key; use_ada_layer_norm),
        Lux.Dense(chn_atom => chn_token, Lux.relu; use_bias=false),
    )
end

function AtomAttentionEncoder(config::NamedTuple)
    return AtomAttentionEncoder(;
        chn_atom=config.c_atom, chn_atom_pair=config.c_atom_pair, chn_token=config.c_token,
        chn_hidden=config.c_hidden, no_heads=config.no_heads, no_blocks=config.no_blocks,
        n_transition=config.n_transition, n_query=config.n_query, n_key=config.n_key,
        add_noisy_pos=config.add_noisy_pos,
        chn_single=hasproperty(config, :c_s) ? config.c_s : nothing,
        chn_pair=hasproperty(config, :c_z) ? config.c_z : nothing,
    )
end

# Noisy-position embedding, dispatched on the embedder field type:
# NoisyPositionEmbedder (add_noisy_pos=true) applies it; NoOpLayer (false) is identity (atom_single=copy(atom_cond)).
_apply_noisy_positions(::Lux.NoOpLayer, batch, atom_cond, atom_pair, s_trunk, z_trunk, atom_pos, Nq, Nk, ps, st) =
    ((; atom_cond, atom_pair, atom_single=copy(atom_cond)), st)

_apply_noisy_positions(npe::NoisyPositionEmbedder, batch, atom_cond, atom_pair, s_trunk, z_trunk, atom_pos, Nq, Nk, ps, st) =
    npe(batch, atom_cond, atom_pair, s_trunk, z_trunk, atom_pos, Nq, Nk, ps, st)

# input-embedder path (no noisy positions)
(l::AtomAttentionEncoder)(batch::NamedTuple, ps, st) =
    l(batch, nothing, nothing, nothing, ps, st)

# diffusion path (with noisy positions + trunk reps)
function (l::AtomAttentionEncoder)(batch::NamedTuple, atom_pos, s_trunk, z_trunk, ps, st)
    Nq, Nk = l.n_query, l.n_key
    chn_atom = l.linear_l.in_dims
    chn_atom_pair = l.linear_l.out_dims

    (atom_cond, atom_pair), st_ref = l.ref_atom_feature_embedder(batch, Nq, Nk, ps.ref_atom_feature_embedder, st.ref_atom_feature_embedder)
    T = eltype(atom_pair)

    (; atom_cond, atom_pair, atom_single), st_noisy = _apply_noisy_positions(
        l.noisy_position_embedder, batch, atom_cond, atom_pair, s_trunk, z_trunk, atom_pos, Nq, Nk,
        ps.noisy_position_embedder, st.noisy_position_embedder,
    )

    # enrich blocked pair from the blocked single conditioning (lines 13–14)
    atom_cond_q, atom_cond_k, block_mask = convert_single_rep_to_blocks(atom_cond, Nq, Nk, batch.atom_mask)
    Nb, B = size(atom_cond_q, 3), size(atom_cond_q, 4)
    atom_cond_q_flat = reshape(Lux.relu.(atom_cond_q), chn_atom, :)
    atom_cond_k_flat = reshape(Lux.relu.(atom_cond_k), chn_atom, :)

    pair_proj_q, st_l = l.linear_l(atom_cond_q_flat, ps.linear_l, st.linear_l)
    pair_proj_k, st_m = l.linear_m(atom_cond_k_flat, ps.linear_m, st.linear_m)

    block_mask_re = reshape(block_mask, 1, Nq, Nk, Nb, B)
    pair_cond_update = reshape(pair_proj_q, chn_atom_pair, Nq, 1, Nb, B) .+ reshape(pair_proj_k, chn_atom_pair, 1, Nk, Nb, B)
    @. pair_cond_update = ifelse(block_mask_re, pair_cond_update, zero(T))
    atom_pair = atom_pair .+ pair_cond_update

    atom_pair_mlp, st_pm = l.pair_mlp(reshape(atom_pair, chn_atom_pair, :), ps.pair_mlp, st.pair_mlp)
    atom_pair = atom_pair .+ reshape(atom_pair_mlp, size(atom_pair))
    @. atom_pair = ifelse(block_mask_re, atom_pair, zero(T))

    # cross-attention transformer (line 15)
    atom_single, st_at = l.atom_transformer((; a=atom_single, s=atom_cond, z=atom_pair, mask=batch.atom_mask), ps.atom_transformer, st.atom_transformer)
    atom_mask = reshape(batch.atom_mask, 1, size(batch.atom_mask)...)
    @. atom_single = ifelse(atom_mask, atom_single, zero(T))

    # aggregate to tokens (line 16)
    atom_single_proj, st_q = l.linear_q(atom_single, ps.linear_q, st.linear_q)
    token_agg = aggregate_atom_feat_to_tokens(batch.token_mask, batch.atom_to_token_index, batch.atom_mask, atom_single_proj; aggregate_fn=static(:mean))

    st_out = merge(st, (;
        ref_atom_feature_embedder=st_ref, noisy_position_embedder=st_noisy,
        linear_l=st_l, linear_m=st_m, pair_mlp=st_pm,
        atom_transformer=st_at, linear_q=st_q,
    ))
    return (; token_agg, atom_single, atom_cond, atom_pair), st_out
end
