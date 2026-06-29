"""
    relpos_complex(batch::NamedTuple, max_relative_idx::Int, max_relative_chain::Int, T=Float32)

Relative position encoding (AF3 Algorithm 3), a 1:1 port of
`openfold3.core.utils.relpos.relpos_complex`. Pure function (no Lux params/state).

Encodes pairwise positional relationships between all token pairs, accounting for
chain boundaries and entity identity — the AF3 multimer/all-atom generalisation of
AF2's `RelativePositionEncoding`.

# Arguments
- `batch`: NamedTuple with integer fields `residue_index`, `token_index`, `asym_id`,
  `sym_id`, `entity_id`, each `[N_token, B]`.
  - `asym_id` — global chain id (determines `same_chain`)
  - `sym_id` — symmetry-copy index within an entity (drives `rel_chain`; ≠ `asym_id`)
- `max_relative_idx`: clip range for residue/token index offsets (AF3 default 32)
- `max_relative_chain`: clip range for chain offsets (AF3 default 2)

# Returns
- `[C_relpos, N_token, N_token, B]` `Float32`, concatenation order
  `[rel_pos, rel_token, same_entity, rel_chain]` along dim 1, with
  `C_relpos = (2*max_relative_idx+2) + (2*max_relative_idx+2) + 1 + (2*max_relative_chain+2)`
  (= 139 for the defaults).
"""
function relpos_complex(batch::NamedTuple, max_relative_idx::Int, max_relative_chain::Int, ::Type{T}=Float32) where T
    res_idx = batch.residue_index   # [N, B]
    tok_idx = batch.token_index
    asym_id = batch.asym_id
    sym_id = batch.sym_id
    entity_id = batch.entity_id
    N, B = size(res_idx)

    # Pairwise conditions [N, N, B] (i = dim 2, j = dim 3 after channel prepend)
    same_chain = reshape(asym_id, N, 1, B) .== reshape(asym_id, 1, N, B)
    same_res = reshape(res_idx, N, 1, B) .== reshape(res_idx, 1, N, B)
    same_entity = reshape(entity_id, N, 1, B) .== reshape(entity_id, 1, N, B)

    rel_pos = _relpos_onehot(res_idx, same_chain, max_relative_idx, T)
    rel_token = _relpos_onehot(tok_idx, same_chain .& same_res, max_relative_idx, T)
    rel_chain = _relpos_onehot(sym_id, same_entity, max_relative_chain, T)
    same_entity_feat = reshape(T.(same_entity), 1, N, N, B)

    return vcat(rel_pos, rel_token, same_entity_feat, rel_chain)
end

# One-hot of clipped relative offset with an overflow bin (matches the Python inner
# `relpos` helper). `pos [N, B]`, `condition [N, N, B]` Bool → `[2*clip+2, N, N, B]`.
function _relpos_onehot(pos::AbstractArray, condition::AbstractArray, clip::Int, ::Type{T}) where T
    N, B = size(pos)
    nbins = 2 * clip + 2
    offset = reshape(pos, N, 1, B) .- reshape(pos, 1, N, B)   # [N, N, B]
    clipped = clamp.(offset .+ clip, 0, 2 * clip)              # 0 .. 2*clip
    final = ifelse.(condition, clipped, 2 * clip + 1)          # overflow bin
    bins = reshape(0:(nbins - 1), nbins, 1, 1, 1)
    return T.(reshape(final, 1, N, N, B) .== bins)             # [nbins, N, N, B]
end
