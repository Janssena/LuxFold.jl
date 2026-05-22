"""
    RelativePositionEncoding(chn_pair, relpos_k=32; is_multimer=false, max_relative_chain=2, use_bias=true)

Relative positional encoding layer implementing Algorithm 4 with both monomer
and multimer variants. Used by `InputEmbedder` and `PreembeddingEmbedder`.

Monomer path: clamp pairwise residue index differences to `[-relpos_k, relpos_k]`,
offset by `+relpos_k`, one-hot encode, and project through a linear layer.

Multimer path: concatenates three features per pair (clipped offset with cross-chain
sentinel, same-entity binary flag, clipped relative sym_id with cross-entity sentinel)
and projects through a linear layer.

# Arguments
- `chn_pair`: Output channel dimension for the pair embedding
- `relpos_k`: Window size for relative positional encoding (default: 32)

# Keyword Arguments
- `is_multimer`: Whether to use multimer chain-relative encoding (default: false)
- `max_relative_chain`: Maximum relative chain offset for multimer (default: 2)
- `use_bias`: Bool or NamedTuple for the linear layer bias (default: true)

# Inputs (Monomer)
- `residue_index`: Integer tensor of shape `[N, B]`

# Inputs (Multimer)
- `residue_index`: Integer tensor of shape `[N, B]`
- `asym_id`: Integer tensor of shape `[N, B]` with chain assignments
- `entity_id`: Integer tensor of shape `[N, B]` with entity assignments
- `sym_id`: Integer tensor of shape `[N, B]` with symmetry IDs

# Returns
- `y`: Pair encoding tensor of shape `[chn_pair, N, N, B]`
- `st`: Updated state containing `linear` state
"""
struct RelativePositionEncoding{M,L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
    relpos_k::Int
    max_relative_chain::Int
    is_multimer::M
end

function RelativePositionEncoding(chn_pair::Int, relpos_k::Int=32; is_multimer=false, max_relative_chain=2, use_bias=true)
    no_bins = is_multimer ? (2 * relpos_k + 2) + 1 + (2 * max_relative_chain + 2) : 2 * relpos_k + 1
    return RelativePositionEncoding(
        Lux.Dense(no_bins => chn_pair; use_bias),
        relpos_k,
        max_relative_chain,
        static(is_multimer),
    )
end

function (l::RelativePositionEncoding{False})(residue_index, ps, st)
    N, B = size(residue_index)
    T = eltype(ps.linear.weight)

    pairwise_diff = T.(reshape(residue_index, N, 1, B) .- reshape(residue_index, 1, N, B))
    clamped_offset = Int.(clamp.(pairwise_diff, -l.relpos_k, l.relpos_k)) .+ (l.relpos_k + 1)

    one_hot = zeros(T, 2 * l.relpos_k + 1, N, N, B)
    for i in CartesianIndices(clamped_offset)
        one_hot[clamped_offset[i], i] = one(T)
    end

    y, st_linear = l.linear(one_hot, ps.linear, st.linear)
    return y, merge(st, (; linear=st_linear))
end

function (l::RelativePositionEncoding{True})(residue_index, asym_id, entity_id, sym_id, ps, st)
    N, B = size(residue_index)
    T = eltype(ps.linear.weight)

    n_bins_offset = 2 * l.relpos_k + 2
    n_bins_sym_id = 2 * l.max_relative_chain + 2

    pairwise_diff = T.(reshape(residue_index, N, 1, B) .- reshape(residue_index, 1, N, B))
    clamped_offset = clamp.(pairwise_diff, -l.relpos_k, l.relpos_k) .+ l.relpos_k
    same_chain = reshape(asym_id, N, 1, B) .== reshape(asym_id, 1, N, B)
    offset_bin = Int.(round.(ifelse.(same_chain, clamped_offset, T(2 * l.relpos_k + 1)))) .+ 1

    oh_offset = zeros(T, n_bins_offset, N, N, B)
    for i in CartesianIndices(offset_bin)
        oh_offset[offset_bin[i], i] = one(T)
    end

    same_entity = T.(reshape(entity_id, N, 1, B) .== reshape(entity_id, 1, N, B))
    same_entity = reshape(same_entity, 1, N, N, B)

    sym_id_diff = T.(reshape(sym_id, N, 1, B) .- reshape(sym_id, 1, N, B))
    clamped_sym_id = clamp.(sym_id_diff, -l.max_relative_chain, l.max_relative_chain) .+ l.max_relative_chain
    same_entity_bool = reshape(entity_id, N, 1, B) .== reshape(entity_id, 1, N, B)
    sym_id_bin = Int.(round.(ifelse.(same_entity_bool, clamped_sym_id, T(2 * l.max_relative_chain + 1)))) .+ 1

    oh_sym_id = zeros(T, n_bins_sym_id, N, N, B)
    for i in CartesianIndices(sym_id_bin)
        oh_sym_id[sym_id_bin[i], i] = one(T)
    end

    relpos_features = vcat(oh_offset, same_entity, oh_sym_id)

    y, st_linear = l.linear(relpos_features, ps.linear, st.linear)
    return y, merge(st, (; linear=st_linear))
end
