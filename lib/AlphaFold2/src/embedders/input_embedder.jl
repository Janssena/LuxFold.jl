"""
    InputEmbedder(chn_target_feat, chn_msa_feat, chn_pair, chn_msa, relpos_k; is_multimer=false, use_bias=true)

Embedder for initial MSA and pair representations from target features, residue indices,
and MSA features (Algorithm 3). Delegates relative positional encoding (Algorithm 4)
to a `RelativePositionEncoding` sub-layer.

# Arguments
- `chn_target_feat`: Channel dimension of target features (e.g., 22 for monomer)
- `chn_msa_feat`: Channel dimension of MSA features (e.g., 49 for monomer)
- `chn_pair`: Channel dimension of the pair embedding (e.g., c_z = 128)
- `chn_msa`: Channel dimension of the MSA embedding (e.g., c_m = 256)
- `relpos_k`: Window size for relative positional encoding (default: 32)

# Keyword Arguments
- `is_multimer`: Forwarded to `RelativePositionEncoding` constructor (default: false)
- `use_bias`: Bool or NamedTuple for linear layer bias (default: true)

# Inputs
- `target_feat`: Target feature tensor of shape `[chn_target_feat, N, B]`
- `residue_index`: Integer tensor of shape `[N, B]`
- `msa_feat`: MSA feature tensor of shape `[chn_msa_feat, N, S, B]`

# Returns
- `m`: MSA embedding tensor of shape `[chn_msa, N, S, B]`
- `z`: Pair embedding tensor of shape `[chn_pair, N, N, B]`
- `st`: Updated state containing states for all sub-layers
"""
struct InputEmbedder{L1,L2,L3,L4,RE} <: Lux.AbstractLuxContainerLayer{(:linear_i, :linear_j, :linear_target_msa, :linear_msa, :relpos_encoding)}
    linear_i::L1
    linear_j::L2
    linear_target_msa::L3
    linear_msa::L4
    relpos_encoding::RE
end

function InputEmbedder(
    chn_target_feat::Int, chn_msa_feat::Int, chn_pair::Int, chn_msa::Int, relpos_k::Int;
    is_multimer=false, use_bias=true
)
    use_bias = resolve_defaults(use_bias, (:linear_i, :linear_j, :linear_target_msa, :linear_msa))

    return InputEmbedder(
        Lux.Dense(chn_target_feat => chn_pair; use_bias=use_bias.linear_i),
        Lux.Dense(chn_target_feat => chn_pair; use_bias=use_bias.linear_j),
        Lux.Dense(chn_target_feat => chn_msa; use_bias=use_bias.linear_target_msa),
        Lux.Dense(chn_msa_feat => chn_msa; use_bias=use_bias.linear_msa),
        RelativePositionEncoding(chn_pair, relpos_k; is_multimer),
    )
end

(l::InputEmbedder)(inputs::NamedTuple, ps, st) = l(
    inputs.target_feat,
    inputs.residue_index,
    inputs.msa_feat,
    ps, st
)

function (l::InputEmbedder)(target_feat, residue_index, msa_feat, ps, st)
    chn_pair, chn_msa = l.linear_i.out_dims, l.linear_target_msa.out_dims
    N, B = size(residue_index)

    z_relpos, st_relpos = l.relpos_encoding(residue_index, ps.relpos_encoding, st.relpos_encoding)

    target_pair_i, st_tpi = l.linear_i(target_feat, ps.linear_i, st.linear_i)
    target_pair_j, st_tpj = l.linear_j(target_feat, ps.linear_j, st.linear_j)
    target_pair_i = reshape(target_pair_i, chn_pair, N, 1, B)
    target_pair_j = reshape(target_pair_j, chn_pair, 1, N, B)
    z = @. z_relpos + target_pair_i + target_pair_j # Can we do this in-place on z_relpos with gradients?

    target_msa, st_tm = l.linear_target_msa(target_feat, ps.linear_target_msa, st.linear_target_msa)
    msa_emb, st_msa = l.linear_msa(msa_feat, ps.linear_msa, st.linear_msa)
    m = msa_emb .+ reshape(target_msa, chn_msa, N, 1, B)

    st_out = merge(st, (;
        linear_i=st_tpi, linear_j=st_tpj,
        linear_target_msa=st_tm, linear_msa=st_msa,
        relpos_encoding=st_relpos,
    ))

    return (msa = m, pair = z), st_out
end
