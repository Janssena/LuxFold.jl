"""
    PreEmbeddingEmbedder(tf_dim, preembedding_dim, c_z, c_m, relpos_k; use_bias=true)

Embeds precomputed per-residue features (e.g. ESM-2) alongside target features, producing a
single-row MSA embedding and a pair embedding.

# Arguments
- `tf_dim`: Channel dimension of the target feature (aatype one-hot)
- `preembedding_dim`: Channel dimension of the pre-embedding (e.g. ESM-2 representation)
- `c_z`: Channel dimension of the pair embedding
- `c_m`: Channel dimension of the MSA embedding
- `relpos_k`: Half-window for relative positional encoding

# Inputs
- `target_feat`: Target feature tensor of shape `[tf_dim, N, B]`
- `residue_index`: Residue index tensor of shape `[N, B]`
- `preembedding`: Pre-embedding tensor of shape `[preembedding_dim, N, B]`

# Returns
- `m`: Single-row MSA embedding tensor of shape `[c_m, 1, N, B]`
- `z`: Pair embedding tensor of shape `[c_z, N, N, B]`
- `st`: Updated state
"""
struct PreEmbeddingEmbedder{L1,L2,L3,L4,R} <: Lux.AbstractLuxContainerLayer{(:linear_target_msa, :linear_preembedding_msa, :linear_preembedding_pair_i, :linear_preembedding_pair_j, :relpos)}
    linear_target_msa::L1
    linear_preembedding_msa::L2
    linear_preembedding_pair_i::L3
    linear_preembedding_pair_j::L4
    relpos::R
end

function PreEmbeddingEmbedder(tf_dim::Int, preembedding_dim::Int, c_z::Int, c_m::Int, relpos_k::Int; use_bias=true)
    return PreEmbeddingEmbedder(
        Lux.Dense(tf_dim => c_m; use_bias),
        Lux.Dense(preembedding_dim => c_m; use_bias),
        Lux.Dense(preembedding_dim => c_z; use_bias),
        Lux.Dense(preembedding_dim => c_z; use_bias),
        RelativePositionEncoding(c_z, relpos_k; use_bias),
    )
end

function (l::PreEmbeddingEmbedder)(target_feat, residue_index, preembedding, ps, st)
    tf_m, st_tfm = l.linear_target_msa(target_feat, ps.linear_target_msa, st.linear_target_msa)
    preemb_m, st_pm = l.linear_preembedding_msa(preembedding, ps.linear_preembedding_msa, st.linear_preembedding_msa)
    m_update = reshape(tf_m, size(tf_m, 1), 1, size(tf_m, 2), size(tf_m, 3)) .+
               reshape(preemb_m, size(preemb_m, 1), 1, size(preemb_m, 2), size(preemb_m, 3))

    z_rpe, st_rpe = l.relpos(residue_index, ps.relpos, st.relpos)
    preemb_i, st_pi = l.linear_preembedding_pair_i(preembedding, ps.linear_preembedding_pair_i, st.linear_preembedding_pair_i)
    preemb_j, st_pj = l.linear_preembedding_pair_j(preembedding, ps.linear_preembedding_pair_j, st.linear_preembedding_pair_j)
    z_update = z_rpe .+ reshape(preemb_i, size(preemb_i, 1), size(preemb_i, 2), 1, size(preemb_i, 3)) .+
               reshape(preemb_j, size(preemb_j, 1), 1, size(preemb_j, 2), size(preemb_j, 3))

    st_out = merge(st, (;
        linear_target_msa=st_tfm, linear_preembedding_msa=st_pm,
        linear_preembedding_pair_i=st_pi, linear_preembedding_pair_j=st_pj,
        relpos=st_rpe,
    ))
    return (m_update, z_update), st_out
end
