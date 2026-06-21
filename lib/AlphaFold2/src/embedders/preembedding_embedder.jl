"""
    PreEmbeddingEmbedder(feat_dim, preemb_dim, chn_z, chn_msa, relpos_k; use_bias=true)

Embeds precomputed per-residue features (e.g. ESM-2) alongside target features, producing a
single-row MSA embedding and a pair embedding.

# Arguments
- `feat_dim`: Channel dimension of the target feature (aatype one-hot)
- `preemb_dim`: Channel dimension of the pre-embedding (e.g. ESM-2 representation)
- `chn_z`: Channel dimension of the pair embedding
- `chn_msa`: Channel dimension of the MSA embedding
- `relpos_k`: Half-window for relative positional encoding

# Keyword Arguments
- `use_bias`: Bool or NamedTuple for per-layer bias control (default: `true`)

# Inputs
- `target_feat`: Target feature tensor of shape `[tf_dim, N, B]`
- `residue_index`: Residue index tensor of shape `[N, B]`
- `preembedding`: Pre-embedding tensor of shape `[preembedding_dim, N, B]`

# Returns
- `m`: Single-row MSA embedding tensor of shape `[chn_m, 1, N, B]`
- `z`: Pair embedding tensor of shape `[chn_z, N, N, B]`
- `st`: Updated state
"""
struct PreEmbeddingEmbedder{L1,L2,L3,L4,R} <: Lux.AbstractLuxContainerLayer{(:linear_target_msa, :linear_preembedding_msa, :linear_preembedding_pair_i, :linear_preembedding_pair_j, :relpos)}
    linear_target_msa::L1
    linear_preembedding_msa::L2
    linear_preembedding_pair_i::L3
    linear_preembedding_pair_j::L4
    relpos::R
end

function PreEmbeddingEmbedder(feat_dim::Int, preemb_dim::Int, chn_z::Int, chn_msa::Int, relpos_k::Int; use_bias=true)
    use_bias = resolve_defaults(use_bias, (:linear_target_msa, :linear_preembedding_msa, :linear_preembedding_pair_i, :linear_preembedding_pair_j, :relpos))
    return PreEmbeddingEmbedder(
        Lux.Dense(feat_dim => chn_msa; use_bias=use_bias.linear_target_msa),
        Lux.Dense(preemb_dim => chn_msa; use_bias=use_bias.linear_preembedding_msa),
        Lux.Dense(preemb_dim => chn_z; use_bias=use_bias.linear_preembedding_pair_i),
        Lux.Dense(preemb_dim => chn_z; use_bias=use_bias.linear_preembedding_pair_j),
        RelativePositionEncoding(chn_z, relpos_k; use_bias=use_bias.relpos),
    )
end

(l::PreEmbeddingEmbedder)(inputs::NamedTuple, ps, st) = l(
    inputs.target_feat, 
    inputs.residue_index, 
    inputs.preembedding, 
    ps, st
)

function (l::PreEmbeddingEmbedder)(target_feat, residue_index, preembedding, ps, st)
    feat_m, st_tfm = l.linear_target_msa(target_feat, ps.linear_target_msa, st.linear_target_msa)
    preemb_m, st_pm = l.linear_preembedding_msa(preembedding, ps.linear_preembedding_msa, st.linear_preembedding_msa)
    
    m_update = feat_m .+ preemb_m
    chn_msa, N, B = size(m_update)
    m_update = reshape(m_update, chn_msa, 1, N, B)

    z_rpe, st_rpe = l.relpos(residue_index, ps.relpos, st.relpos)
    preemb_i, st_pi = l.linear_preembedding_pair_i(preembedding, ps.linear_preembedding_pair_i, st.linear_preembedding_pair_i)
    preemb_j, st_pj = l.linear_preembedding_pair_j(preembedding, ps.linear_preembedding_pair_j, st.linear_preembedding_pair_j)
    
    preemb_dim = size(preemb_i, 1)
    preemb_i_re = reshape(preemb_i, preemb_dim, N, 1, B)
    preemb_j_re = reshape(preemb_j, preemb_dim, 1, N, B)

    z_update = @. z_rpe + preemb_i_re + preemb_j_re

    st_out = merge(st, (;
        linear_target_msa=st_tfm, linear_preembedding_msa=st_pm,
        linear_preembedding_pair_i=st_pi, linear_preembedding_pair_j=st_pj,
        relpos=st_rpe,
    ))
    return (m = m_update, z = z_update), st_out
end

# ==============================================================================
# Canonical config factories
# ==============================================================================

PreEmbeddingEmbedder(s::Symbol; kwargs...) = PreEmbeddingEmbedder(static(s); kwargs...)

PreEmbeddingEmbedder(::StaticSymbol{:monomer}; kwargs...) = 
    PreEmbeddingEmbedder(22, 256, 128, 256, 32; kwargs...)

PreEmbeddingEmbedder(::StaticSymbol{:multimer}; kwargs...) = 
    PreEmbeddingEmbedder(21, 256, 128, 256, 32; merge(kwargs, (is_multimer=true, ))...)