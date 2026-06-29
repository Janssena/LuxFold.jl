
"""
    MSAModuleEmbedder(; chn_m_feats=34, chn_m, chn_s_input)

MSA module input embedder, AF3 Algorithm 8 (lines 1–4) — 1:1 port of
`openfold3...input_embedders.MSAModuleEmbedder` (subsampling omitted; both
`subsample_*` flags default `false` for deterministic parity). Both Linears bias-free.

# Inputs
- `batch.msa [32, N_token, N_msa, B]`, `batch.has_deletion [N_token, N_msa, B]`,
  `batch.deletion_value [N_token, N_msa, B]`, `batch.msa_mask [N_token, N_msa, B]`
- `s_input [chn_s_input, N_token, B]`

# Returns
- `(; msa, msa_mask)` where `msa [chn_m, N_token, N_msa, B]` =
  `linear_m(cat(msa, has_deletion, deletion_value)) + linear_s_input(s_input)`
  (broadcast over the MSA dim); `msa_mask` is passed through
"""
struct MSAModuleEmbedder{LM,LS} <:
       Lux.AbstractLuxContainerLayer{(:linear_m, :linear_s_input)}
    linear_m::LM
    linear_s_input::LS
end

function MSAModuleEmbedder(; chn_m::Int, chn_s_input::Int, chn_m_feats::Int=34)
    return MSAModuleEmbedder(
        Lux.Dense(chn_m_feats => chn_m; use_bias=false),
        Lux.Dense(chn_s_input => chn_m; use_bias=false),
    )
end

"""
    apply_msa_mask!(m, mask)

Zero out masked (token, seq) positions of an MSA representation `m [C, N, S, B]` in place.
Dispatches on `mask::Nothing` (no-op) vs `mask::AbstractArray{Bool}` ([N, S, B]) via `ifelse`.
"""
apply_msa_mask!(m, ::Nothing) = nothing
function apply_msa_mask!(m::AbstractArray{T}, mask::AbstractArray{Bool}) where {T}
    mask_reshaped = reshape(mask, 1, size(mask)...)                 # [1, N, S, B]
    @. m = ifelse(mask_reshaped, m, zero(T))
    return nothing
end

# Default: no output masking → exact openfold parity (openfold uses msa_mask only for the
# omitted subsampling, not to mask `m`).
(l::MSAModuleEmbedder)(batch::NamedTuple, s_input::AbstractArray, ps, st) =
    l(batch, s_input, nothing, ps, st)

function (l::MSAModuleEmbedder)(batch::NamedTuple, s_input::AbstractArray, mask, ps, st)
    has_del = reshape(batch.has_deletion, 1, size(batch.has_deletion)...)    # [1, N, S, B]
    del_val = reshape(batch.deletion_value, 1, size(batch.deletion_value)...)
    msa_feat = cat(batch.msa, has_del, del_val; dims=1)                       # [34, N, S, B]

    msa_proj,     st_m = l.linear_m(msa_feat, ps.linear_m, st.linear_m)             # [chn_m, N, S, B]
    s_input_proj, st_s = l.linear_s_input(s_input, ps.linear_s_input, st.linear_s_input)  # [chn_m, N, B]

    chn_m, N_token, B = size(s_input_proj)
    msa_emb = msa_proj .+ reshape(s_input_proj, chn_m, N_token, 1, B)         # broadcast over S

    apply_msa_mask!(msa_emb, mask)

    st_out = merge(st, (linear_m=st_m, linear_s_input=st_s))
    return (; msa=msa_emb, msa_mask=batch.msa_mask), st_out
end
