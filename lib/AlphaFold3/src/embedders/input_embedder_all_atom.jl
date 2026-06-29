
"""
    InputEmbedderAllAtom(atom_attn_enc; chn_s_input, chn_s, chn_z,
                         max_relative_idx=32, max_relative_chain=2, num_relpos_dims=139)

All-atom input embedder, AF3 Algorithm 2 — 1:1 port of
`openfold3...input_embedders.InputEmbedderAllAtom`. Builds the single-token input
representation `s_input`, the single representation `s`, and the pair representation `z`
from per-atom reference features (via `AtomAttentionEncoder` in input-embedder mode,
`add_noisy_pos=false`) and per-token features.

All five Linears are bias-free.

# Inputs (NamedTuple `batch`)
- atom features required by `AtomAttentionEncoder` (`ref_pos`, `ref_mask`, …)
- `restype [n_restype, N_token, B]`, `profile [n_profile, N_token, B]`,
  `deletion_mean [N_token, B]`, `token_bonds [N_token, N_token, B]`
- relpos integer features (`residue_index`, `token_index`, `asym_id`, `sym_id`, `entity_id`)

# Returns
- `s_input [chn_s_input, N_token, B]`, `s [chn_s, N_token, B]`, `z [chn_z, N_token, N_token, B]`
"""
struct InputEmbedderAllAtom{E,LS,LZI,LZJ,LR,LT} <:
       Lux.AbstractLuxContainerLayer{(:atom_attn_enc, :linear_s, :linear_z_i, :linear_z_j, :linear_relpos, :linear_token_bonds)}
    max_relative_idx::Int
    max_relative_chain::Int
    atom_attn_enc::E
    linear_s::LS
    linear_z_i::LZI
    linear_z_j::LZJ
    linear_relpos::LR
    linear_token_bonds::LT
end

function InputEmbedderAllAtom(atom_attn_enc; chn_s_input::Int, chn_s::Int, chn_z::Int,
                             max_relative_idx::Int=32, max_relative_chain::Int=2,
                             num_relpos_dims::Int=139)
    return InputEmbedderAllAtom(
        max_relative_idx, max_relative_chain, atom_attn_enc,
        Lux.Dense(chn_s_input => chn_s; use_bias=false),
        Lux.Dense(chn_s_input => chn_z; use_bias=false),
        Lux.Dense(chn_s_input => chn_z; use_bias=false),
        Lux.Dense(num_relpos_dims => chn_z; use_bias=false),
        Lux.Dense(1 => chn_z; use_bias=false),
    )
end

"""
    apply_token_mask!(x, mask)

Zero out masked tokens of a single representation `x [C, N, B]` in place. Dispatches on
`mask::Nothing` (no-op) vs `mask::AbstractArray{Bool}` ([N, B]). Uses `ifelse` (never
multiplication, to keep masked positions exactly zero regardless of `x`).
"""
apply_token_mask!(x, ::Nothing) = nothing
function apply_token_mask!(x::AbstractArray{T}, mask::AbstractArray{Bool}) where T
    mask_reshaped = reshape(mask, 1, size(mask)...)                 # [1, N, B]
    _zero = zero(T)
    @. x = ifelse(mask_reshaped, x, _zero)
    return nothing
end

"""
    apply_token_pair_mask!(z, mask)

Zero out masked token pairs of a pair representation `z [C, N, N, B]` in place, where the
pair mask is the outer-AND of the token mask. Dispatches on `Nothing` / `AbstractArray{Bool}`.
"""
apply_token_pair_mask!(z, ::Nothing) = nothing
function apply_token_pair_mask!(z::AbstractArray{T}, mask::AbstractArray{Bool}) where {T}
    N, B = size(mask)
    pair_mask = reshape(mask, N, 1, B) .& reshape(mask, 1, N, B)    # [N, N, B] Bool
    pair_mask = reshape(pair_mask, 1, N, N, B)
    @. z = ifelse(pair_mask, z, zero(T))
    return nothing
end

# Default: no output masking → exact openfold parity (openfold defers token masking).
(l::InputEmbedderAllAtom)(batch::NamedTuple, ps, st) = l(batch, nothing, ps, st)

function (l::InputEmbedderAllAtom)(batch::NamedTuple, mask, ps, st)
    enc_out, st_enc = l.atom_attn_enc(batch, ps.atom_attn_enc, st.atom_attn_enc)
    token_agg = enc_out.token_agg                    # [chn_token, N, B]
    N = size(token_agg, 2)

    deletion = reshape(batch.deletion_mean, 1, size(batch.deletion_mean)...)      # [1, N, B]
    s_input = cat(token_agg, batch.restype, batch.profile, deletion; dims=1)      # [chn_s_input, N, B]

    s,           st_s  = l.linear_s(s_input, ps.linear_s, st.linear_s)            # [chn_s, N, B]
    s_input_i,   st_zi = l.linear_z_i(s_input, ps.linear_z_i, st.linear_z_i)     # [chn_z, N, B]
    s_input_j,   st_zj = l.linear_z_j(s_input, ps.linear_z_j, st.linear_z_j)

    chn_z = size(s_input_i, 1); B = size(s_input_i, 3)
    z = reshape(s_input_i, chn_z, N, 1, B) .+ reshape(s_input_j, chn_z, 1, N, B)  # [chn_z, N, N, B]

    relpos_feats = relpos_complex(batch, l.max_relative_idx, l.max_relative_chain, eltype(z))
    relpos_emb, st_r = l.linear_relpos(relpos_feats, ps.linear_relpos, st.linear_relpos)
    z = z .+ relpos_emb

    tb = reshape(batch.token_bonds, 1, size(batch.token_bonds)...)            # [1, N, N, B]
    tb_emb, st_tb = l.linear_token_bonds(eltype(z).(tb), ps.linear_token_bonds, st.linear_token_bonds)
    z = z .+ tb_emb

    apply_token_mask!(s_input, mask)
    apply_token_mask!(s, mask)
    apply_token_pair_mask!(z, mask)

    st_out = merge(st, (atom_attn_enc=st_enc, linear_s=st_s, linear_z_i=st_zi,
                        linear_z_j=st_zj, linear_relpos=st_r, linear_token_bonds=st_tb))
    return (; s_input, s, z), st_out
end
