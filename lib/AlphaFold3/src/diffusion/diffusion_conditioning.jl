"""
    DiffusionConditioning(; chn_s_input, chn_s, chn_z, chn_fourier_emb,
                          max_relative_idx=32, max_relative_chain=2,
                          sigma_data=16f0, seed_fourier_emb=42, num_relpos_dims=139)

AF3 Algorithm 21 — 1:1 port of `openfold3...diffusion_conditioning.DiffusionConditioning`.
Transforms trunk reps `(s_input, s_trunk, z_trunk)` + noise level `t` (=σ) into conditioning
signals `(s_cond, z_cond)`.

All `Linear`s are bias-free; `layer_norm_{z,s,n}` are offset-free (`create_offset=False` →
`LayerNormNoBias`); the `SwiGLUTransition`s carry their own (offset) `LayerNorm`.

# Inputs
- `batch`: NamedTuple with `residue_index`, `token_index`, `asym_id`, `sym_id`, `entity_id`, `token_mask`
- `t [B]`: noise level σ
- `s_input [chn_s_input, N, B]`
- `s_trunk [chn_s, N, B]`, `z_trunk [chn_z, N, N, B]` — the trunk reps to condition on, **or `nothing`**
  for the unconditioned case (equivalent to Python's `use_conditioning=false`; zeros are
  substituted, with shapes taken from `linear_s`/`linear_z` `out_dims`).

# Returns
- `(; s [chn_s, N, B], z [chn_z, N, N, B])`, `st`
"""
struct DiffusionConditioning{LNZ,LZ,TZ,LNS,LS,FE,LNN,LN,TS} <:
       Lux.AbstractLuxContainerLayer{(:layer_norm_z, :linear_z, :transition_z,
                                      :layer_norm_s, :linear_s, :fourier_emb,
                                      :layer_norm_n, :linear_n, :transition_s)}
    sigma_data::Float32
    max_relative_idx::Int
    max_relative_chain::Int
    layer_norm_z::LNZ
    linear_z::LZ
    transition_z::TZ
    layer_norm_s::LNS
    linear_s::LS
    fourier_emb::FE
    layer_norm_n::LNN
    linear_n::LN
    transition_s::TS
end

function DiffusionConditioning(;
    chn_s_input::Int, chn_s::Int, chn_z::Int, chn_fourier_emb::Int,
    max_relative_idx::Int=32, max_relative_chain::Int=2,
    sigma_data=16f0, seed_fourier_emb::Int=42, num_relpos_dims::Int=139,
)
    cz_cat = num_relpos_dims + chn_z
    cs_cat = chn_s + chn_s_input
    return DiffusionConditioning(
        Float32(sigma_data), max_relative_idx, max_relative_chain,
        LayerNormNoBias((cz_cat, 1, 1); dims=1),                    # zij is 4D [C, N, N, B]
        Lux.Dense(cz_cat => chn_z; use_bias=false),
        ResidualSwiGLUBlock(chn_z, 2; is_4d=true),
        LayerNormNoBias((cs_cat, 1); dims=1),                       # si is 3D [C, N, B]
        Lux.Dense(cs_cat => chn_s; use_bias=false),
        FourierEmbedding(chn_fourier_emb; seed=seed_fourier_emb),
        LayerNormNoBias((chn_fourier_emb,); dims=1),                # n is 2D [C, B]
        Lux.Dense(chn_fourier_emb => chn_s; use_bias=false),
        ResidualSwiGLUBlock(chn_s, 2),
    )
end

# Unconditioned: absent trunk reps ⇒ zeros (shapes from the projection `out_dims` + `s_input`).
function (l::DiffusionConditioning)(batch::NamedTuple, t, s_input::AbstractArray{T}, ::Nothing, ::Nothing, ps, st) where T
    N, B = size(s_input, 2), size(s_input, 3)
    s_trunk = similar(s_input, T, l.linear_s.out_dims, N, B);    s_trunk .= zero(T)
    z_trunk = similar(s_input, T, l.linear_z.out_dims, N, N, B); z_trunk .= zero(T)
    return l(batch, t, s_input, s_trunk, z_trunk, ps, st)
end

function (l::DiffusionConditioning)(batch::NamedTuple, t, s_input::AbstractArray{T}, s_trunk, z_trunk, ps, st) where T
    N, B = size(s_trunk, 2), size(s_trunk, 3)

    # --- pair conditioning ---
    relpos = relpos_complex(batch, l.max_relative_idx, l.max_relative_chain, T)     # [num_relpos, N, N, B]
    z_cond = cat(z_trunk, relpos; dims=1)                                          # [cz_cat, N, N, B]
    z_cond, st_lnz = l.layer_norm_z(z_cond, ps.layer_norm_z, st.layer_norm_z)
    z_cond, st_lz  = l.linear_z(z_cond, ps.linear_z, st.linear_z)                 # [chn_z, N, N, B]

    # --- single conditioning ---
    s_cond = cat(s_trunk, s_input; dims=1)                                         # [cs_cat, N, B]
    s_cond, st_lns = l.layer_norm_s(s_cond, ps.layer_norm_s, st.layer_norm_s)
    s_cond, st_ls  = l.linear_s(s_cond, ps.linear_s, st.linear_s)                 # [chn_s, N, B]

    # --- Fourier noise-level embedding (note the 0.25 factor, matching the Python code) ---
    noise_log = @. T(0.25) * log(t / T(l.sigma_data))                             # [B]
    noise_log = reshape(noise_log, 1, length(noise_log))                           # [1, B]
    noise_emb, st_fe  = l.fourier_emb(noise_log, ps.fourier_emb, st.fourier_emb)  # [chn_fourier, B]
    noise_emb, st_lnn = l.layer_norm_n(noise_emb, ps.layer_norm_n, st.layer_norm_n)
    noise_proj, st_ln = l.linear_n(noise_emb, ps.linear_n, st.linear_n)           # [chn_s, B]
    s_cond = s_cond .+ reshape(noise_proj, size(noise_proj, 1), 1, size(noise_proj, 2))  # broadcast over tokens

    # --- masked residual transitions (Bool masks; SwiGLUTransition masks its output) ---
    token_mask = batch.token_mask                                                  # [N, B] Bool
    pair_mask = reshape(token_mask, N, 1, B) .& reshape(token_mask, 1, N, B)      # [N, N, B] Bool
    z_cond, st_tz = l.transition_z(z_cond, pair_mask, ps.transition_z, st.transition_z)
    s_cond, st_ts = l.transition_s(s_cond, token_mask, ps.transition_s, st.transition_s)

    st_out = merge(st, (layer_norm_z=st_lnz, linear_z=st_lz, transition_z=st_tz,
                        layer_norm_s=st_lns, linear_s=st_ls, fourier_emb=st_fe,
                        layer_norm_n=st_lnn, linear_n=st_ln, transition_s=st_ts))
    return (; s=s_cond, z=z_cond), st_out
end
