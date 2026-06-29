
"""
    DiffusionModule(; diffusion_conditioning, atom_attn_enc, atom_attn_dec,
                    diffusion_transformer, chn_s, chn_token, sigma_data=16f0)

AF3 Algorithm 20 — the full denoising network, a 1:1 port of
`openfold3...structure.diffusion_module.DiffusionModule`. Given noisy atom positions
`xl_noisy` at noise level `t` (=σ), predicts denoised positions via EDM preconditioning.

Composes already-validated sub-layers (`DiffusionConditioning`, `AtomAttentionEncoder`
with `add_noisy_pos=true`, `AtomAttentionDecoder`, self-attention `DiffusionTransformer`).
`layer_norm_s`/`layer_norm_a` are offset-free; `linear_s` is bias-free.

# Inputs
- `batch`: NamedTuple (`atom_mask`, `token_mask`, atom features, relpos integer features)
- `xl_noisy [3, N_atom, B]`, `t [B]`, `s_input [chn_s_input, N, B]`,
  `s_trunk [chn_s, N, B]`, `z_trunk [chn_z, N, N, B]`

# Returns
- `xl_out [3, N_atom, B]`, `st`
"""
struct DiffusionModule{DC,AE,AD,DT,LNS,LS,LNA} <:
       Lux.AbstractLuxContainerLayer{(:diffusion_conditioning, :atom_attn_enc, :diffusion_transformer, :atom_attn_dec, :layer_norm_s, :linear_s, :layer_norm_a)}
    sigma_data::Float32
    diffusion_conditioning::DC
    atom_attn_enc::AE
    diffusion_transformer::DT
    atom_attn_dec::AD
    layer_norm_s::LNS
    linear_s::LS
    layer_norm_a::LNA
end

function DiffusionModule(; diffusion_conditioning, atom_attn_enc, atom_attn_dec,
                         diffusion_transformer, chn_s::Int, chn_token::Int, sigma_data=16f0)
    return DiffusionModule(
        Float32(sigma_data),
        diffusion_conditioning, atom_attn_enc, diffusion_transformer, atom_attn_dec,
        LayerNormNoBias((chn_s, 1); dims=1),                # offset-free
        Lux.Dense(chn_s => chn_token; use_bias=false),
        LayerNormNoBias((chn_token, 1); dims=1),            # offset-free
    )
end

function (l::DiffusionModule)(batch::NamedTuple, xl_noisy::AbstractArray{T}, t, s_input, s_trunk, z_trunk, ps, st) where T
    σd = T(l.sigma_data)
    atom_mask  = batch.atom_mask                                   # [N_atom, B] Bool
    token_mask = batch.token_mask                                  # [N_token, B] Bool

    # 1. Conditioning on the trunk reps. (The encoder below always gets the original
    #    `s_trunk`, matching openfold — so the module conditions unconditionally; the
    #    unconditioned path lives in `DiffusionConditioning` via `nothing` trunk reps.)
    cond, st_cond = l.diffusion_conditioning(
        batch, t, s_input, s_trunk, z_trunk, ps.diffusion_conditioning, st.diffusion_conditioning,
    )
    s_cond, z_cond = cond.s, cond.z

    # 2. EDM input scaling (xl_noisy masked by atom_mask, via ifelse — Bool mask)
    atom_mask3 = reshape(atom_mask, 1, size(atom_mask)...)             # [1, N_atom, B]
    xl_noisy = ifelse.(atom_mask3, xl_noisy, zero(T))
    t_shaped = reshape(t, ntuple(one, ndims(xl_noisy) - 1)..., length(t))   # [1, 1, B]
    xl_scaled = @. xl_noisy / sqrt(t_shaped ^ 2 + σd^2)

    # 3. AtomAttentionEncoder (noisy-position path): original s_trunk, CONDITIONED z_cond
    enc, st_enc = l.atom_attn_enc(batch, xl_scaled, s_trunk, z_cond, ps.atom_attn_enc, st.atom_attn_enc)
    token_agg, atom_single, atom_cond, atom_pair = enc.token_agg, enc.atom_single, enc.atom_cond, enc.atom_pair

    # 4. Add conditioned single rep
    s_cond_norm, st_lns = l.layer_norm_s(s_cond, ps.layer_norm_s, st.layer_norm_s)
    s_cond_proj, st_ls  = l.linear_s(s_cond_norm, ps.linear_s, st.linear_s)
    token_agg = token_agg .+ s_cond_proj

    # 5. Conditioned self-attention transformer
    token_agg, st_dt = l.diffusion_transformer((; a=token_agg, s=s_cond, z=z_cond, mask=token_mask), ps.diffusion_transformer, st.diffusion_transformer)

    # 6. Norm + decode to per-atom position update
    token_agg, st_lna = l.layer_norm_a(token_agg, ps.layer_norm_a, st.layer_norm_a)
    pos_update, st_dec = l.atom_attn_dec(batch, token_agg, atom_single, atom_cond, atom_pair, ps.atom_attn_dec, st.atom_attn_dec)

    # 7. EDM output preconditioning: c_skip·x + c_out·F
    c_skip = @. σd^2 / (σd^2 + t_shaped ^ 2)
    c_out  = @. σd * t_shaped / sqrt(σd^2 + t_shaped^2)
    xl_out = @. c_skip * xl_noisy + c_out * pos_update
    xl_out = ifelse.(atom_mask3, xl_out, zero(T))

    st_out = merge(st, (diffusion_conditioning=st_cond, atom_attn_enc=st_enc,
                        diffusion_transformer=st_dt, atom_attn_dec=st_dec,
                        layer_norm_s=st_lns, linear_s=st_ls, layer_norm_a=st_lna))
    return xl_out, st_out
end
