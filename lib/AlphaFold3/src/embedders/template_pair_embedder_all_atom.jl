
"""
    TemplatePairEmbedderAllAtom(; chn_in, chn_t, chn_dgram=39, chn_aatype=AF3_RESTYPE_NUM)

AF3 Algorithm 16 lines 1–5 (and line 8) — 1:1 port of
`openfold3.core.model.feature_embedders.template_embedders.TemplatePairEmbedderAllAtom`.

Embeds raw template pair features plus the current pair representation `z` into a template
pair embedding `[chn_t, N_token, N_token, N_templ, B]`. All `Dense` layers are bias-free;
`layer_norm_z` is an affine `LayerNorm` over the pair channel.

# Inputs
- `batch`: NamedTuple with `template_distogram [39, N, N, T, B]`, `template_unit_vector [3, N, N, T, B]`,
  `template_restype [chn_aatype, N, T, B]`, `template_pseudo_beta_mask [N, T, B]` (Bool),
  `template_backbone_frame_mask [N, T, B]` (Bool), `asym_id [N, B]`
- `z`: pair representation `[chn_in, N, N, B]`

# Returns
- `t`: `[chn_t, N, N, T, B]` template pair embedding
- `st`: updated state
"""
struct TemplatePairEmbedderAllAtom{LD,LA1,LA2,LP,LX,LY,LZuv,LB,LNZ,LZ} <:
       Lux.AbstractLuxContainerLayer{(:dgram_linear, :aatype_linear_1, :aatype_linear_2,
                                      :pseudo_beta_mask_linear, :x_linear, :y_linear, :z_linear,
                                      :backbone_mask_linear, :layer_norm_z, :linear_z)}
    dgram_linear::LD
    aatype_linear_1::LA1
    aatype_linear_2::LA2
    pseudo_beta_mask_linear::LP
    x_linear::LX
    y_linear::LY
    z_linear::LZuv
    backbone_mask_linear::LB
    layer_norm_z::LNZ
    linear_z::LZ
end

function TemplatePairEmbedderAllAtom(; chn_in::Int, chn_t::Int, chn_dgram::Int=39,
                                     chn_aatype::Int=AF3_RESTYPE_NUM)
    dense(in) = Lux.Dense(in => chn_t; use_bias=false)
    return TemplatePairEmbedderAllAtom(
        dense(chn_dgram), dense(chn_aatype), dense(chn_aatype),
        dense(1), dense(1), dense(1), dense(1), dense(1),
        Lux.LayerNorm((chn_in, 1, 1); dims=1),
        Lux.Dense(chn_in => chn_t; use_bias=false),
    )
end

(l::TemplatePairEmbedderAllAtom)(inputs::NamedTuple, ps, st) = l(inputs.batch, inputs.z, ps, st)

function (l::TemplatePairEmbedderAllAtom)(batch::NamedTuple, z::AbstractArray{T}, ps, st) where T
    restype = batch.template_restype
    N = size(restype, 2)

    same_chain = reshape(batch.asym_id, N, 1, size(batch.asym_id, 2)) .== reshape(batch.asym_id, 1, N, size(batch.asym_id, 2))
    same_chain_pair = reshape(T.(same_chain), 1, N, N, 1, size(batch.asym_id, 2))

    pb_mask = batch.template_pseudo_beta_mask
    bb_mask = batch.template_backbone_frame_mask
    N_templ = size(pb_mask, 2); B = size(pb_mask, 3)
    # TODO: should thus be a .& instead
    pbm_pair = reshape(T.(pb_mask), 1, N, 1, N_templ, B) .* reshape(T.(pb_mask), 1, 1, N, N_templ, B) .* same_chain_pair
    bfm_pair = reshape(T.(bb_mask), 1, N, 1, N_templ, B) .* reshape(T.(bb_mask), 1, 1, N, N_templ, B) .* same_chain_pair

    unit_vec = batch.template_unit_vector
    unit_x = @view unit_vec[1:1, :, :, :, :]
    unit_y = @view unit_vec[2:2, :, :, :, :] 
    unit_z = @view unit_vec[3:3, :, :, :, :]

    restype_i = reshape(restype, size(restype, 1), N, 1, N_templ, B)
    restype_j = reshape(restype, size(restype, 1), 1, N, N_templ, B)

    t_emb, st_d  = l.dgram_linear(batch.template_distogram, ps.dgram_linear, st.dgram_linear)
    t_pb_mask, st_p  = l.pseudo_beta_mask_linear(pbm_pair, ps.pseudo_beta_mask_linear, st.pseudo_beta_mask_linear)
    t_aatype_i, st_a1 = l.aatype_linear_1(T.(restype_i), ps.aatype_linear_1, st.aatype_linear_1)
    t_aatype_j, st_a2 = l.aatype_linear_2(T.(restype_j), ps.aatype_linear_2, st.aatype_linear_2)
    t_unit_x, st_x  = l.x_linear(unit_x, ps.x_linear, st.x_linear)
    t_unit_y, st_y  = l.y_linear(unit_y, ps.y_linear, st.y_linear)
    t_unit_z, st_z  = l.z_linear(unit_z, ps.z_linear, st.z_linear)
    t_bb_mask,st_b  = l.backbone_mask_linear(bfm_pair, ps.backbone_mask_linear, st.backbone_mask_linear)

    t_emb = @. t_emb + t_pb_mask + t_aatype_i + t_aatype_j + t_unit_x + t_unit_y + t_unit_z + t_bb_mask

    z_ln, st_lnz = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)
    z_proj, st_lz = l.linear_z(z_ln, ps.linear_z, st.linear_z)
    t = reshape(z_proj, size(z_proj, 1), N, N, 1, B) .+ t_emb

    st_out = merge(st, (dgram_linear=st_d, aatype_linear_1=st_a1, aatype_linear_2=st_a2,
                        pseudo_beta_mask_linear=st_p, x_linear=st_x, y_linear=st_y,
                        z_linear=st_z, backbone_mask_linear=st_b,
                        layer_norm_z=st_lnz, linear_z=st_lz))
    return t, st_out
end
