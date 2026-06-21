# weight_loading.jl — Pure-Julia NPZ weight loading for AlphaFold2
#
# Loads official DeepMind AlphaFold2 `.npz` parameter files into Julia parameter
# NamedTuples produced by `Lux.setup`.
#
# ## NPZ dimension convention
#
# NPZ.jl reads NumPy C-contiguous (row-major) arrays as Julia column-major arrays
# with ALL dimensions reversed:
#
#   NumPy [M, N]            →  Julia [N, M]
#   NumPy [n_blk, ...]      →  Julia [..., n_blk]   (stacked block dim becomes last)
#
# Key consequences:
#   JAX Dense weight [C_in, C_out]        →  Julia [C_out, C_in]  ✓ (no transform needed)
#   JAX LayerNorm scale/offset [C]        →  Julia [C]            (reshape to [C, 1])
#   JAX MHA Q/K/V weight [H, C_in, C_hd] →  Julia [C_hd, C_in, H]
#   JAX MHA output [H, C_hd, C_out]      →  Julia [C_out, C_hd, H]
#   JAX OPM output [C_hid, C_hid, C_z]   →  Julia [C_z, C_hid, C_hid]
#
# ## Julia parameter paths  (mirrors test sync_* helpers in test/)
#
#   TriMulCore (fused, AF2 default):
#     glu_ab.linear.weight [4*H, C_in]  — rows 1:H = a_proj, H+1:2H = b_proj,
#                                          2H+1:3H = a_gate, 3H+1:4H = b_gate
#     layer_norm_out                    — center LayerNorm
#     glu_out.linear / glu_out.gate    — output projection / output gate
#
#   TriangleAttention:
#     layer_norm, linear, mha
#
#   AttentionPairBias (MSARowAtt):
#     layer_norm_in (Python: layer_norm_m), layer_norm_z, linear_z, mha
#
#   MSAColumnGlobalAttention:
#     layer_norm, linear_q, linear_k, linear_v, linear_g, linear_o
#
#   Block naming:
#     EvoformerStack/ExtraMSAStack blocks → ps.blocks[:block_i]
#     TemplatePairStack blocks            → ps.blocks[:block_i]
#     AngleResnet blocks                  → ps.blocks[:layer_i]   (Lux.Chain)
#     StructureModuleTransition layers    → ps.layers[:layer_i]   (Lux.Chain)
#
# ## NPZ key prefix
#
#   All DeepMind AF2 keys begin with "alphafold/alphafold_iteration/"

import NPZ
import Downloads
import Tar

const _NPZ_PREFIX = "alphafold/alphafold_iteration/"

# Official DeepMind release — single tarball containing all 22 model variants.
const _AF2_PARAMS_URL =
    "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"

# Default weights directory: ~/.alphafold/
# Override with: AlphaFold2.set_weights_dir!("/your/path") or the `dir` kwarg.
const _DEFAULT_WEIGHTS_DIR = Ref(joinpath(homedir(), ".alphafold"))

# ============================================================================
# Primitive transform helpers
# ============================================================================

# Read key from NPZ dict, convert to Float32.
@inline _w(npz, key::String) = Float32.(npz[key])

# Select block slice from stacked array: numpy stores [n_blks, ...],
# NPZ.jl reverses to [..., n_blks], so block i is the last-dim slice.
@inline _block(npz, key, i) = selectdim(_w(npz, key), ndims(npz[key]), i)

# MHA Q/K/V: Julia reads [C_hd, C_in, H] → need [H*C_hd, C_in]
function _mha_qkv_w(w::AbstractArray{T,3}) where T
    C_hd, C_in, H = size(w)
    return reshape(permutedims(w, (3, 1, 2)), H * C_hd, C_in)
end

# MHA output: Julia reads [C_out, C_hd, H] → need [C_out, H*C_hd]
function _mha_out_w(w::AbstractArray{T,3}) where T
    C_out, C_hd, H = size(w)
    return reshape(permutedims(w, (1, 3, 2)), C_out, H * C_hd)
end

# OPM output: Julia reads [C_z, C_hid, C_hid] → need [C_z, C_hid^2]
function _opm_out_w(w::AbstractArray{T,3}) where T
    C_z, a, b = size(w)
    return reshape(w, C_z, a * b)
end

# ============================================================================
# Primitive loaders (non-stacked, direct NPZ key)
# ============================================================================

function _load_dense!(ps, npz, prefix::String)
    ps.weight .= _w(npz, prefix * "/weights")
    if hasproperty(ps, :bias) && haskey(npz, prefix * "/bias")
        ps.bias .= _w(npz, prefix * "/bias")
    end
end

# LayerNorm: reshape the 1D NPZ vector to match the actual parameter shape in ps.
# Lux.LayerNorm((C,)) → ps.scale shape [C, 1]; LayerNorm((C, 1, 1)) → [C, 1, 1], etc.
function _load_layernorm!(ps, npz, prefix::String)
    ps.scale .= reshape(_w(npz, prefix * "/scale"), size(ps.scale))
    if hasproperty(ps, :bias) && haskey(npz, prefix * "/offset")
        ps.bias .= reshape(_w(npz, prefix * "/offset"), size(ps.bias))
    end
end

# MHA self-attention (fused QKV):  qkv.weight = vcat(Q, K, V).
# MHA cross-attention (fused KV):  qkv.q.weight + qkv.kv.weight = vcat(K, V).
function _load_mha!(ps_mha, npz, prefix::String; gated::Bool=true)
    qw = _mha_qkv_w(_w(npz, prefix * "/query_w"))
    kw = _mha_qkv_w(_w(npz, prefix * "/key_w"))
    vw = _mha_qkv_w(_w(npz, prefix * "/value_w"))

    if hasproperty(ps_mha.qkv, :weight)
        ps_mha.qkv.weight .= vcat(qw, kw, vw)          # fused self-attention
    else
        ps_mha.qkv.q.weight  .= qw                       # fused-KV cross-attention
        ps_mha.qkv.kv.weight .= vcat(kw, vw)
    end

    ps_mha.out.weight .= _mha_out_w(_w(npz, prefix * "/output_w"))
    ps_mha.out.bias   .= _w(npz, prefix * "/output_b")

    if gated
        ps_mha.gate.weight .= _mha_qkv_w(_w(npz, prefix * "/gating_w"))
        # gating_b: JAX stores [H, C_head] → Julia reads [C_head, H]
        # permute to [H, C_head] then flatten column-major to get H-first ordering
        # that matches the gating weight row layout [H*C_head, C_in].
        gb = _w(npz, prefix * "/gating_b")
        ps_mha.gate.bias .= ndims(gb) == 2 ? vec(permutedims(gb, (2, 1))) : vec(gb)
    end
end

# ============================================================================
# Stacked-array helpers  (each NPZ array has block index as its last dim)
# ============================================================================

# Same as _load_dense! but reading from stacked arrays.
function _load_dense_s!(ps, npz, prefix::String, i::Int)
    ps.weight .= _block(npz, prefix * "/weights", i)
    if hasproperty(ps, :bias) && haskey(npz, prefix * "/bias")
        ps.bias .= _block(npz, prefix * "/bias", i)
    end
end

function _load_layernorm_s!(ps, npz, prefix::String, i::Int)
    ps.scale .= reshape(_block(npz, prefix * "/scale", i), size(ps.scale))
    if hasproperty(ps, :bias) && haskey(npz, prefix * "/offset")
        ps.bias .= reshape(_block(npz, prefix * "/offset", i), size(ps.bias))
    end
end

function _load_mha_s!(ps_mha, npz, prefix::String, i::Int; gated::Bool=true)
    qw = _mha_qkv_w(_block(npz, prefix * "/query_w", i))
    kw = _mha_qkv_w(_block(npz, prefix * "/key_w", i))
    vw = _mha_qkv_w(_block(npz, prefix * "/value_w", i))

    if hasproperty(ps_mha.qkv, :weight)
        ps_mha.qkv.weight .= vcat(qw, kw, vw)
    else
        ps_mha.qkv.q.weight  .= qw
        ps_mha.qkv.kv.weight .= vcat(kw, vw)
    end

    ps_mha.out.weight .= _mha_out_w(_block(npz, prefix * "/output_w", i))
    ps_mha.out.bias   .= _block(npz, prefix * "/output_b", i)

    if gated
        ps_mha.gate.weight .= _mha_qkv_w(_block(npz, prefix * "/gating_w", i))
        gb = _block(npz, prefix * "/gating_b", i)
        ps_mha.gate.bias .= ndims(gb) == 2 ? vec(permutedims(gb, (2, 1))) : vec(gb)
    end
end

# ============================================================================
# Triangle Multiplication  (TriangleMultiplication → layer_norm + TriMulCore)
#
# Julia fused glu_ab.linear.weight layout [4H, C_in]:
#   rows  1:H        = a_projection   (left_projection for outgoing)
#   rows  H+1:2H     = b_projection   (right_projection for outgoing)
#   rows  2H+1:3H    = a_gate         (left_gate for outgoing)
#   rows  3H+1:4H    = b_gate         (right_gate for outgoing)
#
# For incoming the AF2 JAX checkpoint swaps a↔b, so we swap left↔right.
# (mirrors the swap in openfold/utils/import_weights.py TriMulOutParams)
# ============================================================================

function _load_trimul!(ps, npz, prefix::String; outgoing::Bool=true)
    _load_layernorm!(ps.layer_norm, npz, prefix * "/layer_norm_input")

    H = size(ps.core.glu_ab.linear.weight, 1) ÷ 4

    # NPZ keys (JAX stores [C_in, H], Julia reads [H, C_in] — correct Dense shape)
    if outgoing
        ap = _w(npz, prefix * "/left_projection/weights")
        bp = _w(npz, prefix * "/right_projection/weights")
        ag = _w(npz, prefix * "/left_gate/weights")
        bg = _w(npz, prefix * "/right_gate/weights")
    else
        # incoming: AF2 stores b at left, a at right  (undo the JAX swap)
        ap = _w(npz, prefix * "/right_projection/weights")
        bp = _w(npz, prefix * "/left_projection/weights")
        ag = _w(npz, prefix * "/right_gate/weights")
        bg = _w(npz, prefix * "/left_gate/weights")
    end
    ps.core.glu_ab.linear.weight[1:H,      :] .= ap
    ps.core.glu_ab.linear.weight[H+1:2H,   :] .= bp
    ps.core.glu_ab.linear.weight[2H+1:3H,  :] .= ag
    ps.core.glu_ab.linear.weight[3H+1:4H,  :] .= bg

    _load_layernorm!(ps.core.layer_norm_out, npz, prefix * "/center_layer_norm")
    _load_dense!(ps.core.glu_out.linear, npz, prefix * "/output_projection")
    _load_dense!(ps.core.glu_out.gate,   npz, prefix * "/gating_linear")
end

function _load_trimul_s!(ps, npz, prefix::String, i::Int; outgoing::Bool=true)
    _load_layernorm_s!(ps.layer_norm, npz, prefix * "/layer_norm_input", i)

    H = size(ps.core.glu_ab.linear.weight, 1) ÷ 4
    if outgoing
        ap = _block(npz, prefix * "/left_projection/weights",  i)
        bp = _block(npz, prefix * "/right_projection/weights", i)
        ag = _block(npz, prefix * "/left_gate/weights",  i)
        bg = _block(npz, prefix * "/right_gate/weights", i)
    else
        ap = _block(npz, prefix * "/right_projection/weights", i)
        bp = _block(npz, prefix * "/left_projection/weights",  i)
        ag = _block(npz, prefix * "/right_gate/weights", i)
        bg = _block(npz, prefix * "/left_gate/weights",  i)
    end
    ps.core.glu_ab.linear.weight[1:H,      :] .= ap
    ps.core.glu_ab.linear.weight[H+1:2H,   :] .= bp
    ps.core.glu_ab.linear.weight[2H+1:3H,  :] .= ag
    ps.core.glu_ab.linear.weight[3H+1:4H,  :] .= bg

    _load_layernorm_s!(ps.core.layer_norm_out, npz, prefix * "/center_layer_norm", i)
    _load_dense_s!(ps.core.glu_out.linear, npz, prefix * "/output_projection", i)
    _load_dense_s!(ps.core.glu_out.gate,   npz, prefix * "/gating_linear",     i)
end

# ============================================================================
# Triangle Attention
# ============================================================================

function _load_triatt!(ps, npz, prefix::String)
    _load_layernorm!(ps.layer_norm, npz, prefix * "/query_norm")
    ps.linear.weight .= _w(npz, prefix * "/feat_2d_weights")   # no bias (use_bias=false)
    _load_mha!(ps.mha, npz, prefix * "/attention"; gated=true)
end

function _load_triatt_s!(ps, npz, prefix::String, i::Int)
    _load_layernorm_s!(ps.layer_norm, npz, prefix * "/query_norm", i)
    ps.linear.weight .= _block(npz, prefix * "/feat_2d_weights", i)
    _load_mha_s!(ps.mha, npz, prefix * "/attention", i; gated=true)
end

# ============================================================================
# PairStackBlock  (used by Evoformer, ExtraMSA, and Template pair stacks)
# ============================================================================

function _load_pair_stack_block_s!(ps, npz, prefix::String, i::Int)
    _load_triatt_s!(ps.tri_att_start, npz, prefix * "/triangle_attention_starting_node", i)
    _load_triatt_s!(ps.tri_att_end,   npz, prefix * "/triangle_attention_ending_node",   i)
    _load_trimul_s!(ps.tri_mul_out, npz, prefix * "/triangle_multiplication_outgoing", i; outgoing=true)
    _load_trimul_s!(ps.tri_mul_in,  npz, prefix * "/triangle_multiplication_incoming", i; outgoing=false)

    pt_pfx = prefix * "/pair_transition"
    _load_layernorm_s!(ps.pair_transition.layer_norm, npz, pt_pfx * "/input_layer_norm", i)
    _load_dense_s!(ps.pair_transition.linear_1, npz, pt_pfx * "/transition1", i)
    _load_dense_s!(ps.pair_transition.linear_2, npz, pt_pfx * "/transition2", i)
end

# ============================================================================
# EvoformerBlock  (blocks[:block_i])
# ============================================================================

function _load_evoformer_block_s!(ps, npz, prefix::String, i::Int)
    # MSA row attention (AttentionPairBias):
    #   Julia layer_norm_in ↔ Python layer_norm_m
    row_pfx = prefix * "/msa_row_attention_with_pair_bias"
    _load_layernorm_s!(ps.msa_att_row.layer_norm_in, npz, row_pfx * "/query_norm",    i)
    _load_layernorm_s!(ps.msa_att_row.layer_norm_z,  npz, row_pfx * "/feat_2d_norm",  i)
    ps.msa_att_row.linear_z.weight .= _block(npz, row_pfx * "/feat_2d_weights", i)
    _load_mha_s!(ps.msa_att_row.mha, npz, row_pfx * "/attention", i; gated=true)

    # MSA column attention (MSAColumnAttention):
    #   Julia layer_norm ↔ Python _msa_att.layer_norm_m
    col_pfx = prefix * "/msa_column_attention"
    _load_layernorm_s!(ps.msa_att_col.layer_norm, npz, col_pfx * "/query_norm", i)
    _load_mha_s!(ps.msa_att_col.mha, npz, col_pfx * "/attention", i; gated=true)

    # MSA transition
    mt_pfx = prefix * "/msa_transition"
    _load_layernorm_s!(ps.msa_transition.layer_norm, npz, mt_pfx * "/input_layer_norm", i)
    _load_dense_s!(ps.msa_transition.linear_1, npz, mt_pfx * "/transition1", i)
    _load_dense_s!(ps.msa_transition.linear_2, npz, mt_pfx * "/transition2", i)

    # OuterProductMean
    opm_pfx = prefix * "/outer_product_mean"
    _load_layernorm_s!(ps.outer_product_mean.layer_norm, npz, opm_pfx * "/layer_norm_input", i)
    _load_dense_s!(ps.outer_product_mean.linear1, npz, opm_pfx * "/left_projection",  i)
    _load_dense_s!(ps.outer_product_mean.linear2, npz, opm_pfx * "/right_projection", i)
    ow = _block(npz, opm_pfx * "/output_w", i)
    ps.outer_product_mean.linear_out.weight .= _opm_out_w(ow)
    ps.outer_product_mean.linear_out.bias   .= _block(npz, opm_pfx * "/output_b", i)

    _load_pair_stack_block_s!(ps.pair_stack, npz, prefix, i)
end

# ============================================================================
# ExtraMSABlock  (blocks[:block_i], uses GlobalAttention for column)
# ============================================================================

function _load_extra_msa_block_s!(ps, npz, prefix::String, i::Int)
    # MSA row attention: same as EvoformerBlock
    row_pfx = prefix * "/msa_row_attention_with_pair_bias"
    _load_layernorm_s!(ps.msa_att_row.layer_norm_in, npz, row_pfx * "/query_norm",   i)
    _load_layernorm_s!(ps.msa_att_row.layer_norm_z,  npz, row_pfx * "/feat_2d_norm", i)
    ps.msa_att_row.linear_z.weight .= _block(npz, row_pfx * "/feat_2d_weights", i)
    _load_mha_s!(ps.msa_att_row.mha, npz, row_pfx * "/attention", i; gated=true)

    # MSA column GLOBAL attention (MSAColumnGlobalAttention):
    #   Julia: layer_norm, linear_q, linear_k, linear_v, linear_g, linear_o
    #   Python: layer_norm_m + global_attention.{linear_q,k,v,g,o}
    #   K and V use LinearWeight (not MHA) since global attention is single-head.
    col_pfx = prefix * "/msa_column_global_attention"
    _load_layernorm_s!(ps.msa_att_col.layer_norm, npz, col_pfx * "/query_norm", i)
    att_pfx = col_pfx * "/attention"
    ps.msa_att_col.linear_q.weight .= _mha_qkv_w(_block(npz, att_pfx * "/query_w",  i))
    ps.msa_att_col.linear_k.weight .= _block(npz, att_pfx * "/key_w",   i)  # single-head → no reshape
    ps.msa_att_col.linear_v.weight .= _block(npz, att_pfx * "/value_w", i)
    ps.msa_att_col.linear_g.weight .= _mha_qkv_w(_block(npz, att_pfx * "/gating_w", i))
    let gb = _block(npz, att_pfx * "/gating_b", i)
        ps.msa_att_col.linear_g.bias .= ndims(gb) == 2 ? vec(permutedims(gb, (2, 1))) : vec(gb)
    end
    ow = _block(npz, att_pfx * "/output_w", i)
    ps.msa_att_col.linear_o.weight .= _mha_out_w(ow)
    ps.msa_att_col.linear_o.bias   .= _block(npz, att_pfx * "/output_b", i)

    # MSA transition
    mt_pfx = prefix * "/msa_transition"
    _load_layernorm_s!(ps.msa_transition.layer_norm, npz, mt_pfx * "/input_layer_norm", i)
    _load_dense_s!(ps.msa_transition.linear_1, npz, mt_pfx * "/transition1", i)
    _load_dense_s!(ps.msa_transition.linear_2, npz, mt_pfx * "/transition2", i)

    # OuterProductMean
    opm_pfx = prefix * "/outer_product_mean"
    _load_layernorm_s!(ps.outer_product_mean.layer_norm, npz, opm_pfx * "/layer_norm_input", i)
    _load_dense_s!(ps.outer_product_mean.linear1, npz, opm_pfx * "/left_projection",  i)
    _load_dense_s!(ps.outer_product_mean.linear2, npz, opm_pfx * "/right_projection", i)
    ow = _block(npz, opm_pfx * "/output_w", i)
    ps.outer_product_mean.linear_out.weight .= _opm_out_w(ow)
    ps.outer_product_mean.linear_out.bias   .= _block(npz, opm_pfx * "/output_b", i)

    _load_pair_stack_block_s!(ps.pair_stack, npz, prefix, i)
end

# ============================================================================
# IPA  (InvariantPointAttention)
# ============================================================================

function _load_ipa!(ps, npz, prefix::String; is_multimer::Bool=false)
    ipa_pfx = prefix * "/invariant_point_attention"
    if !is_multimer
        # Monomer: fused KV (linear_kv) and fused KV-points (linear_kv_pts.linear)
        _load_dense!(ps.linear_q,              npz, ipa_pfx * "/q_scalar")
        _load_dense!(ps.linear_kv,             npz, ipa_pfx * "/kv_scalar")
        _load_dense!(ps.linear_q_pts.linear,   npz, ipa_pfx * "/q_point_local")
        _load_dense!(ps.linear_kv_pts.linear,  npz, ipa_pfx * "/kv_point_local")
    else
        # Multimer: separate Q/K/V and separate point projections (LinearWeightMHA)
        ps.linear_q.weight .= _mha_qkv_w(_w(npz, ipa_pfx * "/q_scalar_projection/weights"))
        ps.linear_k.weight .= _mha_qkv_w(_w(npz, ipa_pfx * "/k_scalar_projection/weights"))
        ps.linear_v.weight .= _mha_qkv_w(_w(npz, ipa_pfx * "/v_scalar_projection/weights"))
        _load_dense!(ps.linear_q_pts.linear, npz, ipa_pfx * "/q_point_projection/point_projection")
        _load_dense!(ps.linear_k_pts.linear, npz, ipa_pfx * "/k_point_projection/point_projection")
        _load_dense!(ps.linear_v_pts.linear, npz, ipa_pfx * "/v_point_projection/point_projection")
    end
    _load_dense!(ps.linear_b,   npz, ipa_pfx * "/attention_2d")
    ps.head_weights.w .= _w(npz, ipa_pfx * "/trainable_point_weights")
    _load_dense!(ps.linear_out, npz, ipa_pfx * "/output_projection")
end

# ============================================================================
# StructureModule
# ============================================================================

function _load_structure_module!(ps, npz; is_multimer::Bool=false)
    sm_pfx = _NPZ_PREFIX * "structure_module"

    _load_layernorm!(ps.layer_norm_s, npz, sm_pfx * "/single_layer_norm")
    _load_dense!(ps.linear_in,        npz, sm_pfx * "/initial_projection")
    _load_layernorm!(ps.layer_norm_z, npz, sm_pfx * "/pair_layer_norm")

    fold_pfx = sm_pfx * "/fold_iteration"

    _load_ipa!(ps.fold.ipa, npz, fold_pfx; is_multimer)
    _load_layernorm!(ps.fold.layer_norm_ipa, npz, fold_pfx * "/attention_layer_norm")

    # StructureModuleTransition: layers is a Lux.Chain with one layer_1 block
    _load_dense!(ps.fold.transition.layers.layer_1.linear_1, npz, fold_pfx * "/transition")
    _load_dense!(ps.fold.transition.layers.layer_1.linear_2, npz, fold_pfx * "/transition_1")
    _load_dense!(ps.fold.transition.layers.layer_1.linear_3, npz, fold_pfx * "/transition_2")
    _load_layernorm!(ps.fold.transition.layer_norm, npz, fold_pfx * "/transition_layer_norm")

    # BackboneUpdate
    if !is_multimer
        _load_dense!(ps.fold.backbone_update.linear, npz, fold_pfx * "/affine_update")
    else
        _load_dense!(ps.fold.backbone_update.linear, npz, fold_pfx * "/quat_rigid/rigid")
    end

    # AngleResnet: blocks is a Lux.Chain with layer_1, layer_2 (no_resnet_blocks=2)
    ar_pfx = fold_pfx * "/rigid_sidechain"
    _load_dense!(ps.fold.angle_resnet.linear_in,      npz, ar_pfx * "/input_projection")
    _load_dense!(ps.fold.angle_resnet.linear_initial, npz, ar_pfx * "/input_projection_1")
    _load_dense!(ps.fold.angle_resnet.blocks.layer_1.linear_1, npz, ar_pfx * "/resblock1")
    _load_dense!(ps.fold.angle_resnet.blocks.layer_1.linear_2, npz, ar_pfx * "/resblock2")
    _load_dense!(ps.fold.angle_resnet.blocks.layer_2.linear_1, npz, ar_pfx * "/resblock1_1")
    _load_dense!(ps.fold.angle_resnet.blocks.layer_2.linear_2, npz, ar_pfx * "/resblock2_1")
    _load_dense!(ps.fold.angle_resnet.linear_out, npz, ar_pfx * "/unnormalized_angles")
end

# ============================================================================
# TemplateEmbedder  (monomer)
# ============================================================================

function _load_template_embedder!(ps, npz)
    evo_pfx = _NPZ_PREFIX * "evoformer"

    # TemplateSingleEmbedder: Lux.Chain with layer_1 (relu Dense) + layer_2 (Dense)
    _load_dense!(ps.template_single_embedder.chain.layer_1,
        npz, evo_pfx * "/template_single_embedding")
    _load_dense!(ps.template_single_embedder.chain.layer_2,
        npz, evo_pfx * "/template_projection")

    # TemplatePairEmbedder: single Dense
    _load_dense!(ps.template_pair_embedder.linear,
        npz, evo_pfx * "/template_embedding/single_template_embedding/embedding2d")

    # TemplatePairStack blocks: stacked under __layer_stack_no_state
    tps_pfx = evo_pfx *
        "/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state"
    n_blocks = length(ps.template_pair_stack.blocks)
    for i in 1:n_blocks
        _load_pair_stack_block_s!(
            ps.template_pair_stack.blocks[Symbol("block_$i")],
            npz, tps_pfx, i
        )
    end

    # TemplatePairStack final LayerNorm
    _load_layernorm!(ps.template_pair_stack.layer_norm,
        npz, evo_pfx * "/template_embedding/single_template_embedding/output_layer_norm")

    # TemplatePointwiseAttention: non-gated cross-attention
    #   Julia: mha.qkv.q + mha.qkv.kv (fused KV), mha.out — no gate
    _load_mha!(ps.template_pointwise_att.mha,
        npz, evo_pfx * "/template_embedding/attention"; gated=false)
end

# ============================================================================
# AuxiliaryHeads
# ============================================================================

function _load_aux_heads!(ps, npz; has_tm::Bool=false)
    plddt_pfx = _NPZ_PREFIX * "predicted_lddt_head"
    _load_layernorm!(ps.plddt.layer_norm, npz, plddt_pfx * "/input_layer_norm")
    _load_dense!(ps.plddt.linear_1, npz, plddt_pfx * "/act_0")
    _load_dense!(ps.plddt.linear_2, npz, plddt_pfx * "/act_1")
    _load_dense!(ps.plddt.linear_3, npz, plddt_pfx * "/logits")

    _load_dense!(ps.distogram.linear,
        npz, _NPZ_PREFIX * "distogram_head/half_logits")
    _load_dense!(ps.experimentally_resolved.linear,
        npz, _NPZ_PREFIX * "experimentally_resolved_head/logits")
    _load_dense!(ps.masked_msa.linear,
        npz, _NPZ_PREFIX * "masked_msa_head/logits")

    if has_tm
        _load_dense!(ps.tm.linear,
            npz, _NPZ_PREFIX * "predicted_aligned_error_head/logits")
    end
end

# ============================================================================
# Top-level loader
# ============================================================================

function _load_weights_from_npz!(ps, npz;
    use_templates::Bool=true,
    is_multimer::Bool=false,
    has_tm::Bool=false,
)
    evo_pfx = _NPZ_PREFIX * "evoformer"

    # ── InputEmbedder ──────────────────────────────────────────────────────────
    _load_dense!(ps.input_embedder.linear_i,          npz, evo_pfx * "/left_single")
    _load_dense!(ps.input_embedder.linear_j,          npz, evo_pfx * "/right_single")
    _load_dense!(ps.input_embedder.linear_target_msa, npz, evo_pfx * "/preprocess_1d")
    _load_dense!(ps.input_embedder.linear_msa,        npz, evo_pfx * "/preprocess_msa")
    # Relative position encoding — different key name for multimer
    relpos_key = is_multimer ?
        evo_pfx * "/~_relative_encoding/position_activations" :
        evo_pfx * "/pair_activiations"    # DeepMind typo preserved in NPZ
    _load_dense!(ps.input_embedder.relpos_encoding.linear, npz, relpos_key)

    # ── RecyclingEmbedder ──────────────────────────────────────────────────────
    _load_dense!(    ps.recycling_embedder.linear,          npz, evo_pfx * "/prev_pos_linear")
    _load_layernorm!(ps.recycling_embedder.layer_norm_m,    npz, evo_pfx * "/prev_msa_first_row_norm")
    _load_layernorm!(ps.recycling_embedder.layer_norm_z,    npz, evo_pfx * "/prev_pair_norm")

    # ── ExtraMSAEmbedder ───────────────────────────────────────────────────────
    _load_dense!(ps.extra_msa_embedder.linear, npz, evo_pfx * "/extra_msa_activations")

    # ── ExtraMSAStack ──────────────────────────────────────────────────────────
    n_extra = length(ps.extra_msa_stack.blocks)
    for i in 1:n_extra
        _load_extra_msa_block_s!(
            ps.extra_msa_stack.blocks[Symbol("block_$i")],
            npz, evo_pfx * "/extra_msa_stack", i
        )
    end

    # ── EvoformerStack ─────────────────────────────────────────────────────────
    n_evo = length(ps.evoformer.blocks)
    for i in 1:n_evo
        _load_evoformer_block_s!(
            ps.evoformer.blocks[Symbol("block_$i")],
            npz, evo_pfx * "/evoformer_iteration", i
        )
    end
    _load_dense!(ps.evoformer.single_projection, npz, evo_pfx * "/single_activations")

    # ── TemplateEmbedder ───────────────────────────────────────────────────────
    if use_templates
        _load_template_embedder!(ps.template_embedder, npz)
    end

    # ── StructureModule ────────────────────────────────────────────────────────
    _load_structure_module!(ps.structure_module, npz; is_multimer)

    # ── AuxiliaryHeads ──────────────────────────────────────────────────────────
    _load_aux_heads!(ps.aux_heads, npz; has_tm)

    return ps
end

# ============================================================================
# Public API
# ============================================================================

"""
    load_alphafold2_weights!(ps, path::AbstractString;
                             use_templates::Bool = true,
                             is_multimer::Bool   = false,
                             has_tm::Bool        = false)

Load official AlphaFold2 `.npz` weights from `path` into the Julia parameter
NamedTuple `ps` (in-place). Returns `ps`.

`ps` must come from `Lux.setup(rng, model)` where `model` was built with the
same `use_templates` / `is_multimer` settings.

`has_tm = true` loads the `predicted_aligned_error_head` (present in `_ptm`
variants and all multimer models).

# Example
```julia
model = AlphaFold(:model_1)
ps, st = Lux.setup(Random.Xoshiro(42), model)
load_alphafold2_weights!(ps, "/data/af2/params_model_1.npz")
```
"""
function load_alphafold2_weights!(
    ps, path::AbstractString;
    use_templates::Bool = true,
    is_multimer::Bool   = false,
    has_tm::Bool        = false,
)
    npz = NPZ.npzread(path)
    _load_weights_from_npz!(ps, npz; use_templates, is_multimer, has_tm)
    return ps
end

"""
    weights_dir() -> String

Return the current default directory used to store AlphaFold2 weight files.
Defaults to `~/.alphafold`. Override permanently with `set_weights_dir!`.
"""
weights_dir() = _DEFAULT_WEIGHTS_DIR[]

"""
    set_weights_dir!(path::AbstractString)

Set the default directory for AlphaFold2 weight files for this Julia session.
The directory is created if it does not exist when weights are first downloaded.

```julia
AlphaFold2.set_weights_dir!("/data/alphafold")
```
"""
set_weights_dir!(path::AbstractString) = (_DEFAULT_WEIGHTS_DIR[] = String(path); nothing)

"""
    download_alphafold2_weights(; dir = weights_dir(), force = false)

Download the official DeepMind AlphaFold2 parameter tarball and extract all
`.npz` weight files into `dir` (default: `~/.alphafold`).

The tarball (~4 GB) is downloaded once and then deleted after extraction.
Individual `.npz` files (~380 MB each) are kept in `dir`. Existing files are
skipped unless `force = true`.

# Keyword Arguments
- `dir`:   Directory to extract weights into (default: `weights_dir()`)
- `force`: Re-download and overwrite existing files (default: `false`)

# Example
```julia
download_alphafold2_weights()                         # → ~/.alphafold/
download_alphafold2_weights(dir = "/data/alphafold")  # custom path
```
"""
function download_alphafold2_weights(;
    dir::AbstractString = weights_dir(),
    force::Bool = false,
)
    mkpath(dir)

    # Check whether all expected files are already present.
    expected = [entry.filename for entry in values(AF2_MODEL_REGISTRY)]
    unique!(expected)
    missing_files = filter(f -> !isfile(joinpath(dir, f)), expected)

    if isempty(missing_files) && !force
        @info "All AlphaFold2 weight files already present." dir
        return dir
    end

    tar_path = joinpath(dir, "alphafold_params.tar")
    try
        if !isfile(tar_path) || force
            @info "Downloading AlphaFold2 params (~4 GB)…" url=_AF2_PARAMS_URL dest=tar_path
            Downloads.download(_AF2_PARAMS_URL, tar_path; progress = _download_progress)
        else
            @info "Tarball already downloaded, extracting…" tar_path
        end

        @info "Extracting weight files…" dir
        open(tar_path) do io
            Tar.extract(io, dir) do hdr
                # Only extract the .npz files we actually need.
                basename(hdr.path) in expected
            end
        end
        @info "Extraction complete." dir n_files=length(expected)
    finally
        # Always clean up the tarball — it's ~4 GB we don't need to keep.
        isfile(tar_path) && rm(tar_path)
    end

    return dir
end

# Simple progress callback for Downloads.download.
function _download_progress(total::Integer, now::Integer)
    total > 0 || return
    pct = round(Int, 100 * now / total)
    mb_now   = round(now   / 1e6; digits=1)
    mb_total = round(total / 1e6; digits=1)
    print("\r  $(pct)%  $(mb_now) / $(mb_total) MB   ")
    now == total && println()
end

"""
    load_alphafold2_weights!(ps, variant::Symbol; dir = weights_dir())

Load official DeepMind weights for `variant` (e.g. `:model_1`, `:model_3`,
`:model_2_ptm`, `:model_1_multimer_v3`) from `dir`.

If the weight file is not found in `dir`, a helpful error is raised suggesting
`download_alphafold2_weights()`.

`ps` must have been created from `AlphaFold(variant)` (or an equivalent call).

# Example
```julia
download_alphafold2_weights()          # once — populates ~/.alphafold/

model = AlphaFold(:model_1)
ps, st = Lux.setup(Random.Xoshiro(42), model)
load_alphafold2_weights!(ps, :model_1)

# Or with a custom directory:
load_alphafold2_weights!(ps, :model_1; dir = "/data/alphafold")
```
"""
function load_alphafold2_weights!(ps, variant::Symbol; dir::AbstractString = weights_dir())
    haskey(AF2_MODEL_REGISTRY, variant) ||
        throw(ArgumentError(
            "Unknown model variant $(repr(variant)). " *
            "Known variants: $(sort(collect(keys(AF2_MODEL_REGISTRY))))"
        ))
    entry = AF2_MODEL_REGISTRY[variant]
    path  = joinpath(dir, entry.filename)

    isfile(path) || throw(ArgumentError(
        "Weight file not found: $path\n" *
        "Run `AlphaFold2.download_alphafold2_weights(; dir=$(repr(dir)))` to download."
    ))

    is_multimer = entry.architecture == :multimer
    has_tm      = endswith(string(variant), "_ptm") || is_multimer
    return load_alphafold2_weights!(ps, path;
        use_templates = entry.use_templates,
        is_multimer,
        has_tm,
    )
end
