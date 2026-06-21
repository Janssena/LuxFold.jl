module AlphaFold2

using Lux
using LuxFoldCore
using LuxTriangleAttention
using Static
using Random

include("utils/constants.jl")
export restypes, restype_order, restype_1to3, restype_3to1
export restypes_with_x, restype_order_with_x
export chi_angles_atoms, chi_angles_mask
export atom_types, atom_order
export van_der_waals_radius
export backbone_atoms, backbone_atom_order
export residue_atoms, rigid_group_atom_positions
export restype_name_to_atom14_names
export restype_atom37_to_rigid_group, restype_atom37_mask, restype_atom37_rigid_group_positions
export restype_atom14_to_rigid_group, restype_atom14_mask, restype_atom14_rigid_group_positions
export restype_rigid_group_default_frame
export restype_atom37_to_atom14

include("utils/rigid_utils.jl")
export Rigid, rigid_identity
export quat_to_rotmat, rot_to_quat, quat_multiply_by_vec
export compose_q_update_vec, scale_translation
export to_tensor_7, to_tensor_4x4
export apply_rigid, apply_rotation_only
# Note: `apply(r::Rigid, pts)` and `invert_apply(r::Rigid, pts)` are NOT exported
# because the names are generic; callers should use `AlphaFold2.apply` or the
# qualified `apply_rigid` / existing `invert_apply(rot, trans, pts)` from geometry.

include("utils/geometry.jl")
export make_transform_from_reference, invert_apply

include("layers/transition.jl")
include("layers/pair_stack_block.jl")
export Transition, MSATransition, PairTransition, PairStackBlock

include("structure-module/backbone_update.jl")
export BackboneUpdate

include("structure-module/angle_resnet.jl")
export AngleResnetBlock, AngleResnet

include("structure-module/structure_module_transition.jl")
export StructureModuleTransitionLayer, StructureModuleTransition

include("structure-module/ipa.jl")
export PointProjection, HeadWeights, InvariantPointAttention

include("structure-module/structure_module.jl")
export StructureModuleFold, StructureModule

include("evoformer/msa_attention.jl")
export MSAColumnAttention, MSAColumnGlobalAttention

include("evoformer/evoformer_block.jl")
include("evoformer/extra_msa_block.jl")
export EvoformerBlock, ExtraMSABlock

include("evoformer/evoformer_stack.jl")
include("evoformer/extra_msa_stack.jl")
export EvoformerStack, ExtraMSAStack

include("heads/heads.jl")
export PerResidueLDDTCaPredictor, DistogramHead, TMScoreHead
export MaskedMSAHead, ExperimentallyResolvedHead, AuxiliaryHeads

include("utils/scoring.jl")
export compute_plddt, compute_predicted_aligned_error, compute_tm

include("utils/feats.jl")
export pseudo_beta_fn, atom14_to_atom37, build_extra_msa_feat

include("embedders/relative_position_encoding.jl")
include("embedders/input_embedder.jl")
include("embedders/extra_msa_embedder.jl")
include("embedders/recycling_embedder.jl")
include("embedders/preembedding_embedder.jl")
include("embedders/utils.jl")
include("utils/atom_utils.jl")
export torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos

include("embedders/template_pointwise_attention.jl")
include("embedders/template_pair_stack.jl")
include("embedders/template_embedder.jl")
export RelativePositionEncoding
export InputEmbedder, ExtraMSAEmbedder
export RecyclingEmbedder
export PreEmbeddingEmbedder
export dgram_from_positions, build_template_angle_feat, build_template_pair_feat
export TemplatePointwiseAttention, TemplatePairStackBlock, TemplatePairStack
export TemplateSingleEmbedder, TemplatePairEmbedder, TemplateEmbedder

include("alphafold.jl")
export AlphaFold, predict, iteration, tolerance_reached
export AF2_MODEL_REGISTRY, af2_weight_filename

include("utils/weight_loading.jl")
export load_alphafold2_weights!, download_alphafold2_weights
export weights_dir, set_weights_dir!

end # module AlphaFold2

