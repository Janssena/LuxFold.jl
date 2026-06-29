module AlphaFold3

using Lux
using Static
import Random

using LuxFoldCore
using LuxTriangleAttention

# Include utils (used by atom module + heads + diffusion)
include("utils/constants.jl")
include("utils/atom_attention_block_utils.jl")
include("utils/atomize_utils.jl")
include("utils/relpos.jl")
include("utils/geometry.jl")

# Include embedders (input embedders, ref atom feature embedder, template embedders)
include("embedders/fourier_embedding.jl")
include("embedders/input_embedder_all_atom.jl")
include("embedders/msa_module_embedder.jl")
include("embedders/ref_atom_embedder.jl")
include("embedders/template_pair_embedder_all_atom.jl")
include("embedders/template_pair_block.jl")
include("embedders/template_pair_stack.jl")
include("embedders/template_embedder_all_atom.jl")

# Include shared layers (transition, base blocks, conditioned transition)
include("layers/transition.jl")
include("layers/pair_block.jl")
include("layers/conditioned_transition_block.jl")

# Include pairformer
include("pairformer/pairformer_block.jl")
include("pairformer/pairformer_stack.jl")

# Include MSA module
include("msa-module/msa_block.jl")
include("msa-module/msa_stack.jl")

# Include atom module: sequence-local attention (always first — used by diffusion transformer)
include("atom-module/sequence_local_attention_pair_bias.jl")

# Include diffusion transformer (uses ConditionedTransitionBlock from layers,
# SequenceLocalAttentionPairBias from atom-module)
include("diffusion/diffusion_transformer_block.jl")
include("diffusion/diffusion_transformer.jl")

# Include diffusion noisy position embedder (used by AtomAttentionEncoder)
include("diffusion/noisy_position_embedder.jl")

# Include atom attention encoder/decoder (uses NoisyPositionEmbedder + DiffusionTransformer)
include("atom-module/atom_attention_encoder.jl")
include("atom-module/atom_attention_decoder.jl")

# Include remaining diffusion components
include("diffusion/diffusion_conditioning.jl")
include("diffusion/diffusion_module.jl")
include("diffusion/sample_diffusion.jl")

# Include heads
include("heads/prediction_heads.jl")
include("heads/head_modules.jl")

# Include top-level model (file is `model.jl`, not `alphafold3.jl`, to avoid a
# case-insensitive-filesystem collision with this module file `AlphaFold3.jl`).
include("model.jl")

export AlphaFold3Model, run_trunk
export DistogramHead, PredictedDistanceErrorHead, PredictedAlignedErrorHead
export PerResidueLDDTAllAtom, ExperimentallyResolvedHeadAllAtom
export PairformerEmbedding, AuxiliaryHeadsAllAtom

# Atom module (AF3 Algorithms 5 & 6)
export RefAtomFeatureEmbedder, NoisyPositionEmbedder
export AtomTransformer, AtomAttentionEncoder, AtomAttentionDecoder
export SequenceLocalAttentionPairBias

# Shared layers
export ConditionedTransitionBlock, DiffusionTransformerBlock, DiffusionTransformer

# Utils (af3-utils)
export relpos_complex, sample_rotations, quat_to_rot, centre_random_augmentation
export AF3_RESTYPE_NUM, AF3_ELEMENT_NUM, AF3_ATOM_NAME_CHARS, AF3_ATOM_NAME_NCHARS
export broadcast_token_feat_to_atoms, aggregate_atom_feat_to_tokens
export max_atom_per_token_masked_select
export get_token_representative_atoms, get_token_center_atoms
export get_query_block_padding, get_block_indices, get_pair_atom_block_mask
export convert_single_rep_to_blocks, convert_pair_rep_to_blocks

# Embedders (af3-input-embedder)
export FourierEmbedding, InputEmbedderAllAtom, MSAModuleEmbedder

# Diffusion (af3-diffusion)
export DiffusionConditioning, DiffusionModule, SampleDiffusion, af3_noise_schedule

# Pairformer (af3-pairformer)
export PairBlock, PairFormerBlock, PairFormerStack

# Template module (af3-template-module)
export TemplatePairEmbedderAllAtom, TemplatePairBlock, TemplatePairStack, TemplateEmbedderAllAtom

# MSA module (af3-msa-module)
export MSAModuleBlock, MSAModuleStack

end # module AlphaFold3
