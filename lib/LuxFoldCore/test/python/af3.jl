# The REAL openfold-3 layers behind LuxFoldCore's shared layers. See `af2.jl` for why these are
# imported rather than copied. Needs the openfold-3 pixi env, see `runtests_af3.jl`.
const _of3_apb = pyimport("openfold3.core.model.layers.attention_pair_bias")
const _of3_msa = pyimport("openfold3.core.model.layers.msa")

const PyAF3AdaLN                     = pyimport("openfold3.core.model.primitives.normalization").AdaLN
const PyAF3OuterProductMean          = pyimport("openfold3.core.model.layers.outer_product_mean").OuterProductMean
const PyAF3AttentionPairBias         = _of3_apb.AttentionPairBias
const PyAF3DiffusionAttentionPairBias = _of3_apb.DiffusionAttentionPairBias
const PyAF3CrossAttentionPairBias    = _of3_apb.CrossAttentionPairBias
const PyAF3MSAPairWeightedAveraging  = _of3_msa.MSAPairWeightedAveraging
const PyAF3MSARowAttentionWithPairBias = _of3_msa.MSARowAttentionWithPairBias
