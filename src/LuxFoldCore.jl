module LuxFoldCore

import Lux

using Static

include("utils.jl")
export resolve_defaults, block_array, unblock_array, pad_array

include("layers/primitives.jl")
include("layers/adaln.jl")
export AdaLN, LayerNormNoBias, Activation, ReLU

include("layers/attention.jl")
include("layers/attention_pair_bias.jl")
include("layers/outer_product_mean.jl")
include("layers/pair_weighted_averaging.jl")
export Attention, AttentionPairBias, OuterProductMean, PairWeightedAveraging

end # module LuxFoldCore
