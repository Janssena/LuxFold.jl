module LuxFoldCore

import Lux

using Static
using Reexport

@reexport using LuxTriangleAttention

include("layers/adaln.jl")
export AdaLN

include("layers/attention_pair_bias.jl")
include("layers/crossed_attention_pair_bias.jl")
export AttentionPairBias, MSARowAttentionPairBias, CrossedAttentionPairBias

include("layers/outer_product_mean.jl")
include("layers/pair_weighted_averaging.jl")
export OuterProductMean, PairWeightedAveraging

include("utils.jl")
export pad_and_block, unblock_and_slice

end # module LuxFoldCore
