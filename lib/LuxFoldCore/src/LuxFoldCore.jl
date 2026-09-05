module LuxFoldCore

import Lux

using Static
using Reexport

@reexport using LuxTriangleAttention

include("layers/adaln.jl")
export AdaLN

include("layers/attention_pair_bias.jl")
export AttentionPairBias, MSARowAttentionPairBias

include("layers/outer_product_mean.jl")
include("layers/pair_weighted_averaging.jl")
export OuterProductMean, PairWeightedAveraging

include("utils.jl")
export pad_and_block, unblock_and_slice

include("weights.jl")
export WeightsCache, cache_dir, set_cache_dir!, ModelRegistryEntry, ModelRegistry
export download_file, fetch_weights, read_state_dict, write_state_dict, load_weights!, download_weights
export flatten_params, load_flat_weights!

end # module LuxFoldCore
