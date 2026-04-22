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
export Attention, AttentionPairBias

end # module LuxFoldCore
