module LuxFoldCore

import Lux

using Static

include("utils.jl")
export resolve_defaults, block_array, unblock_array, pad_array

include("layers/primitives.jl")
include("layers/adaln.jl")

export AdaLN, LayerNormNoBias, Activation, ReLU

end # module LuxFoldCore
