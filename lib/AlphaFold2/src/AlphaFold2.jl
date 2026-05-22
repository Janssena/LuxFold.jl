module AlphaFold2

using Lux
using LuxFoldCore
using LuxTriangleAttention
using Static

include("layers/transition.jl")
export Transition, MSATransition, PairTransition

include("embedders/relative_position_encoding.jl")
include("embedders/input_embedder.jl")
include("embedders/extra_msa_embedder.jl")
export RelativePositionEncoding
export InputEmbedder, ExtraMSAEmbedder

end # module AlphaFold2

