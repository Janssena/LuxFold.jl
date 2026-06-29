import AlphaFold3: Lux
import Random

using BenchmarkTools, AlphaFold3

include("setup/python.jl")

include("utils/atomize_utils.bench.jl")
include("utils/atom_attention_block_utils.bench.jl")
