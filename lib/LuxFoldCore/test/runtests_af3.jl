# LuxFoldCore's shared layers against the REAL openfold-3 (AlphaFold3) reference.
#
# Its own process: the interpreter is fixed when PythonCall initialises, so one process cannot hold
# openfold, openfold-3 and boltz at once (three different Python ABIs — 3.10, 3.14, 3.11). The
# driver in `runtests.jl` launches this and its two siblings separately.
repo = abspath(joinpath(@__DIR__, "..", "..", ".."))
pythonpath = joinpath(repo, "python", "openfold-3", ".pixi", "envs", "openfold3-base", "bin", "python")
ENV["JULIA_PYTHONCALL_EXE"]   = pythonpath
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

import PythonTestHelpers: setup
setup(pythonpath)

using PythonCall, PythonTestHelpers
import LuxFoldCore: Lux
import Random
using Test, LuxFoldCore

pyimport("sys").path.insert(0, joinpath(repo, "python", "openfold-3"))
mock_imports("deepspeed", "flash_attn", "attn_core_inplace_cuda")
include("python/af3.jl")

@testset "LuxFoldCore vs openfold-3 (AF3)" begin
    include("layers/adaln.af3.jl")
    include("layers/attention_pair_bias.af3.jl")
    include("layers/outer_product_mean.af3.jl")
    include("layers/pair_weighted_averaging.af3.jl")
end
