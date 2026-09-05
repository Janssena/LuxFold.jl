# LuxFoldCore's shared layers against the REAL boltz (Boltz2) reference.
#
# Its own process: the interpreter is fixed when PythonCall initialises, so one process cannot hold
# openfold, openfold-3 and boltz at once (three different Python ABIs — 3.10, 3.14, 3.11). The
# driver in `runtests.jl` launches this and its two siblings separately.
repo = abspath(joinpath(@__DIR__, "..", "..", ".."))
pythonpath = joinpath(repo, "python", "boltz", ".venv", "bin", "python")
ENV["JULIA_PYTHONCALL_EXE"]   = pythonpath
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

import PythonTestHelpers: setup
setup(pythonpath)

using PythonCall, PythonTestHelpers
import LuxFoldCore: Lux
import Random
using Test, LuxFoldCore

pyimport("sys").path.insert(0, joinpath(repo, "python", "boltz", "src"))
mock_imports("cuequivariance_torch")
include("python/boltz2.jl")

@testset "LuxFoldCore vs boltz (Boltz2)" begin
    include("layers/adaln.boltz2.jl")
    include("layers/attention_pair_bias.boltz2.jl")
    include("layers/outer_product_mean.boltz2.jl")
    include("layers/pair_weighted_averaging.boltz2.jl")
end
