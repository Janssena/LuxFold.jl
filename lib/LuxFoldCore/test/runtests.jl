# LuxFoldCore's suite: pure-Julia tests here, Python parity in three CHILD PROCESSES.
#
# The shared layers (AdaLN, AttentionPairBias, OuterProductMean, PairWeightedAveraging) are ported
# from three different upstream packages, and each is tested against its own real reference, i.e.
# openfold, openfold-3 and boltz. Those live in three independent Python environments
# (3.10, 3.14, 3.11), and PythonCall fixes the interpreter when it initialises, so one process
# cannot reach all three. Hence one child per environment.

import LuxFoldCore: Lux
import Random
using Test, LuxFoldCore

const _TEST_DIR = @__DIR__
const _PROJECT = Base.active_project()

"""Run one per-environment child suite, failing the main suite if the child fails."""
function run_env_suite(name::AbstractString, file::AbstractString)
    @testset "$name" begin
        script = joinpath(_TEST_DIR, file)
        cmd = `$(Base.julia_cmd()) --project=$(_PROJECT) $script`
        ok = success(pipeline(cmd; stdout=stdout, stderr=stderr))
        @test ok
    end
end

@testset "LuxFoldCore.jl" begin
    @testset "Weights" begin
        include("weights/weights.test.jl")
    end

    # Python parity, one child process per reference environment.
    run_env_suite("AlphaFold2 reference (openfold)", "runtests_af2.jl")
    run_env_suite("AlphaFold3 reference (openfold-3)", "runtests_af3.jl")
    run_env_suite("Boltz2 reference (boltz)", "runtests_boltz2.jl")
end
