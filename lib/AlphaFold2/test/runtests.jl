import AlphaFold2: Lux
import Random

using Test, AlphaFold2

include("setup/python.jl");

@testset "AlphaFold2" begin
    # The ordering here is relevant; PairStackBlock test defines helpers that are used in evoformer tests
    @testset "Layers" begin
        include("layers/runtests.jl")
    end

    @testset "Evoformer" begin
        include("evoformer/runtests.jl")
    end

    @testset "Heads" begin
        include("heads/runtests.jl")
    end

    @testset "Embedders" begin
        include("embedders/runtests.jl")
    end

    @testset "Utils" begin
        include("utils/runtests.jl")
    end

    @testset "Structure Module" begin
        include("structure-module/runtests.jl")
    end

    @testset "Integration" begin
        include("integration/runtests.jl")
    end

    @testset "AlphaFold Model" begin
        include("test_alphafold.jl")
    end
end
