import AlphaFold3: Lux, Static
import Random

using Test, AlphaFold3

include("setup/python.jl");

@testset "AlphaFold3" begin
    @testset "Utils" begin
        include("utils/runtests.jl")
    end
    @testset "Embedders" begin
        include("embedders/runtests.jl")
    end
    @testset "Atom Module" begin
        include("atom-module/runtests.jl")
    end
    @testset "Diffusion" begin
        include("diffusion/runtests.jl")
    end
    @testset "Pairformer" begin
        include("pairformer/runtests.jl")
    end
    @testset "MSA Module" begin
        include("msa-module/runtests.jl")
    end
    @testset "Heads" begin
        include("heads/runtests.jl")
    end
    @testset "Full Model" begin
        include("test_alphafold3.jl")
    end
end
