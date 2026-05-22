import AlphaFold2: Lux
import Random

using Test, AlphaFold2

include("setup/python.jl");

@testset "AlphaFold2" begin
    @testset "Layers" begin
        include("layers/runtests.jl")
    end
    
    @testset "Embedders" begin
        include("embedders/runtests.jl")
    end
end