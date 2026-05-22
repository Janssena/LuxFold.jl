import LuxFoldCore: Lux
import Random

using Test, LuxFoldCore

include("python/setup.jl");

@testset "LuxFoldCore.jl" begin
    @testset "Layers" begin
        include("layers/runtests.jl")
    end
end