import StableHLOModels: Lux
import Random

using Test, StableHLOModels

@testset "StableHLOModels.jl" begin
    @testset "Checkpoints" begin
        include("save.test.jl")
        include("save_build.test.jl")
    end
end
