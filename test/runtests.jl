import LuxFoldCore: Lux
import Random

using Test, LuxFoldCore

convert_types(::Type{Float64}) = Lux.f64
convert_types(::Type{Float32}) = Lux.f32
convert_types(::Type{Float16}) = Lux.f16

include("python/setup.jl");

@testset "LuxFoldCore.jl" begin
    @testset "Layers" begin
        include("layers/runtests.jl")
    end
end