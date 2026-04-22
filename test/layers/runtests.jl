@testset "AdaLN" begin
    include("adaln.af3.jl")
    include("adaln.boltz2.jl")
end

@testset "Attention" begin
    include("attention.af3.jl")
end