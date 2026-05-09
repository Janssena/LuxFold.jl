@testset "AdaLN" begin
    include("adaln.af3.jl")
    include("adaln.boltz2.jl")
end

@testset "AttentionPairBias" begin
    include("attention_pair_bias.af3.jl")
    include("attention_pair_bias.boltz2.jl")
    include("crossed_attention_pair_bias.af3.jl")
end

@testset "OuterProductMean" begin
    include("outer_product_mean.af2.jl")
    include("outer_product_mean.af3.jl")
    include("outer_product_mean.boltz2.jl")
end

@testset "PairWeightedAveraging" begin
    include("pair_weighted_averaging.af3.jl")
    include("pair_weighted_averaging.boltz2.jl")
end