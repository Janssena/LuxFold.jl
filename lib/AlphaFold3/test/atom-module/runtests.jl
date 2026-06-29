@testset "SequenceLocalAttentionPairBias" begin
    include("sequence_local_attention_pair_bias.test.jl")
end

include("sync_helpers.jl")
include("atom_attention_test_helpers.jl")

@testset "AtomAttentionEncoder" begin
    include("atom_attention_encoder.test.jl")
end

@testset "AtomAttentionDecoder" begin
    include("atom_attention_decoder.test.jl")
end
