# Shared helpers (convert_types, sync_atom_attention_*, transformer_sync, …)
include("../atom-module/sync_helpers.jl")

@testset "NoisyPositionEmbedder" begin
    include("noisy_position_embedder.test.jl")
end

@testset "DiffusionTransformerBlock" begin
    include("diffusion_transformer_block.test.jl")
end

@testset "DiffusionTransformer" begin
    include("diffusion_transformer.test.jl")
end

@testset "DiffusionConditioning" begin
    include("diffusion_conditioning.test.jl")
end

@testset "DiffusionModule" begin
    include("diffusion_module.test.jl")
end

@testset "SampleDiffusion" begin
    include("sample_diffusion.test.jl")
end
