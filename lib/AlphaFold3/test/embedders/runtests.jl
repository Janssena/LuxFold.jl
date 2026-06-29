# Shared helpers (convert_types, sync_atom_attention_encoder!, …)
include("../atom-module/sync_helpers.jl")

include("template_sync_helpers.jl")

@testset "FourierEmbedding" begin
    include("fourier_embedding.test.jl")
end

@testset "InputEmbedderAllAtom" begin
    include("input_embedder_all_atom.test.jl")
end

@testset "MSAModuleEmbedder" begin
    include("msa_embedder.test.jl")
end

@testset "RefAtomFeatureEmbedder" begin
    include("ref_atom_embedder.test.jl")
end

@testset "TemplatePairEmbedderAllAtom" begin
    include("template_pair_embedder_all_atom.test.jl")
end

@testset "TemplatePairBlock" begin
    include("template_pair_block.test.jl")
end

@testset "TemplatePairStack" begin
    include("template_pair_stack.test.jl")
end

@testset "TemplateEmbedderAllAtom" begin
    include("template_embedder_all_atom.test.jl")
end
