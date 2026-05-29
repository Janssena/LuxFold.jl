const PyPreEmbeddingEmbedder = pyimport("openfold.model.embedders").PreembeddingEmbedder

rng = Random.Xoshiro(42)

@testset "PreEmbeddingEmbedder" begin
    feat_dim = 22
    preemb_dim = 128
    chn_z = 128
    chn_msa = 256
    relpos_k = 32
    @testset "Python parity" begin
        N, B = 8, 2

        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                jl_layer = PreEmbeddingEmbedder(feat_dim, preemb_dim, chn_z, chn_msa, relpos_k)
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyPreEmbeddingEmbedder(feat_dim, preemb_dim, chn_z, chn_msa, relpos_k)

                sync_dense!(py_layer.linear_tf_m, jl_ps.linear_target_msa)
                sync_dense!(py_layer.linear_preemb_m, jl_ps.linear_preembedding_msa)
                sync_dense!(py_layer.linear_preemb_z_i, jl_ps.linear_preembedding_pair_i)
                sync_dense!(py_layer.linear_preemb_z_j, jl_ps.linear_preembedding_pair_j)
                sync_dense!(py_layer.linear_relpos, jl_ps.relpos.linear)

                target_feat = randn(rng, T, feat_dim, N, B)
                residue_index = rand(rng, 1:100, N, B)
                preembedding = randn(rng, T, preemb_dim, N, B)

                (m_jl, z_jl), _ = jl_layer(target_feat, residue_index, preembedding, jl_ps, jl_st)

                tf_py = to_py(target_feat; swap_batch_dim=true)
                ri_py = to_py(residue_index; swap_batch_dim=true)
                preemb_py = to_py(preembedding; swap_batch_dim=true)

                m_py, z_py = py_layer(tf_py, ri_py, preemb_py)

                @testset "MSA parity" begin
                    @test m_jl ≈ to_jl(m_py; swap_batch_dim=true)
                end

                @testset "Pair parity" begin
                    @test z_jl ≈ to_jl(z_py; swap_batch_dim=true)
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred jl_layer(target_feat, residue_index, preembedding, jl_ps, jl_st)
                end
            end
        end
    end
end
