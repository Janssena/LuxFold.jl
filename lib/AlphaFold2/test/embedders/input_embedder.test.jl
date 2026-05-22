const PyInputEmbedder = pyimport("openfold.model.embedders").InputEmbedder
const CHN_TARGET_FEAT, CHN_MSA_FEAT, CHN_PAIR, CHN_MSA, RELPOS_K = 22, 49, 128, 256, 32

rng = Random.Xoshiro(42)

@testset "InputEmbedder" begin
    @testset "Python parity" begin
        N, S, B = 8, 4, 2

        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                jl_layer = InputEmbedder(CHN_TARGET_FEAT, CHN_MSA_FEAT, CHN_PAIR, CHN_MSA, RELPOS_K)
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyInputEmbedder(CHN_TARGET_FEAT, CHN_MSA_FEAT, CHN_PAIR, CHN_MSA, RELPOS_K)

                sync_dense!(py_layer.linear_tf_z_i, jl_ps.linear_i)
                sync_dense!(py_layer.linear_tf_z_j, jl_ps.linear_j)
                sync_dense!(py_layer.linear_tf_m, jl_ps.linear_target_msa)
                sync_dense!(py_layer.linear_msa_m, jl_ps.linear_msa)
                sync_dense!(py_layer.linear_relpos, jl_ps.relpos_encoding.linear)

                target_feat = randn(rng, T, CHN_TARGET_FEAT, N, B)
                residue_index = rand(rng, 1:100, N, B)
                msa_feat = randn(rng, T, CHN_MSA_FEAT, N, S, B)

                (m_jl, z_jl), _ = jl_layer(target_feat, residue_index, msa_feat, jl_ps, jl_st)

                tf_py = to_py(target_feat; swap_batch_dim=true)
                ri_py = to_py(residue_index; swap_batch_dim=true)
                msa_py = to_py(permutedims(msa_feat, (4, 3, 2, 1)); swap_batch_dim=false)

                m_py, z_py = py_layer(tf_py, ri_py, msa_py)

                @testset "MSA parity" begin
                    @test m_jl ≈ permutedims(to_jl(m_py; swap_batch_dim=true), (1, 3, 2, 4))
                end

                @testset "Pair parity" begin
                    @test z_jl ≈ to_jl(z_py; swap_batch_dim=true)
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred jl_layer(target_feat, residue_index, msa_feat, jl_ps, jl_st)
                end
            end
        end
    end
end
