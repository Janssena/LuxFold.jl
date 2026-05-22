const PyRecyclingEmbedder = pyimport("openfold.model.embedders").RecyclingEmbedder
const C_M, C_Z, NO_BINS = 16, 16, 15

rng = Random.Xoshiro(42)

@testset "RecyclingEmbedder" begin
    @testset "Python parity" begin
        N, B = 8, 2
        min_bin, max_bin = 3.25f0, 20.75f0

        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                jl_layer = RecyclingEmbedder(C_M, C_Z; min_bin, max_bin, no_bins=NO_BINS)
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyRecyclingEmbedder(C_M, C_Z, min_bin, max_bin, NO_BINS, T(1e8))

                sync_layernorm!(py_layer.layer_norm_m, jl_ps.layer_norm_m)
                sync_layernorm!(py_layer.layer_norm_z, jl_ps.layer_norm_z)
                sync_dense!(py_layer.linear, jl_ps.linear)

                m_jl = randn(rng, T, C_M, N, B)
                z_jl = randn(rng, T, C_Z, N, N, B)
                x_jl = randn(rng, T, 3, N, B) .* T(5)

                (m_jl_out, z_jl_out), _ = jl_layer(m_jl, z_jl, x_jl, jl_ps, jl_st)

                m_py = to_py(m_jl; swap_batch_dim=true)
                z_py = to_py(z_jl; swap_batch_dim=true)
                x_py = to_py(x_jl; swap_batch_dim=true)

                m_py_out, z_py_out = py_layer(m_py, z_py, x_py)

                @testset "MSA parity" begin
                    @test m_jl_out ≈ to_jl(m_py_out; swap_batch_dim=true)
                end

                @testset "Pair parity" begin
                    @test z_jl_out ≈ to_jl(z_py_out; swap_batch_dim=true)
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred jl_layer(m_jl, z_jl, x_jl, jl_ps, jl_st)
                end
            end
        end
    end
end
