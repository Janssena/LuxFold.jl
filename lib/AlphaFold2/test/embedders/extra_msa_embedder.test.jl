const PyExtraMSAEmbedder = pyimport("openfold.model.embedders").ExtraMSAEmbedder

rng = Random.Xoshiro(42)

@testset "ExtraMSAEmbedder" begin
    @testset "Python parity" begin
        chn_in, chn_out = 25, 64
        N, S, B = 8, 5, 2

        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                jl_layer = ExtraMSAEmbedder(chn_in, chn_out)
                ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyExtraMSAEmbedder(chn_in, chn_out)

                sync_dense!(py_layer.linear, ps.linear)

                x = randn(rng, T, chn_in, N, S, B)

                jl_out, _ = jl_layer(x, ps, st)

                x_py = to_py(permutedims(x, reverse(1:4)); swap_batch_dim=false)
                py_out = py_layer(x_py)

                @testset "Python parity" begin
                    @test jl_out ≈ permutedims(to_jl(py_out; swap_batch_dim=false), reverse(1:4))
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred jl_layer(x, ps, st)
                end
            end
        end
    end
end
