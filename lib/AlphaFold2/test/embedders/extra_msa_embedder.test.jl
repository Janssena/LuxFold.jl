const PyExtraMSAEmbedder = pyimport("openfold.model.embedders").ExtraMSAEmbedder
const C_IN, C_OUT = 25, 64

rng = Random.Xoshiro(42)

@testset "ExtraMSAEmbedder" begin
    @testset "Python parity" begin
        N, S, B = 8, 5, 2

        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                jl_layer = ExtraMSAEmbedder(C_IN, C_OUT)
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyExtraMSAEmbedder(C_IN, C_OUT)

                sync_dense!(py_layer.linear, jl_ps.linear)

                x_jl = randn(rng, T, C_IN, N, S, B)

                jl_out, _ = jl_layer(x_jl, jl_ps, jl_st)

                x_py = to_py(permutedims(x_jl, (4, 3, 2, 1)); swap_batch_dim=false)
                py_out = py_layer(x_py)

                @test jl_out ≈ permutedims(to_jl(py_out; swap_batch_dim=true), (1, 3, 2, 4))

                @test_nowarn @inferred jl_layer(x_jl, jl_ps, jl_st)
            end
        end
    end
end
