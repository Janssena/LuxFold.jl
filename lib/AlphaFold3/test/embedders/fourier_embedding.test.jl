const PyInputEmb = pyimport("openfold3.core.model.feature_embedders.input_embedders")

@testset "FourierEmbedding" begin
    c_fourier, B = 256, 4
    rng = Random.Xoshiro(42)

    for T in [Float64, Float32]
        @testset "Python parity ($T)" begin
            l = FourierEmbedding(c_fourier; seed=42)
            ps, st = Lux.setup(rng, l)

            # Python instance + sync its w/b buffers into Julia state (AF2 style).
            py = PyInputEmb.FourierEmbedding(c=c_fourier, seed=42)
            st = merge(st, (
                w = T.(to_jl(py.w)),   # [c_fourier] buffer
                b = T.(to_jl(py.b)),
            ))

            x = randn(rng, T, 1, B)                          # noise level σ, [1, B]
            y_jl, _ = l(x, ps, st)                           # [c_fourier, B]

            # Python forward: x [B, 1] -> [B, c_fourier]
            x_py = py(to_py(x; swap_batch_dim=true))
            @test size(y_jl) == (c_fourier, B)
            @test y_jl ≈ to_jl(x_py; swap_batch_dim=true)
        end
    end

    @testset "type-stability" begin
        l = FourierEmbedding(c_fourier; seed=42)
        ps, st = Lux.setup(rng, l)
        x = randn(rng, Float32, 1, B)
        @test_nowarn @inferred l(x, ps, st)
    end
end
