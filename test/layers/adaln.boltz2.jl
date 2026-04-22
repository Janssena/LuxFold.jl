include("../python/boltz2.jl");

rng = Random.Xoshiro(42)

@testset "Boltz2" begin
    N, B = 12, 2
    chn_a = 32
    chn_s = 16
    dim_cfg = (
        ("3D tensors", (N, B)),
        ("4D tensors", (N, N, B)),
    )
    
    for (name, dims) in dim_cfg
        @testset "$name" begin
            for T in [Float16, Float32, Float64]
                x = randn(rng, T, chn_a, dims...)
                s = randn(rng, T, chn_s, dims...)
                rank = length(size(x))
                
                affine = (layer_norm_a = false, layer_norm_s = true)
                use_bias = (false, (gate = true,))
                
                jl_layer = AdaLN(chn_a, chn_s; rank, affine, use_bias, epsilon=T(1f-5))
                ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = py"Boltz2AdaLN"(chn_a, chn_s)

                sync_boltz2_adaln!(py_layer, ps)

                y_jl, _ = jl_layer(x, s, ps, st)
                x_py = to_py(x; swap_batch_dim=true)
                s_py = to_py(s; swap_batch_dim=true)
                
                y_py = py_layer(x_py, s_py)

                @testset "Python parity ($T)" begin
                    @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                end

                @testset "Type-stability ($T)" begin
                    @test_nowarn @inferred jl_layer(x, s, ps, st)
                end
            end
        end
    end
end