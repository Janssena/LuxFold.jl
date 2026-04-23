include("../python/alphafold3.jl");

rng = Random.Xoshiro(42)

@testset "AlphaFold3" begin
    N, S, B = 16, 8, 2
    chn_in = 32
    num_heads = 6 # H
    head_dim = 8 # c_hidden in af3

    dims_cfg = (
        ("3D inputs", (N, B)),
        ("4D inputs", (N, S, B)),
    )

    for (dim_name, _dims) in dims_cfg
        @testset "$dim_name" begin
            mask_cfg = (
                ("No mask", nothing),
                ("Random mask", rand(rng, Bool, _dims...)),
                ("All-ones mask", trues(_dims...)),
            )
            @testset "Attention" begin
                for (name, mask) in mask_cfg
                    @testset "$name" begin
                        for T in [Float16, Float32, Float64]
                            inf_val = T == Float16 ? Float16(1e4) : T(1e9)
                            x = randn(rng, T, chn_in, _dims...)
                            bias = randn(rng, T, num_heads, N, N, B)
                            
                            # use_bias = (true, (layer_norm_a = false, layer_norm_s = false, ))
                            jl_layer = Attention(chn_in, head_dim, num_heads; fuse_qkv=false)
                            ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                            (y_jl, _), _ = jl_layer(x, bias, mask, ps, st)

                            py_layer = py"AF3Attention"(chn_in, chn_in, chn_in, head_dim, num_heads)
                            
                            sync_af3_attention!(py_layer, ps)
                    
                            x_py = if length(_dims) == 2
                                 to_py(x; swap_batch_dim=true)
                            else
                                to_py(permutedims(x, (4, 3, 2, 1)); swap_batch_dim=false)
                            end
                            mask_py = to_py(isnothing(mask) ? trues(_dims...) : mask; swap_batch_dim=true).to(py_dtype(T))
                            bias_py = to_py(bias; swap_batch_dim=true)
                            
                            py"""
                            # mask in expected: [B, N] or [B, S, N]
                            # bias in expected: [B, N, N, H]
                            triangle_bias = af3_permute_final_dims($bias_py, (2, 0, 1)) # [B, H, N, N]
                            mask = $inf_val * ($mask_py - 1)
                            if len(mask.shape) == 3:
                                mask_bias = mask[..., :, None, None, :]
                                triangle_bias = triangle_bias.unsqueeze(1)
                            else:
                                mask_bias = mask[:, None, None, :]
                            biases = [mask_bias, triangle_bias]
                            """
                            
                            py_layer.eval()
                            y_py = py_layer(x_py, x_py; biases=py"biases")

                            @testset "Python parity ($T)" begin
                                if length(_dims) == 2
                                    @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                                else
                                    @test y_jl ≈ permutedims(to_jl(y_py; swap_batch_dim=false), (4, 3, 2, 1))
                                end
                            end

                            @testset "Type-stability ($T)" begin
                                @test_nowarn @inferred jl_layer(x, bias, mask, ps, st)
                            end
                        end
                    end
                end
            end
        end
    end
end