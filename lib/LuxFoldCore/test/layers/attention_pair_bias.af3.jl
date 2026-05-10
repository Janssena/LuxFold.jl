include("../python/alphafold3.jl")

rng = Random.Xoshiro(42)

@testset "AlphaFold3" begin
    N, B = 12, 2
    chn_in = 32
    chn_z = 24
    head_dim = 8
    num_heads = 4

    mask_cfg = (
        ("No mask", nothing),
        ("Random mask", rand(rng, Bool, N, B))
    )
    cond_cfg = (
        ("No cond", nothing, nothing),
        ("Random cond", 16, rand(rng, Float32, 16, N, B))
    )

    for (mask_name, mask) in mask_cfg, (cond_name, chn_cond, cond) in cond_cfg
        @testset "$mask_name, $cond_name" begin
            @testset "AttentionPairBias" begin
                for T in [Float16, Float32, Float64]
                    x = randn(rng, T, chn_in, N, B)
                    z = randn(rng, T, chn_z, N, N, B)
                    cond = isnothing(cond) ? nothing : T.(cond)

                    if isnothing(cond)
                        affine = true
                        use_bias = (true, (mha=false,))
                    else
                        affine = (true, (layer_norm_in=(layer_norm_a=false, layer_norm_s=true,),))
                        use_bias = (true, (layer_norm_in=(false, (shift=true, gate=true,)), mha=false,))
                    end

                    jl_layer = AttentionPairBias(chn_in, chn_z, head_dim, num_heads; chn_cond, affine, use_bias, fuse_qkv=false)
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = py"AF3AttentionPairBias"(chn_in, chn_in, chn_in, chn_cond, chn_z, head_dim, num_heads, !isnothing(cond))

                    sync_af3_attention_pair_bias!(py_layer, ps)

                    y_jl, _ = jl_layer(x, z, cond, mask, ps, st)

                    x_py = to_py(x; swap_batch_dim=true)
                    z_py = to_py(z; swap_batch_dim=true)
                    cond_py = isnothing(cond) ? nothing : to_py(cond; swap_batch_dim=true)
                    mask_py = isnothing(mask) ? nothing : to_py(mask; swap_batch_dim=true).to(py_dtype(T))

                    y_py = py_layer(x_py, z_py, cond_py, mask_py)

                    @testset "Python parity ($T)" begin
                        @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                    end

                    @testset "Type-stability ($T)" begin
                        @test_nowarn @inferred jl_layer(x, z, cond, mask, ps, st)
                    end
                end
            end

            if isnothing(cond)
                @testset "MSARowAttentionPairBias" begin
                    S = 6
                    for T in [Float16, Float32, Float64]
                        x = randn(rng, T, chn_in, N, S, B)
                        z = randn(rng, T, chn_z, N, N, B)
                        mask = isnothing(mask) ? nothing : rand(rng, Bool, N, S, B)

                        jl_layer = MSARowAttentionPairBias(chn_in, chn_z, head_dim, num_heads; fuse_qkv=false)
                        ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                        py_layer = py"AF3MSARowAttentionWithPairBias"(chn_in, chn_z, head_dim, num_heads)
                        sync_af3_msa_row_attention_with_pair_bias!(py_layer, ps)

                        y_jl, _ = jl_layer(x, z, mask, ps, st)

                        x_py = to_py(permutedims(x, (4, 3, 2, 1)); swap_batch_dim=false)
                        z_py = to_py(z; swap_batch_dim=true)
                        mask_py = isnothing(mask) ? nothing : to_py(mask; swap_batch_dim=true).to(py_dtype(T))

                        y_py = py_layer(x_py, z_py, mask_py)

                        @testset "Python parity ($T)" begin
                            @test y_jl ≈ permutedims(to_jl(y_py; swap_batch_dim=false), (4, 3, 2, 1))
                        end

                        @testset "Type-stability ($T)" begin
                            @test_nowarn @inferred jl_layer(x, z, mask, ps, st)
                        end
                    end
                end
            end
        end
    end
end
