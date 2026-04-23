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
        ("Random cond", 12, rand(rng, Float32, 12, N, B))
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
                        use_bias = (true, (mha = false, ))
                    else
                        affine = (true, (layer_norm_in = (layer_norm_a = false, layer_norm_s = true, ), ))
                        use_bias = (true, (layer_norm_in = (false, (shift = true, gate = true, )), mha = false, ))
                    end
                    
                    jl_layer = AttentionPairBias(chn_in, chn_z, head_dim, num_heads; chn_cond, affine, use_bias, fuse_qkv=false)
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = py"AF3AttentionPairBias"(chn_in, chn_in, chn_in, chn_cond, chn_z, head_dim, num_heads, !isnothing(cond))
                    
                    sync_af3_attention_pair_bias!(py_layer, ps)
                
                    (y_jl, _), _ = jl_layer(x, z, cond, mask, ps, st)
                    
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
                    for T in [Float16, Float32, Float64]
                        S = 1 # Baseline S=1 to verify logic
                        x = randn(rng, T, chn_in, N, S, B)
                        z = randn(rng, T, chn_z, N, N, B)
                        
                        x_jl = reshape(x, chn_in, N, S * B)
                        
                        jl_layer = MSARowAttentionPairBias(chn_in, chn_z, head_dim, num_heads; fuse_qkv=false)
                        ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)
                        
                        py_layer = py"AF3MSARowAttentionWithPairBias"(chn_in, chn_z, head_dim, num_heads)
                        sync_af3_msa_row_attention_with_pair_bias!(py_layer, ps)
                        
                        (y_jl, _), _ = jl_layer(x_jl, z, ps, st)
                        
                        x_py = torch.from_numpy(permutedims(collect(x), (4, 3, 2, 1))).to(py_dtype(T))
                        z_py = to_py(z; swap_batch_dim=true)
                        
                        y_py = py_layer(x_py, z_py)
                        y_py_jl = permutedims(y_py.detach().cpu().numpy(), (4, 3, 2, 1))
                        
                        @test reshape(y_jl, chn_in, N, S, B) ≈ y_py_jl
                    end
                end
            end
        end
    end

    # @testset "AttentionPairBias (LayerNorm)" begin
    #     for T in [Float16, Float32, Float64]
    #         x = randn(rng, T, chn_in, N, B)
    #         z = randn(rng, T, chn_z, N, N, B)
            
    #         jl_layer = AttentionPairBias(chn_in, chn_z, head_dim, num_heads)
    #         ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)
            
    #         py_layer = py"AF3AttentionPairBias"(chn_in, chn_in, chn_in, chn_in, chn_z, head_dim, num_heads, false)
    #         sync_af3_attention_pair_bias!(py_layer, ps)
            
    #         (y_jl, _), _ = jl_layer(x, z, ps, st)
            
    #         x_py = to_py(x; swap_batch_dim=true)
    #         z_py = to_py(z; swap_batch_dim=true)
            
    #         y_py = py_layer(x_py, z_py)
            
    #         @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
    #     end
    # end

    # @testset "AttentionPairBias (AdaLN)" begin
    #     for T in [Float16, Float32, Float64]
    #         chn_cond = 16
    #         x = randn(rng, T, chn_in, N, B)
    #         z = randn(rng, T, chn_z, N, N, B)
    #         cond = randn(rng, T, chn_cond, N, B)
            
    #         jl_layer = AttentionPairBias(chn_in, chn_z, head_dim, num_heads; chn_cond=chn_cond)
    #         ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)
            
    #         py_layer = py"AF3AttentionPairBias"(chn_in, chn_in, chn_in, chn_cond, chn_z, head_dim, num_heads, true)
    #         sync_af3_attention_pair_bias!(py_layer, ps)
            
    #         (y_jl, _), _ = jl_layer(x, z, ps, st; cond=cond)
            
    #         x_py = to_py(x; swap_batch_dim=true)
    #         z_py = to_py(z; swap_batch_dim=true)
    #         cond_py = to_py(cond; swap_batch_dim=true)
            
    #         y_py = py_layer(x_py, z_py, cond_py)
            
    #         @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
    #     end
    # end

    # @testset "MSARowAttentionPairBias" begin
    #     for T in [Float16, Float32, Float64]
    #         S = 1 # Baseline S=1 to verify logic
    #         x = randn(rng, T, chn_in, N, S, B)
    #         z = randn(rng, T, chn_z, N, N, B)
            
    #         x_jl = reshape(x, chn_in, N, S * B)
            
    #         jl_layer = MSARowAttentionPairBias(chn_in, chn_z, head_dim, num_heads; fuse_qkv=false)
    #         ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)
            
    #         py_layer = py"AF3MSARowAttentionWithPairBias"(chn_in, chn_z, head_dim, num_heads)
    #         sync_af3_msa_row_attention_with_pair_bias!(py_layer, ps)
            
    #         (y_jl, _), _ = jl_layer(x_jl, z, ps, st)
            
    #         x_py = torch.from_numpy(permutedims(collect(x), (4, 3, 2, 1))).to(py_dtype(T))
    #         z_py = to_py(z; swap_batch_dim=true)
            
    #         y_py = py_layer(x_py, z_py)
    #         y_py_jl = permutedims(y_py.detach().cpu().numpy(), (4, 3, 2, 1))
            
    #         @test reshape(y_jl, chn_in, N, S, B) ≈ y_py_jl
    #     end
    # end

    # @testset "CrossAttentionPairBias" begin
    #     for T in [Float16, Float32, Float64]
    #         L = 16
    #         N = 32
    #         B = 1
    #         chn_in = 32
    #         chn_z = 24
    #         head_dim = 8
    #         num_heads = 4
            
    #         x = randn(rng, T, chn_in, N, B)
    #         z = randn(rng, T, chn_z, N, N, B)
            
    #         # 1. No AdaLN
    #         jl_layer = CrossAttentionPairBias(chn_in, chn_z, head_dim, num_heads; 
    #             n_query=L, n_key=L, fuse_qkv=false)
    #         ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)
            
    #         py_layer = py"AF3CrossAttentionPairBias"(chn_in, chn_in, chn_in, chn_in, chn_z, head_dim, num_heads, false, L, L)
    #         sync_af3_cross_attention_pair_bias!(py_layer, ps)
            
    #         (y_jl, _), _ = jl_layer(x, z, ps, st)
            
    #         x_py = to_py(x; swap_batch_dim=true)
    #         z_py = to_py(z; swap_batch_dim=true)
            
    #         y_py = py_layer(x_py, z_py)
            
    #         # Float16 might need higher tolerance
    #         tol = T == Float16 ? 1e-2 : 1e-5
    #         @test y_jl ≈ to_jl(y_py; swap_batch_dim=true) atol=tol
            
    #         # 2. With AdaLN
    #         chn_cond = 16
    #         cond = randn(rng, T, chn_cond, N, B)
            
    #         jl_layer_ada = CrossAttentionPairBias(chn_in, chn_z, head_dim, num_heads; 
    #             chn_cond=chn_cond, n_query=L, n_key=L, fuse_qkv=false)
    #         ps_ada, st_ada = Lux.setup(rng, jl_layer_ada) |> convert_types(T)
            
    #         py_layer_ada = py"AF3CrossAttentionPairBias"(chn_in, chn_in, chn_in, chn_cond, chn_z, head_dim, num_heads, true, L, L)
    #         sync_af3_cross_attention_pair_bias!(py_layer_ada, ps_ada)
            
    #         (y_jl_ada, _), _ = jl_layer_ada(x, z, ps_ada, st_ada; cond=cond)
            
    #         cond_py = to_py(cond; swap_batch_dim=true)
    #         y_py_ada = py_layer_ada(x_py, z_py, cond_py)
            
    #         @test y_jl_ada ≈ to_jl(y_py_ada; swap_batch_dim=true) atol=tol
    #     end
    # end
end
