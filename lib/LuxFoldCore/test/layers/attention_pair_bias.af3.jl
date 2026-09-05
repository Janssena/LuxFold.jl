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

                    # EXACTLY how AlphaFold3 builds each form.
                    if isnothing(cond)
                        # `lib/AlphaFold3/src/pairformer/pairformer_block.jl:49`
                        affine = true
                        use_bias = (layer_norm_in=true, layer_norm_z=true, linear_z=false,
                                    mha=(false, (q=true,)), linear_out=false)
                    else
                        # `lib/AlphaFold3/src/diffusion/diffusion_transformer_block.jl:43`
                        affine = (layer_norm_in=(layer_norm_a=false, layer_norm_s=true),)
                        use_bias = (false, (layer_norm_in=(false, (gate=true, shift=false)),
                                            mha=(false, (q=true,)), linear_out=true))
                    end

                    # `use_layernorm_z=false` in the conditioned form: openfold-3's
                    # `DiffusionAttentionPairBias` has no `layer_norm_z` at all — the enclosing
                    # DiffusionTransformer normalises `z` once for the whole stack. Same as
                    # `lib/AlphaFold3/src/diffusion/diffusion_transformer_block.jl:47`.
                    jl_layer = AttentionPairBias(chn_in, chn_z, head_dim, num_heads;
                        chn_cond, affine, use_bias, fuse_qkv=false,
                        use_layernorm_z=isnothing(cond))
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    # TWO upstream classes, not one flag. openfold-3 splits the conditioned and
                    # unconditioned forms into `DiffusionAttentionPairBias` (AdaLN on `a`, taking
                    # `s`) and `AttentionPairBias` (plain LayerNorm). The hand-copied reference
                    # this test used to run against fused them behind `use_ada_layer_norm`, so the
                    # 8th positional here used to be that flag — upstream it is `gating`.
                    PyCls = isnothing(cond) ? PyAF3AttentionPairBias : PyAF3DiffusionAttentionPairBias
                    py_layer = PyCls(chn_in, chn_in, chn_in, chn_cond, chn_z, head_dim, num_heads)

                    sync_af3_attention_pair_bias!(py_layer, ps)

                    y_jl, _ = jl_layer(x, z, cond, mask, ps, st)

                    x_py = to_py(x; swap_batch_dim=true)
                    z_py = to_py(z; swap_batch_dim=true)
                    cond_py = isnothing(cond) ? nothing : to_py(cond; swap_batch_dim=true)
                    mask_py = isnothing(mask) ? nothing : to_py(mask; swap_batch_dim=true).to(py_dtype(T))

                    # The two upstream classes have DIFFERENT forward signatures:
                    # `AttentionPairBias(a, z, mask, ...)` takes no `s`, while
                    # `DiffusionAttentionPairBias(a, z, s, mask, ...)` does. Passing `s` to the
                    # unconditioned one pushed `mask` into `use_deepspeed_evo_attention`, whose
                    # `or` on a tensor raises "Boolean value of Tensor ... is ambiguous".
                    y_py = isnothing(cond) ? py_layer(x_py, z_py, mask_py) :
                                             py_layer(x_py, z_py, cond_py, mask_py)

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

                        # Matches the REAL openfold-3 `MSARowAttentionWithPairBias`: LayerNorms
                        # affine+biased, `linear_z` BIAS-FREE, and every mha projection
                        # (q/k/v/o/g) bias-free. The hand-copied reference gave `linear_z` a bias.
                        # (AlphaFold2 also builds this layer — `evoformer_block.jl:72` — but
                        # against openfold's own AF2 class, which its own suite covers.)
                        jl_layer = MSARowAttentionPairBias(chn_in, chn_z, head_dim, num_heads;
                            use_bias=(true, (linear_z=false, mha=false)), fuse_qkv=false)
                        ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                        py_layer = PyAF3MSARowAttentionWithPairBias(chn_in, chn_z, head_dim, num_heads)
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
