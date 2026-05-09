include("../python/alphafold3.jl")

rng = Random.Xoshiro(42)

@testset "CrossedAttentionPairBias" begin
    N, B = 12, 2
    chn_in = 32
    chn_z = 24
    head_dim = 8
    num_heads = 4

    configs = [
        (name="Global", n_query=nothing, n_key=nothing),
        (name="Local", n_query=4, n_key=4)
    ]

    mask_cfg = (
        ("No mask", nothing),
        ("Random mask", rand(rng, Bool, N, B))
    )

    cond_cfg = (
        ("No cond", nothing, nothing),
        ("Random cond", 16, rand(rng, Float32, 16, N, B))
    )

    for config in configs, (mask_name, mask) in mask_cfg, (cond_name, chn_cond, cond) in cond_cfg
        @testset "$(config.name), $mask_name, $cond_name" begin
            for T in [Float32, Float64]
                x = randn(rng, T, chn_in, N, B)
                z = randn(rng, T, chn_z, N, N, B)
                cond_val = isnothing(cond) ? nothing : T.(cond)

                use_ada = !isnothing(cond)

                jl_layer = CrossedAttentionPairBias(
                    chn_in, chn_in, chn_in, isnothing(chn_cond) ? 0 : chn_cond, chn_z,
                    head_dim, num_heads;
                    use_ada_layer_norm=use_ada,
                    n_query=config.n_query, n_key=config.n_key
                )
                ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = py"AF3CrossAttentionPairBias"(
                    chn_in, chn_in, chn_in, isnothing(chn_cond) ? 0 : chn_cond,
                    chn_z, head_dim, num_heads, use_ada,
                    config.n_query, config.n_key
                ).to(py_dtype(T))

                # 3. Synchronization
                sync_af3_cross_attention_pair_bias!(py_layer, ps)

                # 4. Forward passes
                y_jl, _ = jl_layer((a=x, z=z, cond=cond_val, mask=mask), ps, st)

                x_py = to_py(x; swap_batch_dim=true).to(py_dtype(T))
                z_py = to_py(z; swap_batch_dim=true).to(py_dtype(T))
                cond_py = isnothing(cond_val) ? nothing : to_py(cond_val; swap_batch_dim=true).to(py_dtype(T))
                mask_py = isnothing(mask) ? nothing : to_py(mask; swap_batch_dim=true).to(py_dtype(T))

                y_py = py_layer(x_py, z_py, cond_py, mask_py)

                @testset "Python parity ($T)" begin
                    @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                end

                @testset "Type-stability ($T)" begin
                    @test_nowarn @inferred jl_layer((a=x, z=z, cond=cond_val, mask=mask), ps, st)
                end
            end
        end
    end
end
