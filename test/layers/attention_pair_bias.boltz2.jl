# Load Boltz2 reference
include("../python/boltz2.jl")

rng = Random.Xoshiro(42)


@testset "Boltz2" begin
    N, B = 12, 2
    chn_in = 32
    chn_z = 24
    head_dim = 8
    num_heads = 4

    mask_cfg = (
        ("No mask", nothing),
        ("Random mask", rand(rng, Bool, N, B))
    )

    for (mask_name, mask) in mask_cfg
        @testset "$mask_name" begin
            @testset "AttentionPairBias" begin
                for T in [Float16, Float32, Float64]
                    x = randn(rng, T, chn_in, N, B)
                    z = randn(rng, T, chn_z, N, N, B)

                    affine = true
                    use_bias = (layer_norm_in=false, layer_norm_z=true, linear_z=false, mha=false, linear_out=false)
                    jl_layer = AttentionPairBias(chn_in, chn_z, head_dim, num_heads; affine, use_bias, use_layernorm_in=false, fuse_qkv=false)
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = py"Boltz2AttentionPairBias"(chn_in, chn_z, num_heads)
                    sync_boltz2_attention_pair_bias!(py_layer, ps)

                    y_jl, _ = jl_layer(x, z, mask, ps, st)

                    x_py = to_py(x; swap_batch_dim=true)
                    z_py = to_py(z; swap_batch_dim=true)
                    mask_py = isnothing(mask) ? to_py(trues(B, N); swap_batch_dim=false).to(py_dtype(T)) : to_py(mask; swap_batch_dim=true).to(py_dtype(T))

                    y_py = py_layer(x_py, z_py, mask_py, k_in=x_py)

                    @testset "Python parity ($T)" begin
                        if T == Float64 # There is an explicit float() cast in the boltz2 code
                            @test y_jl ≈ to_jl(y_py; swap_batch_dim=true) atol = 1e-5
                        else
                            @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                        end
                    end

                    @testset "Type-stability ($T)" begin
                        @test_nowarn @inferred jl_layer(x, z, mask, ps, st)
                    end
                end
            end
        end
    end
end








# @testset "Boltz2" begin
#     N, B = 12, 2
#     chn_in = 32
#     chn_z = 24
#     num_heads = 4
#     head_dim = chn_in ÷ num_heads

#     @testset "AttentionPairBias" begin
#         for T in [Float32, Float64] # Float16 might be tricky with Boltz2's autocast logic
#             for use_mask in [false, true]
#                 mask = use_mask ? rand(rng, Bool, N, B) : nothing

#                 x = randn(rng, T, chn_in, N, B)
#                 z = randn(rng, T, chn_z, N, N, B)

#                 # Boltz2 Bias Configuration:
#                 # proj_q: bias=True
#                 # proj_k, proj_v, proj_g, proj_o, proj_z: bias=False
#                 # layer_norm_z: affine=True (default)
#                 # layer_norm_in: Boltz2 doesn't have it inside, so we use NoOp or LayerNormNoBias with fixed weights

#                 # Boltz2 configuration:
#                 # - No input LN (managed outside)
#                 # - Pair LN (affine=True, bias=True) -> but in Julia we use LayerNorm(affine=True) which has bias too if not explicitly disabled.
#                 # - Pair Linear (bias=False)
#                 # - Attention (Q bias=False in our adjusted ref, K,V,G,O bias=False)

#                 mha = Attention(chn_in, head_dim, num_heads; use_gate=true, use_bias=false, fuse_qkv=false)
#                 jl_layer = AttentionPairBias(
#                     Lux.NoOpLayer(),
#                     Lux.LayerNorm((chn_z, 1, 1); dims=1, affine=true),
#                     Lux.Dense(chn_z => num_heads; use_bias=false),
#                     mha,
#                     Lux.NoOpLayer()
#                 )

#                 ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)


#                 py_layer = py"AttentionPairBias"(chn_in, chn_z, num_heads)
#                 sync_boltz2_attention_pair_bias!(py_layer, ps)

#                 (y_jl, _), _ = jl_layer(x, z, nothing, mask, ps, st)

#                 x_py = to_py(x; swap_batch_dim=true)
#                 z_py = to_py(z; swap_batch_dim=true)

#                 # Boltz2 expects mask [B, N, N]
#                 mask_py = if isnothing(mask)
#                     torch.ones((B, N, N)).to(py_dtype(T))
#                 else
#                     # Expand mask [N, B] to [B, N, N]
#                     # Pass token mask [N, B] -> [B, N]
#                     to_py(mask; swap_batch_dim=true)


#                 end

#                 # Boltz2 forward: forward(self, s, z, mask, k_in, multiplicity=1)
#                 y_py = py_layer(x_py, z_py, mask_py, x_py)

#                 @testset "Python parity ($T, mask=$use_mask)" begin
#                     @test y_jl ≈ to_jl(y_py; swap_batch_dim=true) atol = 1e-5
#                 end

#                 @testset "Type-stability ($T)" begin
#                     @test_nowarn @inferred jl_layer(x, z, nothing, mask, ps, st)
#                 end
#             end
#         end
#     end
# end
