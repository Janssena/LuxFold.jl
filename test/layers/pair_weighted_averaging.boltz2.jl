include("../python/boltz2.jl");

rng = Random.Xoshiro(42)

@testset "Boltz2" begin
    N_seq, N_res, B = 8, 16, 2
    chn_msa = 32
    chn_pair = 64
    head_dim = 8
    num_heads = 4

    mask_cfg = (
        ("Random mask", rand(rng, Bool, N_res, N_res, B)),
    )
    
    @testset "PairWeightedAveraging" begin
        for (name, mask) in mask_cfg
            @testset "$name" begin
                for T in [Float16, Float32, Float64]
                    m = randn(rng, T, chn_msa, N_seq, N_res, B)
                    z = randn(rng, T, chn_pair, N_res, N_res, B)
                    
                    # Boltz2 style: LayerNorm has bias, but Linear layers don't
                    jl_layer = PairWeightedAveraging(chn_msa, chn_pair, head_dim, num_heads; 
                        use_bias=(true, (linear_v=false, linear_z=false, linear_g=false, linear_out=false)))
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    y_jl, _ = jl_layer(m, z, mask, ps, st)

                    py_layer = py"Boltz2PairWeightedAveraging"(chn_msa, chn_pair, head_dim, num_heads)
                    
                    sync_boltz2_pwa!(py_layer, ps)
            
                    m_py = to_py(m; swap_batch_dim=true)
                    z_py = to_py(z; swap_batch_dim=true)
                    mask_py = to_py(permutedims(mask, (2, 1, 3)); swap_batch_dim=true).to(py_dtype(T))
                    
                    py_layer.eval()
                    y_py = py_layer(m_py, z_py, mask=mask_py)

                    @testset "Python parity ($T)" begin
                        @test y_jl ≈ to_jl(y_py; swap_batch_dim=true) rtol=T(1e-2) atol=T(1e-2)
                    end

                    @testset "Type-stability ($T)" begin
                        @test_nowarn @inferred jl_layer(m, z, mask, ps, st)
                    end
                end
            end
        end
    end
end
