include("../python/boltz2.jl");

rng = Random.Xoshiro(42)

@testset "Boltz2" begin
    N_seq, N_res, B = 12, 16, 2
    chn_in = 32
    chn_hidden = 8
    chn_z = 16

    mask_cfg = (
        ("Random mask", rand(rng, Bool, N_res, N_seq, B)),
    )
    
    @testset "OuterProductMean" begin
        for (name, mask) in mask_cfg
            @testset "$name" begin
                for T in [Float16, Float32, Float64]
                    m = randn(rng, T, chn_in, N_res, N_seq, B)
                    
                    # Boltz2 style: bias=false, clamp=true, project_first=false
                    jl_layer = OuterProductMean(chn_in, chn_z, chn_hidden; use_bias=false, use_clamp=true, project_first=false)
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    y_jl, _ = jl_layer(m, mask, ps, st)

                    py_layer = py"Boltz2OuterProductMean"(chn_in, chn_hidden, chn_z)
                    
                    sync_boltz2_opm!(py_layer, ps)
            
                    m_py = to_py(permutedims(m, reverse(1:ndims(m))); swap_batch_dim=false)
                    mask_py = to_py(permutedims(mask, reverse(1:ndims(mask))); swap_batch_dim=false).to(py_dtype(T))
                    
                    py_layer.eval()
                    y_py = py_layer(m_py, mask_py)

                    @testset "Python parity ($T)" begin
                        @test y_jl ≈ to_jl(y_py; swap_batch_dim=true) rtol=T(1e-2) atol=T(1e-2)
                    end

                    @testset "Type-stability ($T)" begin
                        @test_nowarn @inferred jl_layer(m, mask, ps, st)
                    end
                end
            end
        end
    end
end
