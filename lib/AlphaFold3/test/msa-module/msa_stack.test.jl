const PyMSAModuleStack = pyimport("openfold3.core.model.latent.msa_module").MSAModuleStack

include("sync_helpers.jl")

@testset "MSAModuleStack" begin
    rng = Random.Xoshiro(42)
    c_m, c_z, N_seq, N_token, B = 32, 32, 4, 8, 1
    cha, chopm, chmul, chpa = 8, 8, 16, 8
    nhm, nhp, tn, no_blocks = 2, 2, 2, 2

    for T in [Float64, Float32]
        @testset "Python parity ($T, $no_blocks blocks)" begin
            jl = MSAModuleStack(; chn_m=c_m, chn_z=c_z, chn_hidden_msa_att=cha, chn_hidden_opm=chopm, chn_hidden_mul=chmul, chn_hidden_pair_att=chpa, no_heads_msa=nhm, no_heads_pair=nhp, transition_n=tn, no_blocks=no_blocks)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            py = PyMSAModuleStack(c_m=c_m, c_z=c_z, c_hidden_msa_att=cha, c_hidden_opm=chopm,
                                  c_hidden_mul=chmul, c_hidden_pair_att=chpa, no_heads_msa=nhm,
                                  no_heads_pair=nhp, no_blocks=no_blocks, transition_type="swiglu",
                                  transition_n=tn, msa_dropout=0.0, pair_dropout=0.0,
                                  opm_first=false, fuse_projection_weights=false,
                                  blocks_per_ckpt=nothing, inf=1e9, eps=1e-3)
            py.to(py_dtype(T))
            sync_msa_module_stack!(py, ps)

            m = randn(rng, T, c_m, N_token, N_seq, B)
            z = randn(rng, T, c_z, N_token, N_token, B)

            z_jl, _ = jl(m, z, nothing, nothing, ps, st)

            msa_mask_py  = to_py_msamask(ones(Bool, N_token, N_seq, B), T)
            pair_mask_py = to_py_pairmask(ones(Bool, N_token, N_token, B), T)
            z_py = py(to_py_msa(m), to_py(z; swap_batch_dim=true), msa_mask_py, pair_mask_py)

            atol = T == Float64 ? 1e-5 : 5f-3
            @test isapprox(z_jl, to_jl(z_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = MSAModuleStack(; chn_m=c_m, chn_z=c_z, chn_hidden_msa_att=cha, chn_hidden_opm=chopm, chn_hidden_mul=chmul, chn_hidden_pair_att=chpa, no_heads_msa=nhm, no_heads_pair=nhp, transition_n=tn, no_blocks=no_blocks)
        ps, st = Lux.setup(rng, jl) |> convert_types(Float32)
        m = randn(rng, Float32, c_m, N_token, N_seq, B)
        z = randn(rng, Float32, c_z, N_token, N_token, B)
        @test_nowarn @inferred jl(m, z, nothing, nothing, ps, st)
    end
end
