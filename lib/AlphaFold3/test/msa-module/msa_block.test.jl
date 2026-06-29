const PyMSAModuleBlock = pyimport("openfold3.core.model.latent.msa_module").MSAModuleBlock

include("sync_helpers.jl")

@testset "MSAModuleBlock" begin
    rng = Random.Xoshiro(42)
    c_m, c_z, N_seq, N_token, B = 32, 32, 4, 8, 1
    cha, chopm, chmul, chpa = 8, 8, 16, 8
    nhm, nhp, tn = 2, 2, 2

    smask = rand(rng, Bool, N_token, N_seq, B); smask[1, 1, :] .= true
    pmask = rand(rng, Bool, N_token, N_token, B); pmask[1, 1, :] .= true
    mask_cfg = (("No mask", nothing, nothing), ("Random mask", smask, pmask))

    for T in [Float64, Float32], (mask_name, msa_mask, pair_mask) in mask_cfg
        @testset "$T, $mask_name" begin
            jl = MSAModuleBlock(; chn_m=c_m, chn_z=c_z, chn_hidden_msa_att=cha, chn_hidden_opm=chopm, chn_hidden_mul=chmul, chn_hidden_pair_att=chpa, no_heads_msa=nhm, no_heads_pair=nhp, transition_n=tn)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            py = PyMSAModuleBlock(c_m=c_m, c_z=c_z, c_hidden_msa_att=cha, c_hidden_opm=chopm,
                                  c_hidden_mul=chmul, c_hidden_pair_att=chpa, no_heads_msa=nhm,
                                  no_heads_pair=nhp, transition_type="swiglu", transition_n=tn,
                                  msa_dropout=0.0, pair_dropout=0.0, opm_first=false,
                                  fuse_projection_weights=false, inf=1e9, eps=1e-3)
            py.to(py_dtype(T))
            sync_msa_module_block!(py, ps)

            m = randn(rng, T, c_m, N_token, N_seq, B)
            z = randn(rng, T, c_z, N_token, N_token, B)

            out_jl, _ = jl(m, z, msa_mask, pair_mask, ps, st)

            msa_mask_py  = isnothing(msa_mask)  ? to_py_msamask(ones(Bool, N_token, N_seq, B), T) : to_py_msamask(msa_mask, T)
            pair_mask_py = isnothing(pair_mask) ? to_py_pairmask(ones(Bool, N_token, N_token, B), T) : to_py_pairmask(pair_mask, T)
            m_py, z_py = py(to_py_msa(m), to_py(z; swap_batch_dim=true), msa_mask_py, pair_mask_py)

            atol = T == Float64 ? 1e-5 : 1f-3
            @test isapprox(out_jl.m, to_jl_msa(m_py); atol=atol, rtol=atol)
            @test isapprox(out_jl.z, to_jl(z_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = MSAModuleBlock(; chn_m=c_m, chn_z=c_z, chn_hidden_msa_att=cha, chn_hidden_opm=chopm, chn_hidden_mul=chmul, chn_hidden_pair_att=chpa, no_heads_msa=nhm, no_heads_pair=nhp, transition_n=tn)
        ps, st = Lux.setup(rng, jl) |> convert_types(Float32)
        m = randn(rng, Float32, c_m, N_token, N_seq, B)
        z = randn(rng, Float32, c_z, N_token, N_token, B)
        @test_nowarn @inferred jl(m, z, nothing, nothing, ps, st)
        @test_nowarn @inferred jl(m, z, smask, pmask, ps, st)
    end
end
