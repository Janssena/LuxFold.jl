const PyMSAModuleEmbedder = pyimport("openfold3.core.model.feature_embedders.input_embedders").MSAModuleEmbedder

@testset "MSAModuleEmbedder" begin
    rng = Random.Xoshiro(42)
    N_token, N_msa, B = 6, 5, 2
    c_msa_onehot = 32
    c_m_feats = c_msa_onehot + 2     # 34
    c_m, c_s_input = 16, 12

    # msa_mask does NOT affect `m` in openfold (used only for the omitted subsampling), so
    # both configs confirm parity + mask-independence at this stage.
    mask_cfg = (
        ("No mask",     nothing),
        ("Random mask", rand(rng, Bool, N_token, N_msa, B)),
    )

    # openfold does not mask `m`; the Julia layer optionally does. Mask the Python output
    # identically to compare (`nothing` → raw parity).
    _msk_msa(m, ::Nothing) = m
    _msk_msa(m, mask::AbstractArray{Bool}) =
        ifelse.(reshape(mask, 1, size(mask)...), m, zero(eltype(m)))

    for T in [Float64, Float32], (mask_name, mask) in mask_cfg
        @testset "Python parity ($T, $mask_name)" begin
            jl = MSAModuleEmbedder(; chn_m=c_m, chn_s_input=c_s_input, chn_m_feats=c_m_feats)
            jl_ps, jl_st = Lux.setup(rng, jl)
            jl_ps = jl_ps |> convert_types(T)

            py = PyMSAModuleEmbedder(
                c_m_feats=c_m_feats, c_m=c_m, c_s_input=c_s_input,
                subsample_main_msa=false, subsample_all_msa=false,
                min_subsampled_all_msa=1, max_subsampled_all_msa=1,
            )
            sync_dense!(py.linear_m, jl_ps.linear_m)
            sync_dense!(py.linear_s_input, jl_ps.linear_s_input)

            # Julia MSA layout: [C, N_token, N_msa, B]. Masks are Bool in Julia.
            msa            = randn(rng, T, c_msa_onehot, N_token, N_msa, B)
            has_deletion   = T.(rand(rng, Bool, N_token, N_msa, B))   # binary feature (not a mask)
            deletion_value = randn(rng, T, N_token, N_msa, B)
            msa_mask       = isnothing(mask) ? trues(N_token, N_msa, B) : mask   # Bool
            s_input        = randn(rng, T, c_s_input, N_token, B)

            batch_jl = (; msa, has_deletion, deletion_value, msa_mask)

            # Python wants [B, N_msa, N_token, C]; to_py(swap) gives [B, N_token, N_msa, C]
            # so swap the (N_token, N_msa) axes. Masks cast Bool→float at the Python boundary.
            batch_py = Dict(
                "msa"            => to_py(msa; swap_batch_dim=true).permute(0, 2, 1, 3),
                "has_deletion"   => to_py(has_deletion; swap_batch_dim=true),    # [B, N_msa, N_token]
                "deletion_value" => to_py(deletion_value; swap_batch_dim=true),
                "msa_mask"       => to_py(T.(msa_mask); swap_batch_dim=true),
            )
            s_input_py = to_py(s_input; swap_batch_dim=true)                     # [B, N_token, c_s_input]

            emb_jl, _ = jl(batch_jl, s_input, mask, jl_ps, jl_st)
            m_jl = emb_jl.msa
            m_py, _ = py(batch=batch_py, s_input=s_input_py)

            # m_jl [c_m, N_token, N_msa, B]; to_jl(m_py;swap) is [c_m, N_msa, N_token, B]
            @test m_jl ≈ _msk_msa(permutedims(to_jl(m_py; swap_batch_dim=true), (1, 3, 2, 4)), mask)
        end
    end

    @testset "type-stability" begin
        jl = MSAModuleEmbedder(; chn_m=c_m, chn_s_input=c_s_input, chn_m_feats=c_m_feats)
        jl_ps, jl_st = Lux.setup(rng, jl)
        msa = randn(rng, Float32, c_msa_onehot, N_token, N_msa, B)
        msa_mask = rand(rng, Bool, N_token, N_msa, B)
        batch_jl = (; msa,
                    has_deletion=Float32.(rand(rng, Bool, N_token, N_msa, B)),
                    deletion_value=randn(rng, Float32, N_token, N_msa, B),
                    msa_mask)
        s_input = randn(rng, Float32, c_s_input, N_token, B)
        # both the default (no-mask) and masked dispatch paths
        @test_nowarn @inferred jl(batch_jl, s_input, jl_ps, jl_st)
        @test_nowarn @inferred jl(batch_jl, s_input, msa_mask, jl_ps, jl_st)
    end
end
