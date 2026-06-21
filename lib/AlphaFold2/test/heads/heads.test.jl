const PyPerResidueLDDT = pyimport("openfold.model.heads").PerResidueLDDTCaPredictor
const PyDistogramHead = pyimport("openfold.model.heads").DistogramHead
const PyTMScoreHead = pyimport("openfold.model.heads").TMScoreHead
const PyMaskedMSAHead = pyimport("openfold.model.heads").MaskedMSAHead
const PyExpResolvedHead = pyimport("openfold.model.heads").ExperimentallyResolvedHead
const PyAuxiliaryHeads = pyimport("openfold.model.heads").AuxiliaryHeads

rng = Random.Xoshiro(42)

# Inline weight sync helpers
sync_plddt_head!(py_head, jl_ps) = begin
    sync_layernorm!(py_head.layer_norm, jl_ps.layer_norm)
    sync_dense!(py_head.linear_1, jl_ps.linear_1)
    sync_dense!(py_head.linear_2, jl_ps.linear_2)
    sync_dense!(py_head.linear_3, jl_ps.linear_3)
end

sync_single_dense_head!(py_head, jl_ps) = sync_dense!(py_head.linear, jl_ps.linear)

# ==============================================================================
# PerResidueLDDTCaPredictor
# ==============================================================================

@testset "PerResidueLDDTCaPredictor" begin
    chn_s, chn_hidden, no_bins, N, B = 16, 32, 10, 8, 2

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            jl_layer = PerResidueLDDTCaPredictor(chn_s, chn_hidden, no_bins)
            jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

            py_layer = PyPerResidueLDDT(no_bins, chn_s, chn_hidden)
            sync_plddt_head!(py_layer, jl_ps)

            s_jl = randn(rng, T, chn_s, N, B)
            s_py = to_py(s_jl; swap_batch_dim=true)

            y_jl, _ = jl_layer(s_jl, jl_ps, jl_st)
            y_py = py_layer(s_py)

            @testset "Python parity" begin
                @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
            end

            @testset "Type-stability" begin
                @test_nowarn @inferred jl_layer(s_jl, jl_ps, jl_st)
            end
        end
    end
end

# ==============================================================================
# DistogramHead
# ==============================================================================

@testset "DistogramHead" begin
    chn_z, no_bins, N, B = 16, 10, 8, 2

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            jl_layer = DistogramHead(chn_z, no_bins)
            jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

            py_layer = PyDistogramHead(chn_z, no_bins)
            sync_single_dense_head!(py_layer, jl_ps)

            z_jl = randn(rng, T, chn_z, N, N, B)
            z_py = to_py(z_jl; swap_batch_dim=true)

            y_jl, _ = jl_layer(z_jl, jl_ps, jl_st)
            y_py = py_layer(z_py)

            @testset "Python parity" begin
                @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
            end

            @testset "Symmetry" begin
                @test y_jl ≈ permutedims(y_jl, (1, 3, 2, 4))
            end

            @testset "Type-stability" begin
                @test_nowarn @inferred jl_layer(z_jl, jl_ps, jl_st)
            end
        end
    end
end

# ==============================================================================
# TMScoreHead
# ==============================================================================

@testset "TMScoreHead" begin
    chn_z, no_bins, N, B = 16, 10, 8, 2

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            jl_layer = TMScoreHead(chn_z, no_bins)
            jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

            py_layer = PyTMScoreHead(chn_z, no_bins)
            sync_single_dense_head!(py_layer, jl_ps)

            z_jl = randn(rng, T, chn_z, N, N, B)
            z_py = to_py(z_jl; swap_batch_dim=true)

            y_jl, _ = jl_layer(z_jl, jl_ps, jl_st)
            y_py = py_layer(z_py)

            @testset "Python parity" begin
                @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
            end

            @testset "Type-stability" begin
                @test_nowarn @inferred jl_layer(z_jl, jl_ps, jl_st)
            end
        end
    end
end

# ==============================================================================
# MaskedMSAHead
# ==============================================================================

@testset "MaskedMSAHead" begin
    chn_m, chn_out, N, S, B = 16, 10, 8, 4, 2

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            jl_layer = MaskedMSAHead(chn_m, chn_out)
            jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

            py_layer = PyMaskedMSAHead(chn_m, chn_out)
            sync_single_dense_head!(py_layer, jl_ps)

            m_jl = randn(rng, T, chn_m, N, S, B)
            m_py = to_py(m_jl; swap_batch_dim=true)

            y_jl, _ = jl_layer(m_jl, jl_ps, jl_st)
            y_py = py_layer(m_py)

            @testset "Python parity" begin
                @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
            end

            @testset "Type-stability" begin
                @test_nowarn @inferred jl_layer(m_jl, jl_ps, jl_st)
            end
        end
    end
end

# ==============================================================================
# ExperimentallyResolvedHead
# ==============================================================================

@testset "ExperimentallyResolvedHead" begin
    chn_s, chn_out, N, B = 16, 10, 8, 2

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            jl_layer = ExperimentallyResolvedHead(chn_s, chn_out)
            jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

            py_layer = PyExpResolvedHead(chn_s, chn_out)
            sync_single_dense_head!(py_layer, jl_ps)

            s_jl = randn(rng, T, chn_s, N, B)
            s_py = to_py(s_jl; swap_batch_dim=true)

            y_jl, _ = jl_layer(s_jl, jl_ps, jl_st)
            y_py = py_layer(s_py)

            @testset "Python parity" begin
                @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
            end

            @testset "Type-stability" begin
                @test_nowarn @inferred jl_layer(s_jl, jl_ps, jl_st)
            end
        end
    end
end

# ==============================================================================
# AuxiliaryHeads (Container)
# ==============================================================================

@testset "AuxiliaryHeads" begin
    chn_s, chn_z, chn_m = 16, 16, 16
    chn_hidden = 32
    no_bins_lddt, no_bins_dist, no_bins_tm = 10, 10, 10
    chn_out_msa, chn_out_exp = 10, 10
    N, S, B = 8, 4, 2

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            jl_layer = AuxiliaryHeads(
                chn_s, chn_z, chn_m;
                no_bins_lddt, no_bins_dist, no_bins_tm,
                chn_hidden_lddt=chn_hidden, chn_out_msa, chn_out_exp,
            )
            jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

            ConfigDict = pyimport("ml_collections").ConfigDict
            py_lddt_cfg = ConfigDict(Dict("no_bins" => no_bins_lddt, "c_in" => chn_s, "c_hidden" => chn_hidden))
            py_dist_cfg = ConfigDict(Dict("c_z" => chn_z, "no_bins" => no_bins_dist))
            py_tm_cfg = ConfigDict(Dict("c_z" => chn_z, "no_bins" => no_bins_tm, "enabled" => true))
            py_msa_cfg = ConfigDict(Dict("c_m" => chn_m, "c_out" => chn_out_msa))
            py_exp_cfg = ConfigDict(Dict("c_s" => chn_s, "c_out" => chn_out_exp))
            py_config = ConfigDict(Dict(
                "lddt" => py_lddt_cfg, "distogram" => py_dist_cfg,
                "tm" => py_tm_cfg, "masked_msa" => py_msa_cfg,
                "experimentally_resolved" => py_exp_cfg,
            ))
            py_layer = PyAuxiliaryHeads(py_config)

            sync_plddt_head!(py_layer.plddt, jl_ps.plddt)
            sync_single_dense_head!(py_layer.distogram, jl_ps.distogram)
            sync_single_dense_head!(py_layer.tm, jl_ps.tm)
            sync_single_dense_head!(py_layer.masked_msa, jl_ps.masked_msa)
            sync_single_dense_head!(py_layer.experimentally_resolved, jl_ps.experimentally_resolved)

            s_jl = randn(rng, T, chn_s, N, B)
            z_jl = randn(rng, T, chn_z, N, N, B)
            m_jl = randn(rng, T, chn_m, N, S, B)

            s_py = to_py(s_jl; swap_batch_dim=true)
            z_py = to_py(z_jl; swap_batch_dim=true)
            m_py = to_py(m_jl; swap_batch_dim=true)

            outputs_jl = (s=s_jl, z=z_jl, m=m_jl)
            y_jl, _ = jl_layer(outputs_jl, jl_ps, jl_st)

            py_outputs = PyDict()
            sm = PyDict()
            sm["single"] = s_py
            py_outputs["sm"] = sm
            py_outputs["pair"] = z_py
            py_outputs["msa"] = m_py
            py_outputs["single"] = s_py
            y_py = py_layer(py_outputs)

            @testset "Python parity" begin
                @test y_jl.lddt_logits ≈ to_jl(y_py["lddt_logits"]; swap_batch_dim=true)
                plddt_tol = T == Float16 ? 0.5 : 0.001
                @test y_jl.plddt ≈ to_jl(y_py["plddt"]; swap_batch_dim=true) atol=plddt_tol
                @test y_jl.distogram_logits ≈ to_jl(y_py["distogram_logits"]; swap_batch_dim=true)
                @test y_jl.tm_logits ≈ to_jl(y_py["tm_logits"]; swap_batch_dim=true)
                @test y_jl.masked_msa_logits ≈ to_jl(y_py["masked_msa_logits"]; swap_batch_dim=true)
                @test y_jl.experimentally_resolved_logits ≈ to_jl(y_py["experimentally_resolved_logits"]; swap_batch_dim=true)
            end

            @testset "Type-stability" begin
                @test_nowarn @inferred jl_layer(outputs_jl, jl_ps, jl_st)
            end
        end
    end
end
