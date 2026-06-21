# test_alphafold.jl — integration tests for AlphaFold model forward pass
#
# Uses reduced channel dims for speed (chn_msa=32, chn_z=16, chn_s=48).
# Does not test Python parity for the full model (sub-components have their own
# parity tests); validates shapes, finiteness, and recycling state semantics.

N_af, B_af, S_c_af, S_extra_af = 16, 1, 8, 16

rng_af = Random.Xoshiro(42)

# Build small AlphaFold model
const af_model = AlphaFold(;
    chn_msa=32, chn_z=16, chn_s=48,
    chn_target_feat=22, chn_msa_feat=49,
    chn_extra_msa=16, chn_extra_msa_feat=25,
    use_templates=false, use_extra_msa=true,
    evo_no_blocks=2, extra_msa_no_blocks=1,
    sm_no_blocks=2, sm_no_heads_ipa=4, sm_c_ipa=8,
    sm_no_qk_points=2, sm_no_v_points=4, sm_no_resnet_blocks=1, sm_c_resnet=32,
)

# Build synthetic feats NamedTuple
function _make_feats(N, B, S_c, S_extra; rng=Random.Xoshiro(42))
    aatype = rand(rng, 1:20, N, B)
    (;
        target_feat              = randn(rng, Float32, 22, N, B),
        residue_index            = reshape(1:N, N, 1) .* ones(Int, 1, B),
        msa_feat                 = randn(rng, Float32, 49, N, S_c, B),
        seq_mask                 = ones(Float32, N, B),
        msa_mask                 = ones(Float32, N, S_c, B),
        aatype                   = aatype,
        extra_msa                = rand(rng, 0:22, N, S_extra, B),
        extra_msa_deletion_value = rand(rng, Float32, N, S_extra, B),
        extra_msa_has_deletion   = rand(rng, Bool, N, S_extra, B),
        extra_msa_mask           = ones(Float32, N, S_extra, B),
        atom37_atom_exists       = permutedims(restype_atom37_mask, (2,1))[:, aatype],
    )
end

const feats_af = _make_feats(N_af, B_af, S_c_af, S_extra_af; rng=rng_af)

@testset "AlphaFold model" begin

    @testset "Lux.setup — initial state" begin
        ps_af, st_af = Lux.setup(rng_af, af_model)
        @test st_af.recycling_state.m_1_prev === nothing
        @test st_af.recycling_state.z_prev   === nothing
        @test st_af.recycling_state.x_prev   === nothing
        @test size(st_af.residue_constants.default_frames) == (21, 8, 4, 4)
        @test size(st_af.residue_constants.atom_mask)      == (21, 14)
    end

    @testset "iteration — single pass" begin
        ps_af, st_af = Lux.setup(rng_af, af_model)
        outputs, st_new = iteration(af_model, feats_af, ps_af, st_af)
        @test st_new.recycling_state.m_1_prev !== nothing
        @test size(st_new.recycling_state.m_1_prev) == (32, N_af, B_af)
        @test size(st_new.recycling_state.z_prev)   == (16, N_af, N_af, B_af)
        @test size(st_new.recycling_state.x_prev)   == (3, 37, N_af, B_af)
    end

    @testset "predict — output shapes" begin
        ps_af, st_af = Lux.setup(rng_af, af_model)
        outputs, st_new = predict(af_model, feats_af, ps_af, st_af; num_recycles=1)

        @test size(outputs.final_atom_positions)  == (3, 37, N_af, B_af)
        @test size(outputs.final_atom_mask)       == (37, N_af, B_af)
        @test size(outputs.final_affine_tensor)   == (7, N_af, B_af)
        @test size(outputs.s)                     == (48, N_af, B_af)
        @test size(outputs.z)                     == (16, N_af, N_af, B_af)
        @test outputs.num_recycles                == 1
    end

    @testset "predict — no NaN at valid positions" begin
        ps_af, st_af = Lux.setup(rng_af, af_model)
        outputs, _ = predict(af_model, feats_af, ps_af, st_af; num_recycles=1)
        mask3d = repeat(reshape(Bool.(outputs.final_atom_mask), 1, 37, N_af, B_af), 3, 1, 1, 1)
        @test all(isfinite, outputs.final_atom_positions[mask3d])
    end

    @testset "predict — warm start uses recycling state" begin
        ps_af, st_af = Lux.setup(rng_af, af_model)
        outputs_cold, st_warm = predict(af_model, feats_af, ps_af, st_af; num_recycles=1)
        outputs_warm, _       = predict(af_model, feats_af, ps_af, st_warm; num_recycles=1)
        # Warm-start outputs should differ from cold-start
        @test outputs_cold.final_atom_positions != outputs_warm.final_atom_positions
    end

    @testset "tolerance_reached" begin
        N_t, B_t = 8, 1
        rng_t = Random.Xoshiro(99)
        pos1 = randn(rng_t, Float32, 3, 37, N_t, B_t)
        pos2 = randn(rng_t, Float32, 3, 37, N_t, B_t)  # genuinely different structure
        mask = ones(Float32, N_t, B_t)
        # Default threshold < 0 → always false
        @test tolerance_reached(pos1, pos2, mask) == false
        # Identical positions → sqrt(1f-8) ≈ 1e-4 diff; threshold=0.01 > 1e-4 → true
        @test tolerance_reached(pos1, pos1, mask; threshold=0.01) == true
        # Large structural change vs. tight threshold → not converged → false
        @test tolerance_reached(pos1, pos2, mask; threshold=0.001) == false
    end

    @testset "Lux functor — equivalent to predict with num_recycles=3" begin
        ps_af, st_af = Lux.setup(rng_af, af_model)
        out_functor, _  = af_model(feats_af, ps_af, st_af)
        out_predict, _  = predict(af_model, feats_af, ps_af, st_af; num_recycles=3)
        @test out_functor.final_atom_positions ≈ out_predict.final_atom_positions
        @test out_functor.num_recycles == 3
    end

end
