"""
End-to-end integration test: StructureModule pipeline

This test validates the StructureModule layer with dummy Evoformer-like outputs,
focusing on shape correctness and type stability. No Python parity required —
the sub-layers are already tested in isolation.
"""

using Test
import AlphaFold2: Lux
import Random
using AlphaFold2

@testset "End-to-End: StructureModule Pipeline" begin

    rng = Random.Xoshiro(42)

    # Model dimensions (small for fast test)
    N_res = 32                         # sequence length
    chn_s, chn_z = 16, 16                 # single and pair channel dims

    # StructureModule pipeline
    @testset "StructureModule forward pass" begin
        structure_module = StructureModule(
            chn_s, chn_z;
            chn_ipa=8, chn_resnet=16,
            no_heads_ipa=4, no_qk_points=2, no_v_points=4,
            no_blocks=2,  # 2 blocks for speed
            no_transition_layers=1,
            no_resnet_blocks=1,
            no_angles=7,
            trans_scale_factor=10f0,
        )
        sm_ps, sm_st = Lux.setup(rng, structure_module)

        # Dummy Evoformer outputs
        s = randn(Float32, chn_s, N_res, 1)       # [chn_s, N, B]
        z = randn(Float32, chn_z, N_res, N_res, 1)  # [chn_z, N, N, B]
        mask = trues(N_res, 1)                  # [N, B]

        # Forward through StructureModule
        outputs, _ = structure_module(s, z, mask, sm_ps, sm_st)

        # Validate output shapes and types
        @test haskey(outputs, :frames)
        @test haskey(outputs, :angles)
        @test haskey(outputs, :unnormalized_angles)
        @test haskey(outputs, :states)
        @test haskey(outputs, :single)

        @test size(outputs.frames) == (7, N_res, 1, 2)           # [quat+trans, N, B, blocks]
        @test size(outputs.angles) == (2, 7, N_res, 1, 2)        # [sin/cos, angles, N, B, blocks]
        @test size(outputs.unnormalized_angles) == (2, 7, N_res, 1, 2)
        @test size(outputs.states) == (chn_s, N_res, 1, 2)        # [chn_s, N, B, blocks]
        @test size(outputs.single) == (chn_s, N_res, 1)           # [chn_s, N, B]

        # All outputs should be Float32
        @test eltype(outputs.frames) == Float32
        @test eltype(outputs.angles) == Float32
        @test eltype(outputs.single) == Float32

        # Verify outputs are finite (no NaN/Inf at valid positions)
        @test all(isfinite.(outputs.frames))
        @test all(isfinite.(outputs.angles))
        @test all(isfinite.(outputs.single))
    end

    @testset "Type-stability of StructureModule" begin
        # Verify @inferred works through the full pipeline
        structure_module = StructureModule(
            chn_s, chn_z;
            chn_ipa=8, chn_resnet=16,
            no_heads_ipa=4, no_qk_points=2, no_v_points=4,
            no_blocks=1,
            trans_scale_factor=10f0,
        )
        sm_ps, sm_st = Lux.setup(rng, structure_module)

        s = randn(Float32, chn_s, N_res, 1)
        z = randn(Float32, chn_z, N_res, N_res, 1)
        mask = trues(N_res, 1)

        @test_nowarn @inferred structure_module(s, z, mask, sm_ps, sm_st)
    end

end
