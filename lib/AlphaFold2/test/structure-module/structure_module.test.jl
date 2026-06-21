const PyStructureModule =
    pyimport("openfold.model.structure_module").StructureModule

rng = Random.Xoshiro(42)

# ---- sync helpers (test-local) -----------------------------------------------

# Sync a Python StructureModule's sub-layers to a Julia StructureModuleFold ps.
# Python StructureModule has all sub-layers at top level; Julia nests them under `fold`.
function sync_structure_module_fold!(py_sm, jl_ps_fold)
    sync_ipa!(py_sm.ipa, jl_ps_fold.ipa; is_multimer=false)
    sync_layernorm!(py_sm.layer_norm_ipa, jl_ps_fold.layer_norm_ipa)
    # transition
    sync_structure_module_transition!(py_sm.transition, jl_ps_fold.transition)
    # backbone_update
    sync_dense!(py_sm.bb_update.linear, jl_ps_fold.backbone_update.linear)
    # angle_resnet
    sync_angle_resnet!(py_sm.angle_resnet, jl_ps_fold.angle_resnet)
end

function sync_structure_module!(py_sm, jl_ps)
    sync_layernorm!(py_sm.layer_norm_s, jl_ps.layer_norm_s)
    sync_layernorm!(py_sm.layer_norm_z, jl_ps.layer_norm_z)
    sync_dense!(py_sm.linear_in, jl_ps.linear_in)
    sync_structure_module_fold!(py_sm, jl_ps.fold)
end

# Zero masked residue positions before comparison.
# Julia's IPA uses typemin(T) masking → NaN at masked query positions; Python uses a
# finite inf bias → small finite values there.  Zeroing both sides at masked positions
# makes the comparison well-defined (we test valid positions only).
#
# NaN propagates strictly per-residue through LayerNorm/Transition/BackboneUpdate/
# AngleResnet, so the valid-position mask is exact.
function mask_outputs(x::AbstractArray{T}, mask_jl::AbstractArray, dims) where T
    valid = reshape(mask_jl, dims)
    return ifelse.(valid, x, zero(T))
end

# No mask → every position is valid; compare the full output.
mask_outputs(x, ::Nothing, dims) = x

# ---- StructureModuleFold parity -----------------------------------------------

@testset "StructureModuleFold (one block)" begin
    chn_s, chn_z = 16, 8
    chn_ipa, chn_resnet = 8, 16
    no_heads_ipa, no_qk_points, no_v_points = 4, 4, 8
    no_resnet_blocks, no_angles = 2, 7
    N, B = 5, 2

    mask_cfgs = [
        ("No mask",     nothing),
        ("Random mask", let m = rand(rng, Bool, N, B)
                            for b in 1:B; any(@view m[:, b]) || (m[rand(rng, 1:N), b] = true); end
                            m end),
    ]

    for T in (Float64, Float32)
        @testset "$T" begin
            jl_sm = StructureModule(
                chn_s, chn_z;
                chn_ipa, chn_resnet, no_heads_ipa, no_qk_points, no_v_points,
                no_blocks=1,
                no_transition_layers=1,
                no_resnet_blocks, no_angles,
                trans_scale_factor=10f0,
            )
            jl_ps, jl_st = Lux.setup(rng, jl_sm) |> convert_types(T)

            # Python StructureModule: positional args (chn_s, chn_z, chn_ipa, chn_resnet,
            # no_heads_ipa, no_qk_points, no_v_points, dropout_rate, no_blocks,
            # no_transition_layers, no_resnet_blocks, no_angles,
            # trans_scale_factor, epsilon, inf)
            py_sm = PyStructureModule(
                chn_s, chn_z, chn_ipa, chn_resnet, no_heads_ipa, no_qk_points, no_v_points,
                0.0, 1, 1, no_resnet_blocks, no_angles, 10.0, 1e-8, 1e5,
            )
            py_sm.eval()
            sync_structure_module!(py_sm, jl_ps)

            s_jl = randn(rng, T, chn_s, N, B)
            z_jl = randn(rng, T, chn_z, N, N, B)

            for (mask_name, mask_jl) in mask_cfgs
                @testset "$mask_name" begin
                    s_py    = to_py(s_jl; swap_batch_dim=true)
                    z_py    = to_py(z_jl; swap_batch_dim=true)
                    mask_py = if isnothing(mask_jl)
                        to_py(ones(T, N, B); swap_batch_dim=true)
                    else
                        to_py(mask_jl; swap_batch_dim=true).to(py_dtype(T))
                    end

                    outputs_jl, _ = jl_sm(s_jl, z_jl, mask_jl, jl_ps, jl_st)

                    # Python forward requires aatype (unused for frames/angles) — pass zeros
                    aatype_py = pyimport("torch").zeros(B, N, dtype=pyimport("torch").int64)
                    py_out = py_sm(
                        Dict("single" => s_py, "pair" => z_py),
                        aatype_py, mask_py,
                    )

                    # Compare `single` [chn_s, N, B]
                    single_jl = outputs_jl.single
                    single_py = permutedims(to_jl(py_out["single"]; swap_batch_dim=false), (3, 2, 1))
                    @test size(single_jl) == (chn_s, N, B)
                    @testset "Single" begin
                        if T == Float64
                            @test mask_outputs(single_jl, mask_jl, (1, N, B)) ≈ mask_outputs(single_py, mask_jl, (1, N, B)) atol=1e-4
                        else
                            @test mask_outputs(single_jl, mask_jl, (1, N, B)) ≈ mask_outputs(single_py, mask_jl, (1, N, B))
                        end
                    end

                    # Compare `frames` [7, N, B, 1]; Python: [1, B, N, 7]
                    frames_jl  = outputs_jl.frames
                    frames_py  = permutedims(to_jl(py_out["frames"]; swap_batch_dim=false), (4, 3, 2, 1))
                    @test size(frames_jl) == (7, N, B, 1)
                    @testset "Frames" begin
                        if T == Float64
                            @test mask_outputs(frames_jl, mask_jl, (1, N, B, 1)) ≈ mask_outputs(frames_py, mask_jl, (1, N, B, 1)) atol=1e-4
                        else
                            @test mask_outputs(frames_jl, mask_jl, (1, N, B, 1)) ≈ mask_outputs(frames_py, mask_jl, (1, N, B, 1))
                        end
                    end

                    # Compare `angles` [2, no_angles, N, B, 1]; Python: [1, B, N, na, 2]
                    angles_jl  = outputs_jl.angles
                    angles_py  = permutedims(to_jl(py_out["angles"]; swap_batch_dim=false), (5, 4, 3, 2, 1))
                    @test size(angles_jl) == (2, no_angles, N, B, 1)
                    @testset "Angles" begin
                        if T == Float64
                            @test mask_outputs(angles_jl, mask_jl, (1, 1, N, B, 1)) ≈ mask_outputs(angles_py, mask_jl, (1, 1, N, B, 1)) atol=1e-4
                        else
                            @test mask_outputs(angles_jl, mask_jl, (1, 1, N, B, 1)) ≈ mask_outputs(angles_py, mask_jl, (1, 1, N, B, 1))
                        end
                    end
                end
            end

            @testset "Type-stability" begin
                @test_nowarn @inferred jl_sm(s_jl, z_jl, trues(N, B), jl_ps, jl_st)
                @test_nowarn @inferred jl_sm(s_jl, z_jl, nothing, jl_ps, jl_st)
            end
        end
    end
end

# ---- StructureModule multi-block parity ---------------------------------------

@testset "StructureModule (multi-block)" begin
    chn_s, chn_z = 16, 8
    chn_ipa, chn_resnet = 8, 16
    no_heads_ipa, no_qk_points, no_v_points = 4, 4, 8
    no_blocks, no_resnet_blocks, no_angles = 3, 2, 7
    N, B = 5, 2

    mask_cfgs = [
        ("No mask",     nothing),
        ("Random mask", let m = rand(rng, Bool, N, B)
                            for b in 1:B; any(@view m[:, b]) || (m[rand(rng, 1:N), b] = true); end
                            m end),
    ]

    for T in (Float64, Float32)
        @testset "$T" begin
            jl_sm = StructureModule(
                chn_s, chn_z;
                chn_ipa, chn_resnet, no_heads_ipa, no_qk_points, no_v_points,
                no_blocks,
                no_transition_layers=1,
                no_resnet_blocks, no_angles,
                trans_scale_factor=10f0,
            )
            jl_ps, jl_st = Lux.setup(rng, jl_sm) |> convert_types(T)

            py_sm = PyStructureModule(
                chn_s, chn_z, chn_ipa, chn_resnet, no_heads_ipa, no_qk_points, no_v_points,
                0.0, no_blocks, 1, no_resnet_blocks, no_angles, 10.0, 1e-8, 1e5,
            )
            py_sm.eval()
            sync_structure_module!(py_sm, jl_ps)

            s_jl = randn(rng, T, chn_s, N, B)
            z_jl = randn(rng, T, chn_z, N, N, B)

            for (mask_name, mask_jl) in mask_cfgs
                @testset "$mask_name" begin
                    s_py = to_py(s_jl; swap_batch_dim=true)
                    z_py = to_py(z_jl; swap_batch_dim=true)
                    mask_py = if isnothing(mask_jl)
                        to_py(ones(T, N, B); swap_batch_dim=true)
                    else
                        to_py(mask_jl; swap_batch_dim=true).to(py_dtype(T))
                    end

                    outputs_jl, _ = jl_sm(s_jl, z_jl, mask_jl, jl_ps, jl_st)

                    aatype_py = pyimport("torch").zeros(B, N, dtype=pyimport("torch").int64)
                    py_out = py_sm(
                        Dict("single" => s_py, "pair" => z_py),
                        aatype_py, mask_py,
                    )

                    # `single` [chn_s, N, B]
                    single_jl = outputs_jl.single
                    single_py = permutedims(to_jl(py_out["single"]; swap_batch_dim=false), (3, 2, 1))
                    @test size(single_jl) == (chn_s, N, B)
                    @testset "Single" begin
                        if T == Float64
                            @test mask_outputs(single_jl, mask_jl, (1, N, B)) ≈ mask_outputs(single_py, mask_jl, (1, N, B)) atol=1e-4
                        else
                            @test mask_outputs(single_jl, mask_jl, (1, N, B)) ≈ mask_outputs(single_py, mask_jl, (1, N, B))
                        end
                    end

                    # `frames` [7, N, B, no_blocks]; Python: [nb, B, N, 7]
                    frames_jl = outputs_jl.frames
                    frames_py = permutedims(to_jl(py_out["frames"]; swap_batch_dim=false), (4, 3, 2, 1))
                    @test size(frames_jl) == (7, N, B, no_blocks)
                    @testset "Frames" begin
                        if T == Float64
                            @test mask_outputs(frames_jl, mask_jl, (1, N, B, 1)) ≈ mask_outputs(frames_py, mask_jl, (1, N, B, 1)) atol=1e-4
                        else
                            @test mask_outputs(frames_jl, mask_jl, (1, N, B, 1)) ≈ mask_outputs(frames_py, mask_jl, (1, N, B, 1))
                        end
                    end

                    # `angles` [2, no_angles, N, B, no_blocks]; Python: [nb, B, N, na, 2]
                    angles_jl = outputs_jl.angles
                    angles_py = permutedims(to_jl(py_out["angles"]; swap_batch_dim=false), (5, 4, 3, 2, 1))
                    @test size(angles_jl) == (2, no_angles, N, B, no_blocks)
                    @testset "Angles" begin
                        if T == Float64
                            @test mask_outputs(angles_jl, mask_jl, (1, 1, N, B, 1)) ≈ mask_outputs(angles_py, mask_jl, (1, 1, N, B, 1)) atol=1e-4
                        else
                            @test mask_outputs(angles_jl, mask_jl, (1, 1, N, B, 1)) ≈ mask_outputs(angles_py, mask_jl, (1, 1, N, B, 1))
                        end
                    end

                    # `states` [chn_s, N, B, no_blocks]; Python: [nb, B, N, chn_s]
                    states_jl = outputs_jl.states
                    states_py = permutedims(to_jl(py_out["states"]; swap_batch_dim=false), (4, 3, 2, 1))
                    @test size(states_jl) == (chn_s, N, B, no_blocks)
                    @testset "States" begin
                        if T == Float64
                            @test mask_outputs(states_jl, mask_jl, (1, N, B, 1)) ≈ mask_outputs(states_py, mask_jl, (1, N, B, 1)) atol=1e-4
                        else
                            @test mask_outputs(states_jl, mask_jl, (1, N, B, 1)) ≈ mask_outputs(states_py, mask_jl, (1, N, B, 1))
                        end
                    end
                end
            end

            @testset "Type-stability" begin
                @test_nowarn @inferred jl_sm(s_jl, z_jl, trues(N, B), jl_ps, jl_st)
                @test_nowarn @inferred jl_sm(s_jl, z_jl, nothing, jl_ps, jl_st)
            end
        end
    end
end
