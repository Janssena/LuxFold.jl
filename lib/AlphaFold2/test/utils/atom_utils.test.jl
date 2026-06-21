# atom_utils.test.jl — parity + type-stability tests for torsion_angles_to_frames
#                       and frames_and_literature_positions_to_atom14_pos
#
# Reference: openfold/utils/feats.py
# See also: openspec/changes/af2-atom-utilities/specs/atom-position-utilities/spec.md

import LinearAlgebra

const PyRigid_au = pyimport("openfold.utils.rigid_utils").Rigid
const PyRotation_au = pyimport("openfold.utils.rigid_utils").Rotation
const PyFeats = pyimport("openfold.utils.feats")
const PyRC = pyimport("openfold.np.residue_constants")
const torch_au = pyimport("torch")

rng_au = Random.Xoshiro(42)

# ==============================================================================
# Helper: build PyRigid from Julia Rigid (quats + trans, batch dims [N, B])
# → Python Rigid with batch dims [B, N]
# NOTE: OpenFold's Rotation hard-casts to Float32 internally.
# ==============================================================================
function _jl_rigid_to_py(r::Rigid)
    py_quats = to_py(r.quats; swap_batch_dim=true)   # [4,N,B] → [B,N,4]
    py_trans = to_py(r.trans; swap_batch_dim=true)   # [3,N,B] → [B,N,3]
    py_rot = PyRotation_au(rot_mats=nothing, quats=py_quats, normalize_quats=false)
    return PyRigid_au(py_rot, py_trans)
end

# Helper: build PyRigid from Julia frames (rot [3,3,8,N,B], trans [3,8,N,B])
function _jl_frames_to_py(frames)
    rot_perm   = permutedims(frames.rot,   (5, 4, 3, 1, 2))  # [B, N, 8, 3, 3]
    trans_perm = permutedims(frames.trans, (4, 3, 2, 1))      # [B, N, 8, 3]
    py_rot     = PyRotation_au(
        rot_mats=to_py(rot_perm; swap_batch_dim=false),
        quats=nothing,
        normalize_quats=false,
    )
    return PyRigid_au(py_rot, to_py(trans_perm; swap_batch_dim=false))
end

# Helper: convert Python Rigid frames [B,N,8,3,3] / [B,N,8,3] → Julia convention
function _py_frames_to_jl(py_frames)
    py_rot_mat = py_frames.get_rots().get_rot_mats()  # [B, N, 8, 3, 3]
    py_trans   = py_frames.get_trans()                 # [B, N, 8, 3]
    rot_jl   = permutedims(to_jl(py_rot_mat; swap_batch_dim=false), (4, 5, 3, 2, 1))
    trans_jl = permutedims(to_jl(py_trans;   swap_batch_dim=false), (4, 3, 2, 1))
    return (; rot=rot_jl, trans=trans_jl)
end

# Helper: constants as T-typed Julia arrays (simulates device-resident st.residue_constants)
function _make_test_constants(::Type{T}) where T
    default_frames  = T.(restype_rigid_group_default_frame)         # [21, 8, 4, 4]
    group_idx       = restype_atom14_to_rigid_group                  # [21, 14] Int32
    lit_positions   = T.(restype_atom14_rigid_group_positions)       # [21, 14, 3]
    atom_mask       = restype_atom14_mask                              # [21, 14] Bool
    return (; default_frames, group_idx, lit_positions, atom_mask)
end

# ==============================================================================
# §1  Constants parity
# ==============================================================================

@testset "restype_rigid_group_default_frame parity" begin
    py_frames = to_jl(
        torch_au.tensor(PyRC.restype_rigid_group_default_frame); swap_batch_dim=false
    )
    # Julia [21, 8, 4, 4] vs Python [21, 8, 4, 4]
    @test size(restype_rigid_group_default_frame) == (21, 8, 4, 4)
    @test isapprox(Float32.(restype_rigid_group_default_frame), Float32.(py_frames); atol=1f-5)
end

@testset "restype_atom37_to_atom14 parity" begin
    # OpenFold uses RESTYPE_ATOM37_TO_ATOM14 (uppercase) in residue_constants
    py_arr = to_jl(torch_au.tensor(PyRC.RESTYPE_ATOM37_TO_ATOM14); swap_batch_dim=false)
    # Python is 0-based; Julia is 1-based
    @test size(restype_atom37_to_atom14) == (21, 37)
    @test all(1 .<= restype_atom37_to_atom14 .<= 14)
    @test restype_atom37_to_atom14 == Int32.(py_arr .+ 1)
end

# ==============================================================================
# §2  torsion_angles_to_frames
# ==============================================================================

@testset "torsion_angles_to_frames" begin
    N, B = 10, 2
    rng = Random.Xoshiro(42)

    @testset "shape" begin
        r0  = AlphaFold2.rigid_identity(Float32, N, B)
        upd = randn(rng, Float32, 6, N, B) .* 0.3f0
        r   = compose_q_update_vec(r0, upd)
        angles = randn(rng, Float32, 2, 7, N, B)
        aatype = rand(rng, 1:20, N, B)
        cs     = _make_test_constants(Float32)
        frames = torsion_angles_to_frames(r, angles, aatype, cs.default_frames)
        @test size(frames.rot)   == (3, 3, 8, N, B)
        @test size(frames.trans) == (3, 8, N, B)
    end

    @testset "Python parity" begin
        # NOTE: OpenFold's Rotation hard-casts to Float32 internally, so parity tests
        # at all precisions use Float32-level tolerance (atol=1f-4).
        for T in (Float64, Float32)
            @testset "$T" begin
                rng2    = Random.Xoshiro(42)
                r0      = AlphaFold2.rigid_identity(T, N, B)
                upd     = randn(rng2, T, 6, N, B) .* T(0.3)
                r_jl    = compose_q_update_vec(r0, upd)
                angles  = randn(rng2, T, 2, 7, N, B)
                aatype  = rand(rng2, 1:20, N, B)
                cs      = _make_test_constants(T)

                frames_jl = torsion_angles_to_frames(r_jl, angles, aatype, cs.default_frames)

                # --- Python side ---
                # angles: Julia [2, 7, N, B] → Python [B, N, 7, 2]
                angles_py = to_py(
                    permutedims(Float32.(angles), (4, 3, 2, 1));
                    swap_batch_dim=false,
                )
                aatype_py = to_py(
                    permutedims(Int64.(aatype .- 1), (2, 1));  # 0-based, [B, N]
                    swap_batch_dim=false,
                )
                rrgdf_py = to_py(
                    Float32.(restype_rigid_group_default_frame);
                    swap_batch_dim=false,
                )
                py_r      = _jl_rigid_to_py(Rigid(Float32.(r_jl.quats), Float32.(r_jl.trans)))
                py_result = PyFeats.torsion_angles_to_frames(py_r, angles_py, aatype_py, rrgdf_py)

                frames_ref = _py_frames_to_jl(py_result)

                @test isapprox(Float32.(frames_jl.rot),   Float32.(frames_ref.rot);   atol=1f-4)
                @test isapprox(Float32.(frames_jl.trans), Float32.(frames_ref.trans); atol=1f-4)
            end
        end
    end

    @testset "type stability" begin
        r0     = AlphaFold2.rigid_identity(Float32, N, B)
        upd    = randn(rng, Float32, 6, N, B) .* 0.3f0
        r      = compose_q_update_vec(r0, upd)
        angles = randn(rng, Float32, 2, 7, N, B)
        aatype = rand(rng, 1:20, N, B)
        cs     = _make_test_constants(Float32)
        @test_nowarn @inferred torsion_angles_to_frames(r, angles, aatype, cs.default_frames)
    end

    @testset "identity frame + zero angles → default frames" begin
        # With identity backbone and zero angles (sin=0, cos=1 ∀g), the composed
        # frame = default_frame ∘ identity = default_frame.  After applying the
        # identity backbone, global = default_frame.
        r0   = AlphaFold2.rigid_identity(Float32, N, B)
        # zero angles: sin=0, cos=1 for all 7 angles
        angles_zero = similar(r0.rot, Float32, 2, 7, N, B)
        angles_zero[1, :, :, :] .= 0f0  # sin=0
        angles_zero[2, :, :, :] .= 1f0  # cos=1
        aatype = rand(rng, 1:20, N, B)
        cs     = _make_test_constants(Float32)

        frames = torsion_angles_to_frames(r0, angles_zero, aatype, cs.default_frames)

        # Check group 1 (backbone): should be identity rotation, zero translation
        @test isapprox(frames.rot[:, :, 1, :, :],
                       reshape(Matrix{Float32}(LinearAlgebra.I, 3, 3), 3, 3, 1, 1) .* ones(Float32, 1, 1, N, B);
                       atol=1f-5)
        @test isapprox(frames.trans[:, 1, :, :], zeros(Float32, 3, N, B); atol=1f-5)
    end

    @testset "edge case: N=1" begin
        r0     = AlphaFold2.rigid_identity(Float32, 1, 1)
        upd    = randn(rng, Float32, 6, 1, 1) .* 0.3f0
        r      = compose_q_update_vec(r0, upd)
        angles = randn(rng, Float32, 2, 7, 1, 1)
        aatype = ones(Int, 1, 1)
        cs     = _make_test_constants(Float32)
        @test_nowarn torsion_angles_to_frames(r, angles, aatype, cs.default_frames)
    end
end

# ==============================================================================
# §3  frames_and_literature_positions_to_atom14_pos
# ==============================================================================

@testset "frames_and_literature_positions_to_atom14_pos" begin
    N, B = 10, 2
    rng  = Random.Xoshiro(42)

    @testset "shape" begin
        r0     = AlphaFold2.rigid_identity(Float32, N, B)
        upd    = randn(rng, Float32, 6, N, B) .* 0.3f0
        r      = compose_q_update_vec(r0, upd)
        angles = randn(rng, Float32, 2, 7, N, B)
        aatype = rand(rng, 1:20, N, B)
        cs     = _make_test_constants(Float32)
        frames = torsion_angles_to_frames(r, angles, aatype, cs.default_frames)
        pred   = frames_and_literature_positions_to_atom14_pos(
            frames, aatype, cs.group_idx, cs.lit_positions, cs.atom_mask,
        )
        @test size(pred) == (3, 14, N, B)
    end

    @testset "Python parity" begin
        for T in (Float64, Float32)
            @testset "$T" begin
                rng2   = Random.Xoshiro(42)
                r0     = AlphaFold2.rigid_identity(T, N, B)
                upd    = randn(rng2, T, 6, N, B) .* T(0.3)
                r_jl   = compose_q_update_vec(r0, upd)
                angles = randn(rng2, T, 2, 7, N, B)
                aatype = rand(rng2, 1:20, N, B)
                cs     = _make_test_constants(T)

                frames_jl = torsion_angles_to_frames(r_jl, angles, aatype, cs.default_frames)
                pred_jl   = frames_and_literature_positions_to_atom14_pos(
                    frames_jl, aatype, cs.group_idx, cs.lit_positions, cs.atom_mask,
                )

                # --- Python side ---
                aatype_py = to_py(
                    permutedims(Int64.(aatype .- 1), (2, 1));
                    swap_batch_dim=false,
                )
                rrgdf_py = to_py(
                    Float32.(restype_rigid_group_default_frame);
                    swap_batch_dim=false,
                )
                angles_py = to_py(
                    permutedims(Float32.(angles), (4, 3, 2, 1));
                    swap_batch_dim=false,
                )
                py_r      = _jl_rigid_to_py(Rigid(Float32.(r_jl.quats), Float32.(r_jl.trans)))
                py_frames = PyFeats.torsion_angles_to_frames(py_r, angles_py, aatype_py, rrgdf_py)

                pred_py = PyFeats.frames_and_literature_positions_to_atom14_pos(
                    py_frames,
                    aatype_py,
                    rrgdf_py,
                    to_py(Int64.(restype_atom14_to_rigid_group); swap_batch_dim=false),
                    to_py(Float32.(restype_atom14_mask); swap_batch_dim=false),
                    to_py(Float32.(restype_atom14_rigid_group_positions); swap_batch_dim=false),
                )
                # Python [B, N, 14, 3] → Julia [3, 14, N, B]
                pred_ref = permutedims(to_jl(pred_py; swap_batch_dim=false), (4, 3, 2, 1))

                @test isapprox(Float32.(pred_jl), Float32.(pred_ref); atol=1f-4)
            end
        end
    end

    @testset "type stability" begin
        r0     = AlphaFold2.rigid_identity(Float32, N, B)
        upd    = randn(rng, Float32, 6, N, B) .* 0.3f0
        r      = compose_q_update_vec(r0, upd)
        angles = randn(rng, Float32, 2, 7, N, B)
        aatype = rand(rng, 1:20, N, B)
        cs     = _make_test_constants(Float32)
        frames = torsion_angles_to_frames(r, angles, aatype, cs.default_frames)
        @test_nowarn @inferred frames_and_literature_positions_to_atom14_pos(
            frames, aatype, cs.group_idx, cs.lit_positions, cs.atom_mask,
        )
    end

    @testset "glycine: sidechain atoms zeroed" begin
        # GLY (G = index 8 in 1-based Julia restypes: A,R,N,D,C,Q,E,G,...) has 4 backbone
        # atoms in atom14 (N, CA, C, O); sidechain (CB+) are absent → mask is 0.
        # Backbone atoms should be non-zero for a non-identity frame.
        r0  = AlphaFold2.rigid_identity(Float32, 1, 1)
        upd = randn(rng, Float32, 6, 1, 1) .* 0.3f0
        r   = compose_q_update_vec(r0, upd)
        angles = randn(rng, Float32, 2, 7, 1, 1)
        aatype = fill(8, 1, 1)  # GLY = index 8 (1-based): A=1,R=2,N=3,D=4,C=5,Q=6,E=7,G=8
        cs     = _make_test_constants(Float32)
        frames = torsion_angles_to_frames(r, angles, aatype, cs.default_frames)
        pred   = frames_and_literature_positions_to_atom14_pos(
            frames, aatype, cs.group_idx, cs.lit_positions, cs.atom_mask,
        )
        # GLY has 4 backbone atoms in atom14 (N, CA, C, O); atoms 5+ are empty
        n_gly_atoms = sum(cs.atom_mask[8, :])
        @test n_gly_atoms <= 4                         # sanity check
        # Non-masked positions should be non-zero (non-identity frame)
        for a in 1:n_gly_atoms
            @test any(!iszero, pred[:, a, 1, 1])
        end
        # All atoms beyond n_gly_atoms must be exactly zero
        for a in (n_gly_atoms+1):14
            @test all(iszero, pred[:, a, 1, 1])
        end
    end

    @testset "all-masked residue: positions all zero" begin
        # Unk (index 21) has all-zero atom14 mask → all positions should be 0
        r0  = AlphaFold2.rigid_identity(Float32, 1, 1)
        upd = randn(rng, Float32, 6, 1, 1) .* 0.3f0
        r   = compose_q_update_vec(r0, upd)
        angles = randn(rng, Float32, 2, 7, 1, 1)
        aatype = fill(21, 1, 1)  # Unk = index 21
        cs     = _make_test_constants(Float32)
        frames = torsion_angles_to_frames(r, angles, aatype, cs.default_frames)
        pred   = frames_and_literature_positions_to_atom14_pos(
            frames, aatype, cs.group_idx, cs.lit_positions, cs.atom_mask,
        )
        @test all(iszero, pred)
    end
end

# ==============================================================================
# §4  Integration: end-to-end
# ==============================================================================

@testset "Integration: torsion_angles_to_frames → atom14_pos" begin
    N, B = 8, 2
    rng = Random.Xoshiro(42)
    r0     = AlphaFold2.rigid_identity(Float32, N, B)
    upd    = randn(rng, Float32, 6, N, B) .* 0.3f0
    r      = compose_q_update_vec(r0, upd)
    angles = randn(rng, Float32, 2, 7, N, B)
    aatype = rand(rng, 1:20, N, B)
    cs     = _make_test_constants(Float32)

    frames = torsion_angles_to_frames(r, angles, aatype, cs.default_frames)
    pred   = frames_and_literature_positions_to_atom14_pos(
        frames, aatype, cs.group_idx, cs.lit_positions, cs.atom_mask,
    )

    @test size(pred) == (3, 14, N, B)

    # Gather atom14 mask for each residue: [21,14][aatype,:] → [N,B,14] → permute → [14,N,B]
    mask_sel = reshape(
        permutedims(cs.atom_mask[aatype, :], (3, 1, 2)),  # [N,B,14] → [14,N,B]
        1, 14, N, B,
    )
    # Masked positions must be exactly zero; valid positions finite (no NaN/Inf)
    _z = zero(eltype(pred))
    masked_pred   = @. ifelse(mask_sel, _z,   pred)
    unmasked_pred = @. ifelse(mask_sel, pred, _z  )
    @test all(iszero, masked_pred)
    @test all(isfinite, unmasked_pred)
end

# ==============================================================================
# §5  Round-trip test: Rigid(to_tensor_7(r)) ≈ r
# ==============================================================================

@testset "Rigid from_tensor_7 round-trip" begin
    for T in (Float32, Float64)
        rng2 = Random.Xoshiro(42)
        r0   = AlphaFold2.rigid_identity(T, 5, 3)
        upd  = randn(rng2, T, 6, 5, 3) .* T(0.3)
        r    = compose_q_update_vec(r0, upd)
        r2   = Rigid(to_tensor_7(r))
        @test isapprox(r2.quats, r.quats; atol=T(1e-5))
        @test isapprox(r2.trans, r.trans; atol=T(1e-5))
    end
end
