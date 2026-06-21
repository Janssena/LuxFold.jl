using LinearAlgebra: I as IdMat

const PyRigid = pyimport("openfold.utils.rigid_utils").Rigid
const PyRotation = pyimport("openfold.utils.rigid_utils").Rotation

rng = Random.Xoshiro(42)

@testset "Rigid identity" begin
    for T in (Float32, Float64)
        N, B = 5, 2
        r = AlphaFold2.rigid_identity(T, N, B)

        @test all(==(one(T)), r.quats[1, :, :])
        @test all(==(zero(T)), r.quats[2:4, :, :])
        @test all(==(zero(T)), r.trans)

        # rot should be identity at every (n, b)
        for b in 1:B, n in 1:N
            @test r.rot[:, :, n, b] ≈ Matrix{T}(IdMat, 3, 3)
        end
    end
end

@testset "quat_to_rotmat ↔ rot_to_quat round-trip" begin
    # Random rotation: build from a quaternion, convert to rot, back to quat,
    # check we recover the original quaternion (up to sign).
    for T in (Float32, Float64)
        N, B = 4, 3
        raw = randn(rng, T, 4, N, B)
        q   = raw ./ sqrt.(sum(abs2, raw; dims=1))
        # Ensure positive scalar for canonical sign
        for b in 1:B, n in 1:N
            if q[1, n, b] < 0
                @views q[:, n, b] .*= -one(T)
            end
        end
        R   = AlphaFold2.quat_to_rotmat(q)
        q2  = AlphaFold2.rot_to_quat(R)
        # Allow looser tolerance for Float32 since eigh has limited precision.
        atol = T === Float32 ? 1f-4 : 1e-10
        @test isapprox(q2, q; atol=atol)
    end
end

@testset "apply / invert_apply round-trip" begin
    for T in (Float32, Float64), shape in ((3, 5, 2), (3, 7, 4, 5, 2))
        N, B = 5, 2
        # Build a random rotation by composing a random quat update onto identity.
        r0  = AlphaFold2.rigid_identity(T, N, B)
        upd = randn(rng, T, 6, N, B) .* T(0.3)
        r   = AlphaFold2.compose_q_update_vec(r0, upd)

        pts = randn(rng, T, shape...)
        pts_global = AlphaFold2.apply(r, pts)
        pts_back   = AlphaFold2.invert_apply(r, pts_global)
        atol = T === Float32 ? 1f-4 : 1e-10
        @test isapprox(pts, pts_back; atol=atol)
    end
end

@testset "apply parity against Python (3D pts)" begin
    # NOTE: openfold's `Rotation.__init__` hard-casts quats/rot to torch.float32
    # (see rigid_utils.py line ~320), so Python-side parity tests effectively
    # always run at fp32 precision regardless of input dtype. Tolerance is set
    # accordingly.
    for T in (Float64, Float32)
        N, B = 5, 2
        r0  = AlphaFold2.rigid_identity(T, N, B)
        upd = randn(rng, T, 6, N, B) .* T(0.3)
        r   = AlphaFold2.compose_q_update_vec(r0, upd)
        pts = randn(rng, T, 3, N, B)

        out_jl = AlphaFold2.apply(r, pts)

        py_quats = to_py(r.quats; swap_batch_dim=true)
        py_trans = to_py(r.trans; swap_batch_dim=true)
        py_pts   = to_py(pts;     swap_batch_dim=true)
        py_rot   = PyRotation(rot_mats=nothing, quats=py_quats, normalize_quats=false)
        py_rigid = PyRigid(py_rot, py_trans)
        out_py   = py_rigid.apply(py_pts)
        @test isapprox(out_jl, to_jl(out_py; swap_batch_dim=true); atol=1f-4)
    end
end

@testset "compose_q_update_vec parity against Python" begin
    # Note: we deliberately don't use Python's `Rigid.identity` — its
    # `@lru_cache`d helpers can return a Float32 tensor even when float64 is
    # requested. Instead we construct the Python Rigid from explicit quats and
    # trans, mirroring our Julia identity exactly.
    for T in (Float64, Float32)
        N, B = 5, 2
        r0_jl  = AlphaFold2.rigid_identity(T, N, B)
        upd_jl = randn(rng, T, 6, N, B) .* T(0.3)
        r_jl   = AlphaFold2.compose_q_update_vec(r0_jl, upd_jl)

        # Build identity Python Rigid from our explicit quats and trans
        py_quats_in = to_py(r0_jl.quats; swap_batch_dim=true)     # [B, N, 4] Float<T>
        py_trans_in = to_py(r0_jl.trans; swap_batch_dim=true)     # [B, N, 3]
        py_rot_in   = PyRotation(rot_mats=nothing, quats=py_quats_in, normalize_quats=false)
        py_rigid_id = PyRigid(py_rot_in, py_trans_in)

        py_upd   = to_py(upd_jl; swap_batch_dim=true)
        py_rigid = py_rigid_id.compose_q_update_vec(py_upd)

        py_quats = py_rigid.get_rots().get_quats()
        py_trans = py_rigid.get_trans()
        py_rot   = py_rigid.get_rots().get_rot_mats()

        # Python forces fp32 internally (see note in apply parity test above).
        @test isapprox(r_jl.quats, to_jl(py_quats; swap_batch_dim=true); atol=1f-4)
        @test isapprox(r_jl.trans, to_jl(py_trans; swap_batch_dim=true); atol=1f-4)
        py_rot_jl    = to_jl(py_rot; swap_batch_dim=false)
        py_rot_julia = permutedims(py_rot_jl, (3, 4, 2, 1))
        @test isapprox(r_jl.rot, py_rot_julia; atol=1f-4)
    end
end

@testset "scale_translation" begin
    T = Float32
    N, B = 4, 2
    r = AlphaFold2.compose_q_update_vec(
        AlphaFold2.rigid_identity(T, N, B), randn(rng, T, 6, N, B) .* T(0.3)
    )
    r2 = AlphaFold2.scale_translation(r, 10.0)
    @test r2.trans ≈ r.trans .* T(10)
    @test r2.rot   == r.rot
    @test r2.quats == r.quats
end

@testset "to_tensor_7 / to_tensor_4x4" begin
    T = Float32
    N, B = 4, 2
    r = AlphaFold2.compose_q_update_vec(
        AlphaFold2.rigid_identity(T, N, B), randn(rng, T, 6, N, B) .* T(0.3)
    )
    t7 = AlphaFold2.to_tensor_7(r)
    @test t7[1:4, :, :] ≈ r.quats
    @test t7[5:7, :, :] ≈ r.trans

    t44 = AlphaFold2.to_tensor_4x4(r)
    @test t44[1:3, 1:3, :, :] ≈ r.rot
    @test t44[1:3, 4, :, :]   ≈ r.trans
    @test all(==(zero(T)), t44[4, 1:3, :, :])
    @test all(==(one(T)),  t44[4, 4, :, :])
end
