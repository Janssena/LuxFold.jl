const PyRigid = pyimport("openfold3.core.utils.rigid_utils")
const _torch = pyimport("torch")

rng = Random.Xoshiro(42)

@testset "geometry" begin
    @testset "quat_to_rot Python parity" begin
        M = 5
        for T in [Float64, Float32]
            q = randn(rng, T, 4, M)
            q = q ./ sqrt.(sum(abs2, q; dims=1))
            R_jl = quat_to_rot(q)                          # [3, 3, M]
            R_py = PyRigid.quat_to_rot(to_py(q; swap_batch_dim=true))  # [M, 3, 3]
            # swap_batch_dim alone transposes the inner 3×3; restore (i,j) order.
            @test R_jl ≈ permutedims(to_jl(R_py; swap_batch_dim=true), (2, 1, 3))
        end
    end

    @testset "sample_rotations is valid SO(3)" begin
        R = sample_rotations(rng, (4,))                    # [3, 3, 4]
        for k in 1:4
            Rk = R[:, :, k]
            @test Rk * Rk' ≈ I(3) atol = 1e-5
            @test det(Rk) ≈ 1 atol = 1e-5
        end
    end

    @testset "centre_random_augmentation" begin
        N_atom, B = 16, 2
        for T in [Float64, Float32]
            xl   = randn(rng, T, 3, N_atom, B)
            mask = rand(rng, Bool, N_atom, B)
            mask[1, :] .= true                              # ensure ≥1 valid atom per batch

            # pre-sample quaternion + translation so Python can use the same transform.
            q = randn(rng, T, 4, B); q = q ./ sqrt.(sum(abs2, q; dims=1))
            R = quat_to_rot(q)                              # [3, 3, B]
            t = randn(rng, T, 3, 1, B)

            out_jl = centre_random_augmentation(rng, xl, mask; rots=R, trans=t)

            @testset "parity vs openfold transform ($T)" begin
                # Pass q (not R) so Python derives R via its own quat_to_rot — avoids
                # the inner-3×3 transpose ambiguity. Replicate the openfold transform.
                xl_py   = to_py(xl; swap_batch_dim=true)                  # [B, N, 3]
                mask_py = to_py(T.(mask); swap_batch_dim=true)            # [B, N]
                R_py    = PyRigid.quat_to_rot(to_py(q; swap_batch_dim=true))  # [B, 3, 3]
                t_py    = to_py(reshape(t, 3, B); swap_batch_dim=true)    # [B, 3]
                m       = mask_py.unsqueeze(-1)
                mean_xl = (xl_py * m).sum(dim=-2, keepdim=true) / m.sum(dim=-2, keepdim=true)
                xc      = xl_py - mean_xl
                out_py  = (_torch.matmul(xc, R_py.transpose(-1, -2)) + t_py.unsqueeze(-2)) * m
                @test out_jl ≈ to_jl(out_py; swap_batch_dim=true)
            end

            @testset "centering invariant (trans=0) ($T)" begin
                # With no translation the masked centroid is invariant under rotation → ≈ 0.
                out0 = centre_random_augmentation(rng, xl, mask; rots=R, trans=zeros(T, 3, 1, B))
                m3 = reshape(mask, 1, N_atom, B)
                centroid = sum(out0 .* m3; dims=2) ./ sum(m3; dims=2)
                @test all(abs.(centroid) .< (T == Float64 ? 1e-10 : 1e-4))
            end

            @testset "masked-out atoms are zero ($T)" begin
                m3 = reshape(mask, 1, N_atom, B)
                @test all(out_jl[.!repeat(m3, 3, 1, 1)] .== 0)
            end
        end
    end

    @testset "reproducibility" begin
        xl = randn(Random.Xoshiro(1), Float32, 3, 8, 2)
        mask = trues(8, 2)
        a = centre_random_augmentation(Random.Xoshiro(7), xl, mask)
        b = centre_random_augmentation(Random.Xoshiro(7), xl, mask)
        @test a == b
    end

    @testset "type-stability" begin
        xl = randn(rng, Float32, 3, 8, 2); mask = trues(8, 2)

        # Top-level with preset rots/trans
        R = sample_rotations(rng, (2,)); t = randn(rng, Float32, 3, 1, 2)
        @test_nowarn @inferred centre_random_augmentation(rng, xl, mask; rots=R, trans=t)

        # With defaults: calls sample_rotations + randn internally
        rng_stable = Random.Xoshiro(42)
        @test_nowarn @inferred centre_random_augmentation(rng_stable, xl, mask)
    end

    @testset "helper type-stability" begin
        # Test quat_to_rot directly (type-stable with known-shape input)
        q = randn(rng, Float32, 4, 3)
        q = q ./ sqrt.(sum(abs2, q; dims=1))
        @test_nowarn @inferred quat_to_rot(q)

        # sample_rotations: NOT type-stable because shape is a runtime tuple
        # (return dimensions can't be inferred from shape::Tuple at compile time).
        # This is expected; the function is correct but inference is limited.
        R = @test_nowarn sample_rotations(rng, (2,); T=Float32)
        @test typeof(R) == Array{Float32, 3}
    end
end
