const PyRigid = pyimport("openfold.utils.rigid_utils").Rigid

@testset "Geometry" begin
    @testset "make_transform_from_reference" begin
        N, B = 6, 3
        rng = Random.Xoshiro(42)
        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                n_xyz = randn(rng, T, 3, N, B)
                ca_xyz = randn(rng, T, 3, N, B)
                c_xyz = randn(rng, T, 3, N, B)

                jl_rot, jl_trans = make_transform_from_reference(n_xyz, ca_xyz, c_xyz)

                n_py = to_py(n_xyz;  swap_batch_dim=true)  # [3,N,B] → [B,N,3]
                ca_py = to_py(ca_xyz; swap_batch_dim=true)
                c_py = to_py(c_xyz;  swap_batch_dim=true)

                py_rigid = PyRigid.make_transform_from_reference(n_py, ca_py, c_py)
                py_rot = py_rigid.get_rots().get_rot_mats() # [B, N, 3, 3]
                py_trans = py_rigid.get_trans() # [B, N, 3]
                
                # [B, N, 3, 3] → [3, 3, N, B]
                @test jl_rot ≈ permutedims(to_jl(py_rot), (3, 4, 2, 1))
                # [B, N, 3] → [3, N, B]
                @test jl_trans ≈ to_jl(py_trans; swap_batch_dim=true)
            end
        end
    end

    @testset "invert_apply" begin
        N, B = 5, 2
        rng = Random.Xoshiro(77)
        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                n_xyz  = randn(rng, T, 3, N, B)
                ca_xyz = randn(rng, T, 3, N, B)
                c_xyz  = randn(rng, T, 3, N, B)
                rot, trans = make_transform_from_reference(n_xyz, ca_xyz, c_xyz)

                local_pts = randn(rng, T, 3, N, B)
                local_pts_r = reshape(local_pts, 3, 1, N, B)
                global_pts = dropdims(Lux.batched_matmul(rot, local_pts_r); dims=2) .+ trans

                recovered = invert_apply(rot, trans, global_pts)
                @test recovered ≈ local_pts

                n_py  = to_py(n_xyz;  swap_batch_dim=true)
                ca_py = to_py(ca_xyz; swap_batch_dim=true)
                c_py  = to_py(c_xyz;  swap_batch_dim=true)
                py_rigid = PyRigid.make_transform_from_reference(n_py, ca_py, c_py)

                global_pts_py = to_py(global_pts; swap_batch_dim=true)  # [B, N, 3]
                py_recovered = py_rigid.invert_apply(global_pts_py)     # [B, N, 3]

                # Python Rigid computes in float32 internally, so use Float32-level tolerance
                if T == Float64
                    @test recovered ≈ to_jl(py_recovered; swap_batch_dim=true) atol=1e-4
                else
                    @test recovered ≈ to_jl(py_recovered; swap_batch_dim=true)
                end
                # @test isapprox(recovered, to_jl(py_recovered; swap_batch_dim=true), rtol=sqrt(eps(Float32)))
            end
        end
    end
end
