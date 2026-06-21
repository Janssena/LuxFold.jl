const PyBackboneUpdate = pyimport("openfold.model.structure_module").BackboneUpdate

rng = Random.Xoshiro(42)

@testset "BackboneUpdate" begin
    @testset "Python parity" begin
        chn_s, N, B = 16, 5, 2

        for T in (Float64, Float32, Float16)
            @testset "$T" begin
                jl_layer = BackboneUpdate(chn_s)
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyBackboneUpdate(chn_s)
                sync_dense!(py_layer.linear, jl_ps.linear)

                x_jl = randn(rng, T, chn_s, N, B)
                x_py = to_py(x_jl; swap_batch_dim=true)

                y_jl, _ = jl_layer(x_jl, jl_ps, jl_st)
                y_py = py_layer(x_py)

                @testset "Python parity" begin
                    @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred jl_layer(x_jl, jl_ps, jl_st)
                end
            end
        end
    end
end
