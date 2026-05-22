const PyPairTransition = pyimport("openfold.model.pair_transition").PairTransition
const PyMSATransition = pyimport("openfold.model.evoformer").MSATransition

rng = Random.Xoshiro(42)

@testset "Transition" begin
    @testset "Shape preservation" begin
        c_in, N, B = 16, 8, 2

        @testset "Rank 3" begin
            layer = Transition(c_in; n=4, rank=3)
            ps, st = Lux.setup(rng, layer) |> convert_types(Float32)
            x = randn(rng, Float32, c_in, N, B)
            y, _ = layer(x, nothing, ps, st)
            @test size(y) == (c_in, N, B)
        end

        @testset "Rank 4" begin
            layer = Transition(c_in; n=4, rank=4)
            ps, st = Lux.setup(rng, layer) |> convert_types(Float32)
            x = randn(rng, Float32, c_in, N, N, B)
            y, _ = layer(x, nothing, ps, st)
            @test size(y) == (c_in, N, N, B)
        end
    end

    @testset "Mask application" begin
        c_in, N, B = 16, 8, 2
        T = Float32
        layer = Transition(c_in; rank=4)
        ps, st = Lux.setup(rng, layer) |> convert_types(T)
        x = randn(rng, T, c_in, N, N, B)
        mask = trues(N, N, B)
        mask[1, :, :] .= false
        y, _ = layer(x, mask, ps, st)
        @test all(iszero, y[:, 1, :, :])
        @test any(!iszero, y[:, 2:end, :, :])
    end
end

# ==============================================================================
# MSATransition parity tests
# ==============================================================================

@testset "MSATransition" begin
    chn_msa, N, S, B = 16, 8, 4, 2

    mask_cfg = (
        ("No mask", nothing),
        ("Random mask", rand(rng, Bool, N, S, B)),
    )

    for (mask_name, mask_jl) in mask_cfg
        @testset "$mask_name" begin
            for T in [Float64, Float32, Float16]
                @testset "$T" begin
                    jl_layer = MSATransition(chn_msa)
                    jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = PyMSATransition(chn_msa, 4)

                    sync_layernorm!(py_layer.layer_norm, jl_ps.layer_norm)
                    sync_dense!(py_layer.linear_1, jl_ps.linear_1)
                    sync_dense!(py_layer.linear_2, jl_ps.linear_2)

                    x_jl = randn(rng, T, chn_msa, N, S, B)
                    x_py = to_py(x_jl; swap_batch_dim=true)
                    mask_py = isnothing(mask_jl) ? nothing :
                        to_py(permutedims(mask_jl, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T))

                    y_jl, _ = jl_layer(x_jl, mask_jl, jl_ps, jl_st)
                    y_py = py_layer(x_py, mask_py)

                    @testset "Python parity" begin
                        @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                    end

                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer(x_jl, mask_jl, jl_ps, jl_st)
                    end
                end
            end
        end
    end
end

# ==============================================================================
# PairTransition parity tests
# ==============================================================================

@testset "PairTransition" begin
    c_z, N, B = 16, 8, 2

    mask_cfg = (
        ("No mask", nothing),
        ("Random mask", rand(rng, Bool, N, N, B)),
    )

    for (mask_name, mask_jl) in mask_cfg
        @testset "$mask_name" begin
            for T in [Float64, Float32, Float16]
                @testset "$T" begin
                    jl_layer = PairTransition(c_z)
                    jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = PyPairTransition(c_z, 2)

                    sync_layernorm!(py_layer.layer_norm, jl_ps.layer_norm)
                    sync_dense!(py_layer.linear_1, jl_ps.linear_1)
                    sync_dense!(py_layer.linear_2, jl_ps.linear_2)

                    x_jl = randn(rng, T, c_z, N, N, B)
                    x_py = to_py(x_jl; swap_batch_dim=true)
                    mask_py = isnothing(mask_jl) ? nothing :
                        to_py(permutedims(mask_jl, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T))

                    y_jl, _ = jl_layer(x_jl, mask_jl, jl_ps, jl_st)
                    y_py = py_layer(x_py, mask_py)

                    @testset "Python parity" begin
                        @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                    end

                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer(x_jl, mask_jl, jl_ps, jl_st)
                    end
                end
            end
        end
    end
end