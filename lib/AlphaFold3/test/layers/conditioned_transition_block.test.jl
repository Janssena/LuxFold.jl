const PyConditionedTransitionBlock = pyimport("openfold3.core.model.layers.transition").ConditionedTransitionBlock

function sync_conditioned_transition_block!(py, ps)
    sync_af3_adaln!(py.layer_norm, ps.layer_norm)
    sync_dense!(py.swiglu.linear_a, ps.swiglu.gate)
    sync_dense!(py.swiglu.linear_b, ps.swiglu.linear)
    sync_dense!(py.linear_g, ps.linear_g)
    sync_dense!(py.linear_out, ps.linear_out)
    return nothing
end

@testset "ConditionedTransitionBlock" begin
    rng = Random.Xoshiro(42)
    c_a, c_s, n, N, B = 16, 12, 2, 8, 2

    mask = rand(rng, Bool, N, B); mask[1, :] .= true
    mask_cfg = (("No mask", nothing), ("Random mask", mask))

    for T in (Float64, Float32, Float16), (mask_name, amask) in mask_cfg
        @testset "$T, $mask_name" begin
            jl = ConditionedTransitionBlock(c_a, c_s, n)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            py = PyConditionedTransitionBlock(c_a, c_s, n)
            py.to(py_dtype(T))
            sync_conditioned_transition_block!(py, ps)

            a = randn(rng, T, c_a, N, B)
            s = randn(rng, T, c_s, N, B)

            a_jl, _ = jl(a, s, amask, ps, st)

            mask_py = isnothing(amask) ? nothing : to_py(T.(amask); swap_batch_dim=true)
            a_py = py(to_py(a; swap_batch_dim=true), to_py(s; swap_batch_dim=true), mask_py)

            atol = T == Float64 ? 1e-5 : (T == Float32 ? 1f-3 : 5f-2)
            @test isapprox(a_jl, to_jl(a_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = ConditionedTransitionBlock(c_a, c_s, n)
        ps, st = Lux.setup(rng, jl) |> convert_types(Float32)
        a = randn(rng, Float32, c_a, N, B)
        s = randn(rng, Float32, c_s, N, B)
        @test_nowarn @inferred jl(a, s, nothing, ps, st)
        @test_nowarn @inferred jl(a, s, mask, ps, st)
    end
end
