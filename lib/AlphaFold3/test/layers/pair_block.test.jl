const PyPairBlock = pyimport("openfold3.core.model.latent.base_blocks").PairBlock

include("../pairformer/sync_helpers.jl")

@testset "PairBlock" begin
    rng = Random.Xoshiro(42)
    c_z, N, B = 64, 8, 2
    c_hidden_mul, c_hidden_pair_att, no_heads_pair = 16, 16, 4
    transition_n = 2
    inf = 1f9

    pmask = rand(rng, Bool, N, N, B); pmask[1, 1, :] .= true
    mask_cfg = (("No mask", nothing), ("Random mask", pmask))

    for T in (Float64, Float32), (mask_name, pair_mask) in mask_cfg
        @testset "$T, $mask_name" begin
            jl = PairBlock(; chn_z=c_z, chn_hidden_mul=c_hidden_mul,
                chn_hidden_pair_att=c_hidden_pair_att, no_heads_pair, transition_n)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            py = PyPairBlock(c_z, c_hidden_mul, c_hidden_pair_att, no_heads_pair,
                "swiglu", transition_n, 0.0, false, Float64(inf))
            py.to(py_dtype(T))
            sync_pair_block!(py, ps)

            z = randn(rng, T, c_z, N, N, B)

            z_jl, _ = jl(z, pair_mask, ps, st)

            pmask_py = isnothing(pair_mask) ? ones(T, N, N, B) : T.(pair_mask)
            z_py = py(
                to_py(z; swap_batch_dim=true),
                to_py(permutedims(pmask_py, (3, 1, 2)); swap_batch_dim=false),
            )

            atol = T == Float64 ? 1e-5 : 1f-3
            @test isapprox(z_jl, to_jl(z_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = PairBlock(; chn_z=c_z, chn_hidden_mul=c_hidden_mul,
            chn_hidden_pair_att=c_hidden_pair_att, no_heads_pair, transition_n)
        ps, st = Lux.setup(rng, jl) |> convert_types(Float32)
        z = randn(rng, Float32, c_z, N, N, B)
        @test_nowarn @inferred jl(z, nothing, ps, st)
        @test_nowarn @inferred jl(z, pmask, ps, st)
    end
end
