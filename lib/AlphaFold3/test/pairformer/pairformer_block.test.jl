const PyPairFormerBlock = pyimport("openfold3.core.model.latent.pairformer").PairFormerBlock

include("sync_helpers.jl")

@testset "PairFormerBlock" begin
    rng = Random.Xoshiro(42)
    c_s, c_z, N, B = 64, 64, 8, 1
    c_hidden_pair_bias, no_heads_pair_bias = 16, 4
    c_hidden_mul, c_hidden_pair_att, no_heads_pair = 16, 16, 4
    transition_n = 2
    inf = 1f9

    # ≥1 valid token / pair per batch to avoid all-masked attention rows
    smask = rand(rng, Bool, N, B); smask[1, :] .= true
    pmask = rand(rng, Bool, N, N, B); pmask[1, 1, :] .= true
    # mask passed to Julia (Bool / nothing); the Python boundary uses float masks (all-ones for "no mask")
    mask_cfg = (("No mask", nothing, nothing), ("Random mask", smask, pmask))

    for T in [Float64, Float32], (mask_name, single_mask, pair_mask) in mask_cfg
        @testset "$T, $mask_name" begin
            jl = PairFormerBlock(; chn_s=c_s, chn_z=c_z, chn_hidden_pair_bias=c_hidden_pair_bias, no_heads_pair_bias, chn_hidden_mul=c_hidden_mul, chn_hidden_pair_att=c_hidden_pair_att, no_heads_pair, transition_n)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            # Python: PairFormerBlock(c_s, c_z, c_hidden_pair_bias, no_heads_pair_bias,
            #   c_hidden_mul, c_hidden_pair_att, no_heads_pair, transition_type, transition_n,
            #   pair_dropout, fuse_projection_weights, inf)
            py = PyPairFormerBlock(c_s, c_z, c_hidden_pair_bias, no_heads_pair_bias,
                                   c_hidden_mul, c_hidden_pair_att, no_heads_pair,
                                   "swiglu", transition_n, 0.0, false, Float64(inf))
            py.to(py_dtype(T))
            sync_pairformer_block!(py, ps)

            s = randn(rng, T, c_s, N, B)
            z = randn(rng, T, c_z, N, N, B)

            out_jl, _ = jl(s, z, single_mask, pair_mask, ps, st)
            s_jl, z_jl = out_jl.s, out_jl.z

            # Python masks are required (positional); synthesise all-ones for the "no mask" config.
            # Single mask [N,B] → py [B,N] via swap_batch_dim. Pair mask [i,j,B] must map to
            # py [B,i,j] aligned with z's [B,i,j,C] — `permutedims(_,(3,1,2))` (NOT swap_batch_dim,
            # which would transpose i↔j for a 3D array).
            smask_py = isnothing(single_mask) ? ones(T, N, B) : T.(single_mask)
            pmask_jl = isnothing(pair_mask) ? ones(T, N, N, B) : T.(pair_mask)
            s_py, z_py = py(
                to_py(s; swap_batch_dim=true),
                to_py(z; swap_batch_dim=true),
                to_py(smask_py; swap_batch_dim=true),
                to_py(permutedims(pmask_jl, (3, 1, 2)); swap_batch_dim=false),
            )

            atol = T == Float64 ? 1e-5 : 1f-3
            @test isapprox(s_jl, to_jl(s_py; swap_batch_dim=true); atol=atol, rtol=atol)
            @test isapprox(z_jl, to_jl(z_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = PairFormerBlock(; chn_s=c_s, chn_z=c_z, chn_hidden_pair_bias=c_hidden_pair_bias, no_heads_pair_bias, chn_hidden_mul=c_hidden_mul, chn_hidden_pair_att=c_hidden_pair_att, no_heads_pair, transition_n)
        ps, st = Lux.setup(rng, jl) |> convert_types(Float32)
        s = randn(rng, Float32, c_s, N, B)
        z = randn(rng, Float32, c_z, N, N, B)
        @test_nowarn @inferred jl(s, z, nothing, nothing, ps, st)
        @test_nowarn @inferred jl(s, z, smask, pmask, ps, st)
    end
end
