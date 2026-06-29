const PyPairFormerStack = pyimport("openfold3.core.model.latent.pairformer").PairFormerStack

include("sync_helpers.jl")

function sync_pairformer_stack!(py, ps)
    py_blocks = collect(py.blocks)
    for (i, name) in enumerate(keys(ps.blocks))
        sync_pairformer_block!(py_blocks[i], ps.blocks[name])
    end
    return nothing
end

@testset "PairFormerStack" begin
    rng = Random.Xoshiro(42)
    c_s, c_z, N, B = 32, 32, 6, 1
    c_hidden_pair_bias, no_heads_pair_bias = 8, 2
    c_hidden_mul, c_hidden_pair_att, no_heads_pair = 8, 8, 2
    transition_n, no_blocks = 2, 2
    inf = 1f9

    for T in [Float64, Float32]
        @testset "Python parity ($T, $no_blocks blocks)" begin
            jl = PairFormerStack(; chn_s=c_s, chn_z=c_z, chn_hidden_pair_bias=c_hidden_pair_bias, no_heads_pair_bias, chn_hidden_mul=c_hidden_mul, chn_hidden_pair_att=c_hidden_pair_att, no_heads_pair, transition_n, no_blocks)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            # Python: PairFormerStack(c_s, c_z, c_hidden_pair_bias, no_heads_pair_bias,
            #   c_hidden_mul, c_hidden_pair_att, no_heads_pair, no_blocks, transition_type,
            #   transition_n, pair_dropout, fuse_projection_weights, blocks_per_ckpt, inf)
            py = PyPairFormerStack(c_s, c_z, c_hidden_pair_bias, no_heads_pair_bias,
                                   c_hidden_mul, c_hidden_pair_att, no_heads_pair, no_blocks,
                                   "swiglu", transition_n, 0.0, false, nothing, Float64(inf))
            py.to(py_dtype(T))
            sync_pairformer_stack!(py, ps)

            s = randn(rng, T, c_s, N, B)
            z = randn(rng, T, c_z, N, N, B)

            out_jl, _ = jl(s, z, nothing, nothing, ps, st)
            s_jl, z_jl = out_jl.s, out_jl.z

            s_py, z_py = py(
                to_py(s; swap_batch_dim=true),
                to_py(z; swap_batch_dim=true),
                to_py(ones(T, N, B); swap_batch_dim=true),
                to_py(ones(T, N, N, B); swap_batch_dim=true),
            )

            atol = T == Float64 ? 1e-5 : 5f-3   # accumulated over block depth
            @test isapprox(s_jl, to_jl(s_py; swap_batch_dim=true); atol=atol, rtol=atol)
            @test isapprox(z_jl, to_jl(z_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = PairFormerStack(; chn_s=c_s, chn_z=c_z, chn_hidden_pair_bias=c_hidden_pair_bias, no_heads_pair_bias, chn_hidden_mul=c_hidden_mul, chn_hidden_pair_att=c_hidden_pair_att, no_heads_pair, transition_n, no_blocks)
        ps, st = Lux.setup(rng, jl) |> convert_types(Float32)
        s = randn(rng, Float32, c_s, N, B)
        z = randn(rng, Float32, c_z, N, N, B)
        @test_nowarn @inferred jl(s, z, nothing, nothing, ps, st)
    end
end
