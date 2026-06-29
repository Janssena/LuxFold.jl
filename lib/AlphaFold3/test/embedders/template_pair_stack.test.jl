const PyTemplatePairStack = _py_templ_mod.TemplatePairStack

using AlphaFold3: TemplatePairStack

@testset "TemplatePairStack" begin
    rng = Random.Xoshiro(42)
    N, Tpl, B = 12, 2, 1
    c_t, c_hidden_mul, c_hidden_att, no_heads, transition_n, no_blocks = 32, 16, 16, 2, 2, 2

    rand_mask = rand(rng, Bool, N, N, Tpl, B); rand_mask[1, 1, :, :] .= true
    mask_cfg = (("No mask", nothing), ("Random mask", rand_mask))

    for T in [Float64, Float32], (mask_name, mask) in mask_cfg
        @testset "$T, $mask_name" begin
            jl = TemplatePairStack(; chn_t=c_t, chn_hidden_mul=c_hidden_mul,
                                   chn_hidden_pair_att=c_hidden_att, no_heads, transition_n, no_blocks)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            py = PyTemplatePairStack(c_t=c_t, c_hidden_tri_att=c_hidden_att,
                                     c_hidden_tri_mul=c_hidden_mul, no_blocks=no_blocks,
                                     no_heads=no_heads, transition_type="swiglu",
                                     pair_transition_n=transition_n, dropout_rate=0.0,
                                     tri_mul_first=true, fuse_projection_weights=false,
                                     blocks_per_ckpt=nothing, ckpt_per_template=false, inf=1e9)
            py.to(py_dtype(T))
            sync_template_pair_stack!(py, ps)

            t = randn(rng, T, c_t, N, N, Tpl, B)
            out_jl, _ = jl(t, mask, ps, st)

            t_py = to_py5(t)
            mask_py = isnothing(mask) ? ones(T, B, Tpl, N, N) : permutedims(T.(mask), (4, 3, 1, 2))
            out_py = py(t_py, to_py(mask_py; swap_batch_dim=false))

            atol = T == Float64 ? 1e-5 : 5f-3
            @test isapprox(out_jl, to_jl5(out_py); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        rng = Random.Xoshiro(42)
        N, Tpl, B = 12, 2, 1
        c_t, c_hidden_mul, c_hidden_att, no_heads, transition_n, no_blocks = 32, 16, 16, 2, 2, 2
        jl = TemplatePairStack(; chn_t=c_t, chn_hidden_mul=c_hidden_mul,
                               chn_hidden_pair_att=c_hidden_att, no_heads, transition_n, no_blocks)
        ps, st = Lux.setup(rng, jl) |> convert_types(Float32)
        t = randn(rng, Float32, c_t, N, N, Tpl, B)
        @test_nowarn @inferred jl(t, nothing, ps, st)
    end
end
