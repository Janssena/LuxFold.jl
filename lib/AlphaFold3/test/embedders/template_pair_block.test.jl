const _py_templ_mod = pyimport("openfold3.core.model.latent.template_module")
const PyTemplatePairBlock = _py_templ_mod.TemplatePairBlock

using AlphaFold3: TemplatePairBlock

@testset "TemplatePairBlock" begin
    rng = Random.Xoshiro(42)
    N, B = 16, 1
    c_t, c_hidden_mul, c_hidden_att, no_heads, transition_n = 64, 32, 16, 4, 2

    rand_mask = rand(rng, Bool, N, N, B); rand_mask[1, 1, :] .= true
    mask_cfg = (("No mask", nothing), ("Random mask", rand_mask))

    for T in [Float64, Float32], (mask_name, mask) in mask_cfg
        @testset "$T, $mask_name" begin
            jl = TemplatePairBlock(; chn_t=c_t, chn_hidden_mul=c_hidden_mul,
                                   chn_hidden_pair_att=c_hidden_att, no_heads, transition_n)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            py = PyTemplatePairBlock(c_t=c_t, c_hidden_tri_mul=c_hidden_mul,
                                     c_hidden_tri_att=c_hidden_att, no_heads=no_heads,
                                     transition_type="swiglu", pair_transition_n=transition_n,
                                     dropout_rate=0.0, tri_mul_first=true,
                                     fuse_projection_weights=false, ckpt_per_template=false, inf=1e9)
            py.to(py_dtype(T))
            sync_template_pair_block!(py, ps)

            z = randn(rng, T, c_t, N, N, B)
            out_jl, _ = jl(z, mask, ps, st)

            t_py = to_py5(reshape(z, c_t, N, N, 1, B))
            mask_py = isnothing(mask) ? ones(T, B, 1, N, N) :
                      reshape(permutedims(T.(mask), (3, 1, 2)), B, 1, N, N)
            out_py = py(t_py, to_py(mask_py; swap_batch_dim=false))
            out_py_jl = dropdims(to_jl5(out_py); dims=4)

            atol = T == Float64 ? 1e-5 : 1f-3
            @test isapprox(out_jl, out_py_jl; atol=atol, rtol=atol)
        end
    end
end
