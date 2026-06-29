const PyTemplatePairEmbedderAllAtom =
    pyimport("openfold3.core.model.feature_embedders.template_embedders").TemplatePairEmbedderAllAtom

using AlphaFold3: TemplatePairEmbedderAllAtom

@testset "TemplatePairEmbedderAllAtom" begin
    rng = Random.Xoshiro(42)
    N, Tpl, B = 8, 2, 1
    c_z, c_t = 16, 12

    for T in [Float64, Float32]
        @testset "Python parity ($T)" begin
            jl = TemplatePairEmbedderAllAtom(; chn_in=c_z, chn_t=c_t)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            py = PyTemplatePairEmbedderAllAtom(c_in=c_z, c_dgram=39, c_aatype=32, c_out=c_t)
            py.to(py_dtype(T))
            sync_template_pair_embedder!(py, ps)

            batch_jl, batch_py = template_batch(rng, T, N, Tpl, B)
            z = randn(rng, T, c_z, N, N, B)

            t_jl, _ = jl(batch_jl, z, ps, st)
            t_py = py(batch_py, to_py(z; swap_batch_dim=true))

            atol = T == Float64 ? 1e-5 : 1f-3
            @test isapprox(t_jl, to_jl5(t_py); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        rng = Random.Xoshiro(42)
        N, Tpl, B = 8, 2, 1
        c_z, c_t = 16, 12
        jl = TemplatePairEmbedderAllAtom(; chn_in=c_z, chn_t=c_t)
        ps, st = Lux.setup(rng, jl) |> convert_types(Float32)
        batch_jl, _ = template_batch(rng, Float32, N, Tpl, B)
        z = randn(rng, Float32, c_z, N, N, B)
        @test_nowarn @inferred jl(batch_jl, z, ps, st)
    end
end
