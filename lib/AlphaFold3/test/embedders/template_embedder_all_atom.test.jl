const PyTemplateEmbedderAllAtom =
    pyimport("openfold3.core.model.latent.template_module").TemplateEmbedderAllAtom
const _mlc_templ = pyimport("ml_collections")

using AlphaFold3: TemplateEmbedderAllAtom

@testset "TemplateEmbedderAllAtom" begin
    rng = Random.Xoshiro(42)
    N, Tpl, B = 8, 2, 1
    c_z, c_t = 16, 12
    no_blocks, no_heads = 2, 2
    c_hidden_tri_mul, c_hidden_tri_att, transition_n = 8, 8, 2

    _py_config() = _mlc_templ.ConfigDict(Dict(
        "c_t" => c_t, "c_z" => c_z,
        "template_pair_embedder" => Dict(
            "c_in" => c_z, "c_dgram" => 39, "c_aatype" => 32, "c_out" => c_t),
        "template_pair_stack" => Dict(
            "c_t" => c_t, "c_hidden_tri_att" => c_hidden_tri_att,
            "c_hidden_tri_mul" => c_hidden_tri_mul, "no_blocks" => no_blocks,
            "no_heads" => no_heads, "transition_type" => "swiglu",
            "pair_transition_n" => transition_n, "dropout_rate" => 0.0,
            "tri_mul_first" => true, "fuse_projection_weights" => false,
            "blocks_per_ckpt" => nothing, "ckpt_per_template" => false, "inf" => 1e9),
    ))

    rand_mask = rand(rng, Bool, N, N, B); rand_mask[1, 1, :] .= true
    mask_cfg = (("No mask", nothing), ("Random mask", rand_mask))

    for T in [Float64, Float32], (mask_name, mask) in mask_cfg
        @testset "$T, $mask_name" begin
            jl = TemplateEmbedderAllAtom(; chn_in=c_z, chn_t=c_t, chn_z=c_z, no_blocks, no_heads,
                                         chn_hidden_tri_mul=c_hidden_tri_mul,
                                         chn_hidden_tri_att=c_hidden_tri_att, transition_n)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            py = PyTemplateEmbedderAllAtom(_py_config())
            py.to(py_dtype(T))
            sync_template_embedder!(py, ps)

            batch_jl, batch_py = template_batch(rng, T, N, Tpl, B)
            z = randn(rng, T, c_z, N, N, B)

            z_jl, _ = jl(batch_jl, z, mask, ps, st)

            mask_py = isnothing(mask) ? ones(T, B, N, N) : permutedims(T.(mask), (3, 1, 2))
            z_py = py(batch=batch_py, z=to_py(z; swap_batch_dim=true),
                      pair_mask=to_py(mask_py; swap_batch_dim=false), chunk_size=nothing)

            atol = T == Float64 ? 1e-5 : 1f-2
            @test isapprox(z_jl, to_jl(z_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = TemplateEmbedderAllAtom(; chn_in=c_z, chn_t=c_t, chn_z=c_z, no_blocks, no_heads,
                                     chn_hidden_tri_mul=c_hidden_tri_mul,
                                     chn_hidden_tri_att=c_hidden_tri_att, transition_n)
        ps, st = Lux.setup(rng, jl) |> convert_types(Float32)
        batch_jl, _ = template_batch(rng, Float32, N, Tpl, B)
        z = randn(rng, Float32, c_z, N, N, B)
        @test_nowarn @inferred jl(batch_jl, z, nothing, ps, st)
    end
end
