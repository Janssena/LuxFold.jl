const PyDiffusionTransformerBlock = pyimport("openfold3.core.model.layers.diffusion_transformer").DiffusionTransformerBlock

include("../setup/transformer_sync.jl")

using AlphaFold3: DiffusionTransformerBlock

rng = Random.Xoshiro(42)

@testset "DiffusionTransformerBlock (self-attention, conditioned)" begin
    N, B = 10, 2
    chn_a, chn_cond, chn_pair, chn_hidden = 8, 8, 6, 4
    no_heads, n_transition = 2, 2
    inf = 1f9

    rand_mask = rand(rng, Bool, N, B)
    rand_mask[1, :] .= true   # ≥1 valid token per batch (avoid all-masked attention rows)
    mask_cfg = (("No mask", nothing), ("Random mask", rand_mask))

    @testset "Python parity ($T, $mask_name)" for T in [Float64, Float32, Float16],
            (mask_name, mask) in mask_cfg
        jl_block = DiffusionTransformerBlock(;
            chn_a, chn_cond, chn_pair, chn_hidden, no_heads, n_transition,
        )
        jl_ps, jl_st = Lux.setup(rng, jl_block) |> convert_types(T)

        py_block = PyDiffusionTransformerBlock(
            chn_a, chn_cond, chn_pair, chn_hidden, no_heads,
            n_transition, true, nothing, nothing, inf,
        )
        sync_af3_diffusion_transformer_block!(py_block, jl_ps; cross_attention=false)

        a = randn(rng, T, chn_a, N, B)
        s = randn(rng, T, chn_cond, N, B)
        z = randn(rng, T, chn_pair, N, N, B)

        y_jl, _ = jl_block(a, s, z, mask, jl_ps, jl_st)

        mask_py = isnothing(mask) ? nothing : to_py(T.(mask); swap_batch_dim=true)
        y_py = py_block(
            to_py(a; swap_batch_dim=true),
            to_py(s; swap_batch_dim=true),
            to_py(z; swap_batch_dim=true),
            mask_py,
        )

        @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
    end

    @testset "Type stability" begin
        jl_block = DiffusionTransformerBlock(;
            chn_a, chn_cond, chn_pair, chn_hidden, no_heads, n_transition,
        )
        jl_ps, jl_st = Lux.setup(rng, jl_block)
        a = randn(rng, Float32, chn_a, N, B)
        s = randn(rng, Float32, chn_cond, N, B)
        z = randn(rng, Float32, chn_pair, N, N, B)
        @test_nowarn @inferred jl_block(a, s, z, nothing, jl_ps, jl_st)
        @test_nowarn @inferred jl_block(a, s, z, rand(rng, Bool, N, B), jl_ps, jl_st)
    end
end
