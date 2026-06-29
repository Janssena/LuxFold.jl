const PyDiffusionTransformer = pyimport("openfold3.core.model.layers.diffusion_transformer").DiffusionTransformer

include("../setup/transformer_sync.jl")

using AlphaFold3: DiffusionTransformer

rng = Random.Xoshiro(42)

@testset "DiffusionTransformer (self-attention, conditioned)" begin
    @testset "Python parity" begin
        N, B = 10, 2
        chn_a, chn_cond, chn_pair, chn_hidden = 8, 8, 6, 4
        no_heads, no_blocks, n_transition = 2, 2, 2
        inf = 1f9

        # openfold masks BOTH attention and the transition output (`_mask_trans=True`),
        # so random masks hold parity directly — no Python-side reconciliation. Masks are
        # Bool in Julia, cast to float at the Python boundary.
        rand_mask = rand(rng, Bool, N, B)
        rand_mask[1, :] .= true   # ≥1 valid token per batch (avoid all-masked attention rows)
        mask_cfg = (("No mask", nothing), ("Random mask", rand_mask))

        for T in [Float64, Float32, Float16], (mask_name, mask) in mask_cfg
            @testset "$T, $mask_name" begin
                jl_dt = DiffusionTransformer(;
                    chn_a, chn_cond, chn_pair, chn_hidden,
                    no_heads, no_blocks, n_transition, use_ada_layer_norm=true,
                )
                jl_ps, jl_st = Lux.setup(rng, jl_dt) |> convert_types(T)

                # Python: DiffusionTransformer(c_a, c_s, c_z, c_hidden, no_heads,
                #   no_blocks, n_transition, use_ada_layer_norm, n_query, n_key, inf)
                py_dt = PyDiffusionTransformer(
                    chn_a, chn_cond, chn_pair, chn_hidden, no_heads,
                    no_blocks, n_transition, true, nothing, nothing, inf,
                )
                sync_diffusion_transformer!(py_dt, jl_ps; cross_attention=false)

                a = randn(rng, T, chn_a, N, B)
                s = randn(rng, T, chn_cond, N, B)
                z = randn(rng, T, chn_pair, N, N, B)

                y_jl, _ = jl_dt((; a, s, z, mask), jl_ps, jl_st)

                mask_py = isnothing(mask) ? nothing : to_py(T.(mask); swap_batch_dim=true)
                y_py = py_dt(
                    to_py(a; swap_batch_dim=true),
                    to_py(s; swap_batch_dim=true),
                    to_py(z; swap_batch_dim=true),
                    mask_py,   # Python attention does inf*(mask-1); None ⇒ no masking
                )

                @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
            end
        end
    end
end
