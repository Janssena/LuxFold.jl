const PyCrossAttentionPairBias = pyimport("openfold3.core.model.layers.attention_pair_bias").CrossAttentionPairBias

rng = Random.Xoshiro(42)

@testset "SequenceLocalAttentionPairBias" begin
    @testset "Python parity (CrossAttentionPairBias)" begin
        N_atom, B = 16, 2
        n_query, n_key = 4, 8
        N_blocks = cld(N_atom, n_query)
        chn_atom, chn_atom_pair = 8, 4
        head_dim, no_heads = 4, 2

        # Random atom mask, but keep the first atom of each query block valid so every
        # query's centered key-window has ≥1 valid key (avoids all-masked-window NaN in
        # softmax). Masks are Bool in Julia, cast to float at the Python boundary.
        rand_mask = rand(rng, Bool, N_atom, B)
        for b in 1:B, blk in 0:(N_blocks - 1)
            rand_mask[blk * n_query + 1, b] = true
        end
        mask_cfg = (("No mask", nothing), ("Random mask", rand_mask))

        for T in [Float64, Float32, Float16], (mask_name, mask) in mask_cfg
            # Skip Float16 + masking: the Julia layer is Float16-safe (masks with
            # `-floatmax(T)`), but the Python reference (`CrossAttentionPairBias`) masks via
            # `inf*(mask-1)` (inf=1e9) → `Inf16*0 = NaN` for valid keys, so there is no valid
            # reference to compare against. Masked parity is validated at Float64/Float32.
            T === Float16 && !isnothing(mask) && continue
            @testset "$T, $mask_name" begin
                jl_layer = SequenceLocalAttentionPairBias(
                    chn_atom, chn_atom, chn_atom, chn_atom, chn_atom_pair,
                    head_dim, no_heads, n_query, n_key; use_adaln=true,
                )
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyCrossAttentionPairBias(
                    chn_atom, chn_atom, chn_atom, chn_atom, chn_atom_pair,
                    head_dim, no_heads;
                    use_ada_layer_norm=true, n_query=n_query, n_key=n_key,
                )
                sync_af3_cross_attention_pair_bias!(py_layer, jl_ps)

                a         = randn(rng, T, chn_atom, N_atom, B)
                atom_pair = randn(rng, T, chn_atom_pair, n_query, n_key, N_blocks, B)
                cond      = randn(rng, T, chn_atom, N_atom, B)

                y_jl, _ = jl_layer((x=a, z=atom_pair, cond=cond, mask=mask), jl_ps, jl_st)

                mask_py = isnothing(mask) ? nothing : to_py(T.(mask); swap_batch_dim=true)
                y_py = py_layer(
                    to_py(a; swap_batch_dim=true),
                    to_py(permutedims(atom_pair, (1, 4, 2, 3, 5)); swap_batch_dim=true),  # [B, N_blocks, n_query, n_key, C]
                    to_py(cond; swap_batch_dim=true),
                    mask_py,   # Python does inf*(mask-1); None ⇒ no masking
                )

                @testset "parity" begin
                    @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                end
            end
        end
    end
end
