const _saa_dec = pyimport("openfold3.core.model.layers.sequence_local_atom_attention")
const PyAtomAttentionDecoder = _saa_dec.AtomAttentionDecoder

using AlphaFold3: AtomAttentionDecoder

rng_dec = Random.Xoshiro(42)

@testset "AtomAttentionDecoder" begin
    mask_cfg = (("No mask", nothing), ("Random mask", _rand_masks(rng_dec)))
    for T in [Float64, Float32], (mask_name, masks) in mask_cfg
        @testset "$T, $mask_name" begin
            jl_dec = AtomAttentionDecoder(;
                chn_atom=_ca, chn_atom_pair=_cap, chn_token=_ct, chn_hidden=_ch,
                no_heads=_no_heads, no_blocks=_no_blocks, n_transition=_n_trans,
                n_query=_nq, n_key=_nk,
            )
            jl_ps, jl_st = Lux.setup(rng_dec, jl_dec) |> convert_types(T)

            py_dec = PyAtomAttentionDecoder(
                c_atom=_ca, c_atom_pair=_cap, c_token=_ct, c_hidden=_ch,
                no_heads=_no_heads, no_blocks=_no_blocks, n_transition=_n_trans,
                n_query=_nq, n_key=_nk, use_ada_layer_norm=true,
            )
            sync_atom_attention_decoder!(py_dec, jl_ps)

            batch_jl, batch_py = _make_batch(rng_dec, T, masks)
            token_agg   = randn(rng_dec, T, _ct, _N_token, _B)
            atom_single = randn(rng_dec, T, _ca, _N_atom, _B)
            atom_cond   = randn(rng_dec, T, _ca, _N_atom, _B)
            atom_pair   = randn(rng_dec, T, _cap, _nq, _nk, _N_blocks, _B)

            pos_update_jl, _ = jl_dec(batch_jl, token_agg, atom_single, atom_cond, atom_pair, jl_ps, jl_st)

            pos_update_py = py_dec(
                batch=batch_py,
                ai=to_py(token_agg; swap_batch_dim=true),
                ql=to_py(atom_single; swap_batch_dim=true),
                cl=to_py(atom_cond; swap_batch_dim=true),
                plm=to_py(permutedims(atom_pair, (1, 4, 2, 3, 5)); swap_batch_dim=true),
            )

            @test pos_update_jl ≈ to_jl(pos_update_py; swap_batch_dim=true)
        end
    end
end
