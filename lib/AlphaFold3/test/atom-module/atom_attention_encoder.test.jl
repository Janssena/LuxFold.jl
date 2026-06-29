const _saa_enc = pyimport("openfold3.core.model.layers.sequence_local_atom_attention")
const PyAtomAttentionEncoder = _saa_enc.AtomAttentionEncoder
const _mlc = pyimport("ml_collections")

using AlphaFold3: AtomAttentionEncoder

_c_atom_ref() = _mlc.ConfigDict(Dict("element" => _cre, "name_chars" => _crnc))

rng = Random.Xoshiro(42)

@testset "AtomAttentionEncoder (input-embedder path)" begin
    mask_cfg = (("No mask", nothing), ("Random mask", _rand_masks(rng)))
    for T in [Float64, Float32], (mask_name, masks) in mask_cfg
        @testset "$T, $mask_name" begin
            jl_enc = AtomAttentionEncoder(;
                chn_atom=_ca, chn_atom_pair=_cap, chn_token=_ct, chn_hidden=_ch,
                no_heads=_no_heads, no_blocks=_no_blocks, n_transition=_n_trans,
                n_query=_nq, n_key=_nk, add_noisy_pos=false,
                chn_ref_element=_cre, chn_ref_name_chars=_crnc,
            )
            jl_ps, jl_st = Lux.setup(rng, jl_enc) |> convert_types(T)

            py_enc = PyAtomAttentionEncoder(
                c_atom_ref=_c_atom_ref(), c_atom=_ca, c_atom_pair=_cap, c_token=_ct,
                c_hidden=_ch, add_noisy_pos=false, no_heads=_no_heads, no_blocks=_no_blocks,
                n_transition=_n_trans, n_query=_nq, n_key=_nk, use_ada_layer_norm=true,
            )
            sync_atom_attention_encoder!(py_enc, jl_ps; add_noisy_pos=false)

            batch_jl, batch_py = _make_batch(rng, T, masks)
            out_jl, _ = jl_enc(batch_jl, jl_ps, jl_st)
            token_agg_py, atom_single_py, atom_cond_py, atom_pair_py = py_enc(batch=batch_py)

            @test out_jl.atom_cond   ≈ to_jl(atom_cond_py; swap_batch_dim=true)
            @test out_jl.atom_single ≈ to_jl(atom_single_py; swap_batch_dim=true)
            @test out_jl.token_agg   ≈ to_jl(token_agg_py; swap_batch_dim=true)
            @test out_jl.atom_pair   ≈ permutedims(to_jl(atom_pair_py; swap_batch_dim=true), (1, 3, 4, 2, 5))
        end
    end
end

@testset "AtomAttentionEncoder (diffusion path, add_noisy_pos)" begin
    cs, cz = 16, 16
    mask_cfg = (("No mask", nothing), ("Random mask", _rand_masks(rng)))
    for T in [Float64, Float32], (mask_name, masks) in mask_cfg
        @testset "$T, $mask_name" begin
            jl_enc = AtomAttentionEncoder(;
                chn_atom=_ca, chn_atom_pair=_cap, chn_token=_ct, chn_hidden=_ch,
                no_heads=_no_heads, no_blocks=_no_blocks, n_transition=_n_trans,
                n_query=_nq, n_key=_nk, add_noisy_pos=true, chn_single=cs, chn_pair=cz,
                chn_ref_element=_cre, chn_ref_name_chars=_crnc,
            )
            jl_ps, jl_st = Lux.setup(rng, jl_enc) |> convert_types(T)

            py_enc = PyAtomAttentionEncoder(
                c_atom_ref=_c_atom_ref(), c_atom=_ca, c_atom_pair=_cap, c_token=_ct,
                c_hidden=_ch, add_noisy_pos=true, no_heads=_no_heads, no_blocks=_no_blocks,
                n_transition=_n_trans, n_query=_nq, n_key=_nk, use_ada_layer_norm=true,
                c_s=cs, c_z=cz,
            )
            sync_atom_attention_encoder!(py_enc, jl_ps; add_noisy_pos=true)

            batch_jl, batch_py = _make_batch(rng, T, masks)
            atom_pos  = randn(rng, T, 3, _N_atom, _B)
            s_trunk   = randn(rng, T, cs, _N_token, _B)
            z_trunk   = randn(rng, T, cz, _N_token, _N_token, _B)

            out_jl, _ = jl_enc(batch_jl, atom_pos, s_trunk, z_trunk, jl_ps, jl_st)
            token_agg_py, atom_single_py, atom_cond_py, atom_pair_py = py_enc(
                batch=batch_py,
                rl=to_py(atom_pos; swap_batch_dim=true),
                si_trunk=to_py(s_trunk; swap_batch_dim=true),
                zij_trunk=to_py(z_trunk; swap_batch_dim=true),
            )

            @test out_jl.atom_cond   ≈ to_jl(atom_cond_py; swap_batch_dim=true)
            @test out_jl.atom_single ≈ to_jl(atom_single_py; swap_batch_dim=true)
            @test out_jl.token_agg   ≈ to_jl(token_agg_py; swap_batch_dim=true)
            @test out_jl.atom_pair   ≈ permutedims(to_jl(atom_pair_py; swap_batch_dim=true), (1, 3, 4, 2, 5))
        end
    end
end
