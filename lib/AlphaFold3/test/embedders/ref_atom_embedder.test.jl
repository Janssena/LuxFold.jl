const _saa = pyimport("openfold3.core.model.layers.sequence_local_atom_attention")
const PyRefAtomFeatureEmbedder = _saa.RefAtomFeatureEmbedder
const _mlc = pyimport("ml_collections")

rng = Random.Xoshiro(42)

@testset "RefAtomFeatureEmbedder" begin
    @testset "Python parity" begin
        N, B = 16, 2
        n_query, n_key = 4, 8
        chn_atom, chn_atom_pair = 8, 4
        chn_ref_element, chn_ref_name_chars = 128, 256

        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                jl_layer = RefAtomFeatureEmbedder(chn_atom, chn_atom_pair; chn_ref_element, chn_ref_name_chars)
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                c_atom_ref = _mlc.ConfigDict(Dict("element" => chn_ref_element, "name_chars" => chn_ref_name_chars))
                py_layer = PyRefAtomFeatureEmbedder(c_atom_ref, chn_atom, chn_atom_pair)

                sync_dense!(py_layer.linear_ref_pos,        jl_ps.linear_ref_pos)
                sync_dense!(py_layer.linear_ref_charge,     jl_ps.linear_ref_charge)
                sync_dense!(py_layer.linear_ref_mask,       jl_ps.linear_ref_mask)
                sync_dense!(py_layer.linear_ref_element,    jl_ps.linear_ref_element)
                sync_dense!(py_layer.linear_ref_atom_chars, jl_ps.linear_ref_atom_chars)
                sync_dense!(py_layer.linear_ref_offset,     jl_ps.linear_ref_offset)
                sync_dense!(py_layer.linear_inv_sq_dists,   jl_ps.linear_inv_sq_dists)
                sync_dense!(py_layer.linear_valid_mask,     jl_ps.linear_valid_mask)

                # Inputs (Julia channel-first, batch-last)
                ref_pos       = randn(rng, T, 3, N, B)
                ref_charge    = randn(rng, T, 1, N, B)
                ref_mask      = rand(rng, Bool, 1, N, B)
                ref_element   = randn(rng, T, chn_ref_element, N, B)
                ref_chars     = randn(rng, T, chn_ref_name_chars, N, B)
                ref_space_uid = rand(rng, 1:5, 1, N, B)
                atom_mask     = trues(N, B)

                batch_jl = (; ref_pos, ref_charge, ref_mask, ref_element,
                            ref_atom_name_chars=ref_chars, ref_space_uid, atom_mask)

                out_jl, _ = jl_layer(batch_jl, n_query, n_key, jl_ps, jl_st)
                atom_cond_jl, atom_pair_jl = out_jl.atom_cond, out_jl.atom_pair

                # Python batch (batch-first, channel-last)
                batch_py = Dict(
                    "ref_pos"             => to_py(ref_pos; swap_batch_dim=true),
                    "ref_charge"          => to_py(ref_charge; swap_batch_dim=true).squeeze(-1),
                    "ref_mask"            => to_py(ref_mask; swap_batch_dim=true).squeeze(-1),
                    "ref_element"         => to_py(ref_element; swap_batch_dim=true),
                    "ref_atom_name_chars" => to_py(ref_chars; swap_batch_dim=true).reshape(B, N, 4, 64),
                    "ref_space_uid"       => to_py(ref_space_uid; swap_batch_dim=true).squeeze(-1),
                    "atom_mask"           => to_py(atom_mask; swap_batch_dim=true),
                )
                cl_py, plm_py = py_layer(batch_py, n_query, n_key)

                @testset "Single (atom_cond) parity" begin
                    @test atom_cond_jl ≈ to_jl(cl_py; swap_batch_dim=true)
                end
                @testset "Pair (atom_pair) parity" begin
                    @test atom_pair_jl ≈ permutedims(to_jl(plm_py; swap_batch_dim=true), (1, 3, 4, 2, 5))
                end
                @testset "Type-stability" begin
                    @test_nowarn @inferred jl_layer(batch_jl, n_query, n_key, jl_ps, jl_st)
                end
            end
        end
    end
end
