const _saa_npe = pyimport("openfold3.core.model.layers.sequence_local_atom_attention")
const PyNoisyPositionEmbedder = _saa_npe.NoisyPositionEmbedder

rng = Random.Xoshiro(42)

@testset "NoisyPositionEmbedder" begin
    @testset "Python parity" begin
        N_token, N_atom, B = 4, 16, 2
        n_query, n_key = 4, 8
        N_blocks = cld(N_atom, n_query)
        chn_single, chn_pair, chn_atom, chn_atom_pair = 16, 16, 8, 4
        atoms_per_token = N_atom ÷ N_token

        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                jl_layer = NoisyPositionEmbedder(chn_single, chn_pair, chn_atom, chn_atom_pair)
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyNoisyPositionEmbedder(chn_single, chn_pair, chn_atom, chn_atom_pair)

                sync_layernorm!(py_layer.layer_norm_s, jl_ps.layer_norm_s)
                sync_dense!(py_layer.linear_s, jl_ps.linear_s)
                sync_layernorm!(py_layer.layer_norm_z, jl_ps.layer_norm_z)
                sync_dense!(py_layer.linear_z, jl_ps.linear_z)
                sync_dense!(py_layer.linear_r, jl_ps.linear_r)

                # batch indexing arrays
                num_atoms_per_token = fill(atoms_per_token, N_token, B)
                atom_to_token_index = repeat(reduce(vcat, [fill(t, atoms_per_token) for t in 1:N_token]), 1, B)  # 1-based
                token_mask = trues(N_token, B)
                atom_mask  = trues(N_atom, B)
                batch_jl = (; token_mask, num_atoms_per_token, atom_to_token_index, atom_mask)

                atom_cond  = randn(rng, T, chn_atom, N_atom, B)
                atom_pair  = randn(rng, T, chn_atom_pair, n_query, n_key, N_blocks, B)
                si_trunk   = randn(rng, T, chn_single, N_token, B)
                zij_trunk  = randn(rng, T, chn_pair, N_token, N_token, B)
                atom_pos   = randn(rng, T, 3, N_atom, B)

                out_jl, _ = jl_layer(batch_jl, atom_cond, atom_pair, si_trunk, zij_trunk, atom_pos, n_query, n_key, jl_ps, jl_st)

                batch_py = Dict(
                    "token_mask"          => to_py(token_mask; swap_batch_dim=true),
                    "num_atoms_per_token" => to_py(num_atoms_per_token; swap_batch_dim=true),
                    "atom_to_token_index" => to_py(atom_to_token_index .- 1; swap_batch_dim=true),  # 0-based
                    "atom_mask"           => to_py(atom_mask; swap_batch_dim=true),
                )
                atom_cond_py, atom_pair_py, atom_single_py = py_layer(
                    batch_py,
                    to_py(atom_cond; swap_batch_dim=true),
                    to_py(permutedims(atom_pair, (1, 4, 2, 3, 5)); swap_batch_dim=true),  # -> [B, N_blocks, n_query, n_key, C]
                    to_py(si_trunk; swap_batch_dim=true),
                    to_py(zij_trunk; swap_batch_dim=true),
                    to_py(atom_pos; swap_batch_dim=true),
                    n_query, n_key,
                )

                @testset "atom_cond parity" begin
                    @test out_jl.atom_cond ≈ to_jl(atom_cond_py; swap_batch_dim=true)
                end
                @testset "atom_pair parity" begin
                    @test out_jl.atom_pair ≈ permutedims(to_jl(atom_pair_py; swap_batch_dim=true), (1, 3, 4, 2, 5))
                end
                @testset "atom_single parity" begin
                    @test out_jl.atom_single ≈ to_jl(atom_single_py; swap_batch_dim=true)
                end
            end
        end
    end
end
