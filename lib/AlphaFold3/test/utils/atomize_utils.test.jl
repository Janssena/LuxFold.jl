const PyAtomize = pyimport("openfold3.core.utils.atomize_utils")

rng_az = Random.Xoshiro(42)

@testset "atomize_utils" begin
    N_token, N_atom, B = 8, 64, 2
    apt = N_atom ÷ N_token   # 8 atoms per token, uniform

    token_mask_full     = trues(N_token, B)
    num_atoms_per_token = fill(apt, N_token, B)
    atom_to_token_index = repeat(reduce(vcat, [fill(t, apt) for t in 1:N_token]), 1, B)
    start_atom_index    = repeat(reshape(Int.(1:apt:N_atom), N_token, 1), 1, B)
    atom_mask_rand      = rand(rng_az, Bool, N_atom, B)
    batch = (; token_mask=token_mask_full, num_atoms_per_token, atom_to_token_index,
               start_atom_index, atom_mask=trues(N_atom, B))

    # Helpers: Julia [C,N,B] ↔ Python (B,N,C); Julia [N,B] ↔ Python (B,N)
    _jl3_to_py(x) = to_py(permutedims(x, (3, 2, 1)))
    _py3_to_jl(x) = permutedims(to_jl(x; swap_batch_dim=false), (3, 2, 1))
    _jl2_to_py(x) = to_py(Float32.(x); swap_batch_dim=true)

    t_mask_py = _jl2_to_py(token_mask_full)
    n_apt_py  = _jl2_to_py(Float32.(num_atoms_per_token))
    ati_py    = _jl2_to_py(Float32.(atom_to_token_index .- 1))   # 0-indexed for Python

    # atom_mask configurations: nothing → Python side uses all-true float mask
    am_full_py   = _jl2_to_py(trues(N_atom, B))
    am_rand_py   = _jl2_to_py(atom_mask_rand)
    atom_mask_cfg = (("No mask", nothing, am_full_py), ("Random mask", atom_mask_rand, am_rand_py))

    @testset "broadcast_token_feat_to_atoms" begin
        # token_mask: nothing vs all-true (random token mask would change packed output shape)
        tok_mask_cfg = (("No mask", nothing), ("All-true mask", token_mask_full))

        @testset "packed ($T, $tok_name)" for T in [Float64, Float32],
                (tok_name, tok_mask) in tok_mask_cfg
            @testset "$T, $tok_name" begin
                t_feat = randn(rng_az, T, 4, N_token, B)
                out_jl = broadcast_token_feat_to_atoms(tok_mask, num_atoms_per_token, t_feat)
                out_py = PyAtomize.broadcast_token_feat_to_atoms(
                    t_mask_py, n_apt_py, _jl3_to_py(t_feat), -2)
                @test out_jl ≈ _py3_to_jl(out_py) atol=1e-5
            end
        end

        @testset "padded max_apt=8 ($T, $tok_name)" for T in [Float64, Float32],
                (tok_name, tok_mask) in tok_mask_cfg
            @testset "$T, $tok_name" begin
                t_feat = randn(rng_az, T, 4, N_token, B)
                out_jl = broadcast_token_feat_to_atoms(tok_mask, num_atoms_per_token, t_feat;
                                                       max_num_atoms_per_token=8)
                out_py = PyAtomize.broadcast_token_feat_to_atoms(
                    t_mask_py, n_apt_py, _jl3_to_py(t_feat), -2, 8)
                @test out_jl ≈ _py3_to_jl(out_py) atol=1e-5
            end
        end
    end

    @testset "aggregate_atom_feat_to_tokens" begin
        for agg_fn in ["mean", "sum"], (am_name, am_jl, am_py) in atom_mask_cfg
            @testset "$agg_fn, $am_name" begin
                for T in [Float64, Float32]
                    @testset "$T" begin
                        a_feat = randn(rng_az, T, 4, N_atom, B)
                        out_jl = aggregate_atom_feat_to_tokens(
                            token_mask_full, atom_to_token_index, am_jl, a_feat;
                            aggregate_fn=Static.static(Symbol(agg_fn)))
                        out_py = PyAtomize.aggregate_atom_feat_to_tokens(
                            t_mask_py, ati_py, am_py, _jl3_to_py(a_feat), -2, agg_fn)
                        @test out_jl ≈ _py3_to_jl(out_py) atol=1e-5
                    end
                end
            end
        end
    end

    @testset "get_token_representative_atoms" begin
        for (am_name, am_jl, _) in atom_mask_cfg
            am_concrete = isnothing(am_jl) ? trues(N_atom, B) : am_jl
            @testset "$am_name" begin
                for T in [Float64, Float32]
                    @testset "$T" begin
                        x = randn(rng_az, T, 4, N_atom, B)
                        rep_x, rep_am = get_token_representative_atoms(batch, x, am_jl)
                        @test size(rep_x) == (4, N_token, B)
                        for t in 1:N_token, b in 1:B
                            idx = start_atom_index[t, b]
                            @test rep_x[:, t, b] ≈ x[:, idx, b]
                            @test rep_am[t, b] == am_concrete[idx, b]
                        end
                    end
                end
            end
        end
    end

    @testset "max_atom_per_token_masked_select" begin
        max_apt = 10
        feat    = randn(rng_az, Float32, 4, N_token * max_apt, B)

        # All valid: compact output = input
        out_all = max_atom_per_token_masked_select(feat, trues(N_token * max_apt, B))
        @test size(out_all) == (4, N_token * max_apt, B)

        # Sparse: first apt atoms per token block valid
        mask_s = falses(N_token * max_apt, B)
        for t in 1:N_token
            mask_s[(t-1)*max_apt+1:(t-1)*max_apt+apt, :] .= true
        end
        out_s = max_atom_per_token_masked_select(feat, mask_s)
        @test size(out_s) == (4, N_atom, B)
        for t in 1:N_token
            @test out_s[:, (t-1)*apt+1:t*apt, :] ≈ feat[:, (t-1)*max_apt+1:(t-1)*max_apt+apt, :]
        end
    end
end
