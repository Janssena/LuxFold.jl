const PyBlockUtils = pyimport("openfold3.core.utils.atom_attention_block_utils")
const _torch_bu    = pyimport("torch")

rng_bu = Random.Xoshiro(42)

@testset "atom_attention_block_utils" begin
    N_atom, B      = 32, 2
    n_query, n_key = 4, 8

    atom_mask_rand = rand(rng_bu, Bool, N_atom, B)
    am_full_py     = to_py(ones(Float32, B, N_atom); swap_batch_dim=false)   # (B, N_atom) all-true
    am_rand_py     = to_py(Float32.(atom_mask_rand); swap_batch_dim=true)    # (B, N_atom)

    num_blocks = ceil(Int, N_atom / n_query)
    pad_len    = get_query_block_padding(N_atom, n_query)

    # atom_mask configurations: nothing → Python uses all-true float mask
    atom_mask_cfg = (("No mask", nothing, am_full_py), ("Random mask", atom_mask_rand, am_rand_py))

    # Conversion helpers: Julia [n_key, num_blocks, B] 1-indexed ↔ Python (B, num_blocks, n_key) 0-indexed
    _jl_si_to_py(si) = _torch_bu.tensor(permutedims(Int32.(si) .- Int32(1), (3, 2, 1)))
    _jl_im_to_py(im) = _torch_bu.tensor(permutedims(im, (3, 2, 1)))
    _py_si_to_jl(si) = permutedims(Int.(to_jl(si; swap_batch_dim=false)), (3, 2, 1)) .+ 1
    _py_im_to_jl(im) = permutedims(Bool.(to_jl(im; swap_batch_dim=false)), (3, 2, 1))

    @testset "get_block_indices" begin
        for (am_name, am_jl, am_py) in atom_mask_cfg
            @testset "$am_name" begin
                am_concrete = isnothing(am_jl) ? trues(N_atom, B) : am_jl
                si_jl, im_jl = get_block_indices(am_concrete, n_query, n_key)
                # nothing dispatches the N_atom,B overload — must equal the all-true result
                if isnothing(am_jl)
                    si_none, im_none = get_block_indices(N_atom, B, n_query, n_key)
                    @test si_none == si_jl
                    @test im_none == im_jl
                end
                si_py, im_py = PyBlockUtils.get_block_indices(am_py, n_query, n_key, _torch_bu.device("cpu"))
                @test si_jl == _py_si_to_jl(si_py)
                @test im_jl == _py_im_to_jl(im_py)
            end
        end
    end

    @testset "get_pair_atom_block_mask" begin
        for (am_name, am_jl, am_py) in atom_mask_cfg
            @testset "$am_name" begin
                am_concrete = isnothing(am_jl) ? trues(N_atom, B) : am_jl
                si_jl, im_jl = get_block_indices(am_concrete, n_query, n_key)
                mask_jl = isnothing(am_jl) ?
                    get_pair_atom_block_mask(nothing, N_atom, B, num_blocks, n_query, n_key, si_jl, im_jl) :
                    get_pair_atom_block_mask(am_jl, num_blocks, n_query, n_key, si_jl, im_jl)

                mask_py = PyBlockUtils.get_pair_atom_block_mask(
                    am_py, num_blocks, n_query, n_key, pad_len,
                    _jl_si_to_py(si_jl), _jl_im_to_py(im_jl))
                # Python (B, num_blocks, n_query, n_key) → Julia [n_query, n_key, num_blocks, B]
                mask_py_jl = permutedims(to_jl(mask_py; swap_batch_dim=false), (3, 4, 2, 1))
                @test Bool.(mask_jl) == Bool.(mask_py_jl)
            end
        end
    end

    @testset "convert_single_rep_to_blocks" begin
        for (am_name, am_jl, am_py) in atom_mask_cfg
            @testset "$am_name" begin
                for T in [Float64, Float32]
                    @testset "$T" begin
                        atom_single = randn(rng_bu, T, 4, N_atom, B)
                        atom_single_q_jl, atom_single_k_jl, pm_jl = convert_single_rep_to_blocks(atom_single, n_query, n_key, am_jl)
                        atom_single_q_py, atom_single_k_py, pm_py  = PyBlockUtils.convert_single_rep_to_blocks(
                            to_py(permutedims(atom_single, (3, 2, 1))), n_query, n_key, am_py)
                        # Python (B, num_blocks, n_query/n_key, C) → Julia [C, n_query/n_key, num_blocks, B]
                        atom_single_q_py_jl = permutedims(to_jl(atom_single_q_py; swap_batch_dim=false), (4, 3, 2, 1))
                        atom_single_k_py_jl = permutedims(to_jl(atom_single_k_py; swap_batch_dim=false), (4, 3, 2, 1))
                        pm_py_jl            = permutedims(to_jl(pm_py;            swap_batch_dim=false), (3, 4, 2, 1))
                        @test atom_single_q_jl ≈ T.(atom_single_q_py_jl) atol=1e-5
                        @test atom_single_k_jl ≈ T.(atom_single_k_py_jl) atol=1e-5
                        @test Bool.(pm_jl) == Bool.(pm_py_jl)
                    end
                end
            end
        end
    end

    @testset "convert_pair_rep_to_blocks" begin
        N_token = 8
        apt     = N_atom ÷ N_token
        C_z     = 4
        ati     = repeat(reduce(vcat, [fill(t, apt) for t in 1:N_token]), 1, B)
        ati_py  = to_py(Int64.(ati .- 1); swap_batch_dim=true)   # (B, N_atom) 0-indexed long

        for (am_name, am_jl, am_py) in atom_mask_cfg
            @testset "$am_name" begin
                for T in [Float64, Float32]
                    @testset "$T" begin
                        zij    = randn(rng_bu, T, C_z, N_token, N_token, B)
                        plm_jl = convert_pair_rep_to_blocks(
                            (; atom_to_token_index=ati, atom_mask=am_jl), zij, n_query, n_key)

                        # Python: zij (B, N_token_qi, N_token_ki, C_z) — preserve qi/ki order
                        zij_py   = to_py(permutedims(zij, (4, 2, 3, 1)))
                        py_batch = Dict("atom_to_token_index" => ati_py, "atom_mask" => am_py)
                        plm_py   = PyBlockUtils.convert_pair_rep_to_blocks(py_batch, zij_py, n_query, n_key)
                        # Python (B, num_blocks, n_query, n_key, C_z) → Julia [C_z, n_query, n_key, num_blocks, B]
                        plm_py_jl = permutedims(to_jl(plm_py; swap_batch_dim=false), (5, 3, 4, 2, 1))
                        @test plm_jl ≈ T.(plm_py_jl) atol=1e-5
                    end
                end
            end
        end
    end

    @testset "edge case: N_atom < n_key" begin
        si, im = get_block_indices(trues(4, B), n_query, n_key)
        @test all(1 .<= si .<= 4)
        @test any(im)   # some positions invalid when N_atom < n_key
    end
end
