const _PyBlockBench = pyimport("openfold3.core.utils.atom_attention_block_utils")
const _torch_bench  = pyimport("torch")

println("\n", "="^60)
println("atom_attention_block_utils Benchmarks")
println("="^60)

for (N_atom, n_query, n_key, B) in [(64, 4, 8, 4), (512, 32, 64, 4), (2048, 32, 64, 1)]
    rng = Random.Xoshiro(42)
    T   = Float32

    atom_mask    = trues(N_atom, B)
    atom_mask_py = to_py(Float32.(atom_mask); swap_batch_dim=true)   # [B, N_atom]

    println("\n--- get_block_indices  N_atom=$N_atom n_query=$n_query n_key=$n_key B=$B ---")
    get_block_indices(atom_mask, n_query, n_key)
    _PyBlockBench.get_block_indices(atom_mask_py, n_query, n_key, _torch_bench.device("cpu"))

    jl_trial = @benchmark get_block_indices($atom_mask, $n_query, $n_key)
    py_trial = @benchmark $_PyBlockBench.get_block_indices($atom_mask_py, $n_query, $n_key,
                                                            $_torch_bench.device("cpu"))
    println("Julia: $(median(jl_trial))")
    println("Python: $(median(py_trial))")
    display(judge(median(jl_trial), median(py_trial)))

    # --- convert_single_rep_to_blocks ---
    C  = 16
    atom_single    = randn(rng, T, C, N_atom, B)
    atom_single_py = to_py(permutedims(atom_single, (3, 2, 1)))   # [B, N_atom, C]

    println("\n--- convert_single_rep_to_blocks  N_atom=$N_atom n_query=$n_query n_key=$n_key B=$B ---")
    convert_single_rep_to_blocks(atom_single, n_query, n_key, atom_mask)
    _PyBlockBench.convert_single_rep_to_blocks(atom_single_py, n_query, n_key, atom_mask_py)

    jl_trial = @benchmark convert_single_rep_to_blocks($atom_single, $n_query, $n_key, $atom_mask)
    py_trial = @benchmark $_PyBlockBench.convert_single_rep_to_blocks($atom_single_py, $n_query, $n_key,
                                                                        $atom_mask_py)
    println("Julia: $(median(jl_trial))")
    println("Python: $(median(py_trial))")
    display(judge(median(jl_trial), median(py_trial)))

    # --- convert_pair_rep_to_blocks ---
    N_token = max(1, N_atom ÷ 8)
    apt     = N_atom ÷ N_token
    C_z     = 8
    zij     = randn(rng, T, C_z, N_token, N_token, B)
    ati     = repeat(reduce(vcat, [fill(t, apt) for t in 1:N_token]), 1, B)

    batch_b = (; atom_to_token_index=ati, atom_mask)
    zij_py  = to_py(permutedims(zij, (4, 3, 2, 1)))  # [B, N_token, N_token, C_z]
    py_batch = Dict(
        "atom_to_token_index" => to_py(Float32.(ati .- 1); swap_batch_dim=true),
        "atom_mask"           => atom_mask_py,
    )

    println("\n--- convert_pair_rep_to_blocks  N_atom=$N_atom N_token=$N_token B=$B ---")
    convert_pair_rep_to_blocks(batch_b, zij, n_query, n_key)
    _PyBlockBench.convert_pair_rep_to_blocks(py_batch, zij_py, n_query, n_key)

    jl_trial = @benchmark convert_pair_rep_to_blocks($batch_b, $zij, $n_query, $n_key)
    py_trial = @benchmark $_PyBlockBench.convert_pair_rep_to_blocks($py_batch, $zij_py,
                                                                      $n_query, $n_key)
    println("Julia: $(median(jl_trial))")
    println("Python: $(median(py_trial))")
    display(judge(median(jl_trial), median(py_trial)))
end
