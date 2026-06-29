const _PyAtomizeBench = pyimport("openfold3.core.utils.atomize_utils")

println("\n", "="^60)
println("atomize_utils Benchmarks")
println("="^60)

for (N_token, N_atom, B) in [(8, 64, 4), (64, 512, 4), (256, 2048, 1)]
    rng   = Random.Xoshiro(42)
    T     = Float32
    apt   = N_atom ÷ N_token

    token_mask          = trues(N_token, B)
    num_atoms_per_token = fill(apt, N_token, B)
    atom_to_token_index = repeat(reduce(vcat, [fill(t, apt) for t in 1:N_token]), 1, B)
    atom_mask           = trues(N_atom, B)
    C                   = 16

    t_feat   = randn(rng, T, C, N_token, B)
    a_feat   = randn(rng, T, C, N_atom,  B)

    # Python layouts: (B, N, C) for feature tensors, (B, N) for masks
    t_mask_py = to_py(Float32.(token_mask); swap_batch_dim=true)
    n_apt_py  = to_py(Float32.(num_atoms_per_token); swap_batch_dim=true)
    a_mask_py = to_py(Float32.(atom_mask); swap_batch_dim=true)
    ati_py    = to_py(Float32.(atom_to_token_index .- 1); swap_batch_dim=true)
    t_feat_py = to_py(permutedims(t_feat, (3, 2, 1)))   # (B, N_token, C)
    a_feat_py = to_py(permutedims(a_feat, (3, 2, 1)))   # (B, N_atom,  C)

    println("\n--- broadcast_token_feat_to_atoms  N_token=$N_token  N_atom=$N_atom  B=$B ---")
    broadcast_token_feat_to_atoms(token_mask, num_atoms_per_token, t_feat)   # compile
    _PyAtomizeBench.broadcast_token_feat_to_atoms(t_mask_py, n_apt_py, t_feat_py, -2)

    jl_trial = @benchmark broadcast_token_feat_to_atoms($token_mask, $num_atoms_per_token, $t_feat)
    py_trial = @benchmark $_PyAtomizeBench.broadcast_token_feat_to_atoms($t_mask_py, $n_apt_py, $t_feat_py, -2)
    println("Julia: $(median(jl_trial))")
    println("Python: $(median(py_trial))")
    display(judge(median(jl_trial), median(py_trial)))

    println("\n--- aggregate_atom_feat_to_tokens  N_token=$N_token  N_atom=$N_atom  B=$B ---")
    aggregate_atom_feat_to_tokens(token_mask, atom_to_token_index, atom_mask, a_feat)
    _PyAtomizeBench.aggregate_atom_feat_to_tokens(t_mask_py, ati_py, a_mask_py, a_feat_py, -2)

    jl_trial = @benchmark aggregate_atom_feat_to_tokens($token_mask, $atom_to_token_index, $atom_mask, $a_feat)
    py_trial = @benchmark $_PyAtomizeBench.aggregate_atom_feat_to_tokens($t_mask_py, $ati_py, $a_mask_py, $a_feat_py, -2)
    println("Julia: $(median(jl_trial))")
    println("Python: $(median(py_trial))")
    display(judge(median(jl_trial), median(py_trial)))
end
