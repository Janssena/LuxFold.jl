const _PyFeats = pyimport("openfold.utils.feats")

# Python wrappers — include to_py conversion (realistic production cost)

function _py_dgram_from_positions(pos)
    _PyFeats.dgram_from_positions(to_py(pos; swap_batch_dim=true), 3.25, 50.75, 39, 1e8)
end

function _py_build_template_angle_feat(aatype, tsc, atsc, tmask)
    batch = pyimport("builtins").dict()
    batch["template_aatype"]                      = to_py(aatype; swap_batch_dim=true).long()
    batch["template_torsion_angles_sin_cos"]      = to_py(permutedims(tsc,   (4, 3, 2, 1)))
    batch["template_alt_torsion_angles_sin_cos"]  = to_py(permutedims(atsc,  (4, 3, 2, 1)))
    batch["template_torsion_angles_mask"]         = to_py(permutedims(tmask, (3, 2, 1)))
    _PyFeats.build_template_angle_feat(batch)
end

function _py_build_template_pair_feat(pos, mask, aatype, all_pos, all_mask)
    batch = pyimport("builtins").dict()
    batch["template_pseudo_beta"]       = to_py(pos;  swap_batch_dim=true)
    batch["template_pseudo_beta_mask"]  = to_py(mask; swap_batch_dim=true)
    batch["template_aatype"]            = to_py(aatype; swap_batch_dim=true)
    batch["template_all_atom_positions"] = to_py(permutedims(all_pos,  (4, 3, 1, 2)))
    batch["template_all_atom_mask"]      = to_py(permutedims(all_mask, (3, 2, 1)))
    _PyFeats.build_template_pair_feat(batch, 3.25, 50.75, 39, false, 1e-20, 1e8)
end

println("\n", "="^60)
println("Utils Benchmarks")
println("="^60)

for (N, B) in [(64, 4), (128, 4), (256, 4)]
    rng = Random.Xoshiro(42)
    T   = Float32

    pos     = randn(rng, T, 3, N, B) .* T(10)
    mask    = rand(rng, T, N, B) .> T(0.3)
    aatype  = rand(rng, 0:21, N, B)
    all_pos  = randn(rng, T, 37, 3, N, B) .* T(10)
    all_mask = rand(rng, T, 37, N, B) .> T(0.3)

    tsc   = randn(rng, T, 2, 7, N, B)
    atsc  = randn(rng, T, 2, 7, N, B)
    tmask = rand(rng, T, 7, N, B) .> T(0.3)

    println("\n--- dgram_from_positions  N=$N  B=$B ---")
    dgram_from_positions(pos)                   # compile
    _py_dgram_from_positions(pos)               # compile

    jl_trial = @benchmark dgram_from_positions($pos)
    py_trial = @benchmark _py_dgram_from_positions($pos)

    # println("Julia:"); display(jl_trial)
    # println("Python:"); display(py_trial)
    println("Julia: $(median(jl_trial))")
    println("Python $(median(py_trial))")
    display(judge(median(jl_trial), median(py_trial)))

    println("\n--- build_template_angle_feat  N=$N  B=$B ---")
    build_template_angle_feat(aatype, tsc, atsc, tmask)             # compile
    _py_build_template_angle_feat(aatype, tsc, atsc, tmask)         # compile

    jl_trial = @benchmark build_template_angle_feat($aatype, $tsc, $atsc, $tmask)
    py_trial = @benchmark _py_build_template_angle_feat($aatype, $tsc, $atsc, $tmask)

    # println("Julia:"); display(jl_trial)
    # println("Python:"); display(py_trial)
    println("Julia: $(median(jl_trial))")
    println("Python $(median(py_trial))")
    display(judge(median(jl_trial), median(py_trial)))

    println("\n--- build_template_pair_feat  N=$N  B=$B ---")
    build_template_pair_feat(pos, mask, aatype, all_pos, all_mask)             # compile
    _py_build_template_pair_feat(pos, mask, aatype, all_pos, all_mask)         # compile

    jl_trial = @benchmark build_template_pair_feat($pos, $mask, $aatype, $all_pos, $all_mask)
    py_trial = @benchmark _py_build_template_pair_feat($pos, $mask, $aatype, $all_pos, $all_mask)

    # println("Julia:"); display(jl_trial)
    # println("Python:"); display(py_trial)
    println("Julia: $(median(jl_trial))")
    println("Python $(median(py_trial))")
    display(judge(median(jl_trial), median(py_trial)))
end
