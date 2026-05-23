const _PyRigid = pyimport("openfold.utils.rigid_utils").Rigid

# Python wrappers — include to_py conversion (realistic production cost)

function _py_make_transform_from_reference(n_xyz, ca_xyz, c_xyz)
    _PyRigid.make_transform_from_reference(
        to_py(n_xyz;  swap_batch_dim=true),
        to_py(ca_xyz; swap_batch_dim=true),
        to_py(c_xyz;  swap_batch_dim=true),
    )
end

function _py_invert_apply(py_rigid, global_pts)
    py_rigid.invert_apply(to_py(global_pts; swap_batch_dim=true))
end

println("\n", "="^60)
println("Geometry Benchmarks")
println("="^60)

for (N, B) in [(64, 4), (128, 4), (256, 4)]
    rng = Random.Xoshiro(42)
    T   = Float32

    n_xyz  = randn(rng, T, 3, N, B)
    ca_xyz = randn(rng, T, 3, N, B)
    c_xyz  = randn(rng, T, 3, N, B)

    rot, trans = make_transform_from_reference(n_xyz, ca_xyz, c_xyz)
    local_pts  = randn(rng, T, 3, N, B)
    global_pts = dropdims(Lux.batched_matmul(rot, reshape(local_pts, 3, 1, N, B)); dims=2) .+ trans

    # Pre-build Python Rigid once so invert_apply benchmark excludes construction cost
    py_rigid = _py_make_transform_from_reference(n_xyz, ca_xyz, c_xyz)

    println("\n--- make_transform_from_reference  N=$N  B=$B ---")
    make_transform_from_reference(n_xyz, ca_xyz, c_xyz)          # compile
    _py_make_transform_from_reference(n_xyz, ca_xyz, c_xyz)      # compile

    jl_trial = @benchmark make_transform_from_reference($n_xyz, $ca_xyz, $c_xyz)
    py_trial = @benchmark _py_make_transform_from_reference($n_xyz, $ca_xyz, $c_xyz)

    # println("Julia:"); display(jl_trial)
    # println("Python:"); display(py_trial)
    println("Julia: $(median(jl_trial))")
    println("Python $(median(py_trial))")
    display(judge(median(jl_trial), median(py_trial)))

    println("\n--- invert_apply  N=$N  B=$B ---")
    invert_apply(rot, trans, global_pts)                          # compile
    _py_invert_apply(py_rigid, global_pts)                       # compile

    jl_trial = @benchmark invert_apply($rot, $trans, $global_pts)
    py_trial = @benchmark _py_invert_apply($py_rigid, $global_pts)

    # println("Julia:"); display(jl_trial)
    # println("Python:"); display(py_trial)
    println("Julia: $(median(jl_trial))")
    println("Python $(median(py_trial))")
    display(judge(median(jl_trial), median(py_trial)))
end
