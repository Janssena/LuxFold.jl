"""
Performance benchmark comparing Julia geometry functions against Python reference implementations.
Tests dgram_from_positions, make_transform_from_reference, and build_template_pair_feat.
"""

using AlphaFold2
using Random
using BenchmarkTools
using PythonCall

const PyFeats = pyimport("openfold.utils.feats")
const PyRigid = pyimport("openfold.utils.rigid_utils")
const PyUtils = pyimport("openfold.utils.feats")

struct BenchmarkResult
    name::String
    julia_time::Float64  # in milliseconds
    python_time::Float64  # in milliseconds
    speedup::Float64
end

function to_py(arr; swap_batch_dim=false)
    if swap_batch_dim
        # Move batch dim from last to first
        perm = [ndims(arr), 1:ndims(arr)-1...]
        return PyArray(permutedims(arr, perm))
    else
        return PyArray(arr)
    end
end

function to_jl(py_arr; swap_batch_dim=false)
    arr = Array(py_arr)
    if swap_batch_dim
        # Move batch dim from first to last
        perm = [2:ndims(arr)..., 1]
        return permutedims(arr, perm)
    else
        return arr
    end
end

function benchmark_dgram(sizes::Vector{Tuple{Int,Int}}=[(8, 2), (16, 4), (32, 2)])
    println("\n=== dgram_from_positions Benchmark ===")
    results = BenchmarkResult[]
    rng = Random.Xoshiro(42)

    for (N, B) in sizes
        pos = randn(rng, 3, N, B) .* 10

        # Julia benchmark
        julia_time = @elapsed for _ in 1:100
            _ = dgram_from_positions(pos)
        end
        julia_time /= 100 * 1000  # Convert to ms

        # Python benchmark
        pos_py = to_py(pos; swap_batch_dim=true)
        python_time = @elapsed for _ in 1:100
            _ = PyFeats.dgram_from_positions(pos_py, 3.25, 50.75, 39, 1e8)
        end
        python_time /= 100 * 1000  # Convert to ms

        speedup = python_time / julia_time
        result = BenchmarkResult("dgram_from_positions (N=$N, B=$B)", julia_time, python_time, speedup)
        push!(results, result)

        println("  N=$N, B=$B:")
        println("    Julia:  $(round(julia_time, digits=3)) ms")
        println("    Python: $(round(python_time, digits=3)) ms")
        println("    Speedup: $(round(speedup, digits=2))x")
    end

    return results
end

function benchmark_make_transform(sizes::Vector{Tuple{Int,Int}}=[(8, 2), (16, 4), (32, 2)])
    println("\n=== make_transform_from_reference Benchmark ===")
    results = BenchmarkResult[]
    rng = Random.Xoshiro(42)

    for (N, B) in sizes
        n_xyz = randn(rng, 3, N, B)
        ca_xyz = randn(rng, 3, N, B) .* 10
        c_xyz = randn(rng, 3, N, B) .* 10

        # Julia benchmark
        julia_time = @elapsed for _ in 1:100
            _ = make_transform_from_reference(n_xyz, ca_xyz, c_xyz)
        end
        julia_time /= 100 * 1000

        # Python benchmark (Rigid.make_transform_from_reference)
        n_py = to_py(n_xyz; swap_batch_dim=true)
        ca_py = to_py(ca_xyz; swap_batch_dim=true)
        c_py = to_py(c_xyz; swap_batch_dim=true)

        python_time = @elapsed for _ in 1:100
            rigid = PyRigid.Rigid.make_transform_from_reference(n_py, ca_py, c_py)
        end
        python_time /= 100 * 1000

        speedup = python_time / julia_time
        result = BenchmarkResult("make_transform_from_reference (N=$N, B=$B)", julia_time, python_time, speedup)
        push!(results, result)

        println("  N=$N, B=$B:")
        println("    Julia:  $(round(julia_time, digits=3)) ms")
        println("    Python: $(round(python_time, digits=3)) ms")
        println("    Speedup: $(round(speedup, digits=2))x")
    end

    return results
end

function benchmark_build_template_pair(sizes::Vector{Tuple{Int,Int}}=[(4, 2), (8, 2), (16, 2)])
    println("\n=== build_template_pair_feat Benchmark ===")
    results = BenchmarkResult[]
    rng = Random.Xoshiro(42)

    for (N, B) in sizes
        pseudo_beta = randn(rng, 3, N, B) .* 10
        pseudo_beta_mask = rand(rng, N, B) .> 0.3
        aatype = rand(rng, 0:21, N, B)
        all_atom_positions = randn(rng, 3, 37, N, B) .* 10
        all_atom_mask = rand(rng, 37, N, B) .> 0.3

        # Julia benchmark
        julia_time = @elapsed for _ in 1:10
            _ = build_template_pair_feat(pseudo_beta, pseudo_beta_mask, aatype, all_atom_positions, all_atom_mask)
        end
        julia_time /= 10 * 1000

        # Python benchmark
        pseudo_beta_py = to_py(pseudo_beta; swap_batch_dim=true)
        pseudo_beta_mask_py = to_py(pseudo_beta_mask; swap_batch_dim=true)
        aatype_py = to_py(aatype; swap_batch_dim=true).long()
        all_atom_pos_py = to_py(permutedims(all_atom_positions, (4, 3, 2, 1)))
        all_atom_mask_py = to_py(permutedims(all_atom_mask, (3, 2, 1)))

        batch_dict = PyDict()
        batch_dict["template_pseudo_beta"] = pseudo_beta_py
        batch_dict["template_pseudo_beta_mask"] = pseudo_beta_mask_py
        batch_dict["template_aatype"] = aatype_py
        batch_dict["template_all_atom_positions"] = all_atom_pos_py
        batch_dict["template_all_atom_mask"] = all_atom_mask_py

        python_time = @elapsed for _ in 1:10
            _ = PyUtils.build_template_pair_feat(batch_dict, 3.25, 50.75, 39, false, 1e-20, 1e8)
        end
        python_time /= 10 * 1000

        speedup = python_time / julia_time
        result = BenchmarkResult("build_template_pair_feat (N=$N, B=$B)", julia_time, python_time, speedup)
        push!(results, result)

        println("  N=$N, B=$B:")
        println("    Julia:  $(round(julia_time, digits=3)) ms")
        println("    Python: $(round(python_time, digits=3)) ms")
        println("    Speedup: $(round(speedup, digits=2))x")
    end

    return results
end

function main()
    println("\n" * "="^60)
    println("Geometry Functions Performance Benchmark")
    println("Testing vectorized Julia implementations vs Python reference")
    println("="^60)

    results = BenchmarkResult[]

    try
        append!(results, benchmark_dgram())
    catch e
        println("Warning: dgram benchmark failed: $e")
    end

    try
        append!(results, benchmark_make_transform())
    catch e
        println("Warning: make_transform benchmark failed: $e")
    end

    try
        append!(results, benchmark_build_template_pair())
    catch e
        println("Warning: build_template_pair benchmark failed: $e")
    end

    # Summary
    if !isempty(results)
        println("\n" * "="^60)
        println("Summary")
        println("="^60)
        avg_speedup = mean(r.speedup for r in results)
        min_speedup = minimum(r.speedup for r in results)
        max_speedup = maximum(r.speedup for r in results)
        println("Average speedup: $(round(avg_speedup, digits=2))x")
        println("Min speedup: $(round(min_speedup, digits=2))x")
        println("Max speedup: $(round(max_speedup, digits=2))x")
        println("="^60 * "\n")
    end

    return results
end

# Run benchmarks if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
