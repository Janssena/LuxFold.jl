const PyComputePLDDT = pyimport("openfold.utils.loss").compute_plddt
const PyComputePAE = pyimport("openfold.utils.loss").compute_predicted_aligned_error
const PyComputeTM = pyimport("openfold.utils.loss").compute_tm


rng = Random.Xoshiro(42)

# ==============================================================================
# compute_plddt
# ==============================================================================

@testset "compute_plddt" begin
    no_bins, N, B = 10, 8, 2

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            logits_jl = randn(rng, T, no_bins, N, B)
            result_jl = AlphaFold2.compute_plddt(logits_jl)

            logits_py = to_py(logits_jl; swap_batch_dim=true)
            result_py = PyComputePLDDT(logits_py)

            @test result_jl ≈ to_jl(result_py; swap_batch_dim=true)
        end
    end
end

# ==============================================================================
# compute_predicted_aligned_error
# ==============================================================================

@testset "compute_predicted_aligned_error" begin
    no_bins, N, B = 10, 8, 2

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            logits_jl = randn(rng, T, no_bins, N, N, B)
            result_jl = AlphaFold2.compute_predicted_aligned_error(logits_jl; max_bin=31, no_bins)

            logits_py = to_py(logits_jl; swap_batch_dim=true)
            result_py = PyComputePAE(logits_py, max_bin=31, no_bins=no_bins)

            py_pae = T.(permutedims(to_jl(result_py["predicted_aligned_error"]; swap_batch_dim=false), (2, 3, 1)))
            @test result_jl.predicted_aligned_error ≈ py_pae
            py_max = to_jl(result_py["max_predicted_aligned_error"]; swap_batch_dim=false)
            @test T(result_jl.max_predicted_aligned_error) ≈ T(first(py_max))
        end
    end
end

# ==============================================================================
# compute_tm (pTM)
# ==============================================================================

@testset "compute_tm" begin
    no_bins, N, B = 10, 8, 2

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            logits_jl = randn(rng, T, no_bins, N, N, B)

            residue_weights = rand(rng, T, N, B) .+ 1.0
            logits_py = to_py(logits_jl; swap_batch_dim=true).to(py_dtype(T))
            rw_py = to_py(residue_weights; swap_batch_dim=true).to(py_dtype(T))

            result_jl = AlphaFold2.compute_tm(
                logits_jl; residue_weights, max_bin=31, no_bins,
            )

            result_py = PyComputeTM(logits_py, rw_py; max_bin=31, no_bins=no_bins)

            @test T(result_jl) ≈ T(first(to_jl(result_py; swap_batch_dim=false))) atol=0.001
        end
    end
end

@testset "compute_tm with interface" begin
    no_bins, N, B = 10, 8, 2

    asym_id = zeros(Int32, N, B)
    asym_id[1:4, :] .= 1
    asym_id[5:8, :] .= 2

    for T in [Float64, Float32]
        @testset "$T" begin
            logits_jl = randn(rng, T, no_bins, N, N, B)
            logits_py = to_py(logits_jl; swap_batch_dim=true).to(py_dtype(T))
            asym_id_py = to_py(asym_id; swap_batch_dim=true).to(py_dtype(T))

            result_jl = AlphaFold2.compute_tm(
                logits_jl; asym_id, interface=true, max_bin=31, no_bins,
            )

            result_py = PyComputeTM(logits_py; asym_id=asym_id_py, max_bin=31, no_bins=no_bins, interface=true)

            @test T(result_jl) ≈ T(first(to_jl(result_py; swap_batch_dim=false))) atol=0.001
        end
    end
end
