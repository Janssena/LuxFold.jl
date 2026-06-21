# feats.test.jl — parity + type-stability tests for pseudo_beta_fn,
#                  atom14_to_atom37, and build_extra_msa_feat
#
# Reference: openfold/utils/feats.py

const PyFeats_ft    = pyimport("openfold.utils.feats")
const torch_ft      = pyimport("torch")
const nn_functional = pyimport("torch.nn.functional")

N_ft, B_ft, S_extra_ft = 12, 2, 16
rng_ft = Random.Xoshiro(42)

# ==============================================================================
# §1  pseudo_beta_fn
# ==============================================================================

@testset "pseudo_beta_fn" begin
    @testset "Python parity" begin
        for T in (Float64, Float32)
            @testset "$T (no mask)" begin
                rng2     = Random.Xoshiro(42)
                aatype   = rand(rng2, 1:20, N_ft, B_ft)
                positions = randn(rng2, T, 3, 37, N_ft, B_ft)

                pb_jl = pseudo_beta_fn(aatype, positions)

                # Python: aatype [B, N] (0-based), positions [B, N, 37, 3]
                aatype_py   = to_py(permutedims(Int64.(aatype .- 1), (2, 1)); swap_batch_dim=false)
                pos_py      = to_py(permutedims(Float32.(positions), (4, 3, 2, 1)); swap_batch_dim=false)
                # Python returns a single tensor (not a tuple) when mask=nothing
                pb_py = PyFeats_ft.pseudo_beta_fn(aatype_py, pos_py, nothing)

                # Python [B, N, 3] → Julia [3, N, B]
                pb_ref = permutedims(to_jl(pb_py; swap_batch_dim=false), (3, 2, 1))
                @test isapprox(Float32.(pb_jl), Float32.(pb_ref); atol=1f-5)
            end

            @testset "$T (with mask)" begin
                rng2     = Random.Xoshiro(42)
                aatype   = rand(rng2, 1:20, N_ft, B_ft)
                positions = randn(rng2, T, 3, 37, N_ft, B_ft)
                masks     = rand(rng2, Bool, 37, N_ft, B_ft)

                result_jl = pseudo_beta_fn(aatype, positions, masks)

                aatype_py = to_py(permutedims(Int64.(aatype .- 1), (2, 1)); swap_batch_dim=false)
                pos_py    = to_py(permutedims(Float32.(positions), (4, 3, 2, 1)); swap_batch_dim=false)
                masks_py  = to_py(permutedims(Float32.(masks), (3, 2, 1)); swap_batch_dim=false)

                pb_py, pbm_py = PyFeats_ft.pseudo_beta_fn(aatype_py, pos_py, masks_py)

                pb_ref  = permutedims(to_jl(pb_py;  swap_batch_dim=false), (3, 2, 1))
                pbm_ref = permutedims(to_jl(pbm_py; swap_batch_dim=false), (2, 1))

                @test isapprox(Float32.(result_jl.pseudo_beta),      Float32.(pb_ref);  atol=1f-5)
                @test isapprox(Float32.(result_jl.pseudo_beta_mask),  Float32.(pbm_ref); atol=1f-5)
            end
        end
    end

    @testset "type stability" begin
        aatype    = rand(rng_ft, 1:20, N_ft, B_ft)
        positions = randn(rng_ft, Float64, 3, 37, N_ft, B_ft)
        masks     = rand(rng_ft, Bool, 37, N_ft, B_ft)
        @test_nowarn @inferred pseudo_beta_fn(aatype, positions)
        @test_nowarn @inferred pseudo_beta_fn(aatype, positions, masks)
    end

    @testset "glycine uses Cα" begin
        # All-Gly: pseudo_beta should equal Cα positions exactly
        gly_idx = restype_order["G"]
        ca_idx  = atom_order["CA"] + 1
        aatype    = fill(gly_idx, N_ft, B_ft)
        positions = randn(rng_ft, Float32, 3, 37, N_ft, B_ft)
        pb = pseudo_beta_fn(aatype, positions)
        @test pb ≈ positions[:, ca_idx, :, :]
    end
end

# ==============================================================================
# §2  atom14_to_atom37
# ==============================================================================

@testset "atom14_to_atom37" begin
    @testset "Python parity" begin
        for T in (Float64, Float32)
            @testset "$T" begin
                rng2   = Random.Xoshiro(42)
                aatype = rand(rng2, 1:20, N_ft, B_ft)

                # Build residx and exists from aatype (convenience overload path)
                idx_perm  = permutedims(restype_atom37_to_atom14, (2, 1))   # [37, 21]
                mask_perm = permutedims(restype_atom37_mask, (2, 1))         # [37, 21] Bool
                residx    = idx_perm[:, aatype]    # [37, N, B] 1-based
                exists    = mask_perm[:, aatype]   # [37, N, B] Bool

                atom14 = randn(rng2, T, 3, 14, N_ft, B_ft)
                result_jl = atom14_to_atom37(atom14, residx, exists)

                # Python: atom14 [B, N, 14, 3], residx [B, N, 37] (0-based), exists [B, N, 37]
                atom14_py  = to_py(permutedims(Float32.(atom14), (4, 3, 2, 1)); swap_batch_dim=false)
                residx_py  = to_py(permutedims(Int64.(residx .- 1), (3, 2, 1)); swap_batch_dim=false)
                exists_py  = to_py(permutedims(Float32.(exists), (3, 2, 1)); swap_batch_dim=false)

                batch_py = Dict("residx_atom37_to_atom14" => residx_py,
                                "atom37_atom_exists"      => exists_py)
                result_py = PyFeats_ft.atom14_to_atom37(atom14_py, batch_py)

                # Python [B, N, 37, 3] → Julia [3, 37, N, B]
                ref = permutedims(to_jl(result_py; swap_batch_dim=false), (4, 3, 2, 1))
                @test isapprox(Float32.(result_jl), Float32.(ref); atol=1f-5)
            end
        end
    end

    @testset "type stability" begin
        rng2   = Random.Xoshiro(42)
        aatype = rand(rng2, 1:20, N_ft, B_ft)
        idx_perm  = permutedims(restype_atom37_to_atom14, (2, 1))
        mask_perm = permutedims(restype_atom37_mask, (2, 1))   # Bool
        residx = idx_perm[:, aatype]
        exists = mask_perm[:, aatype]   # [37, N, B] Bool
        atom14 = randn(rng2, Float64, 3, 14, N_ft, B_ft)
        @test_nowarn @inferred atom14_to_atom37(atom14, residx, exists)
    end

    @testset "masked atoms are zero" begin
        rng2   = Random.Xoshiro(42)
        atom14 = randn(rng2, Float32, 3, 14, N_ft, B_ft)
        residx = ones(Int32, 37, N_ft, B_ft)   # all map to atom14 slot 1
        exists = trues(37, N_ft, B_ft)
        exists[5, :, :] .= false               # slot 5 masked out
        result = atom14_to_atom37(atom14, residx, exists)
        @test all(iszero, result[:, 5, :, :])
        @test any(!iszero, result[:, 1, :, :])
    end

    @testset "convenience overload (aatype)" begin
        rng2   = Random.Xoshiro(42)
        aatype = rand(rng2, 1:20, N_ft, B_ft)
        atom14 = randn(rng2, Float32, 3, 14, N_ft, B_ft)
        result = atom14_to_atom37(atom14, aatype)
        @test size(result) == (3, 37, N_ft, B_ft)
        @test all(isfinite, result)
    end
end

# ==============================================================================
# §3  build_extra_msa_feat
# ==============================================================================

@testset "build_extra_msa_feat" begin
    @testset "Python parity" begin
        for T in (Float64, Float32)
            @testset "$T" begin
                rng2     = Random.Xoshiro(42)
                extra_msa    = rand(rng2, 0:22, N_ft, S_extra_ft, B_ft)
                del_val      = T.(rand(rng2, Float32, N_ft, S_extra_ft, B_ft))
                has_del      = rand(rng2, Bool, N_ft, S_extra_ft, B_ft)

                result_jl = build_extra_msa_feat(extra_msa, del_val, has_del)

                # Python: extra_msa [B, S_extra, N], has_del [B, S_extra, N], del_val [B, S_extra, N]
                # Output: [B, S_extra, N, 25]
                extra_msa_py = to_py(permutedims(Int64.(extra_msa), (3, 2, 1)); swap_batch_dim=false)
                has_del_py   = to_py(permutedims(Float32.(has_del), (3, 2, 1)); swap_batch_dim=false)
                del_val_py   = to_py(permutedims(Float32.(del_val), (3, 2, 1)); swap_batch_dim=false)

                batch_py = Dict("extra_msa"            => extra_msa_py,
                                "extra_has_deletion"   => has_del_py,
                                "extra_deletion_value" => del_val_py)
                result_py = PyFeats_ft.build_extra_msa_feat(batch_py)

                # Python [B, S_extra, N, 25] → Julia [25, N, S_extra, B]
                ref = permutedims(to_jl(result_py; swap_batch_dim=false), (4, 3, 2, 1))
                @test isapprox(Float32.(result_jl), Float32.(ref); atol=1f-5)
            end
        end
    end

    @testset "type stability" begin
        rng2      = Random.Xoshiro(42)
        extra_msa = rand(rng2, 0:22, N_ft, S_extra_ft, B_ft)
        del_val   = randn(rng2, Float64, N_ft, S_extra_ft, B_ft)
        has_del   = rand(rng2, Bool, N_ft, S_extra_ft, B_ft)
        @test_nowarn @inferred build_extra_msa_feat(extra_msa, del_val, has_del)
    end

    @testset "output shape" begin
        rng2      = Random.Xoshiro(42)
        extra_msa = rand(rng2, 0:22, N_ft, S_extra_ft, B_ft)
        del_val   = randn(rng2, Float32, N_ft, S_extra_ft, B_ft)
        has_del   = rand(rng2, Bool, N_ft, S_extra_ft, B_ft)
        result    = build_extra_msa_feat(extra_msa, del_val, has_del)
        @test size(result) == (25, N_ft, S_extra_ft, B_ft)
    end
end
