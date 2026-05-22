const PyFeats = pyimport("openfold.utils.feats")
const Rng = Random.Xoshiro(42)
const ATOL = Dict(Float64 => 1e-10, Float32 => 1e-6, Float16 => 1f-2)

function _rand_pair_feat_inputs(rng, T, N, B)
    pos = randn(rng, T, 3, N, B) .* T(10)
    mask = rand(rng, T, N, B) .> T(0.3)
    aatype = rand(rng, 0:21, N, B)
    all_pos = randn(rng, T, 3, 37, N, B) .* T(10)
    all_mask = rand(rng, T, 37, N, B) .> T(0.3)
    return pos, mask, aatype, all_pos, all_mask
end

function _rand_angle_feat_inputs(rng, T, N, B)
    aatype = rand(rng, 0:21, N, B)
    tsc = randn(rng, T, 2, 7, N, B)
    atsc = randn(rng, T, 2, 7, N, B)
    tmask = rand(rng, T, 7, N, B) .> T(0.3)
    return aatype, tsc, atsc, tmask
end

function _to_py_pair_feat(pos, mask, aatype, all_pos, all_mask)
    pos_py = to_py(pos; swap_batch_dim=true)
    mask_py = to_py(mask; swap_batch_dim=true)
    aatype_py = to_py(aatype; swap_batch_dim=true)
    all_pos_py = to_py(permutedims(all_pos, (4, 3, 2, 1)))
    all_mask_py = to_py(permutedims(all_mask, (3, 2, 1)))
    return pos_py, mask_py, aatype_py, all_pos_py, all_mask_py
end

function _to_py_angle_feat(aatype, tsc, atsc, tmask)
    tsc_py = to_py(permutedims(tsc, (4, 3, 2, 1)))
    atsc_py = to_py(permutedims(atsc, (4, 3, 2, 1)))
    tmask_py = to_py(permutedims(tmask, (3, 2, 1)))
    aatype_py = to_py(aatype; swap_batch_dim=true).long()
    return aatype_py, tsc_py, atsc_py, tmask_py
end

function _make_pydict(; kwargs...)
    d = pyimport("builtins").dict()
    for (k, v) in kwargs
        d[k] = v
    end
    return d
end

@testset "Utils" begin
    @testset "dgram_from_positions" begin
        N, B = 8, 2
        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                pos = randn(Rng, T, 3, N, B) .* T(10)
                jl_out = dgram_from_positions(pos; min_bin=3.25, max_bin=20.75, no_bins=15)
                pos_py = to_py(pos; swap_batch_dim=true)
                py_out = PyFeats.dgram_from_positions(pos_py, 3.25, 20.75, 15, 1e8)
                jl_py = to_jl(py_out; swap_batch_dim=true)
                ndiff = sum(jl_out .!= jl_py)
                if ndiff > 0
                    println("dgram $T: $ndiff differences out of $(length(jl_out)) elements")
                end
                @test ndiff == 0
            end
        end
    end

    @testset "build_template_angle_feat" begin
        N, B = 8, 2
        for T in [Float64, Float32]
            @testset "$T" begin
                aatype, tsc, atsc, tmask = _rand_angle_feat_inputs(Rng, T, N, B)
                jl_out = build_template_angle_feat(aatype, tsc, atsc, tmask)
                @test size(jl_out) == (57, N, B)

                aatype_py, tsc_py, atsc_py, tmask_py = _to_py_angle_feat(aatype, tsc, atsc, tmask)
                batch = _make_pydict(
                    template_aatype=aatype_py,
                    template_torsion_angles_sin_cos=tsc_py,
                    template_alt_torsion_angles_sin_cos=atsc_py,
                    template_torsion_angles_mask=tmask_py,
                )
                py_out = PyFeats.build_template_angle_feat(batch)
                @test jl_out ≈ to_jl(py_out; swap_batch_dim=true) rtol=ATOL[T]
            end
        end
    end

    @testset "build_template_pair_feat" begin
        N, B = 4, 2
        for T in [Float64, Float32]
            @testset "$T" begin
                pos, mask, aatype, all_pos, all_mask = _rand_pair_feat_inputs(Rng, T, N, B)
                jl_out = build_template_pair_feat(pos, mask, aatype, all_pos, all_mask)
                @test size(jl_out) == (88, N, N, B)

                pos_py, mask_py, aatype_py, all_pos_py, all_mask_py = _to_py_pair_feat(
                    pos, mask, aatype, all_pos, all_mask)
                batch = _make_pydict(
                    template_pseudo_beta=pos_py,
                    template_pseudo_beta_mask=mask_py,
                    template_aatype=aatype_py,
                    template_all_atom_positions=all_pos_py,
                    template_all_atom_mask=all_mask_py,
                )
                py_out = PyFeats.build_template_pair_feat(
                    batch, 3.25, 50.75, 39, false, 1e-20, 1e8)
                jl_py = to_jl(py_out; swap_batch_dim=true)
                @test jl_out ≈ jl_py rtol=ATOL[T]
            end
        end
    end
end
