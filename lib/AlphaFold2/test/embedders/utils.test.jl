const PyFeats = pyimport("openfold.utils.feats")

function _rand_pair_feat_inputs(rng, T, N, B)
    pos      = randn(rng, T, 3, N, B) .* T(10)
    mask     = rand(rng, Bool, N, B)
    aatype   = rand(rng, 0:21, N, B)
    all_pos  = randn(rng, T, 37, 3, N, B) .* T(10)
    all_mask = rand(rng, Bool, 37, N, B)
    # Backbone atoms (N, CA, C at indices 1, 2, 3) are always present in real structures
    all_mask[1:3, :, :] .= true
    return pos, mask, aatype, all_pos, all_mask
end

# 5D variant: inputs carry an extra N_templ dimension.
function _rand_pair_feat_inputs_5d(rng, T, N, N_templ, B)
    pos      = randn(rng, T, 3,  N, N_templ, B) .* T(10)
    mask     = rand(rng, Bool, N, N_templ, B)
    aatype   = rand(rng, 0:21,   N, N_templ, B)
    all_pos  = randn(rng, T, 37, 3, N, N_templ, B) .* T(10)
    all_mask = rand(rng, Bool, 37, N, N_templ, B)
    # Backbone atoms (N, CA, C at indices 1, 2, 3) are always present in real structures
    all_mask[1:3, :, :, :] .= true
    return pos, mask, aatype, all_pos, all_mask
end

function _rand_angle_feat_inputs(rng, T, N, B)
    aatype = rand(rng, 0:21, N, B)
    tsc = randn(rng, T, 2, 7, N, B)
    atsc = randn(rng, T, 2, 7, N, B)
    tmask = rand(rng, Bool, 7, N, B)
    return aatype, tsc, atsc, tmask
end

# 5D variant: inputs carry an extra N_templ dimension.
function _rand_angle_feat_inputs_5d(rng, T, N, N_templ, B)
    aatype = rand(rng, 0:21, N, N_templ, B)
    tsc    = randn(rng, T, 2, 7, N, N_templ, B)
    atsc   = randn(rng, T, 2, 7, N, N_templ, B)
    tmask  = rand(rng, Bool, 7, N, N_templ, B)
    return aatype, tsc, atsc, tmask
end

function _to_py_pair_feat(pos, mask, aatype, all_pos, all_mask)
    pos_py = to_py(pos; swap_batch_dim=true)
    mask_py = to_py(mask; swap_batch_dim=true).to(py_dtype(eltype(pos)))
    aatype_py = to_py(aatype; swap_batch_dim=true)
    all_pos_py = to_py(permutedims(all_pos, (4, 3, 1, 2)))
    all_mask_py = to_py(permutedims(all_mask, (3, 2, 1))).to(py_dtype(eltype(pos)))
    return pos_py, mask_py, aatype_py, all_pos_py, all_mask_py
end

function _to_py_angle_feat(aatype, tsc, atsc, tmask)
    tsc_py = to_py(permutedims(tsc, (4, 3, 2, 1)))
    atsc_py = to_py(permutedims(atsc, (4, 3, 2, 1)))
    tmask_py = to_py(permutedims(tmask, (3, 2, 1))).to(py_dtype(eltype(tsc)))
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
    rng = Random.Xoshiro(42)
    @testset "dgram_from_positions" begin
        N, B = 8, 2
        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                pos = randn(rng, T, 3, N, B) .* T(10)
                jl_out = dgram_from_positions(pos; min_bin=3.25, max_bin=20.75, no_bins=15)
                pos_py = to_py(pos; swap_batch_dim=true)
                py_out = PyFeats.dgram_from_positions(pos_py, 3.25, 20.75, 15, 1e8)

                @testset "Python parity" begin
                    @test jl_out ≈ to_jl(py_out; swap_batch_dim=true)
                end

                @testset "Type stability" begin
                    @test_nowarn @inferred dgram_from_positions(pos; min_bin=3.25, max_bin=20.75, no_bins=15)
                end
            end
        end
    end

    @testset "build_template_angle_feat" begin
        N, B = 8, 2
        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                aatype, tsc, atsc, tmask = _rand_angle_feat_inputs(rng, T, N, B)
                jl_out = build_template_angle_feat(aatype, tsc, atsc, tmask)

                aatype_py, tsc_py, atsc_py, tmask_py = _to_py_angle_feat(aatype, tsc, atsc, tmask)
                batch = _make_pydict(
                    template_aatype=aatype_py,
                    template_torsion_angles_sin_cos=tsc_py,
                    template_alt_torsion_angles_sin_cos=atsc_py,
                    template_torsion_angles_mask=tmask_py,
                )
                py_out = PyFeats.build_template_angle_feat(batch)
                
                @testset "Python parity" begin
                    @test jl_out ≈ to_jl(py_out; swap_batch_dim=true)
                end

                @testset "Type stability" begin
                    @test_nowarn @inferred build_template_angle_feat(aatype, tsc, atsc, tmask)
                end
            end
        end
    end

    @testset "build_template_pair_feat" begin
        N, B = 4, 2
        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                pos, mask, aatype, all_pos, all_mask = _rand_pair_feat_inputs(rng, T, N, B)
                jl_out = build_template_pair_feat(pos, mask, aatype, all_pos, all_mask)

                pos_py, mask_py, aatype_py, all_pos_py, all_mask_py = _to_py_pair_feat(
                    pos, mask, aatype, all_pos, all_mask
                )
                batch = _make_pydict(
                    template_pseudo_beta=pos_py,
                    template_pseudo_beta_mask=mask_py,
                    template_aatype=aatype_py,
                    template_all_atom_positions=all_pos_py,
                    template_all_atom_mask=all_mask_py,
                )
                py_out = PyFeats.build_template_pair_feat(
                    batch, 3.25, 50.75, 39, false, 1e-20, 1e8)

                @testset "Python parity" begin
                    @test jl_out ≈ to_jl(py_out; swap_batch_dim=true)
                end

                @testset "Type stability" begin
                    @test_nowarn @inferred build_template_pair_feat(pos, mask, aatype, all_pos, all_mask)
                end
            end
        end
    end

    @testset "build_template_pair_feat (use_unit_vector=true)" begin
        N, B = 4, 2
        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                pos, mask, aatype, all_pos, all_mask = _rand_pair_feat_inputs(rng, T, N, B)

                jl_out = build_template_pair_feat(
                    pos, mask, aatype, all_pos, all_mask; use_unit_vector=true)

                pos_py, mask_py, aatype_py, all_pos_py, all_mask_py = _to_py_pair_feat(
                    pos, mask, aatype, all_pos, all_mask
                )
                batch = _make_pydict(
                    template_pseudo_beta=pos_py,
                    template_pseudo_beta_mask=mask_py,
                    template_aatype=aatype_py,
                    template_all_atom_positions=all_pos_py,
                    template_all_atom_mask=all_mask_py,
                )
                py_out = PyFeats.build_template_pair_feat(
                    batch, 3.25, 50.75, 39, true, 1e-20, 1e8)

                # Python Rigid computes in float32 internally — only compare at Float32 precision
                if T == Float32
                    @testset "Python parity" begin
                        @test jl_out ≈ to_jl(py_out; swap_batch_dim=true)
                    end
                end
                
                @testset "Type stability" begin
                    @test_nowarn @inferred build_template_pair_feat(
                        pos, mask, aatype, all_pos, all_mask; use_unit_vector=true
                    )
                end
                    
                @testset "Unit vector channels are non-zero" begin
                    @test any(!iszero, jl_out[85:87, :, :, :])
                end
            end
        end
    end

    # ===  5D vectorised paths  ===
    #
    # The 5D versions of build_template_pair_feat and build_template_angle_feat process
    # all N_templ templates in a single call by treating N_templ as an extra batch dim.
    # Strategy: compare the 5D output slice-by-slice against the known-good 4D per-template
    # call. Since the 4D path already has Python parity, this validates the 5D path.
    #
    # Julia 5D layout: [C, N, N, N_templ, B]  (pair) / [C, N, N_templ, B]  (angle)
    # Julia 4D layout: [C, N, N, B]           (pair) / [C, N, B]            (angle)

    @testset "build_template_pair_feat (5D vectorised)" begin
        N, N_templ, B = 4, 3, 2
        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                pos5, mask5, aa5, apos5, amask5 = _rand_pair_feat_inputs_5d(rng, T, N, N_templ, B)

                feat_5d = build_template_pair_feat(pos5, mask5, aa5, apos5, amask5)

                @testset "Consistency with 4D per-template" begin
                    for t in 1:N_templ
                        # Extract the t-th template slice from each 5D input
                        pos_t   = pos5[:, :, t, :]
                        mask_t  = mask5[:, t, :]
                        aa_t    = aa5[:, t, :]
                        apos_t  = apos5[:, :, :, t, :]
                        amask_t = amask5[:, :, t, :]

                        feat_4d = build_template_pair_feat(pos_t, mask_t, aa_t, apos_t, amask_t)
                        
                        @testset "Python parity" begin
                            @test feat_5d[:, :, :, t, :] ≈ feat_4d
                        end

                        @testset "Type stability" begin
                            @test_nowarn @inferred build_template_pair_feat(
                                pos5, mask5, aa5, apos5, amask5
                            )
                        end
                    end
                end
            end
        end
    end

    @testset "build_template_angle_feat (5D vectorised)" begin
        N, N_templ, B = 8, 3, 2
        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                aa5, tsc5, atsc5, tmask5 = _rand_angle_feat_inputs_5d(rng, T, N, N_templ, B)

                feat_5d = build_template_angle_feat(aa5, tsc5, atsc5, tmask5)

                @testset "Consistency with 4D per-template" begin
                    for t in 1:N_templ
                        # tsc/atsc: [2, 7, N, N_templ, B] → slice t → [2, 7, N, B]
                        # tmask:    [7,    N, N_templ, B] → slice t → [7,    N, B]
                        # aa:       [N,       N_templ, B] → slice t → [N,       B]
                        aa_t    = aa5[:, t, :]
                        tsc_t   = tsc5[:, :, :, t, :]
                        atsc_t  = atsc5[:, :, :, t, :]
                        tmask_t = tmask5[:, :, t, :]

                        feat_4d = build_template_angle_feat(aa_t, tsc_t, atsc_t, tmask_t)
                        
                        @testset "Python parity" begin
                            @test feat_5d[:, :, t, :] ≈ feat_4d
                        end

                        @testset "Type stability" begin
                            @test_nowarn @inferred build_template_angle_feat(
                                aa5, tsc5, atsc5, tmask5
                            )
                        end
                    end
                end
            end
        end
    end
end
