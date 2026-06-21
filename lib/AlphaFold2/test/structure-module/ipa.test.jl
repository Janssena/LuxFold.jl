const PyInvariantPointAttention =
    pyimport("openfold.model.structure_module").InvariantPointAttention
const PyPointProjection =
    pyimport("openfold.model.structure_module").PointProjection
const PyRigid = pyimport("openfold.utils.rigid_utils").Rigid
const PyRotation = pyimport("openfold.utils.rigid_utils").Rotation

rng = Random.Xoshiro(42)

# Build a Python Rigid from explicit Julia quats + trans (mirrors the trick used in
# rigid_utils.test.jl — openfold hard-casts to float32 inside Rotation.__init__).
function _py_rigid_from_julia(quats_jl::AbstractArray{T,3}, trans_jl::AbstractArray{T,3}) where T
    # quats_jl: [4, N, B] → Python wants [B, N, 4]
    # trans_jl: [3, N, B] → Python wants [B, N, 3]
    quats_py = to_py(permutedims(quats_jl, (3, 2, 1)); swap_batch_dim=false)
    trans_py = to_py(permutedims(trans_jl, (3, 2, 1)); swap_batch_dim=false)
    py_rot = PyRotation(quats=quats_py, normalize_quats=true)
    return PyRigid(rots=py_rot, trans=trans_py)
end

# ---- sync helpers (test-local; not generic) ---------------------------------

function sync_point_projection!(py_pp::PyObject, jl_ps::NamedTuple)
    sync_dense!(py_pp.linear, jl_ps.linear)
end

function sync_ipa!(py_ipa::PyObject, jl_ps::NamedTuple; is_multimer::Bool=false)
    sync_dense!(py_ipa.linear_q, jl_ps.linear_q)
    if is_multimer
        sync_dense!(py_ipa.linear_k, jl_ps.linear_k)
        sync_dense!(py_ipa.linear_v, jl_ps.linear_v)
        sync_point_projection!(py_ipa.linear_k_points, jl_ps.linear_k_pts)
        sync_point_projection!(py_ipa.linear_v_points, jl_ps.linear_v_pts)
    else
        sync_dense!(py_ipa.linear_kv, jl_ps.linear_kv)
        sync_point_projection!(py_ipa.linear_kv_points, jl_ps.linear_kv_pts)
    end
    sync_point_projection!(py_ipa.linear_q_points, jl_ps.linear_q_pts)
    sync_dense!(py_ipa.linear_b, jl_ps.linear_b)
    # head_weights is a plain nn.Parameter — copy directly.
    copy_jl_ps_to_py!(py_ipa.head_weights, jl_ps.head_weights.w)
    sync_dense!(py_ipa.linear_out, jl_ps.linear_out)
end

# Variants exercised by every parity test below. `is_multimer=false` is the AF2
# monomer flavour (joint KV); `true` is the multimer flavour (separate K/V).
const IPA_VARIANTS = (("monomer", false), ("multimer", true))

# ---- PointProjection parity --------------------------------------------------

@testset "PointProjection" begin
    chn_in, num_points, no_heads, N, B = 16, 4, 2, 5, 2

    for (variant, is_multimer) in IPA_VARIANTS
        @testset "$variant" begin
            # NOTE: openfold hard-casts quats/rot to float32 inside Rotation.__init__,
            # so Float64 Julia output cannot match Python to float64 precision.
            # Python parity is tested at Float32 only.
            for T in (Float32,)
                @testset "$T" begin
                    jl_layer = PointProjection(
                        chn_in, num_points, no_heads; is_multimer=is_multimer,
                    )
                    jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = PyPointProjection(chn_in, num_points, no_heads, is_multimer)
                    sync_point_projection!(py_layer, jl_ps)

                    s_jl = randn(rng, T, chn_in, N, B)
                    s_py = to_py(s_jl; swap_batch_dim=true)

                    quats_jl = randn(rng, T, 4, N, B)
                    quats_jl ./= sqrt.(sum(abs2, quats_jl; dims=1))
                    trans_jl = randn(rng, T, 3, N, B)
                    r_jl = Rigid(quats_jl, trans_jl)
                    r_py = _py_rigid_from_julia(quats_jl, trans_jl)

                    y_jl, _ = jl_layer(s_jl, r_jl, jl_ps, jl_st)
                    y_py = py_layer(s_py, r_py)

                    @testset "Python parity" begin
                        # Julia: [3, P, H, N, B]; Python: [B, N, H, P, 3] → permute
                        @test y_jl ≈ permutedims(to_jl(y_py; swap_batch_dim=false), (5, 4, 3, 2, 1))
                    end

                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer(s_jl, r_jl, jl_ps, jl_st)
                    end
                end
            end
        end
    end
end

# ---- IPA parity --------------------------------------------------------------

@testset "InvariantPointAttention" begin
    chn_s, chn_z, chn_hidden, no_heads = 16, 8, 8, 4
    no_qk_points, no_v_points = 4, 8
    N, B = 5, 2

    for (variant, is_multimer) in IPA_VARIANTS
        @testset "$variant" begin
            # NOTE: openfold's Rigid hard-casts to float32 internally, so Float64
            # Julia outputs differ from Python at float32 precision (~1e-4), which
            # is too large for the default ≈ tolerance.
            #
            # NOTE: Float16 is skipped because Lux.scaled_dot_product_attention uses
            # typemin(T) = -Inf for masked positions.  For Float16 this causes
            # NaN outputs at query positions where all keys are masked
            # (softmax(-Inf, …, -Inf) = 0/0), making comparison with openfold
            # (which uses a finite -1e5 bias) unreliable.
            for T in [Float32]
                @testset "$T" begin
                    jl_layer = InvariantPointAttention(
                        chn_s, chn_z, chn_hidden, no_heads, no_qk_points, no_v_points;
                        is_multimer=is_multimer,
                    )
                    jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = PyInvariantPointAttention(
                        chn_s, chn_z, chn_hidden, no_heads, no_qk_points, no_v_points;
                        is_multimer=is_multimer,
                    )
                    py_layer.eval()
                    sync_ipa!(py_layer, jl_ps; is_multimer=is_multimer)

                    s_jl = randn(rng, T, chn_s, N, B)
                    z_jl = randn(rng, T, chn_z, N, N, B)

                    quats_jl = randn(rng, T, 4, N, B)
                    quats_jl ./= sqrt.(sum(abs2, quats_jl; dims=1))
                    trans_jl = randn(rng, T, 3, N, B)
                    r_jl = Rigid(quats_jl, trans_jl)
                    r_py = _py_rigid_from_julia(quats_jl, trans_jl)

                    s_py = to_py(s_jl; swap_batch_dim=true)
                    z_py = to_py(z_jl; swap_batch_dim=true)

                    # Random mask with at least one valid residue per batch element,
                    # mirroring real-world usage (a protein chain always has ≥1 residue;
                    # an all-zero column would mean an empty sequence).
                    mask_rand = rand(rng, Bool, N, B)
                    for b in 1:B
                        any(@view mask_rand[:, b]) || (mask_rand[rand(rng, 1:N), b] = true)
                    end
                    # `nothing` (no masking) ⇔ Python all-ones mask.
                    mask_cfg = (("No mask", nothing), ("Random mask", mask_rand))

                    for (mask_name, mask_jl) in mask_cfg
                        @testset "$mask_name" begin
                            mask_py = isnothing(mask_jl) ?
                                to_py(ones(T, N, B); swap_batch_dim=true) :
                                to_py(mask_jl; swap_batch_dim=true).to(py_dtype(T))

                            y_jl, _ = jl_layer(s_jl, z_jl, r_jl, mask_jl, jl_ps, jl_st)
                            y_py    = py_layer(s_py, z_py, r_py, mask_py)
                            y_py_jl = to_jl(y_py; swap_batch_dim=true)

                            @testset "Output" begin
                                # SDPA uses -Inf for masked positions, causing NaN in softmax
                                # at invalid query rows. Python produces -Inf too (via large
                                # bias), giving identical NaN positions. Compare only at valid
                                # queries (all valid when there is no mask).
                                valid = isnothing(mask_jl) ? trues(1, N, B) :
                                        reshape(mask_jl, 1, N, B)
                                y_jl_valid = ifelse.(valid, y_jl, zero(T))
                                y_py_valid = ifelse.(valid, y_py_jl, zero(T))
                                @test y_jl_valid ≈ y_py_valid
                            end

                            @testset "Type-stability" begin
                                @test_nowarn @inferred jl_layer(s_jl, z_jl, r_jl, mask_jl, jl_ps, jl_st)
                            end
                        end
                    end
                end
            end
        end
    end
end
