const PyEvoformerBlock = pyimport("openfold.model.evoformer").EvoformerBlock


# ===  Sync helpers  ===
# sync_pair_stack_block!          — defined in pair_stack.test.jl
# sync_msa_column_attention!      — defined in msa_attention.test.jl
# sync_msa_column_global_attention! — defined in msa_attention.test.jl

function sync_af2_msa_row_att!(py_mar::PyObject, jl_ps::NamedTuple)
    # Python MSARowAttentionWithPairBias: layer_norm_m, layer_norm_z, linear_z, mha
    # Julia AttentionPairBias:            layer_norm_in, layer_norm_z, linear_z, mha
    sync_layernorm!(py_mar.layer_norm_m, jl_ps.layer_norm_in)
    sync_layernorm!(py_mar.layer_norm_z, jl_ps.layer_norm_z)
    sync_dense!(py_mar.linear_z, jl_ps.linear_z)
    sync_af3_attention!(py_mar.mha, jl_ps.mha)
end

function sync_msa_transition!(py_mt::PyObject, jl_ps::NamedTuple)
    # Python MSATransition: layer_norm, linear_1, linear_2
    # Julia Transition:     layer_norm, linear_1, linear_2
    sync_layernorm!(py_mt.layer_norm, jl_ps.layer_norm)
    sync_dense!(py_mt.linear_1, jl_ps.linear_1)
    sync_dense!(py_mt.linear_2, jl_ps.linear_2)
end

function sync_evoformer_block!(py_block::PyObject, jl_ps::NamedTuple)
    sync_af2_msa_row_att!(py_block.msa_att_row, jl_ps.msa_att_row)
    sync_msa_column_attention!(py_block.msa_att_col, jl_ps.msa_att_col)
    sync_msa_transition!(py_block.msa_transition, jl_ps.msa_transition)
    sync_af3_opm!(py_block.outer_product_mean, jl_ps.outer_product_mean)
    sync_pair_stack_block!(py_block.pair_stack, jl_ps.pair_stack)
end

# ===  Tensor layout  ===
#
# Julia MSA:       [C_m, N_res, N_seq, B]     mask: [N_res, N_seq, B]
# Python MSA:      [B, N_seq, N_res, C_m]     mask: [B, N_seq, N_res] (float)
# Julia pair:      [C_z, N_res, N_res, B]     mask: [N_res, N_res, B]
# Python pair:     [B, N_res, N_res, C_z]     mask: [B, N_res, N_res] (float)
#
# Reuse jl_to_py_msa / py_to_jl_msa / jl_to_py_mask from msa_attention.test.jl
# Reuse make_pair_mask / jl_to_py_pair_mask from pair_stack.test.jl
# Reuse make_global_mask from msa_attention.test.jl

# ===  Shared dims  ===
const EVOBLOCK_C_m            = 16
const EVOBLOCK_C_z            = 8
const EVOBLOCK_C_hidden_msa   = 4
const EVOBLOCK_C_hidden_opm   = 4
const EVOBLOCK_C_hidden_mul   = 8
const EVOBLOCK_C_hidden_pair  = 4
const EVOBLOCK_no_heads_msa   = 2
const EVOBLOCK_no_heads_pair  = 2
const EVOBLOCK_transition_n   = 2
const EVOBLOCK_N_res          = 8
const EVOBLOCK_N_seq          = 4
const EVOBLOCK_B              = 2

# ===  EvoformerBlock tests  ===

@testset "EvoformerBlock" begin
    rng = Random.Xoshiro(42)

    C_m = EVOBLOCK_C_m; C_z = EVOBLOCK_C_z
    C_hidden_msa = EVOBLOCK_C_hidden_msa; C_hidden_opm = EVOBLOCK_C_hidden_opm
    C_hidden_mul = EVOBLOCK_C_hidden_mul; C_hidden_pair = EVOBLOCK_C_hidden_pair
    no_heads_msa = EVOBLOCK_no_heads_msa; no_heads_pair = EVOBLOCK_no_heads_pair
    transition_n = EVOBLOCK_transition_n
    N_res = EVOBLOCK_N_res; N_seq = EVOBLOCK_N_seq; B = EVOBLOCK_B

    # MSA mask: ensure each residue column has ≥1 valid sequence (avoids NaN in column attention)
    mask_cfg = (
        ("No mask",     (nothing, nothing)),
        ("Random mask", (make_global_mask(rng, N_res, N_seq, B),
                         make_pair_mask(rng, N_res, B)))
    )

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            for (name, (msa_mask, pair_mask)) in mask_cfg
                @testset "$name" begin
                    jl_layer = EvoformerBlock(C_m, C_z;
                        chn_hidden_msa_att=C_hidden_msa,
                        chn_hidden_opm=C_hidden_opm,
                        chn_hidden_mul=C_hidden_mul,
                        chn_hidden_pair_att=C_hidden_pair,
                        no_heads_msa, no_heads_pair, transition_n,
                        opm_first=false, tri_mul_first=true, epsilon=1f-5)
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = PyEvoformerBlock(
                        c_m=C_m, c_z=C_z,
                        c_hidden_msa_att=C_hidden_msa,
                        c_hidden_opm=C_hidden_opm,
                        c_hidden_mul=C_hidden_mul,
                        c_hidden_pair_att=C_hidden_pair,
                        no_heads_msa=no_heads_msa,
                        no_heads_pair=no_heads_pair,
                        transition_n=transition_n,
                        msa_dropout=0.0, pair_dropout=0.0,
                        no_column_attention=false,
                        opm_first=false,
                        fuse_projection_weights=true,
                        inf=1e9, eps=1e-8,
                    )
                    sync_evoformer_block!(py_layer, ps)

                    m  = randn(rng, T, C_m, N_res, N_seq, B)
                    z  = randn(rng, T, C_z, N_res, N_res, B)

                    # Julia → Python tensor conversion
                    m_py = jl_to_py_msa(m)
                    z_py = to_py(z; swap_batch_dim=true)
                    msa_mask_py = isnothing(msa_mask) ? nothing :
                                  jl_to_py_mask(msa_mask, T)
                    pair_mask_py = if isnothing(pair_mask)
                        to_py(ones(T, B, N_res, N_res); swap_batch_dim=false)
                    else
                        jl_to_py_pair_mask(pair_mask, T)
                    end

                    # Julia forward: returns (; m, z) plus pass-through masks
                    (out, _) = jl_layer((; m, z, msa_mask, pair_mask), ps, st)

                    # Python forward: returns (m_out, z_out) tuple
                    m_py_out, z_py_out = py_layer(
                        m_py, z_py, msa_mask_py, pair_mask_py;
                        chunk_size=nothing,
                        use_deepspeed_evo_attention=false,
                        use_cuequivariance_attention=false,
                        use_cuequivariance_multiplicative_update=false,
                        use_lma=false, use_flash=false,
                        inplace_safe=false, _mask_trans=true,
                    )

                    # Block-level tolerance: more generous than individual-layer tests,
                    # since errors accumulate across multiple sub-layers.
                    # max(..., sqrt(eps(T))) preserves Float16's default tolerance.
                    tol = max(T(1e-4), sqrt(eps(T)))
                    @testset "Python parity (m)" begin
                        @test out.m ≈ py_to_jl_msa(m_py_out) rtol=tol atol=tol
                    end
                    @testset "Python parity (z)" begin
                        @test out.z ≈ to_jl(z_py_out; swap_batch_dim=true) rtol=tol atol=tol
                    end
                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer((; m, z, msa_mask, pair_mask), ps, st)
                    end
                end
            end
        end
    end
end