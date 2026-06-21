const PyExtraMSABlock  = pyimport("openfold.model.evoformer").ExtraMSABlock

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

function sync_extra_msa_block!(py_block::PyObject, jl_ps::NamedTuple)
    sync_af2_msa_row_att!(py_block.msa_att_row, jl_ps.msa_att_row)
    sync_msa_column_global_attention!(py_block.msa_att_col, jl_ps.msa_att_col)
    sync_msa_transition!(py_block.msa_transition, jl_ps.msa_transition)
    sync_af3_opm!(py_block.outer_product_mean, jl_ps.outer_product_mean)
    sync_pair_stack_block!(py_block.pair_stack, jl_ps.pair_stack)
end

@testset "ExtraMSABlock" begin
    rng = Random.Xoshiro(42)

    # ExtraMSA defaults: c_m=64, c_hidden_msa_att=8 — use small dims here
    C_m = 16
    C_z = 8
    C_hidden_msa = 4
    C_hidden_opm = 4
    C_hidden_mul = 8 
    C_hidden_pair = 4
    no_heads_msa = 2 
    no_heads_pair = 2
    transition_n = 2
    N_res = 8
    N_seq = 4
    B = 2

    # Global attention requires ≥1 valid sequence per residue column
    mask_cfg = (
        ("No mask",     (nothing, nothing)),
        ("Random mask", (make_global_mask(rng, N_res, N_seq, B),
                         make_pair_mask(rng, N_res, B)))
    )

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            for (name, (msa_mask, pair_mask)) in mask_cfg
                @testset "$name" begin
                    jl_layer = ExtraMSABlock(C_m, C_z;
                        chn_hidden_msa_att=C_hidden_msa,
                        chn_hidden_opm=C_hidden_opm,
                        chn_hidden_mul=C_hidden_mul,
                        chn_hidden_pair_att=C_hidden_pair,
                        no_heads_msa, no_heads_pair, transition_n,
                        opm_first=false, tri_mul_first=true, epsilon=1f-5)
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = PyExtraMSABlock(
                        c_m=C_m, c_z=C_z,
                        c_hidden_msa_att=C_hidden_msa,
                        c_hidden_opm=C_hidden_opm,
                        c_hidden_mul=C_hidden_mul,
                        c_hidden_pair_att=C_hidden_pair,
                        no_heads_msa=no_heads_msa,
                        no_heads_pair=no_heads_pair,
                        transition_n=transition_n,
                        msa_dropout=0.0, pair_dropout=0.0,
                        opm_first=false,
                        fuse_projection_weights=true,
                        inf=1e9, eps=1e-8,
                        ckpt=false,
                    )
                    sync_extra_msa_block!(py_layer, ps)

                    m  = randn(rng, T, C_m, N_res, N_seq, B)
                    z  = randn(rng, T, C_z, N_res, N_res, B)

                    # GlobalAttention needs a float mask; substitute trues when nothing
                    msa_mask_actual = isnothing(msa_mask) ? trues(N_res, N_seq, B) : msa_mask
                    m_py = jl_to_py_msa(m)
                    z_py = to_py(z; swap_batch_dim=true)
                    msa_mask_py   = jl_to_py_mask(msa_mask_actual, T)
                    pair_mask_py  = if isnothing(pair_mask)
                        to_py(ones(T, B, N_res, N_res); swap_batch_dim=false)
                    else
                        jl_to_py_pair_mask(pair_mask, T)
                    end

                    (out, _) = jl_layer((; m, z, msa_mask, pair_mask), ps, st)

                    # use_lma=true forces use_memory_efficient_kernel=False in Python's
                    # ExtraMSABlock.forward (which computes: not(use_lma or ...)).
                    # The memory-efficient kernel only supports Float32 and produces
                    # ValueError for Float64/Float16, and numerically different results.
                    m_py_out, z_py_out = py_layer(
                        m_py, z_py, msa_mask_py, pair_mask_py;
                        chunk_size=nothing,
                        use_deepspeed_evo_attention=false,
                        use_cuequivariance_attention=false,
                        use_cuequivariance_multiplicative_update=false,
                        use_lma=true,
                        inplace_safe=false, _mask_trans=true,
                    )

                    # tol = max(T(1e-4), sqrt(eps(T)))
                    @testset "Python parity (m)" begin
                        @test out.m ≈ py_to_jl_msa(m_py_out) # rtol=tol atol=tol
                    end
                    @testset "Python parity (z)" begin
                        if T == Float64
                            @test out.z ≈ to_jl(z_py_out; swap_batch_dim=true) atol=5e-2
                        else
                            @test out.z ≈ to_jl(z_py_out; swap_batch_dim=true)
                        end
                    end
                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer((; m, z, msa_mask, pair_mask), ps, st)
                    end
                end
            end
        end
    end
end