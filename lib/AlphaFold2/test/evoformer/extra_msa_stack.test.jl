const PyExtraMSAStack  = pyimport("openfold.model.evoformer").ExtraMSAStack

function sync_extra_msa_stack!(py_stack::PyObject, jl_ps::NamedTuple)
    for (i, py_block) in enumerate(py_stack.blocks)
        name = Symbol("block_$i")
        sync_extra_msa_block!(py_block, jl_ps.blocks[name])
    end
end

@testset "ExtraMSAStack" begin
    rng = Random.Xoshiro(42)

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
    no_blocks = 2

    mask_cfg = (
        ("No mask",     (nothing, nothing)),
        ("Random mask", (make_global_mask(rng, N_res, N_seq, B),
                         make_pair_mask(rng, N_res, B)))
    )

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            for (name, (msa_mask, pair_mask)) in mask_cfg
                @testset "$name" begin
                    jl_layer = ExtraMSAStack(C_m, C_z;
                        no_blocks,
                        chn_hidden_msa_att=C_hidden_msa,
                        chn_hidden_opm=C_hidden_opm,
                        chn_hidden_mul=C_hidden_mul,
                        chn_hidden_pair_att=C_hidden_pair,
                        no_heads_msa, no_heads_pair, transition_n,
                        opm_first=false, tri_mul_first=true, epsilon=1f-5)
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = PyExtraMSAStack(
                        c_m=C_m, c_z=C_z,
                        c_hidden_msa_att=C_hidden_msa,
                        c_hidden_opm=C_hidden_opm,
                        c_hidden_mul=C_hidden_mul,
                        c_hidden_pair_att=C_hidden_pair,
                        no_heads_msa=no_heads_msa,
                        no_heads_pair=no_heads_pair,
                        no_blocks=no_blocks,
                        transition_n=transition_n,
                        msa_dropout=0.0, pair_dropout=0.0,
                        opm_first=false,
                        fuse_projection_weights=true,
                        inf=1e9, eps=1e-8,
                        ckpt=false,
                    )
                    sync_extra_msa_stack!(py_layer, ps)

                    m = randn(rng, T, C_m, N_res, N_seq, B)
                    z = randn(rng, T, C_z, N_res, N_res, B)

                    # GlobalAttention always needs a float mask
                    msa_mask_actual = isnothing(msa_mask) ? trues(N_res, N_seq, B) : msa_mask
                    m_py = jl_to_py_msa(m)
                    z_py = to_py(z; swap_batch_dim=true)
                    msa_mask_py  = jl_to_py_mask(msa_mask_actual, T)
                    pair_mask_py = if isnothing(pair_mask)
                        to_py(ones(T, B, N_res, N_res); swap_batch_dim=false)
                    else
                        jl_to_py_pair_mask(pair_mask, T)
                    end

                    # Julia: returns z_out only
                    z_jl, _ = jl_layer(m, z, msa_mask, pair_mask, ps, st)

                    # use_lma=true forces use_memory_efficient_kernel=False in each
                    # ExtraMSABlock.forward (which computes: not(use_lma or ...)).
                    # The memory-efficient kernel only supports Float32 and produces
                    # ValueError for Float64/Float16, and numerically different results.
                    z_py_out = py_layer(
                        m_py, z_py, msa_mask_py, pair_mask_py;
                        chunk_size=nothing,
                        use_deepspeed_evo_attention=false,
                        use_cuequivariance_attention=false,
                        use_cuequivariance_multiplicative_update=false,
                        use_lma=true,
                        inplace_safe=false, _mask_trans=true,
                    )

                    # tol = max(T(5e-2), sqrt(eps(T)))
                    @testset "Python parity (z)" begin
                        if T == Float64
                            @test z_jl ≈ to_jl(z_py_out; swap_batch_dim=true) atol=5e-2
                        else
                            @test z_jl ≈ to_jl(z_py_out; swap_batch_dim=true)
                        end
                    end
                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer(m, z, msa_mask, pair_mask, ps, st)
                    end
                end
            end
        end
    end
end
