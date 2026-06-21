const PyEvoformerStack = pyimport("openfold.model.evoformer").EvoformerStack

# ===  Sync helpers  ===
# sync_evoformer_block! / sync_extra_msa_block! — defined in evoformer_block.test.jl

function sync_evoformer_stack!(py_stack::PyObject, jl_ps::NamedTuple, no_blocks::Int)
    for (i, py_block) in enumerate(py_stack.blocks)
        name = Symbol("block_$i")
        sync_evoformer_block!(py_block, jl_ps.blocks[name])
    end
    sync_dense!(py_stack.linear, jl_ps.single_projection)
end

# ===  EvoformerStack tests  ===

@testset "EvoformerStack" begin
    rng = Random.Xoshiro(42)

    C_m = EVOBLOCK_C_m; C_z = EVOBLOCK_C_z; C_s = 12
    C_hidden_msa = EVOBLOCK_C_hidden_msa; C_hidden_opm = EVOBLOCK_C_hidden_opm
    C_hidden_mul = EVOBLOCK_C_hidden_mul; C_hidden_pair = EVOBLOCK_C_hidden_pair
    no_heads_msa = EVOBLOCK_no_heads_msa; no_heads_pair = EVOBLOCK_no_heads_pair
    transition_n = EVOBLOCK_transition_n
    N_res = EVOBLOCK_N_res; N_seq = EVOBLOCK_N_seq; B = EVOBLOCK_B
    no_blocks = 2  # small for speed

    mask_cfg = (
        ("No mask",     (nothing, nothing)),
        ("Random mask", (make_global_mask(rng, N_res, N_seq, B),
                         make_pair_mask(rng, N_res, B)))
    )

    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            for (name, (msa_mask, pair_mask)) in mask_cfg
                @testset "$name" begin
                    jl_layer = EvoformerStack(C_m, C_z, C_s;
                        no_blocks,
                        chn_hidden_msa_att=C_hidden_msa,
                        chn_hidden_opm=C_hidden_opm,
                        chn_hidden_mul=C_hidden_mul,
                        chn_hidden_pair_att=C_hidden_pair,
                        no_heads_msa, no_heads_pair, transition_n,
                        opm_first=false, tri_mul_first=true, epsilon=1f-5)
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = PyEvoformerStack(
                        c_m=C_m, c_z=C_z, c_s=C_s,
                        c_hidden_msa_att=C_hidden_msa,
                        c_hidden_opm=C_hidden_opm,
                        c_hidden_mul=C_hidden_mul,
                        c_hidden_pair_att=C_hidden_pair,
                        no_heads_msa=no_heads_msa,
                        no_heads_pair=no_heads_pair,
                        no_blocks=no_blocks,
                        transition_n=transition_n,
                        msa_dropout=0.0, pair_dropout=0.0,
                        no_column_attention=false,
                        opm_first=false,
                        fuse_projection_weights=true,
                        blocks_per_ckpt=nothing,
                        inf=1e9, eps=1e-8,
                    )
                    sync_evoformer_stack!(py_layer, ps, no_blocks)

                    m = randn(rng, T, C_m, N_res, N_seq, B)
                    z = randn(rng, T, C_z, N_res, N_res, B)

                    m_py = jl_to_py_msa(m)
                    z_py = to_py(z; swap_batch_dim=true)
                    msa_mask_py = isnothing(msa_mask) ? nothing :
                                  jl_to_py_mask(msa_mask, T)
                    pair_mask_py = if isnothing(pair_mask)
                        to_py(ones(T, B, N_res, N_res); swap_batch_dim=false)
                    else
                        jl_to_py_pair_mask(pair_mask, T)
                    end

                    # Julia: returns (m_out, z_out, s)
                    (m_jl, z_jl, s_jl), _ = jl_layer(m, z, msa_mask, pair_mask, ps, st)

                    # Python: returns (m_out, z_out, s)
                    m_py_out, z_py_out, s_py_out = py_layer(
                        m_py, z_py, msa_mask_py, pair_mask_py;
                        chunk_size=nothing,
                        use_deepspeed_evo_attention=false,
                        use_cuequivariance_attention=false,
                        use_cuequivariance_multiplicative_update=false,
                        use_lma=false, use_flash=false,
                        inplace_safe=false, _mask_trans=true,
                    )

                    # Stack-level tolerance: errors compound across no_blocks blocks,
                    # so we allow a more generous tolerance than individual block tests.
                    # max(..., sqrt(eps(T))) preserves Float16's default tolerance floor.
                    tol = max(T(5e-2), sqrt(eps(T)))
                    @testset "Python parity (m)" begin
                        @test m_jl ≈ py_to_jl_msa(m_py_out) rtol=tol atol=tol
                    end
                    @testset "Python parity (z)" begin
                        @test z_jl ≈ to_jl(z_py_out; swap_batch_dim=true) rtol=tol atol=tol
                    end
                    @testset "Python parity (s)" begin
                        # Python s: [B, N_res, C_s] → Julia [C_s, N_res, B]
                        @test s_jl ≈ to_jl(s_py_out; swap_batch_dim=true) rtol=tol atol=tol
                    end
                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer(m, z, msa_mask, pair_mask, ps, st)
                    end
                end
            end
        end
    end
end
