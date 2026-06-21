const PyPairStack = pyimport("openfold.model.evoformer").PairStack

# ===  Sync helpers  ===

function sync_pair_stack_block!(py_ps::PyObject, jl_ps::NamedTuple)
    sync_triangle_multiplication!(py_ps.tri_mul_out, jl_ps.tri_mul_out)
    sync_triangle_multiplication!(py_ps.tri_mul_in,  jl_ps.tri_mul_in)
    sync_triangle_attention!(py_ps.tri_att_start, jl_ps.tri_att_start)
    sync_triangle_attention!(py_ps.tri_att_end,   jl_ps.tri_att_end)
    sync_layernorm!(py_ps.pair_transition.layer_norm, jl_ps.pair_transition.layer_norm)
    sync_dense!(py_ps.pair_transition.linear_1, jl_ps.pair_transition.linear_1)
    sync_dense!(py_ps.pair_transition.linear_2, jl_ps.pair_transition.linear_2)
end

# ===  Tensor helpers  ===

# Pair mask [N_res, N_res, B] → Python [B, N_res, N_res] float
function jl_to_py_pair_mask(mask, T)
    to_py(permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T))
end

# Random pair mask with diagonal forced True (avoids all-masked rows in attention)
function make_pair_mask(rng, N_res, B)
    mask = rand(rng, Bool, N_res, N_res, B)
    for b in 1:B, i in 1:N_res
        mask[i, i, b] = true
    end
    return mask
end

# ===  PairStackBlock tests  ===

@testset "PairStackBlock" begin
    rng = Random.Xoshiro(42)
    C_z          = 16
    C_hidden_mul = 16
    C_hidden_att = 4
    no_heads     = 2
    transition_n = 2
    N_res, B     = 8, 2

    mask_cfg = (
        ("No mask",     nothing),
        ("Random mask", make_pair_mask(rng, N_res, B))
    )

    # tri_mul_first=true matches the Python PairStack operation order:
    # tri_mul_out → tri_mul_in → tri_att_start → tri_att_end → pair_transition
    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            for (name, mask) in mask_cfg
                @testset "$name" begin
                    jl_layer = PairStackBlock(C_z, C_hidden_mul, C_hidden_att, no_heads, transition_n;
                        tri_mul_first=true, epsilon=1f-5)
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = PyPairStack(
                        c_z=C_z,
                        c_hidden_mul=C_hidden_mul,
                        c_hidden_pair_att=C_hidden_att,
                        no_heads_pair=no_heads,
                        transition_n=transition_n,
                        pair_dropout=0.0,
                        fuse_projection_weights=true,
                        inf=1e9,
                        eps=1e-8,
                    )
                    sync_pair_stack_block!(py_layer, ps)

                    z = randn(rng, T, C_z, N_res, N_res, B)

                    # Julia [C, N, N, B] → Python [B, N, N, C] via standard swap
                    z_py = to_py(z; swap_batch_dim=true)
                    mask_py = if isnothing(mask)
                        to_py(ones(T, B, N_res, N_res); swap_batch_dim=false)
                    else
                        jl_to_py_pair_mask(mask, T)
                    end

                    z_jl, _ = jl_layer(z, mask, ps, st)
                    z_py_out = py_layer(z_py, mask_py;
                        chunk_size=nothing,
                        use_deepspeed_evo_attention=false,
                        use_cuequivariance_attention=false,
                        use_cuequivariance_multiplicative_update=false,
                        use_lma=false,
                        inplace_safe=false,
                        _mask_trans=true
                    )

                    @testset "Python parity" begin
                        # Python output [B, N, N, C] → Julia [C, N, N, B] via standard swap
                        @test z_jl ≈ to_jl(z_py_out; swap_batch_dim=true)
                    end
                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer(z, mask, ps, st)
                        @test_nowarn @inferred jl_layer((; z, pair_mask=mask), ps, st)
                    end
                end
            end
        end
    end
end
