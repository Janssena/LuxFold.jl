const PyTemplatePairStackBlock = pyimport("openfold.model.template").TemplatePairStackBlock
const PyTemplatePairStack = pyimport("openfold.model.template").TemplatePairStack

function sync_template_pair_stack_block!(py_block::PyObject, jl_ps::NamedTuple)
    sync_triangle_attention!(py_block.tri_att_start, jl_ps.tri_att_start)
    sync_triangle_attention!(py_block.tri_att_end,   jl_ps.tri_att_end)
    sync_triangle_multiplication!(py_block.tri_mul_out, jl_ps.tri_mul_out)
    sync_triangle_multiplication!(py_block.tri_mul_in,  jl_ps.tri_mul_in)
    sync_layernorm!(py_block.pair_transition.layer_norm, jl_ps.pair_transition.layer_norm)
    sync_dense!(py_block.pair_transition.linear_1,       jl_ps.pair_transition.linear_1)
    sync_dense!(py_block.pair_transition.linear_2,       jl_ps.pair_transition.linear_2)
end

function sync_template_pair_stack!(py_stack::PyObject, jl_ps::NamedTuple)
    for (i, py_block) in enumerate(py_stack.blocks)
        name = Symbol("block_$i")
        sync_template_pair_stack_block!(py_block, jl_ps.blocks[name])
    end
    sync_layernorm!(py_stack.layer_norm, jl_ps.layer_norm)
end

# Note: mask with diagonal forced True (avoids NaN in attention softmax from fully-masked rows)
function make_mask(rng, N_res, N_templ, B)
    mask = rand(rng, Bool, N_res, N_res, N_templ, B)
    for b in 1:B, t in 1:N_templ, i in 1:N_res
        mask[i, i, t, b] = true
    end
    return mask
end

@testset "TemplatePairStackBlock" begin
    rng = Random.Xoshiro(42)
    chn_templ = 16
    chn_hidden_tri_att = 4
    chn_hidden_tri_mul = 16
    no_heads = 4
    pair_transition_n = 2
    N_res, N_templ, B = 12, 6, 2

    for tri_mul_first in [false, true]
        @testset "tri_mul_first=$tri_mul_first" begin
            for T in [Float64, Float32, Float16]
                @testset "$T" begin
                    mask_cfg = (
                        ("No mask", nothing),
                        ("Random mask", make_mask(rng, N_res, N_templ, B))
                    )
                    for (name, mask) in mask_cfg
                        @testset "$name" begin
                            jl_layer = TemplatePairStackBlock(
                                chn_templ, chn_hidden_tri_att, chn_hidden_tri_mul, no_heads, pair_transition_n;
                                tri_mul_first, epsilon=1f-5
                            )
                            ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                            py_layer = PyTemplatePairStackBlock(
                                c_t=chn_templ,
                                c_hidden_tri_att=chn_hidden_tri_att,
                                c_hidden_tri_mul=chn_hidden_tri_mul,
                                no_blocks=1,
                                no_heads=no_heads,
                                pair_transition_n=pair_transition_n,
                                dropout_rate=0.0,
                                tri_mul_first=tri_mul_first,
                                fuse_projection_weights=true,
                                inf=1e9
                            )
                            sync_template_pair_stack_block!(py_layer, ps)

                            z = randn(rng, T, chn_templ, N_res, N_res, N_templ, B)
                            z_init = copy(z)
                            mask_init = isnothing(mask) ? nothing : copy(mask)

                            # Python expects [B, N_templ, Ni, Nj, C] and [B, N_templ, Ni, Nj]
                            z_py    = to_py(permutedims(z, (5, 4, 2, 3, 1)); swap_batch_dim=false)
                            mask_py = isnothing(mask) ? trues(N_res, N_res, N_templ, B) : mask
                            mask_py = to_py(permutedims(mask_py, (4, 3, 1, 2)); swap_batch_dim=false).to(py_dtype(T))

                            (y_jl, mask_out), _ = jl_layer((; z, mask), ps, st)
                            y_py    = py_layer(z_py, mask_py, chunk_size=nothing,
                                               use_deepspeed_evo_attention=false,
                                               use_cuequivariance_attention=false,
                                               use_cuequivariance_multiplicative_update=false,
                                               use_lma=false,
                                               inplace_safe=false,
                                               _mask_trans=true)

                            @testset "Python parity" begin
                                # Python output [B, N_templ, Ni, Nj, C] → Julia [C, Ni, Nj, N_templ, B]
                                @test y_jl ≈ permutedims(to_jl(y_py; swap_batch_dim=false), (5, 3, 4, 2, 1))
                            end

                            @testset "No inplace mutation of z and mask in julia" begin
                                @test isequal(z_init, z)
                                @test isequal(mask_init, mask_out)
                            end

                            @testset "Type-stability" begin
                                @test_nowarn @inferred jl_layer(z, mask, ps, st)
                                @test_nowarn @inferred jl_layer((; z, mask), ps, st)
                            end
                        end
                    end
                end
            end
        end
    end
end


@testset "TemplatePairStack" begin
    rng = Random.Xoshiro(42)

    chn_templ = 16
    chn_hidden_tri_att = 4
    chn_hidden_tri_mul = 16
    pair_transition_n = 2
    no_heads = 8
    N_res, N_templ, B = 6, 3, 2
    no_blocks = 10

    for tri_mul_first in [false, true]
        @testset "tri_mul_first = $tri_mul_first" begin
            for T in [Float64, Float32, Float16]
                @testset "$T" begin
                    mask_cfg = (
                        ("No mask", nothing),
                        ("Random mask", make_mask(rng, N_res, N_templ, B))
                    )
                    for (name, mask) in mask_cfg
                        @testset "$name" begin
                            jl_layer = TemplatePairStack(
                                chn_templ, chn_hidden_tri_att, chn_hidden_tri_mul,
                                no_blocks, no_heads, pair_transition_n;  # no_blocks = 1
                                tri_mul_first=false, epsilon=1f-5
                            )
                            ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                            py_layer = PyTemplatePairStack(
                                c_t=chn_templ,
                                c_hidden_tri_att=chn_hidden_tri_att,
                                c_hidden_tri_mul=chn_hidden_tri_mul,
                                no_blocks=no_blocks,
                                no_heads=no_heads,
                                pair_transition_n=pair_transition_n,
                                dropout_rate=0.0,
                                tri_mul_first=false,
                                fuse_projection_weights=true,
                                blocks_per_ckpt=nothing,
                                inf=1e9
                            )

                            sync_template_pair_stack!(py_layer, ps)

                            template = randn(rng, T, chn_templ, N_res, N_res, N_templ, B)

                            # Python expects [B, N_templ, Ni, Nj, C] and mask [B, N_templ, Ni, Nj]
                            template_py = to_py(permutedims(template, (5, 4, 2, 3, 1)); swap_batch_dim=false)
                            mask_py = if isnothing(mask) 
                                to_py(ones(T, B, N_templ, N_res, N_res); swap_batch_dim=false)
                            else
                                to_py(permutedims(mask, (4, 3, 1, 2)); swap_batch_dim=false).to(py_dtype(T))
                            end

                            y_jl, _ = jl_layer(template, mask, ps, st)
                            y_py = py_layer(template_py, mask_py, chunk_size=nothing)

                            @testset "Python parity" begin
                                # Python output [B, N_templ, Ni, Nj, C] → Julia [C, Ni, Nj, N_templ, B]
                                @test y_jl ≈ permutedims(to_jl(y_py; swap_batch_dim=false), (5, 3, 4, 2, 1))
                            end

                            @testset "Type-stability" begin
                                @test_nowarn @inferred jl_layer(template, mask, ps, st)
                            end
                        end
                    end
                end
            end
        end
    end
end
