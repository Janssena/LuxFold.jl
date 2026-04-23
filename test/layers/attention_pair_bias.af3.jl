include("../python/alphafold3.jl");

function sync_af3_attention!(py::PyObject, ps::NamedTuple)
    if :weight ∈ keys(ps.qkv) # is_fused
        throw(ErrorException("Not implemented."))
    else
        sync_dense!(py.linear_q, ps.qkv.q)
        sync_dense!(py.linear_k, ps.qkv.k)
        sync_dense!(py.linear_v, ps.qkv.v)
    end
    sync_dense!(py.linear_o, ps.out)

    if !isempty(ps.gate)
        sync_dense!(py.linear_g, ps.gate)
    end

    return nothing
end

function sync_af3_attention_pair_bias!(py::PyObject, ps::NamedTuple)   
    if isempty(ps.linear_out)
        sync_layernorm!(py.layer_norm_a, ps.layer_norm_in)
    else
        sync_af3_adaln!(py.layer_norm_a, ps.layer_norm_in)
        sync_dense!(py.linear_ada_out, ps.linear_out)    
    end
    sync_layernorm!(py.layer_norm_z, ps.layer_norm_z)
    sync_dense!(py.linear_z, ps.linear_z)

    sync_af3_attention!(py.mha, ps.mha)

    return nothing
end

rng = Random.Xoshiro(42)

@testset "AlphaFold3" begin
    N, B = 12, 2
    chn_in = 32 # chn_a / chn_q etc.
    chn_z = 24 # chn_z
    head_dim = 8 # chn_hidden
    num_heads = 4
    
    mask_cfg = (
        ("No mask", nothing),
        ("Random mask", rand(rng, Bool, N, B)),
        ("All-ones mask", trues(N, B)),
    )

    for (name, mask) in mask_cfg
        @testset "$name" begin
            @testset "Standard AttentionPairBias" begin
                for use_cond in [true, false], use_gate in [true, false]
                    chn_cond = use_cond ? 12 : nothing # chn_s
                    @testset "Use conditioning = $use_cond, use_gate = $(use_gate)" begin
                        for T in [Float16, Float32, Float64]
                            x = randn(rng, T, chn_in, N, B)
                            z = randn(rng, T, chn_z, N, N, B)
                            cond = isnothing(chn_cond) ? nothing : randn(rng, T, chn_cond, N, B)

                            adaln_affine = (layer_norm_a = false, layer_norm_s = true)
                            affine = use_cond ? (layer_norm_in = adaln_affine, layer_norm_z = true) : true
                            use_bias = use_cond ? (true, (layer_norm_in = (false, (shift = true, gate = true, )), mha = false, )) : (true, (mha = false, ))
                            
                            jl_layer = AttentionPairBias(chn_in, chn_z, head_dim, num_heads; chn_cond, use_gate, affine, use_bias, fuse_qkv=false)
                            ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)
                            py_layer = py"AF3AttentionPairBias"(chn_in, chn_in, chn_in, chn_cond, chn_z, head_dim, num_heads, use_ada_layer_norm=use_cond, gating=use_gate)
                            
                            sync_af3_attention_pair_bias!(py_layer, ps)

                            # Forward Julia
                            (y_jl, _), _ = jl_layer(x, z, cond, mask, ps, st)
                            
                            # Forward Python
                            x_py = to_py(x; swap_batch_dim=true)
                            z_py = to_py(z; swap_batch_dim=true)
                            cond_py = isnothing(cond) ? nothing : to_py(cond; swap_batch_dim=true)
                            mask_py = isnothing(mask) ? nothing : to_py(mask; swap_batch_dim=true).to(py_dtype(T))
                            y_py = py_layer(x_py, z_py, cond_py, mask_py)

                            @testset "Python parity ($T)" begin
                                @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                            end

                            @testset "Type-stability ($T)" begin
                                @test_nowarn @inferred jl_layer(x, z, cond, ps, st)
                            end
                        end
                    end
                end
            end

            @testset "CrossAttentionPairBias" begin
                # TODO:
            end
        end
    end
end
