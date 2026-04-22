include("../python/alphafold3.jl");

function copy_weights_to_af3_attention!(
    py_layer::PyObject,
    ps::NamedTuple
)   
    if :weight ∈ keys(ps.qkv) # is_fused
        throw(ErrorException("Not implemented."))
    else
        sync_dense!(py_layer.linear_q, ps.qkv.q)
        sync_dense!(py_layer.linear_k, ps.qkv.k)
        sync_dense!(py_layer.linear_v, ps.qkv.v)
    end
    sync_dense!(py_layer.linear_o, ps.out)

    if !isempty(ps.gate)
        sync_dense!(py_layer.linear_g, ps.gate)
    end

    return nothing
end

rng = Random.Xoshiro(42)

@testset "AlphaFold3" begin
    N, B = 16, 2
    chn_in = 32
    num_heads = 6 # H
    head_dim = 8 # c_hidden in af3

    mask_cfg = (
        ("No mask", nothing),
        ("Random mask", rand(rng, Bool, N, B)),
        ("All-ones mask", trues(N, B)),
    )
    
    @testset "Attention" begin
        for (name, mask) in mask_cfg
            @testset "$name" begin
                for T in [Float16, Float32, Float64]
                    inf_val = T == Float16 ? Float16(1e4) : T(1e9)
                    x = randn(rng, T, chn_in, N, B)
                    bias = randn(rng, T, num_heads, N, N, B)
                    
                    # use_bias = (true, (layer_norm_a = false, layer_norm_s = false, ))
                    jl_layer = Attention(chn_in, head_dim, num_heads; fuse_qkv=false)
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    (y_jl, _), _ = jl_layer(x, bias, mask, ps, st)

                    py_layer = py"AF3Attention"(chn_in, chn_in, chn_in, head_dim, num_heads)
                    
                    copy_weights_to_af3_attention!(py_layer, ps)
            
                    x_py = to_py(x; swap_batch_dim=true)
                    mask_py = to_py(isnothing(mask) ? trues(N, B) : mask; swap_batch_dim=true).to(py_dtype(T))
                    bias_py = to_py(bias; swap_batch_dim=true)
                    
                    py"""
                    # mask in expected: [B, N]
                    # bias in expected: [B, N, N, H]
                    triangle_bias = af3_permute_final_dims($bias_py, (2, 0, 1)) # [B, H, N, N]
                    mask = $inf_val * ($mask_py - 1)
                    mask_bias = mask[:, None, None, :]
                    biases = [mask_bias, triangle_bias]
                    """
                    
                    py_layer.eval()
                    y_py = py_layer(x_py, x_py; biases=py"biases")

                    @testset "Python parity ($T)" begin
                        @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                    end

                    @testset "Type-stability ($T)" begin
                        @test_nowarn @inferred jl_layer(x, bias, mask, ps, st)
                    end
                end
            end
        end
    end
end