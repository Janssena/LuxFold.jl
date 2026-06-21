# ===  Python imports  ===
const PyMSAColumnAttention       = pyimport("openfold.model.msa").MSAColumnAttention
const PyMSAColumnGlobalAttention = pyimport("openfold.model.msa").MSAColumnGlobalAttention

rng = Random.Xoshiro(42)

# Tensor layout conventions:
#   Julia MSA:  [C, N_res, N_seq, B]   (channel-first, batch-last)
#   Python MSA: [B, N_seq, N_res, C]   (batch-first, channel-last)
#
#   Julia msa_mask:  [N_res, N_seq, B]
#   Python mask:     [B, N_seq, N_res]  (float: 0.0 = masked, 1.0 = valid)
#
# Julia → Python:  permutedims(m, (4,3,2,1))   [C,N_res,N_seq,B] → [B,N_seq,N_res,C]
# Python → Julia:  to_jl(swap=true) gives [C,N_seq,N_res,B]; permute (1,3,2,4) → [C,N_res,N_seq,B]

function jl_to_py_msa(m)
    # [C, N_res, N_seq, B] → Python [B, N_seq, N_res, C]
    to_py(permutedims(m, (4, 3, 2, 1)); swap_batch_dim=false)
end

function jl_to_py_mask(mask, T)
    # [N_res, N_seq, B] → Python [B, N_seq, N_res]
    to_py(permutedims(mask, (3, 2, 1)); swap_batch_dim=false).to(py_dtype(T))
end

function py_to_jl_msa(m_py)
    # Python [B, N_seq, N_res, C] → Julia [C, N_res, N_seq, B]
    permutedims(to_jl(m_py; swap_batch_dim=true), (1, 3, 2, 4))
end

# ===  Sync helpers  ===

function sync_msa_column_attention!(py_layer, jl_ps)
    # Python: py_layer._msa_att.layer_norm_m, py_layer._msa_att.mha
    sync_layernorm!(py_layer._msa_att.layer_norm_m, jl_ps.layer_norm)
    sync_af3_attention!(py_layer._msa_att.mha, jl_ps.mha)
end

function sync_msa_column_global_attention!(py_layer, jl_ps)
    # Python: py_layer.layer_norm_m + py_layer.global_attention.{linear_q,k,v,g,o}
    sync_layernorm!(py_layer.layer_norm_m, jl_ps.layer_norm)
    sync_dense!(py_layer.global_attention.linear_q, jl_ps.linear_q)
    sync_dense!(py_layer.global_attention.linear_k, jl_ps.linear_k)
    sync_dense!(py_layer.global_attention.linear_v, jl_ps.linear_v)
    sync_dense!(py_layer.global_attention.linear_g, jl_ps.linear_g)
    sync_dense!(py_layer.global_attention.linear_o, jl_ps.linear_o)
end

# ===  Shared dimensions  ===
C_m      = 16
C_hidden = 4
no_heads = 2
N_res    = 8   # residues (dim 2 of MSA)
N_seq    = 5   # sequences (dim 3 of MSA)
B        = 2

# ===  MSAColumnAttention tests  ===

@testset "MSAColumnAttention" begin
    mask_configs = (
        ("No mask",     nothing),
        ("Random mask", rand(rng, Bool, N_res, N_seq, B))
    )
    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            jl_layer = MSAColumnAttention(C_m, C_hidden, no_heads)
            ps, st   = Lux.setup(rng, jl_layer) |> convert_types(T)

            py_layer = PyMSAColumnAttention(C_m, C_hidden, no_heads)
            sync_msa_column_attention!(py_layer, ps)

            for (name, mask) in mask_configs
                @testset "$name" begin
                    m_jl = randn(rng, T, C_m, N_res, N_seq, B)  # [C, N_res, N_seq, B]
                    m_py = jl_to_py_msa(m_jl)

                    mask_py = isnothing(mask) ? nothing :
                              jl_to_py_mask(mask, T)

                    m_out_jl, _ = jl_layer(m_jl, mask, ps, st)
                    m_out_py = py_layer(m_py;
                        mask=mask_py, chunk_size=nothing,
                        use_deepspeed_evo_attention=false,
                        use_cuequivariance_attention=false,
                        use_lma=false, use_flash=false)

                    @testset "Python parity" begin
                        @test m_out_jl ≈ py_to_jl_msa(m_out_py)
                    end
                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer(m_jl, mask, ps, st)
                    end
                end
            end
        end
    end
end

# ===  MSAColumnGlobalAttention tests  ===
# ExtraMSA defaults: c_in=64, c_hidden=8, no_heads=8 — use small dims here.

C_g      = 16
C_hid_g  = 4
heads_g  = 2

# Global attention requires at least one valid sequence per batch×residue.
# Defined at top level so it's available to evoformer_block.test.jl and friends.
function make_global_mask(rng, N_res, N_seq, B)
    mask = rand(rng, Bool, N_res, N_seq, B)
    for b in 1:B, r in 1:N_res
        mask[r, :, b] .= any(mask[r, :, b]) ? mask[r, :, b] : trues(N_seq)
    end
    return mask
end

@testset "MSAColumnGlobalAttention" begin
    mask_configs = (
        ("No mask",     nothing),
        ("Random mask", make_global_mask(rng, N_res, N_seq, B))
    )
    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            jl_layer = MSAColumnGlobalAttention(C_g, C_hid_g, heads_g)
            ps, st   = Lux.setup(rng, jl_layer) |> convert_types(T)

            py_layer = PyMSAColumnGlobalAttention(C_g, C_hid_g, heads_g)
            sync_msa_column_global_attention!(py_layer, ps)

            for (name, mask) in mask_configs
                @testset "$name" begin
                    m_jl = randn(rng, T, C_g, N_res, N_seq, B)
                    m_py = jl_to_py_msa(m_jl)

                    # GlobalAttention needs a float mask (0/1); Julia Bool → Python float.
                    # The Nothing dispatch fills trues internally, so pass the actual mask.
                    mask_actual = isnothing(mask) ? trues(N_res, N_seq, B) : mask
                    mask_py = jl_to_py_mask(mask_actual, T)

                    m_out_jl, _ = jl_layer(m_jl, mask, ps, st)
                    m_out_py = py_layer(m_py;
                        mask=mask_py, chunk_size=nothing, use_lma=false)

                    @testset "Python parity" begin
                        @test m_out_jl ≈ py_to_jl_msa(m_out_py)
                    end
                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer(m_jl, mask, ps, st)
                    end
                end
            end
        end
    end
end
