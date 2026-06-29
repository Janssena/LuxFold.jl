# Python weight-sync helpers + layout converters for the AF3 MSA module.
# Reuses sync_pair_block! / sync_swiglu_transition! from the pairformer helpers.

if !@isdefined(_AF3_MSA_SYNC_LOADED)
    const _AF3_MSA_SYNC_LOADED = true

    isdefined(@__MODULE__, :sync_pair_block!) || include("../pairformer/sync_helpers.jl")

    # MSA tensor layout: Julia [c_m, N_token, N_seq, B] <-> Python [B, N_seq, N_token, c_m]
    to_py_msa(m) = to_py(permutedims(m, (4, 3, 2, 1)); swap_batch_dim=false)
    to_jl_msa(y) = permutedims(to_jl(y; swap_batch_dim=false), (4, 3, 2, 1))
    # MSA mask: Julia [N_token, N_seq, B] -> Python [B, N_seq, N_token]
    to_py_msamask(mask, T) = to_py(permutedims(T.(mask), (3, 2, 1)); swap_batch_dim=false)
    # pair mask: Julia [N, N, B] -> Python [B, N, N]
    to_py_pairmask(mask, T) = to_py(permutedims(T.(mask), (3, 1, 2)); swap_batch_dim=false)

    # MSAPairWeightedAveraging (AF3 Alg 10): layer_norm_m/z, linear_z/v/g, linear_o.
    sync_msa_pair_weighted_averaging!(py, ps) = sync_pwa!(py, ps;
        ref=(layer_norm_m=:layer_norm_m, layer_norm_z=:layer_norm_z, linear_z=:linear_z,
             linear_v=:linear_v, linear_g=:linear_g, linear_out=:linear_o))

    function sync_msa_module_block!(py, ps)
        sync_msa_pair_weighted_averaging!(py.msa_att_row, ps.msa_att_row)
        sync_swiglu_transition!(py.msa_transition, ps.msa_transition)
        sync_af3_opm!(py.outer_product_mean, ps.outer_product_mean)
        sync_pair_block!(py.pair_stack, ps.pair_block)   # Python attr `pair_stack`
        return nothing
    end

    function sync_msa_module_stack!(py, ps)
        py_blocks = collect(py.blocks)
        for (i, name) in enumerate(keys(ps.blocks))
            sync_msa_module_block!(py_blocks[i], ps.blocks[name])
        end
        return nothing
    end
end
