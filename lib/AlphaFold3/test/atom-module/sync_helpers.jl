# Model-specific Python weight-sync helpers for the AF3 atom module.

# Shared DiffusionTransformer sync helpers (sync_conditioned_transition!, sync_diffusion_transformer!)
include("../setup/transformer_sync.jl")

_py_linears(seq) = [m for m in seq if pytypeof(m).__name__ == "Linear"]

function sync_ref_atom_feature_embedder!(py, ps)
    for n in (:linear_ref_pos, :linear_ref_charge, :linear_ref_mask, :linear_ref_element,
              :linear_ref_atom_chars, :linear_ref_offset, :linear_inv_sq_dists, :linear_valid_mask)
        sync_dense!(getproperty(py, n), getproperty(ps, n))
    end
    return nothing
end

function sync_noisy_position_embedder!(py, ps)
    sync_layernorm!(py.layer_norm_s, ps.layer_norm_s)
    sync_dense!(py.linear_s, ps.linear_s)
    sync_layernorm!(py.layer_norm_z, ps.layer_norm_z)
    sync_dense!(py.linear_z, ps.linear_z)
    sync_dense!(py.linear_r, ps.linear_r)
    return nothing
end

function sync_atom_attention_encoder!(py, ps; add_noisy_pos=false)
    sync_ref_atom_feature_embedder!(py.ref_atom_feature_embedder, ps.ref_atom_feature_embedder)
    if add_noisy_pos
        sync_noisy_position_embedder!(py.noisy_position_embedder, ps.noisy_position_embedder)
    end
    sync_dense!(py.linear_l, ps.linear_l)
    sync_dense!(py.linear_m, ps.linear_m)
    pml = _py_linears(py.pair_mlp)               # 3 Linears in order
    sync_dense!(pml[1], ps.pair_mlp.layer_2)
    sync_dense!(pml[2], ps.pair_mlp.layer_3)
    sync_dense!(pml[3], ps.pair_mlp.layer_4)
    sync_diffusion_transformer!(py.atom_transformer, ps.atom_transformer)
    sync_dense!(_py_linears(py.linear_q)[1], ps.linear_q)
    return nothing
end

function sync_atom_attention_decoder!(py, ps)
    sync_dense!(py.linear_q_in, ps.linear_q_in)
    sync_diffusion_transformer!(py.atom_transformer, ps.atom_transformer)
    sync_layernorm!(py.layer_norm, ps.layer_norm)
    sync_dense!(py.linear_q_out, ps.linear_q_out)
    return nothing
end
