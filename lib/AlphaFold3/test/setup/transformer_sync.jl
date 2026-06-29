# Shared Python weight-sync helpers for the AF3 DiffusionTransformer stack.
# Used by both atom_module (cross-attention) and diffusion (self-attention) tests.
# Guarded so it can be `include`d from multiple test files in one session.

if !@isdefined(_AF3_TRANSFORMER_SYNC_LOADED)
    const _AF3_TRANSFORMER_SYNC_LOADED = true

    # ConditionedTransitionBlock (AF3 Algorithm 25)
    function sync_conditioned_transition!(py, ps)
        sync_af3_adaln!(py.layer_norm, ps.layer_norm)
        # Python SwiGLU: swish(linear_a(x)) * linear_b(x).
        # Shared (LuxTriangleAttention) SwiGLU, unfused: v=linear(x), g=gate(x), y=v*swish(g)
        #   ⇒ gate ↔ linear_a (swish branch), linear ↔ linear_b (value branch).
        sync_dense!(py.swiglu.linear_a, ps.swiglu.gate)
        sync_dense!(py.swiglu.linear_b, ps.swiglu.linear)
        sync_dense!(py.linear_g, ps.linear_g)
        sync_dense!(py.linear_out, ps.linear_out)
        return nothing
    end

    # One DiffusionTransformerBlock — self- or cross-attention.
    function sync_af3_diffusion_transformer_block!(py, ps; cross_attention::Bool)
        if cross_attention
            sync_af3_cross_attention_pair_bias!(py.attention_pair_bias, ps.attention_pair_bias)
        else
            sync_af3_attention_pair_bias!(py.attention_pair_bias, ps.attention_pair_bias)
        end
        sync_conditioned_transition!(py.conditioned_transition, ps.conditioned_transition)
        return nothing
    end

    # Full DiffusionTransformer stack. `cross_attention=true` (atom transformer)
    # also syncs the stack-level offset-free `layer_norm_z`.
    function sync_diffusion_transformer!(py, ps; cross_attention::Bool=true)
        if cross_attention
            sync_layernorm!(py.layer_norm_z, ps.layer_norm_z)
        end
        py_blocks = collect(py.blocks)
        for (i, name) in enumerate(keys(ps.blocks))
            sync_af3_diffusion_transformer_block!(py_blocks[i], ps.blocks[name]; cross_attention)
        end
        return nothing
    end
end
