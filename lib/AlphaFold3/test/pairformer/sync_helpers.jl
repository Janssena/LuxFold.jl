# Python weight-sync helpers for the AF3 Pairformer.
# Guarded so it can be `include`d from multiple test files in one session.

if !@isdefined(_AF3_PAIRFORMER_SYNC_LOADED)
    const _AF3_PAIRFORMER_SYNC_LOADED = true

    # openfold-3 TriangleAttention: layer_norm (affine), linear_z, mha (q/k/v/gate/out).
    function sync_af3_triangle_attention!(py, ps)
        sync_layernorm!(py.layer_norm, ps.layer_norm)
        sync_dense!(py.linear_z, ps.linear)
        sync_af3_attention!(py.mha, ps.mha)
        return nothing
    end

    # openfold-3 TriangleMultiplicativeUpdate (non-fused): two input GLUs (a/b),
    # an output LayerNorm, and an output GLU (gated projection back to c_z).
    function sync_af3_triangle_multiplication!(py, ps)
        sync_layernorm!(py.layer_norm_in, ps.layer_norm)
        sync_glu!(py, ps.core.glu_a; ref=(linear=:linear_a_p, gate=:linear_a_g))
        sync_glu!(py, ps.core.glu_b; ref=(linear=:linear_b_p, gate=:linear_b_g))
        sync_layernorm!(py.layer_norm_out, ps.core.layer_norm_out)
        sync_glu!(py, ps.core.glu_out; ref=(linear=:linear_z, gate=:linear_g))
        return nothing
    end

    # SwiGLUTransition: LayerNorm → SwiGLU (linear_a=swish/gate, linear_b=value) → linear_out.
    function sync_swiglu_transition!(py, ps)
        sync_layernorm!(py.layer_norm, ps.layer_norm)
        sync_dense!(py.swiglu.linear_a, ps.swiglu.gate)
        sync_dense!(py.swiglu.linear_b, ps.swiglu.linear)
        sync_dense!(py.linear_out, ps.linear_out)
        return nothing
    end

    # PairBlock (the "pair stack").
    function sync_pair_block!(py, ps)
        sync_af3_triangle_multiplication!(py.tri_mul_out, ps.tri_mul_out)
        sync_af3_triangle_multiplication!(py.tri_mul_in, ps.tri_mul_in)
        sync_af3_triangle_attention!(py.tri_att_start, ps.tri_att_start)
        sync_af3_triangle_attention!(py.tri_att_end, ps.tri_att_end)
        sync_swiglu_transition!(py.pair_transition, ps.pair_transition)
        return nothing
    end

    # PairFormerBlock — note the Python attribute is `pair_stack`, the Julia field `pair_block`.
    function sync_pairformer_block!(py, ps)
        sync_pair_block!(py.pair_stack, ps.pair_block)
        sync_af3_attention_pair_bias!(py.attn_pair_bias, ps.attn_pair_bias)
        sync_swiglu_transition!(py.single_transition, ps.single_transition)
        return nothing
    end
end
