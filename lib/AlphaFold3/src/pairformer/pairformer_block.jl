
"""
    PairFormerBlock(; chn_s, chn_z, chn_hidden_pair_bias, no_heads_pair_bias,
                    chn_hidden_mul, chn_hidden_pair_att, no_heads_pair, transition_n)

One iteration of AF3 Algorithm 17 — 1:1 port of
`openfold3.core.model.latent.pairformer.PairFormerBlock`. Jointly refines the single
representation `s [chn_s, N, B]` and the pair representation `z [chn_z, N, N, B]`:

1. `z = pair_block(z, pair_mask)`                       — pair stack update
2. `s += attn_pair_bias(s, z, mask=single_mask)`        — single attention with pair bias
3. `s += single_transition(s, single_mask)`             — single SwiGLU transition

`attn_pair_bias` runs without conditioning (`use_ada_layer_norm=false`, `chn_cond=nothing`)
and with gating enabled, matching the Python reference. Both `layer_norm_a`/`layer_norm_z`
are affine `LayerNorm`s (with offset), `linear_z` is bias-free, and the MHA carries a bias
only on the query projection (`att_pair_bias_mha_init`).

# Inputs
- `s`: Single embedding `[chn_s, N, B]`
- `z`: Pair embedding `[chn_z, N, N, B]`
- `single_mask`: `AbstractArray{Bool}` `[N, B]` or `nothing`
- `pair_mask`: `AbstractArray{Bool}` `[N, N, B]` or `nothing`

# Returns
- `(s, z)`: Updated single `[chn_s, N, B]` and pair `[chn_z, N, N, B]` embeddings
- `st`: Updated state
"""
struct PairFormerBlock{PB,APB,ST} <:
       Lux.AbstractLuxContainerLayer{(:pair_block, :attn_pair_bias, :single_transition)}
    pair_block::PB
    attn_pair_bias::APB
    single_transition::ST
end

function PairFormerBlock(;
    chn_s::Int, chn_z::Int, chn_hidden_pair_bias::Int, no_heads_pair_bias::Int,
    chn_hidden_mul::Int, chn_hidden_pair_att::Int, no_heads_pair::Int, transition_n::Int,
)
    pair_block = PairBlock(;
        chn_z, chn_hidden_mul, chn_hidden_pair_att, no_heads_pair, transition_n,
    )

    # openfold AttentionPairBias (use_ada_layer_norm=False, gating=True):
    #   layer_norm_a / layer_norm_z affine LayerNorms (with offset), linear_z bias=false,
    #   mha unfused with q bias=true (k/v/gate/out bias=false).
    attn_pair_bias = AttentionPairBias(
        chn_s, chn_z, chn_hidden_pair_bias, no_heads_pair_bias;
        chn_cond=nothing, use_gate=true, fuse_qkv=false, affine=true,
        use_bias=(layer_norm_in=true, layer_norm_z=true, linear_z=false,
                  mha=(false, (q=true,)), linear_out=false),
    )

    single_transition = SwiGLUTransition(chn_s, transition_n)   # rank-3 single [chn_s, N, B]

    return PairFormerBlock(pair_block, attn_pair_bias, single_transition)
end

# config-NamedTuple compatibility shim
PairFormerBlock(config::NamedTuple) = PairFormerBlock(;
    chn_s=config.c_s, chn_z=config.c_z,
    chn_hidden_pair_bias=config.c_hidden_pair_bias, no_heads_pair_bias=config.no_heads_pair_bias,
    chn_hidden_mul=config.c_hidden_mul, chn_hidden_pair_att=config.c_hidden_pair_att,
    no_heads_pair=config.no_heads_pair, transition_n=config.transition_n,
)

# NamedTuple dispatch — threads `(; s, z, single_mask, pair_mask)` through `Lux.Chain`
# (used by `PairFormerStack`). The positional forward returns only the updated fields
# `(; s, z)`; `merge(inputs, out)` passes the masks through unchanged.
function (l::PairFormerBlock)(inputs::NamedTuple, ps, st)
    out, st_new = l(
        inputs.s, inputs.z,
        get(inputs, :single_mask, nothing), 
        get(inputs, :pair_mask, nothing),
        ps, st,
    )
    return merge(inputs, out), st_new
end

function (l::PairFormerBlock)(s, z, single_mask, pair_mask, ps, st)
    # 1. pair stack update
    z, st_pb = l.pair_block(z, pair_mask, ps.pair_block, st.pair_block)

    # 2. single attention with pair bias (no conditioning ⇒ single_mask is the attention mask)
    s_att, st_apb = l.attn_pair_bias((; x=s, z, mask=single_mask), ps.attn_pair_bias, st.attn_pair_bias)
    s = s .+ s_att

    # 3. single SwiGLU transition (output masked, matching `_mask_trans=true`)
    s_trans, st_sgl_tr = l.single_transition(s, single_mask, ps.single_transition, st.single_transition)
    s = s .+ s_trans

    return (; s, z), merge(st, (;
        pair_block=st_pb, attn_pair_bias=st_apb, single_transition=st_sgl_tr,
    ))
end
