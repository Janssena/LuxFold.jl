
# openfold-3 bias conventions (see default_linear_init_config.py):
#   TriangleMultiplication (tri_mul_init): all linears bias=false; non-fused core + non-fused GLUs.
#   TriangleAttention (tri_att_init = mha_bias_init): layer_norm affine (+offset), linear_z bias=false,
#     mha (q/k/v/gate/out) all bias=false.
const _AF3_TRI_ATT_USE_BIAS = (false, (layer_norm=true,))   # ⇒ layer_norm=true, linear=false, mha=false

"""
    PairBlock(; chn_z, chn_hidden_mul, chn_hidden_pair_att, no_heads_pair, transition_n)
    PairBlock(config::NamedTuple)

AF3 pair-representation update block (the "pair stack" shared by the Pairformer and MSA module).
1:1 port of `openfold3.core.model.latent.base_blocks.PairBlock` with `transition_type="swiglu"`,
`fuse_projection_weights=false`. Applies, as residual updates on `z [chn_z, N, N, B]`:

1. `z += tri_mul_out(z, pair_mask)`   — triangle multiplication (outgoing)
2. `z += tri_mul_in(z, pair_mask)`    — triangle multiplication (incoming)
3. `z += tri_att_start(z, pair_mask)` — triangle attention (starting node)
4. `z += tri_att_end(z, pair_mask)`   — triangle attention (ending node; transpose handled internally)
5. `z += pair_transition(z, pair_mask)` — SwiGLU pair transition (output masked, `_mask_trans=true`)

# Inputs
- `z`: Pair embedding `[chn_z, N, N, B]`
- `pair_mask`: `AbstractArray{Bool}` `[N, N, B]` or `nothing`

# Returns
- `z`: Updated pair embedding `[chn_z, N, N, B]`
- `st`: Updated state
"""
struct PairBlock{TMO,TMI,TAS,TAE,PT} <:
       Lux.AbstractLuxContainerLayer{(:tri_mul_out, :tri_mul_in, :tri_att_start, :tri_att_end, :pair_transition)}
    tri_mul_out::TMO
    tri_mul_in::TMI
    tri_att_start::TAS
    tri_att_end::TAE
    pair_transition::PT
end

function PairBlock(; chn_z::Int, chn_hidden_mul::Int, chn_hidden_pair_att::Int,
                   no_heads_pair::Int, transition_n::Int)
    return PairBlock(
        # openfold-3 tri_mul_init: all GLU linears bias=false; LayerNorms stay affine (always have offset).
        TriangleMultiplication(chn_z, chn_hidden_mul; is_outgoing=true,  use_bias=false, fused=false, fused_glu=false),
        TriangleMultiplication(chn_z, chn_hidden_mul; is_outgoing=false, use_bias=false, fused=false, fused_glu=false),
        TriangleAttention(chn_z, chn_hidden_pair_att, no_heads_pair; is_starting=true,  use_bias=_AF3_TRI_ATT_USE_BIAS),
        TriangleAttention(chn_z, chn_hidden_pair_att, no_heads_pair; is_starting=false, use_bias=_AF3_TRI_ATT_USE_BIAS),
        SwiGLUTransition(chn_z, transition_n; is_4d=true),
    )
end

# config-NamedTuple compatibility shim (used by the MSA module)
PairBlock(config::NamedTuple) = PairBlock(;
    chn_z=config.c_z, chn_hidden_mul=config.c_hidden_mul,
    chn_hidden_pair_att=config.c_hidden_pair_att, no_heads_pair=config.no_heads_pair,
    transition_n=config.transition_n,
)

function (l::PairBlock)(z, pair_mask, ps, st)
    z_mul_out, st_tmo = l.tri_mul_out(z, pair_mask, ps.tri_mul_out, st.tri_mul_out)
    z = z .+ z_mul_out

    z_mul_in, st_tmi = l.tri_mul_in(z, pair_mask, ps.tri_mul_in, st.tri_mul_in)
    z = z .+ z_mul_in

    z_att_start, st_tas = l.tri_att_start(z, pair_mask, ps.tri_att_start, st.tri_att_start)
    z = z .+ z_att_start

    z_att_end, st_tae = l.tri_att_end(z, pair_mask, ps.tri_att_end, st.tri_att_end)
    z = z .+ z_att_end

    z_trans, st_pt = l.pair_transition(z, pair_mask, ps.pair_transition, st.pair_transition)
    z = z .+ z_trans

    return z, merge(st, (;
        tri_mul_out=st_tmo, tri_mul_in=st_tmi,
        tri_att_start=st_tas, tri_att_end=st_tae, pair_transition=st_pt,
    ))
end
