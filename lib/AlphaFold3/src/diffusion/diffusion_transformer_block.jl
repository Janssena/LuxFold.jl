
"""
    DiffusionTransformerBlock(; chn_a, chn_cond, chn_pair, chn_hidden, no_heads,
                              n_transition, n_query=nothing, n_key=nothing,
                              use_ada_layer_norm=true)

A single diffusion transformer block (AF3 Algorithm 23): `AttentionPairBias`
(self-attention, when `n_query`/`n_key` are `nothing`) or `SequenceLocalAttentionPairBias`
(sequence-local cross-attention, when `n_query`/`n_key` are given) followed by a
`ConditionedTransitionBlock`, both as residual updates on `a`.

# Keyword Arguments
- `chn_a`: Channel dim of the activation
- `chn_cond`: Channel dim of the single conditioning (`s`)
- `chn_pair`: Channel dim of the pair representation (`z`)
- `chn_hidden`: Head dim for attention
- `no_heads`: Number of attention heads
- `n_transition`: Width multiplier for the SwiGLU hidden dim
- `n_query`, `n_key`: Block sizes for sequence-local attention (both `nothing` → global self-attention)
- `use_ada_layer_norm`: Use AdaLN in `SequenceLocalAttentionPairBias` (default: `true`)

# Inputs
- `a`: Activation `[chn_a, N, B]`
- `s`: Single conditioning `[chn_cond, N, B]`
- `z`: Pair representation `[chn_pair, N, N, B]`
- `mask`: `AbstractArray{Bool}` `[N, B]` or `nothing`

# Returns
- `a`: Updated activation `[chn_a, N, B]`
- `st`: Updated state
"""
struct DiffusionTransformerBlock{A,T} <: Lux.AbstractLuxContainerLayer{(:attention_pair_bias, :conditioned_transition)}
    attention_pair_bias::A
    conditioned_transition::T
end

function DiffusionTransformerBlock(;
    chn_a::Int, chn_cond::Int, chn_pair::Int, chn_hidden::Int, no_heads::Int,
    n_transition::Int, n_query=nothing, n_key=nothing,
    use_ada_layer_norm::Bool=true,
)
    attention_pair_bias = if isnothing(n_query)
        AttentionPairBias(
            chn_a, chn_pair, chn_hidden, no_heads;
            chn_cond, use_gate=true, fuse_qkv=false,
            affine=(layer_norm_in=(layer_norm_a=false, layer_norm_s=true), layer_norm_z=true),
            use_bias=(false, (layer_norm_in=(false, (gate=true, shift=false)),
                              mha=(false, (q=true,)), linear_out=true)),
        )
    else
        SequenceLocalAttentionPairBias(
            chn_a, chn_a, chn_a, chn_cond, chn_pair, chn_hidden, no_heads, n_query, n_key;
            use_adaln=use_ada_layer_norm,
        )
    end

    return DiffusionTransformerBlock(
        attention_pair_bias,
        ConditionedTransitionBlock(chn_a, chn_cond, n_transition),
    )
end

function DiffusionTransformerBlock(config::NamedTuple; chn_a=nothing, chn_cond=nothing, chn_pair=nothing)
    chn_a    = something(chn_a,    config.c_a)
    chn_cond = something(chn_cond, config.c_s)
    chn_pair = something(chn_pair, config.c_z)
    chn_hidden = hasproperty(config, :c_hidden_att) ? config.c_hidden_att : config.c_hidden
    return DiffusionTransformerBlock(;
        chn_a, chn_cond, chn_pair, chn_hidden,
        no_heads=config.no_heads, n_transition=config.n_transition,
        n_query=config.n_query, n_key=config.n_key,
    )
end

# NamedTuple dispatch — used when chained inside DiffusionTransformer
function (l::DiffusionTransformerBlock)(inputs::NamedTuple, ps, st)
    a, st_new = l(
        inputs.a,
        inputs.s,
        inputs.z,
        get(inputs, :mask, nothing),
        ps, st,
    )
    return merge(inputs, (; a)), st_new
end

function (l::DiffusionTransformerBlock)(a, s, z, mask, ps, st)
    a_attn, st_apb = l.attention_pair_bias((; x=a, z, cond=s, mask), ps.attention_pair_bias, st.attention_pair_bias)
    a = a .+ a_attn

    a_cond_trans, st_ctr = l.conditioned_transition((; a, s, mask), ps.conditioned_transition, st.conditioned_transition)
    a = a .+ a_cond_trans

    return a, merge(st, (; attention_pair_bias=st_apb, conditioned_transition=st_ctr))
end
