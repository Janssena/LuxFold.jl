
"""
    DiffusionTransformer(; chn_a, chn_cond, chn_pair, chn_hidden, no_heads,
                         no_blocks, n_transition, n_query=nothing, n_key=nothing,
                         use_ada_layer_norm=true)

Stack of `no_blocks` `DiffusionTransformerBlock`s (AF3 Algorithm 23). On the
sequence-local cross-attention path (`n_query`/`n_key` given), applies an offset-free
`LayerNorm` to `z` before the blocks; on the self-attention path (`n_query=nothing`),
`layer_norm_z` is a `NoOpLayer`.

# Inputs
- `a`: Activation `[chn_a, N, B]`
- `s`: Single conditioning `[chn_cond, N, B]`
- `z`: Pair representation `[chn_pair, N, N, B]` (or `[chn_pair, n_query, n_key, N_blocks, B]` for cross-attention)
- `mask`: `AbstractArray{Bool}` `[N, B]` or `nothing`

# Returns
- `a`: Updated activation `[chn_a, N, B]`
- `st`: Updated state
"""
struct DiffusionTransformer{LZ,B} <: Lux.AbstractLuxContainerLayer{(:layer_norm_z, :blocks)}
    layer_norm_z::LZ
    blocks::B
end

function DiffusionTransformer(;
    chn_a::Int, chn_cond::Int, chn_pair::Int, chn_hidden::Int, no_heads::Int,
    no_blocks::Int, n_transition::Int, n_query=nothing, n_key=nothing,
    use_ada_layer_norm::Bool=true,
)
    layer_norm_z = if !isnothing(n_query)
        LayerNormNoBias((chn_pair, 1, 1, 1); dims=1)
    else
        Lux.NoOpLayer()
    end

    blocks = Tuple(
        DiffusionTransformerBlock(;
            chn_a, chn_cond, chn_pair, chn_hidden, no_heads, n_transition,
            n_query, n_key, use_ada_layer_norm,
        ) for _ in 1:no_blocks
    )

    return DiffusionTransformer(layer_norm_z, Lux.Chain(blocks...))
end

function DiffusionTransformer(config::NamedTuple; chn_a=nothing, chn_cond=nothing, chn_pair=nothing)
    chn_a    = something(chn_a,    config.c_a)
    chn_cond = something(chn_cond, config.c_s)
    chn_pair = something(chn_pair, config.c_z)
    chn_hidden = hasproperty(config, :c_hidden_att) ? config.c_hidden_att : config.c_hidden
    return DiffusionTransformer(;
        chn_a, chn_cond, chn_pair, chn_hidden,
        no_heads=config.no_heads, no_blocks=config.no_blocks,
        n_transition=config.n_transition, n_query=config.n_query, n_key=config.n_key,
    )
end

(l::DiffusionTransformer)(inputs::NamedTuple, ps, st) = l(
    inputs.a,
    inputs.s,
    inputs.z,
    get(inputs, :mask, nothing),
    ps, st,
)

function (l::DiffusionTransformer)(a, s, z, mask, ps, st)
    z, st_ln_z = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)
    out, st_blocks = l.blocks((; a, s, z, mask), ps.blocks, st.blocks)
    return out.a, merge(st, (; layer_norm_z=st_ln_z, blocks=st_blocks))
end
