
"""
    PairFormerStack(; chn_s, chn_z, chn_hidden_pair_bias, no_heads_pair_bias,
                    chn_hidden_mul, chn_hidden_pair_att, no_heads_pair, transition_n,
                    no_blocks)

Full AF3 Pairformer trunk (Algorithm 17) — 1:1 port of
`openfold3.core.model.latent.pairformer.PairFormerStack`. Runs `no_blocks`
`PairFormerBlock`s sequentially, threading the single `s` and pair `z` representations
through each block. Sub-block state is keyed `block_1, …, block_n`.

# Inputs
- `s`: Single embedding `[chn_s, N, B]`
- `z`: Pair embedding `[chn_z, N, N, B]`
- `single_mask`: `AbstractArray{Bool}` `[N, B]` or `nothing`
- `pair_mask`: `AbstractArray{Bool}` `[N, N, B]` or `nothing`

# Returns
- `(s, z)`: Updated single `[chn_s, N, B]` and pair `[chn_z, N, N, B]` embeddings
- `st`: Updated state
"""
struct PairFormerStack{B} <: Lux.AbstractLuxContainerLayer{(:blocks,)}
    blocks::B    # Lux.Chain of PairFormerBlocks, keyed block_1 … block_n
end

function PairFormerStack(;
    chn_s::Int, chn_z::Int, chn_hidden_pair_bias::Int, no_heads_pair_bias::Int,
    chn_hidden_mul::Int, chn_hidden_pair_att::Int, no_heads_pair::Int, transition_n::Int,
    no_blocks::Int,
)
    block_nt = NamedTuple{Tuple(Symbol("block_$i") for i in 1:no_blocks)}(
        ntuple(_ -> PairFormerBlock(;
            chn_s, chn_z, chn_hidden_pair_bias, no_heads_pair_bias,
            chn_hidden_mul, chn_hidden_pair_att, no_heads_pair, transition_n,
        ), no_blocks)
    )
    return PairFormerStack(Lux.Chain(block_nt))
end

# config-NamedTuple compatibility shim
PairFormerStack(config::NamedTuple) = PairFormerStack(;
    chn_s=config.c_s, chn_z=config.c_z,
    chn_hidden_pair_bias=config.c_hidden_pair_bias, no_heads_pair_bias=config.no_heads_pair_bias,
    chn_hidden_mul=config.c_hidden_mul, chn_hidden_pair_att=config.c_hidden_pair_att,
    no_heads_pair=config.no_heads_pair, transition_n=config.transition_n,
    no_blocks=config.no_blocks,
)

# NamedTuple-input dispatch → positional forward (matches the AF2 stack convention).
(l::PairFormerStack)(inputs::NamedTuple, ps, st) = l(
    inputs.s, inputs.z,
    get(inputs, :single_mask, nothing), get(inputs, :pair_mask, nothing),
    ps, st,
)

function (l::PairFormerStack)(s, z, single_mask, pair_mask, ps, st)
    # Thread `(; s, z, single_mask, pair_mask)` through the chain of blocks; `Lux.Chain`'s
    # recursion keeps this type-stable (masks pass through each block unchanged).
    out, st_blocks = l.blocks((; s, z, single_mask, pair_mask), ps.blocks, st.blocks)
    return (; s=out.s, z=out.z), merge(st, (; blocks=st_blocks))
end
