
"""
    MSAModuleStack(; chn_m, chn_z, chn_hidden_msa_att, chn_hidden_opm, chn_hidden_mul, chn_hidden_pair_att,
                   no_heads_msa, no_heads_pair, transition_n, no_blocks, opm_first=false)

Full AF3 MSA module (Algorithm 8) — 1:1 port of
`openfold3.core.model.latent.msa_module.MSAModuleStack`. Runs `no_blocks` `MSAModuleBlock`s
sequentially and returns **only** the updated pair representation `z`; the MSA representation `m`
is discarded after the final block. Block state is keyed `block_1 … block_n`.

# Inputs
- `m`: MSA representation `[chn_m, N_token, N_seq, B]`
- `z`: pair representation `[chn_z, N_token, N_token, B]`
- `msa_mask`: `AbstractArray{Bool}` `[N_token, N_seq, B]` or `nothing`
- `pair_mask`: `AbstractArray{Bool}` `[N_token, N_token, B]` or `nothing`

# Returns
- `z`: updated pair representation `[chn_z, N_token, N_token, B]` (`m` is NOT returned)
- `st`: updated state
"""
struct MSAModuleStack{B} <: Lux.AbstractLuxContainerLayer{(:blocks,)}
    blocks::B
end

function MSAModuleStack(; chn_m::Int, chn_z::Int, chn_hidden_msa_att::Int, chn_hidden_opm::Int,
                        chn_hidden_mul::Int, chn_hidden_pair_att::Int, no_heads_msa::Int,
                        no_heads_pair::Int, transition_n::Int, no_blocks::Int,
                        opm_first::Bool=false)
    block_nt = NamedTuple{Tuple(Symbol("block_$i") for i in 1:no_blocks)}(
        ntuple(_ -> MSAModuleBlock(;
            chn_m, chn_z, chn_hidden_msa_att, chn_hidden_opm, chn_hidden_mul, chn_hidden_pair_att,
            no_heads_msa, no_heads_pair, transition_n, opm_first,
        ), no_blocks)
    )
    return MSAModuleStack(Lux.Chain(block_nt))
end

# config-NamedTuple compatibility shim
MSAModuleStack(config::NamedTuple) = MSAModuleStack(;
    chn_m=config.c_m, chn_z=config.c_z, chn_hidden_msa_att=config.c_hidden_msa_att,
    chn_hidden_opm=config.c_hidden_opm, chn_hidden_mul=config.c_hidden_mul,
    chn_hidden_pair_att=config.c_hidden_pair_att, no_heads_msa=config.no_heads_msa,
    no_heads_pair=config.no_heads_pair, transition_n=config.transition_n,
    no_blocks=config.no_blocks, opm_first=get(config, :opm_first, false),
)

# NamedTuple-input dispatch → positional forward
(l::MSAModuleStack)(inputs::NamedTuple, ps, st) = l(
    inputs.m, inputs.z,
    get(inputs, :msa_mask, nothing), get(inputs, :pair_mask, nothing),
    ps, st,
)

function (l::MSAModuleStack)(m, z, msa_mask, pair_mask, ps, st)
    # Thread `(; m, z, msa_mask, pair_mask)` through the block chain (type-stable);
    # discard the MSA representation at the end, returning only the pair representation.
    out, st_blocks = l.blocks((; m, z, msa_mask, pair_mask), ps.blocks, st.blocks)
    return out.z, merge(st, (; blocks=st_blocks))
end
