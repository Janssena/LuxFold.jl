"""
    EvoformerStack(chn_m, chn_z, chn_s; no_blocks=48, kwargs...)

Full Evoformer trunk: chains `no_blocks` `EvoformerBlock` iterations via `Lux.Chain`
and produces the single representation `s` by projecting the first MSA row.

# Arguments
- `chn_m`: MSA channel dimension (default: 256)
- `chn_z`: Pair channel dimension (default: 128)
- `chn_s`: Single representation dimension (default: 384)

# Keyword Arguments
- `no_blocks`: Number of Evoformer blocks (default: 48; use fewer for testing)
- All keyword arguments of `EvoformerBlock` are forwarded to each block

# Inputs
- `m`: MSA tensor `[chn_m, N_res, N_seq, B]`
- `z`: Pair tensor `[chn_z, N_res, N_res, B]`
- `msa_mask`: Bool mask `[N_res, N_seq, B]` or `nothing`
- `pair_mask`: Bool mask `[N_res, N_res, B]` or `nothing`

# Returns
- `((; m, z, s), st_new)` — updated MSA, pair, and single representations
"""
struct EvoformerStack{B, SP} <: Lux.AbstractLuxContainerLayer{(:blocks, :single_projection)}
    blocks::B            # Lux.Chain of EvoformerBlocks
    single_projection::SP
end

function EvoformerStack(
    chn_m::Int=256, chn_z::Int=128, chn_s::Int=384;
    no_blocks::Int=48,
    kwargs...
)
    block_nt = NamedTuple{Tuple(Symbol("block_$i") for i in 1:no_blocks)}(
        ntuple(_ -> EvoformerBlock(chn_m, chn_z; kwargs...), no_blocks)
    )
    blocks = Lux.Chain(block_nt)
    single_projection = Lux.Dense(chn_m => chn_s; use_bias=true)
    return EvoformerStack(blocks, single_projection)
end

# NamedTuple outer dispatch
(l::EvoformerStack)(inputs::NamedTuple, ps, st) = l(
    inputs.m, 
    inputs.z, 
    get(inputs, :msa_mask, nothing), 
    get(inputs, :pair_mask, nothing), 
    ps, st
)

function (l::EvoformerStack)(m, z, msa_mask, pair_mask, ps, st)
    output, st_blocks = l.blocks(
        (; m, z, msa_mask, pair_mask),
        ps.blocks, st.blocks
    )

    # Single representation: project first MSA row (first sequence, dim 3 index 1).
    # m layout [chn_m, N_res, N_seq, B] → m[:, :, 1, :] = [chn_m, N_res, B]
    s, st_sp = l.single_projection(output.m[:, :, 1, :], ps.single_projection, st.single_projection)

    st_new = merge(st, (; blocks=st_blocks, single_projection=st_sp))
    return (; m=output.m, z=output.z, s), st_new
end

# Canonical config factories — identical dims for monomer and multimer
EvoformerStack(s::Symbol; kwargs...) = EvoformerStack(static(s); kwargs...)
EvoformerStack(::StaticSymbol; kwargs...) = EvoformerStack(256, 128, 384; merge((no_blocks=48,), kw)...)