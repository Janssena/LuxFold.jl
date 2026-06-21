"""
    ExtraMSAStack(chn_m, chn_z; no_blocks=4, kwargs...)

ExtraMSA trunk: chains `no_blocks` `ExtraMSABlock` iterations via `Lux.Chain`
and returns ONLY the updated pair representation `z` (MSA is discarded).

# Arguments
- `chn_m`: ExtraMSA channel dimension (default: 64)
- `chn_z`: Pair channel dimension (default: 128)

# Keyword Arguments
- `no_blocks`: Number of ExtraMSA blocks (default: 4; use fewer for testing)
- All keyword arguments of `ExtraMSABlock` are forwarded to each block

# Inputs
- `m`: ExtraMSA tensor `[chn_m, N_res, N_seq, B]`
- `z`: Pair tensor `[chn_z, N_res, N_res, B]`
- `msa_mask`: Bool mask `[N_res, N_seq, B]` or `nothing`
- `pair_mask`: Bool mask `[N_res, N_res, B]` or `nothing`

# Returns
- `(z_out, st_new)` — updated pair representation (MSA `m` is discarded)
"""
struct ExtraMSAStack{B} <: Lux.AbstractLuxContainerLayer{(:blocks,)}
    blocks::B    # Lux.Chain of ExtraMSABlocks
end

function ExtraMSAStack(
    chn_m::Int=64, chn_z::Int=128;
    no_blocks::Int=4,
    kwargs...
)
    block_nt = NamedTuple{Tuple(Symbol("block_$i") for i in 1:no_blocks)}(
        ntuple(_ -> ExtraMSABlock(chn_m, chn_z; kwargs...), no_blocks)
    )
    blocks = Lux.Chain(block_nt)
    return ExtraMSAStack(blocks)
end

(l::ExtraMSAStack)(m, z, msa_mask, pair_mask, ps, st) = l(
    (; m, z, msa_mask, pair_mask), ps, st
)

function (l::ExtraMSAStack)(inputs::NamedTuple, ps, st)
    output, st_blocks = l.blocks(inputs, ps.blocks, st.blocks)
    # Only z is returned; ExtraMSA discards m after processing
    st_new = merge(st, (; blocks=st_blocks))
    return output.z, st_new
end

# Canonical config factories — identical dims for monomer and multimer
ExtraMSAStack(s::Symbol; kwargs...) = ExtraMSAStack(static(s); kwargs...)
ExtraMSAStack(::StaticSymbol; kwargs...) = ExtraMSAStack(64, 128; merge((no_blocks=4,), kw)...)