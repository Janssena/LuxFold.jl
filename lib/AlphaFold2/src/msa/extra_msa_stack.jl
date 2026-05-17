struct ExtraMsaStack <: Lux.AbstractExplicitContainerLayer{(:blocks,)}
    blocks::NTuple{N, ExtraMsaBlock} where N
end

function ExtraMsaStack(config::NamedTuple)
    num_blocks = config.num_extra_msa
    return ExtraMsaStack(Tuple(ExtraMsaBlock(config) for _ in 1:num_blocks))
end

function (m::ExtraMsaStack)(msa, pair, msa_mask, pair_mask, ps, st)
    st_blocks = []
    for (i, block) in enumerate(m.blocks)
        msa, pair, st_b = block(msa, pair, msa_mask, pair_mask, ps.blocks[i], st.blocks[i])
        push!(st_blocks, st_b)
    end
    return (msa=msa, pair=pair), (blocks=Tuple(st_blocks),)
end
