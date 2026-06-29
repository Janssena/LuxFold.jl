
struct TemplatePairBlock{P} <: Lux.AbstractLuxContainerLayer{(:pair_block,)}
    pair_block::P
end

function TemplatePairBlock(; chn_t::Int, chn_hidden_mul::Int, chn_hidden_pair_att::Int,
                           no_heads::Int, transition_n::Int)
    return TemplatePairBlock(PairBlock(;
        chn_z=chn_t, chn_hidden_mul, chn_hidden_pair_att, no_heads_pair=no_heads, transition_n,
    ))
end

function (l::TemplatePairBlock)(inputs::NamedTuple, ps, st)
    z, st_new = l(inputs.z, get(inputs, :pair_mask, nothing), ps, st)
    return merge(inputs, (; z)), st_new
end

function (l::TemplatePairBlock)(z, pair_mask, ps, st)
    z, st_pb = l.pair_block(z, pair_mask, ps.pair_block, st.pair_block)
    return z, merge(st, (; pair_block=st_pb))
end
