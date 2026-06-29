
_expand_templ_mask(::Nothing, Ni, Nj, Tpl, B) = nothing
_expand_templ_mask(mask::AbstractArray{Bool,4}, Ni, Nj, Tpl, B) = reshape(mask, Ni, Nj, Tpl * B)
_expand_templ_mask(mask::AbstractArray{Bool,3}, Ni, Nj, Tpl, B) =
    reshape(reshape(mask, Ni, Nj, 1, B) .& trues(1, 1, Tpl, 1), Ni, Nj, Tpl * B)

struct TemplatePairStack{B,LN} <: Lux.AbstractLuxContainerLayer{(:blocks, :layer_norm)}
    blocks::B
    layer_norm::LN
end

function TemplatePairStack(; chn_t::Int, chn_hidden_mul::Int, chn_hidden_pair_att::Int,
                           no_heads::Int, transition_n::Int, no_blocks::Int)
    block_nt = NamedTuple{Tuple(Symbol("block_$i") for i in 1:no_blocks)}(
        ntuple(_ -> TemplatePairBlock(;
            chn_t, chn_hidden_mul, chn_hidden_pair_att, no_heads, transition_n,
        ), no_blocks)
    )
    return TemplatePairStack(Lux.Chain(block_nt), Lux.LayerNorm((chn_t, 1, 1); dims=1))
end

TemplatePairStack(config::NamedTuple) = TemplatePairStack(;
    chn_t=config.c_t, chn_hidden_mul=config.c_hidden_tri_mul,
    chn_hidden_pair_att=config.c_hidden_tri_att, no_heads=config.no_heads,
    transition_n=config.pair_transition_n, no_blocks=config.no_blocks,
)

function (l::TemplatePairStack)(t, pair_mask, ps, st)
    C, Ni, Nj, Tpl, B = size(t)
    t4 = reshape(t, C, Ni, Nj, Tpl * B)
    mask4 = _expand_templ_mask(pair_mask, Ni, Nj, Tpl, B)

    out, st_blocks = l.blocks((; z=t4, pair_mask=mask4), ps.blocks, st.blocks)
    t4 = out.z

    t4, st_ln = l.layer_norm(t4, ps.layer_norm, st.layer_norm)

    return reshape(t4, C, Ni, Nj, Tpl, B), merge(st, (; blocks=st_blocks, layer_norm=st_ln))
end
