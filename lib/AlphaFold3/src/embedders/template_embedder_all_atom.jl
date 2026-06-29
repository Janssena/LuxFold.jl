
struct TemplateEmbedderAllAtom{E,S,LT} <:
       Lux.AbstractLuxContainerLayer{(:template_pair_embedder, :template_pair_stack, :linear_t)}
    template_pair_embedder::E
    template_pair_stack::S
    linear_t::LT
end

function TemplateEmbedderAllAtom(; chn_in::Int, chn_t::Int, chn_z::Int, chn_dgram::Int=39,
                                 chn_aatype::Int=AF3_RESTYPE_NUM, no_blocks::Int, no_heads::Int,
                                 chn_hidden_tri_mul::Int, chn_hidden_tri_att::Int,
                                 transition_n::Int)
    return TemplateEmbedderAllAtom(
        TemplatePairEmbedderAllAtom(; chn_in, chn_t, chn_dgram, chn_aatype),
        TemplatePairStack(; chn_t, chn_hidden_mul=chn_hidden_tri_mul, chn_hidden_pair_att=chn_hidden_tri_att,
                          no_heads, transition_n, no_blocks),
        Lux.Dense(chn_t => chn_z; use_bias=false),
    )
end

(l::TemplateEmbedderAllAtom)(inputs::NamedTuple, ps, st) =
    l(inputs.batch, inputs.z, get(inputs, :pair_mask, nothing), ps, st)

function (l::TemplateEmbedderAllAtom)(batch::NamedTuple, z, pair_mask, ps, st)
    t, st_e = l.template_pair_embedder(batch, z, ps.template_pair_embedder, st.template_pair_embedder)
    t, st_s = l.template_pair_stack(t, pair_mask, ps.template_pair_stack, st.template_pair_stack)

    Tpl = size(t, 4)
    t = dropdims(sum(t; dims=4); dims=4) ./ eltype(t)(Tpl)
    t = Lux.relu.(t)

    z_update, st_l = l.linear_t(t, ps.linear_t, st.linear_t)

    st_out = merge(st, (template_pair_embedder=st_e, template_pair_stack=st_s, linear_t=st_l))
    return z_update, st_out
end
