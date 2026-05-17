struct AlphaFold2Model <: Lux.AbstractLuxContainerLayer{(:embedder, :extra_msa, :template_pair, :template_attn, :evoformer, :structure_module, :heads)}
    embedder::InputEmbedder
    extra_msa::ExtraMsaStack
    template_pair::TemplatePair
    template_attn::TemplatePointwiseAttention
    evoformer::NTuple{N,EvoformerBlock} where N
    structure_module::StructureModule
    heads::NamedTuple # Tuple of heads
end

function AlphaFold2Model(config::NamedTuple)
    heads = (
        distogram=DistogramHead(config),
        plddt=PLDDTHead(config),
        masked_msa=MaskedMSAHead(config),
        exp_resolved=ExperimentallyResolvedHead(config)
    )

    evoformer_blocks = Tuple(EvoformerBlock(config) for _ in 1:config.num_evoformer_blocks)

    return AlphaFold2Model(
        InputEmbedder(config),
        ExtraMsaStack(config),
        TemplatePair(config),
        TemplatePointwiseAttention(config),
        evoformer_blocks,
        StructureModule(config),
        heads
    )
end

function (m::AlphaFold2Model)(inputs::NamedTuple, ps, st)
    # inputs: (target_feat, msa_feat, residue_index, template_feat, ...)

    # recycling loop
    num_cycles = get(inputs, :num_cycles, 3)
    num_ensembles = get(inputs, :num_ensembles, 1)

    # Initialize recycling states
    m_cycle, z_cycle = nothing, nothing

    st_cycles = []
    for cycle in 1:num_cycles
        # Ensembling (Simplified: assuming num_ensembles=1 for now)
        # Embedding
        (m_cycle, z_cycle), st_emb = m.embedder(inputs.target_feat, inputs.residue_index, inputs.msa_feat, ps.embedder, st.embedder)

        # Extra MSA
        (m_cycle, z_cycle), st_extra = m.extra_msa(m_cycle, z_cycle, inputs.msa_mask, nothing, ps.extra_msa, st.extra_msa)

        # Templates
        t_feat, st_tp = m.template_pair(inputs.template_feat, inputs.template_mask, ps.template_pair, st.template_pair)
        z_template, st_ta = m.template_attn(t_feat, z_cycle, nothing, ps.template_attn, st.template_attn)
        z_cycle = z_cycle .+ z_template

        # Evoformer Trunk
        for (i, block) in enumerate(m.evoformer)
            m_cycle, z_cycle, st_b = block(m_cycle, z_cycle, inputs.msa_mask, nothing, ps.evoformer[i], st.evoformer[i])
        end

        # Stop gradient if not last cycle
        if cycle < num_cycles
            m_cycle = Lux.Zygote.ignore(() -> m_cycle)
            z_cycle = Lux.Zygote.ignore(() -> z_cycle)
        end
    end

    # Structure Module
    res_struct, st_sm = m.structure_module(z_cycle, ps.structure_module, st.structure_module)

    # Heads
    out_dist, st_dist = m.heads.distogram(z_cycle, ps.heads.distogram, st.heads.distogram)
    out_plddt, st_plddt = m.heads.plddt(res_struct.s, ps.heads.plddt, st.heads.plddt)

    return (
        structure=res_struct,
        distogram=out_dist,
        plddt=out_plddt
    ), st
end
