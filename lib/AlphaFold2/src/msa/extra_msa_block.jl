struct ExtraMsaBlock <: Lux.AbstractExplicitContainerLayer{(:msa_row_att, :msa_col_att, :msa_transition, :outer_mean, :tri_mult_out, :tri_mult_in, :tri_att_start, :tri_att_end, :pair_transition)}
    msa_row_att::AttentionPairBias # From LuxFoldCore
    msa_col_att::MSAColumnGlobalAttention
    msa_transition::MSATransition
    outer_mean::OuterProductMean # From LuxFoldCore
    tri_mult_out::TriangleMultiplication # From LuxTriangleAttention
    tri_mult_in::TriangleMultiplication
    tri_att_start::TriangleAttention
    tri_att_end::TriangleAttention
    pair_transition::PairTransition
end

function ExtraMsaBlock(config::NamedTuple)
    c_e = config.c_e
    c_z = config.c_z
    
    # We use config overrides for Extra MSA if they exist
    extra_msa_trans_n = getattr(config, :extra_msa_transition_n, config.msa_transition_n)
    extra_msa_outer_product_dim = getattr(config, :extra_msa_outer_product_dim, config.outer_product_dim)
    
    return ExtraMsaBlock(
        AttentionPairBias(c_e, c_z, config.num_heads, config.extra_msa_dim; is_row=true),
        MSAColumnGlobalAttention(config; c_in=c_e),
        MSATransition(config; c_in=c_e, n=extra_msa_trans_n),
        OuterProductMean(c_e, c_z, extra_msa_outer_product_dim),
        TriangleMultiplication(c_z, config.triangle_mult_c; is_outgoing=true),
        TriangleMultiplication(c_z, config.triangle_mult_c; is_outgoing=false),
        TriangleAttention(c_z, config.triangle_dim, config.num_heads; is_starting=true),
        TriangleAttention(c_z, config.triangle_dim, config.num_heads; is_starting=false),
        PairTransition(config)
    )
end

function (m::ExtraMsaBlock)(msa, pair, msa_mask, pair_mask, ps, st)
    # Row Attention
    y, st_row = m.msa_row_att((m=msa, z=pair, mask=msa_mask), ps.msa_row_att, st.msa_row_att)
    msa = msa .+ y # Dropout can be added
    
    # Column Attention
    y, st_col = m.msa_col_att(msa, msa_mask, ps.msa_col_att, st.msa_col_att)
    msa = msa .+ y
    
    # MSA Transition
    y, st_trans = m.msa_transition(msa, ps.msa_transition, st.msa_transition)
    msa = msa .+ y
    
    # Outer Product Mean
    y, st_opm = m.outer_mean((m=msa, mask=msa_mask), ps.outer_mean, st.outer_mean)
    pair = pair .+ y
    
    # Triangle Multiplication Outgoing
    y, st_tmo = m.tri_mult_out(pair, pair_mask, ps.tri_mult_out, st.tri_mult_out)
    pair = pair .+ y
    
    # Triangle Multiplication Incoming
    y, st_tmi = m.tri_mult_in(pair, pair_mask, ps.tri_mult_in, st.tri_mult_in)
    pair = pair .+ y
    
    # Triangle Attention Starting
    y, st_tas = m.tri_att_start(pair, pair_mask, ps.tri_att_start, st.tri_att_start)
    pair = pair .+ y
    
    # Triangle Attention Ending
    y, st_tae = m.tri_att_end(pair, pair_mask, ps.tri_att_end, st.tri_att_end)
    pair = pair .+ y
    
    # Pair Transition
    y, st_pt = m.pair_transition(pair, ps.pair_transition, st.pair_transition)
    pair = pair .+ y
    
    return msa, pair, (msa_row_att=st_row, msa_col_att=st_col, msa_transition=st_trans, outer_mean=st_opm, tri_mult_out=st_tmo, tri_mult_in=st_tmi, tri_att_start=st_tas, tri_att_end=st_tae, pair_transition=st_pt)
end
