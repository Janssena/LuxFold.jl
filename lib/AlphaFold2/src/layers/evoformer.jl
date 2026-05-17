struct EvoformerBlock <: Lux.AbstractExplicitContainerLayer{(:msa_row_att, :msa_col_att, :msa_transition, :outer_mean, :tri_mult_out, :tri_mult_in, :tri_att_start, :tri_att_end, :pair_transition)}
    msa_row_att::AttentionPairBias
    msa_col_att::Attention # Non-global, acts on columns
    msa_transition::MSATransition
    outer_mean::OuterProductMean
    tri_mult_out::TriangleMultiplication
    tri_mult_in::TriangleMultiplication
    tri_att_start::TriangleAttention
    tri_att_end::TriangleAttention
    pair_transition::PairTransition
    
    msa_dropout::Float32
    pair_dropout::Float32
end

function EvoformerBlock(config::NamedTuple)
    c_m = config.c_m
    c_z = config.c_z
    num_heads = config.num_heads
    head_dim = config.dim
    
    return EvoformerBlock(
        MSARowAttentionPairBias(c_m, c_z, head_dim, num_heads),
        Attention(c_m, head_dim, num_heads; use_gate=true, use_bias=false), # Column attention
        MSATransition(config),
        OuterProductMean(c_m, c_z, config.outer_product_dim),
        TriangleMultiplication(c_z, config.triangle_mult_c; is_outgoing=true),
        TriangleMultiplication(c_z, config.triangle_mult_c; is_outgoing=false),
        TriangleAttention(c_z, config.triangle_dim, num_heads; is_starting=true),
        TriangleAttention(c_z, config.triangle_dim, num_heads; is_starting=false),
        PairTransition(config),
        Float32(config.evoformer_msa_dropout),
        Float32(config.evoformer_pair_dropout)
    )
end

function (m::EvoformerBlock)(msa, pair, msa_mask, pair_mask, ps, st)
    # msa: [c_m, N_seq, N_res, batch]
    # pair: [c_z, N_res, N_res, batch]
    # msa_mask: [N_seq, N_res, batch]
    # pair_mask: [N_res, N_res, batch]
    
    # 1. MSA Row Attention with Pair Bias
    # AttentionPairBias expects x: [C, S, N, B] and z: [C, N, N, B]
    y, st_row = m.msa_row_att((x=msa, z=pair, mask=msa_mask), ps.msa_row_att, st.msa_row_att)
    msa = msa .+ dropout_rowwise(y, m.msa_dropout, st.training)
    
    # 2. MSA Column Attention
    # We need to swap seq and res dimensions for column attention
    # [C, S, N, B] -> [C, N, S, B]
    msa_col = permutedims(msa, (1, 3, 2, 4))
    mask_col = isnothing(msa_mask) ? nothing : permutedims(msa_mask, (2, 1, 3))
    
    (y_col, _), st_col = m.msa_col_att(msa_col, nothing, mask_col, ps.msa_col_att, st.msa_col_att)
    
    # [C, N, S, B] -> [C, S, N, B]
    msa = msa .+ permutedims(y_col, (1, 3, 2, 4))
    
    # 3. MSA Transition
    y, st_trans = m.msa_transition(msa, ps.msa_transition, st.msa_transition)
    msa = msa .+ y
    
    # 4. Outer Product Mean
    y, st_opm = m.outer_mean((m=msa, mask=msa_mask), ps.outer_mean, st.outer_mean)
    pair = pair .+ y
    
    # 5. Triangle Multiplicative Updates
    y, st_tmo = m.tri_mult_out(pair, pair_mask, ps.tri_mult_out, st.tri_mult_out)
    pair = pair .+ dropout_rowwise(y, m.pair_dropout, st.training)
    
    y, st_tmi = m.tri_mult_in(pair, pair_mask, ps.tri_mult_in, st.tri_mult_in)
    pair = pair .+ dropout_rowwise(y, m.pair_dropout, st.training)
    
    # 6. Triangle Self-Attention
    y, st_tas = m.tri_att_start(pair, pair_mask, ps.tri_att_start, st.tri_att_start)
    pair = pair .+ dropout_rowwise(y, m.pair_dropout, st.training)
    
    y, st_tae = m.tri_att_end(pair, pair_mask, ps.tri_att_end, st.tri_att_end)
    pair = pair .+ dropout_columnwise(y, m.pair_dropout, st.training)
    
    # 7. Pair Transition
    y, st_pt = m.pair_transition(pair, ps.pair_transition, st.pair_transition)
    pair = pair .+ y
    
    return msa, pair, (msa_row_att=st_row, msa_col_att=st_col, msa_transition=st_trans, outer_mean=st_opm, tri_mult_out=st_tmo, tri_mult_in=st_tmi, tri_att_start=st_tas, tri_att_end=st_tae, pair_transition=st_pt)
end

# Placeholder for dropout functions (should be implemented in utils or geometry)
function dropout_rowwise(x, p, training)
    # Placeholder: implementation should zero out rows with probability p
    return x
end

function dropout_columnwise(x, p, training)
    # Placeholder: implementation should zero out columns with probability p
    return x
end
