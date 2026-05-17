struct TemplatePairBlock <: Lux.AbstractLuxContainerLayer{(:tri_att_start, :tri_att_end, :tri_mult_out, :tri_mult_in, :pair_transition)}
    tri_att_start::TriangleAttention
    tri_att_end::TriangleAttention
    tri_mult_out::TriangleMultiplication
    tri_mult_in::TriangleMultiplication
    pair_transition::PairTransition
end

function TemplatePairBlock(config::NamedTuple)
    c_z = config.c_z
    tri_attn_c = config.template_triangle_attn_c
    tri_attn_heads = config.template_triangle_attn_num_heads
    tri_mult_c = config.template_triangle_mult_c
    pair_trans_n = config.template_pair_transition_n

    return TemplatePairBlock(
        TriangleAttention(c_z, tri_attn_c, tri_attn_heads; is_starting=true),
        TriangleAttention(c_z, tri_attn_c, tri_attn_heads; is_starting=false),
        TriangleMultiplication(c_z, tri_mult_c; is_outgoing=true),
        TriangleMultiplication(c_z, tri_mult_c; is_outgoing=false),
        PairTransition(config; n=pair_trans_n)
    )
end

function (m::TemplatePairBlock)(x, mask, ps, st)
    # x: [c_z, N_res, N_res, batch*N_templ]

    # We follow Algorithm 16 block order
    # Note: minAlphaFold2 applies residual and dropout here
    # In Lux, we need to handle that manually if not part of the layer

    # Starting Attention
    y, st_tas = m.tri_att_start(x, mask, ps.tri_att_start, st.tri_att_start)
    x = x .+ y # Residual (dropout should be added if config says so)

    # Ending Attention
    y, st_tae = m.tri_att_end(x, mask, ps.tri_att_end, st.tri_att_end)
    x = x .+ y

    # Outgoing Multiplication
    y, st_tmo = m.tri_mult_out(x, mask, ps.tri_mult_out, st.tri_mult_out)
    x = x .+ y

    # Incoming Multiplication
    y, st_tmi = m.tri_mult_in(x, mask, ps.tri_mult_in, st.tri_mult_in)
    x = x .+ y

    # Pair Transition
    y, st_pt = m.pair_transition(x, ps.pair_transition, st.pair_transition)
    x = x .+ y

    return x, (tri_att_start=st_tas, tri_att_end=st_tae, tri_mult_out=st_tmo, tri_mult_in=st_tmi, pair_transition=st_pt)
end

struct TemplatePair <: Lux.AbstractLuxContainerLayer{(:layer_norm, :linear_in, :blocks, :final_layer_norm)}
    layer_norm::Lux.LayerNorm
    linear_in::Lux.Dense
    blocks::NTuple{N,TemplatePairBlock} where N
    final_layer_norm::Lux.LayerNorm
end

function TemplatePair(config::NamedTuple)
    num_blocks = config.template_pair_num_blocks
    c_t = config.c_t
    c_z = config.c_z

    blocks = Tuple(TemplatePairBlock(config) for _ in 1:num_blocks)

    return TemplatePair(
        Lux.LayerNorm((c_t,)),
        Lux.Dense(c_t, c_z),
        blocks,
        Lux.LayerNorm((c_z,))
    )
end

function (m::TemplatePair)(template_feat, pair_mask, ps, st)
    # template_feat: [c_t, N_res, N_res, N_templ, batch]
    # pair_mask: [N_res, N_res, N_templ, batch]

    c_t, n_i, n_j, t, b = size(template_feat)

    # Pre-processing
    # Shape: [c_t, N_res, N_res, N_templ * batch]
    x = reshape(template_feat, c_t, n_i, n_j, t * b)

    x_norm, st_ln = m.layer_norm(x, ps.layer_norm, st.layer_norm)
    x, st_lin = m.linear_in(x_norm, ps.linear_in, st.linear_in)

    # Mask flattening
    mask = nothing
    if !isnothing(pair_mask)
        mask = reshape(pair_mask, n_i, n_j, t * b)
        x = x .* reshape(mask, 1, n_i, n_j, t * b)
    end

    # Block loop
    st_blocks = []
    for (i, block) in enumerate(m.blocks)
        x, st_b = block(x, mask, ps.blocks[i], st.blocks[i])
        push!(st_blocks, st_b)
    end

    # Final LayerNorm
    x, st_fln = m.final_layer_norm(x, ps.final_layer_norm, st.final_layer_norm)
    if !isnothing(mask)
        x = x .* reshape(mask, 1, n_i, n_j, t * b)
    end

    # Restore dimensions
    out = reshape(x, size(x, 1), n_i, n_j, t, b)

    return out, (layer_norm=st_ln, linear_in=st_lin, blocks=Tuple(st_blocks), final_layer_norm=st_fln)
end

struct TemplatePointwiseAttention <: Lux.AbstractLuxContainerLayer{(:linear_q, :linear_k, :linear_v, :linear_output)}
    num_heads::Int
    head_dim::Int
    linear_q::Lux.Dense
    linear_k::Lux.Dense
    linear_v::Lux.Dense
    linear_output::Lux.Dense
end

function TemplatePointwiseAttention(config::NamedTuple)
    head_dim = config.template_pointwise_attention_dim
    num_heads = config.template_pointwise_num_heads
    total_dim = head_dim * num_heads

    return TemplatePointwiseAttention(
        num_heads, head_dim,
        Lux.Dense(config.c_z, total_dim; use_bias=false),
        Lux.Dense(config.c_z, total_dim; use_bias=false),
        Lux.Dense(config.c_z, total_dim; use_bias=false),
        Lux.Dense(total_dim, config.c_z)
    )
end

function (m::TemplatePointwiseAttention)(template_feat, pair_representation, template_mask, ps, st)
    # template_feat: [c_z, N_res, N_res, N_templ, batch]
    # pair_representation: [c_z, N_res, N_res, batch]
    # template_mask: [N_templ, batch]

    c_z, n_i, n_j, t, b = size(template_feat)
    h = m.num_heads
    d = m.head_dim

    # Projections
    q, st_q = m.linear_q(pair_representation, ps.linear_q, st.linear_q) # [h*d, N_res, N_res, batch]
    k, st_k = m.linear_k(template_feat, ps.linear_k, st.linear_k)       # [h*d, N_res, N_res, N_templ, batch]
    v, st_v = m.linear_v(template_feat, ps.linear_v, st.linear_v)       # [h*d, N_res, N_res, N_templ, batch]

    q = reshape(q, d, h, n_i, n_j, b)
    k = reshape(k, d, h, n_i, n_j, t, b)
    v = reshape(v, d, h, n_i, n_j, t, b)

    # Attention scores: [t, h, N_res, N_res, batch]
    q_p = reshape(q, d, h * n_i * n_j * b)
    k_p = reshape(permutedims(k, (1, 5, 2, 3, 4, 6)), d, t, h * n_i * n_j * b)

    scores = Lux.batched_matmul(q_p, k_p; lhs_contracting_dim=1, rhs_contracting_dim=1)
    scores = reshape(scores, t, h, n_i, n_j, b)
    scores = scores ./ sqrt(d)

    if !isnothing(template_mask)
        # template_mask: [t, b]
        scores = scores .+ reshape(template_mask, t, 1, 1, 1, b) .* 0 .- (1 .- reshape(template_mask, t, 1, 1, 1, b)) .* 1f9
    end

    attn = softmax(scores; dims=1) # Softmax over templates

    # Weighted sum: [d, h, N_res, N_res, batch]
    attn_p = reshape(attn, t, h * n_i * n_j * b)
    v_p = reshape(permutedims(v, (5, 1, 2, 3, 4, 6)), t, d, h * n_i * n_j * b)

    out_v = Lux.batched_matmul(attn_p, v_p; lhs_contracting_dim=1, rhs_contracting_dim=1)
    out_v = reshape(out_v, d, h, n_i, n_j, b)

    out_v = reshape(out_v, h * d, n_i, n_j, b)

    out, st_out = m.linear_output(out_v, ps.linear_output, st.linear_output)

    return out, (linear_q=st_q, linear_k=st_k, linear_v=st_v, linear_output=st_out)
end
