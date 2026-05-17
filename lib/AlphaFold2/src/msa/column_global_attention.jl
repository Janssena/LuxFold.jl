struct MSAColumnGlobalAttention <: Lux.AbstractLuxContainerLayer{(:linear_q, :linear_k, :linear_v, :linear_gate, :linear_output)}
    num_heads::Int
    head_dim::Int
    linear_q::Lux.Dense
    linear_k::Lux.Dense
    linear_v::Lux.Dense
    linear_gate::Lux.Dense
    linear_output::Lux.Dense
end

function MSAColumnGlobalAttention(config::NamedTuple; c_in=nothing)
    c_in = isnothing(c_in) ? config.c_s : c_in
    head_dim = config.msa_column_global_attention_dim
    num_heads = config.num_heads
    total_dim = head_dim * num_heads

    return MSAColumnGlobalAttention(
        num_heads, head_dim,
        Lux.Dense(c_in, total_dim; use_bias=false),
        Lux.Dense(c_in, head_dim; use_bias=false),
        Lux.Dense(c_in, head_dim; use_bias=false),
        Lux.Dense(c_in, total_dim),
        Lux.Dense(total_dim, c_in)
    )
end

function (m::MSAColumnGlobalAttention)(msa_representation, msa_mask, ps, st)
    c_in, s, i, b = size(msa_representation)
    h = m.num_heads
    d = m.head_dim

    q, st_q = m.linear_q(msa_representation, ps.linear_q, st.linear_q)
    k, st_k = m.linear_k(msa_representation, ps.linear_k, st.linear_k)
    v, st_v = m.linear_v(msa_representation, ps.linear_v, st.linear_v)
    g, st_g = m.linear_gate(msa_representation, ps.linear_gate, st.linear_gate)

    q = reshape(q, d, h, s, i, b)
    g = sigmoid.(reshape(g, d, h, s, i, b))

    if !isnothing(msa_mask)
        q_masked = q .* reshape(msa_mask, 1, 1, s, i, b)
        q_mean = sum(q_masked, dims=3) ./ (sum(msa_mask, dims=1) .+ 1f-8)
    else
        q_mean = mean(q, dims=3)
    end

    # Attention scores: [s, h, i, b]
    q_mean_reshaped = reshape(q_mean, d, h, i * b)
    k_reshaped = reshape(k, d, s, i * b)

    scores = Lux.batched_matmul(q_mean_reshaped, k_reshaped; lhs_contracting_dim=1, rhs_contracting_dim=1)
    scores = reshape(permutedims(scores, (2, 1, 3)), s, h, i, b)
    scores = scores ./ sqrt(d)

    if !isnothing(msa_mask)
        scores = scores .- (1 .- reshape(msa_mask, s, 1, i, b)) .* 1f9
    end

    attn = softmax(scores; dims=1)

    # Weighted sum: [d, h, i, b]
    attn_reshaped = reshape(attn, s, h, i * b)
    v_reshaped = reshape(v, d, s, i * b)

    result = Lux.batched_matmul(attn_reshaped, v_reshaped; lhs_contracting_dim=1, rhs_contracting_dim=2)
    result = reshape(permutedims(result, (2, 1, 3)), d, h, i, b)

    output = g .* reshape(result, d, h, 1, i, b)
    output = reshape(output, h * d, s, i, b)

    out, st_out = m.linear_output(output, ps.linear_output, st.linear_output)

    if !isnothing(msa_mask)
        out = out .* reshape(msa_mask, 1, s, i, b)
    end

    return out, (linear_q=st_q, linear_k=st_k, linear_v=st_v, linear_gate=st_g, linear_output=st_out)
end
