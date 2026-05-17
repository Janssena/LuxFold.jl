struct RelPos <: Lux.AbstractLuxContainerLayer{(:linear,)}
    max_rel::Int
    linear::Lux.Dense
end

function RelPos(config::NamedTuple; max_rel=32)
    return RelPos(max_rel, Lux.Dense(2 * max_rel + 1, config.c_z; use_bias=true))
end

function (m::RelPos)(residue_index::AbstractMatrix{Int}, ps, st)
    # residue_index: (N_res, batch) - Julia uses (dim, batch)
    N_res, batch = size(residue_index)

    # Compute relative positions: (N_res, N_res, batch)
    d = reshape(residue_index, 1, N_res, batch) .- reshape(residue_index, N_res, 1, batch)

    # Clamp and shift to [0, 2*max_rel]
    d = clamp.(d, -m.max_rel, m.max_rel) .+ m.max_rel .+ 1 # 1-based indexing

    # One-hot encode: (2*max_rel+1, N_res, N_res, batch)
    # We can use a more efficient way than full one-hot if needed, 
    # but for now let's keep it simple.
    # Lux.Dense expects (dim, ..., batch)

    oh = zeros(eltype(ps.linear.weight), 2 * m.max_rel + 1, N_res, N_res, batch)
    for b in 1:batch, j in 1:N_res, i in 1:N_res
        oh[d[i, j, b], i, j, b] = 1.0
    end

    # Project: (c_z, N_res, N_res, batch)
    out, st_linear = m.linear(oh, ps.linear, st.linear)

    return out, (linear=st_linear,)
end

struct InputEmbedder <: Lux.AbstractLuxContainerLayer{(:linear_tf1, :linear_tf2, :linear_tf3, :linear_msa, :rel_pos)}
    linear_tf1::Lux.Dense
    linear_tf2::Lux.Dense
    linear_tf3::Lux.Dense
    linear_msa::Lux.Dense
    rel_pos::RelPos
end

function InputEmbedder(config::NamedTuple)
    return InputEmbedder(
        Lux.Dense(22, config.c_z), # linear_target_feat_1
        Lux.Dense(22, config.c_z), # linear_target_feat_2
        Lux.Dense(22, config.c_m), # linear_target_feat_3
        Lux.Dense(49, config.c_m), # linear_msa
        RelPos(config)
    )
end

function (m::InputEmbedder)(target_feat, residue_index, msa_feat, ps, st)
    # target_feat: (22, N_res, batch)
    # residue_index: (N_res, batch)
    # msa_feat: (49, N_cluster, N_res, batch)

    # Pair representation z: (c_z, N_res, N_res, batch)
    a, st_tf1 = m.linear_tf1(target_feat, ps.linear_tf1, st.linear_tf1) # (c_z, N_res, batch)
    b, st_tf2 = m.linear_tf2(target_feat, ps.linear_tf2, st.linear_tf2) # (c_z, N_res, batch)

    # Outer sum: z_ij = a_i + b_j
    z = reshape(a, size(a, 1), size(a, 2), 1, size(a, 3)) .+ reshape(b, size(b, 1), 1, size(b, 2), size(b, 3))

    rel_pos_out, st_rp = m.rel_pos(residue_index, ps.rel_pos, st.rel_pos)
    z = z .+ rel_pos_out

    # MSA representation m: (c_m, N_cluster, N_res, batch)
    tf3_out, st_tf3 = m.linear_tf3(target_feat, ps.linear_tf3, st.linear_tf3) # (c_m, N_res, batch)
    msa_out, st_msa = m.linear_msa(msa_feat, ps.linear_msa, st.linear_msa)    # (c_m, N_cluster, N_res, batch)

    m_out = reshape(tf3_out, size(tf3_out, 1), 1, size(tf3_out, 2), size(tf3_out, 3)) .+ msa_out

    return (m=m_out, z=z), (linear_tf1=st_tf1, linear_tf2=st_tf2, linear_tf3=st_tf3, linear_msa=st_msa, rel_pos=st_rp)
end


struct PairTransition <: Lux.AbstractLuxContainerLayer{(:layer_norm, :linear_up, :linear_down)}
    layer_norm::Lux.LayerNorm
    linear_up::Lux.Dense
    linear_down::Lux.Dense
end

function PairTransition(config::NamedTuple; n=nothing)
    n = isnothing(n) ? config.pair_transition_n : n
    return PairTransition(
        Lux.LayerNorm((config.c_z,)),
        Lux.Dense(config.c_z, n * config.c_z),
        Lux.Dense(n * config.c_z, config.c_z)
    )
end

function (m::PairTransition)(x, ps, st)
    x_norm, st_ln = m.layer_norm(x, ps.layer_norm, st.layer_norm)
    x_up, st_up = m.linear_up(x_norm, ps.linear_up, st.linear_up)
    x_down, st_down = m.linear_down(relu.(x_up), ps.linear_down, st.linear_down)
    return x_down, (layer_norm=st_ln, linear_up=st_up, linear_down=st_down)
end
