struct InvariantPointAttention <: Lux.AbstractLuxContainerLayer{(:linear_q, :linear_k, :linear_v, :linear_q_pts, :linear_k_pts, :linear_v_pts, :linear_z, :linear_out)}
    num_heads::Int
    num_queries_pts::Int
    num_values_pts::Int
    head_dim::Int

    linear_q::Lux.Dense
    linear_k::Lux.Dense
    linear_v::Lux.Dense
    linear_q_pts::Lux.Dense
    linear_k_pts::Lux.Dense
    linear_v_pts::Lux.Dense
    linear_z::Lux.Dense
    linear_out::Lux.Dense
end

function InvariantPointAttention(config::NamedTuple)
    c_s = config.c_s
    c_z = config.c_z
    num_heads = config.ipa_num_heads
    head_dim = config.ipa_head_dim
    num_queries_pts = config.ipa_num_queries_pts
    num_values_pts = config.ipa_num_values_pts

    return InvariantPointAttention(
        num_heads, num_queries_pts, num_values_pts, head_dim,
        Lux.Dense(c_s, num_heads * head_dim; use_bias=false),
        Lux.Dense(c_s, num_heads * head_dim; use_bias=false),
        Lux.Dense(c_s, num_heads * head_dim; use_bias=false),
        Lux.Dense(c_s, num_heads * num_queries_pts * 3; use_bias=false),
        Lux.Dense(c_s, num_heads * num_queries_pts * 3; use_bias=false),
        Lux.Dense(c_s, num_heads * num_values_pts * 3; use_bias=false),
        Lux.Dense(c_z, num_heads; use_bias=false),
        Lux.Dense(num_heads * (head_dim + c_z + num_values_pts * 4), c_s) # Simplified output dim for now
    )
end

function (m::InvariantPointAttention)(s, z, r, mask, ps, st)
    # s: [c_s, N_res, batch]
    # z: [c_z, N_res, N_res, batch]
    # r: (R, T) - Current frames (Rotation and Translation)
    # mask: [N_res, batch]

    c_s, n, b = size(s)
    h = m.num_heads
    d = m.head_dim
    np = m.num_queries_pts
    nv = m.num_values_pts

    # 1. Projections
    q, st_q = m.linear_q(s, ps.linear_q, st.linear_q) # [h*d, n, b]
    k, st_k = m.linear_k(s, ps.linear_k, st.linear_k) # [h*d, n, b]
    v, st_v = m.linear_v(s, ps.linear_v, st.linear_v) # [h*d, n, b]

    q = reshape(q, d, h, n, b)
    k = reshape(k, d, h, n, b)
    v = reshape(v, d, h, n, b)

    # 2. Point Projections
    q_pts_local, st_qp = m.linear_q_pts(s, ps.linear_q_pts, st.linear_q_pts) # [h*np*3, n, b]
    k_pts_local, st_kp = m.linear_k_pts(s, ps.linear_k_pts, st.linear_k_pts) # [h*np*3, n, b]
    v_pts_local, st_vp = m.linear_v_pts(s, ps.linear_v_pts, st.linear_v_pts) # [h*nv*3, n, b]

    q_pts_local = reshape(q_pts_local, 3, np, h, n, b)
    k_pts_local = reshape(k_pts_local, 3, np, h, n, b)
    v_pts_local = reshape(v_pts_local, 3, nv, h, n, b)

    # 3. Transform points to global frame
    # r.R: [3, 3, n, b], r.T: [3, n, b]
    q_pts = apply_rigid_transform(r, q_pts_local)
    k_pts = apply_rigid_transform(r, k_pts_local)
    v_pts = apply_rigid_transform(r, v_pts_local)

    # 4. Attention Scores
    # Scalar Part
    q_p = reshape(permutedims(q, (3, 1, 2, 4)), n, d, h * b)
    k_p = reshape(permutedims(k, (3, 1, 2, 4)), n, d, h * b)
    scores_scalar = Lux.batched_matmul(q_p, k_p; lhs_contracting_dim=2, rhs_contracting_dim=2)
    scores_scalar = reshape(scores_scalar, n, n, h, b)
    scores_scalar = scores_scalar ./ sqrt(d)

    # Pair Part
    z_proj, st_z = m.linear_z(z, ps.linear_z, st.linear_z) # [h, n, n, b]
    scores_pair = permutedims(z_proj, (2, 3, 1, 4))

    # Point Part
    gamma = 1.0 / (3 * np) # Placeholder for learned weight
    # Euclidean distance squared: ||q_pts - k_pts||^2 = ||q||^2 + ||k||^2 - 2q·k
    q_pts_p = reshape(permutedims(q_pts, (4, 1, 2, 3, 5)), n, 3 * np, h * b)
    k_pts_p = reshape(permutedims(k_pts, (4, 1, 2, 3, 5)), n, 3 * np, h * b)
    qk = Lux.batched_matmul(q_pts_p, k_pts_p; lhs_contracting_dim=2, rhs_contracting_dim=2)

    q_norm_sq = reshape(sum(q_pts .^ 2, dims=(1, 2)), n, 1, h, b)
    k_norm_sq = reshape(sum(k_pts .^ 2, dims=(1, 2)), 1, n, h, b)
    scores_pts = -0.5 * gamma .* (q_norm_sq .+ k_norm_sq .- 2 .* reshape(qk, n, n, h, b))

    # Combined Scores
    scores = scores_scalar .+ scores_pair .+ scores_pts

    if !isnothing(mask)
        scores = scores .- (1 .- reshape(mask, 1, n, 1, b)) .* 1f9
    end

    attn = softmax(scores; dims=2) # Softmax over j

    # 5. Weighted Sums
    # Scalar result: [d, h, i, b]
    attn_p = reshape(attn, n, n, h * b)
    v_p = reshape(permutedims(v, (3, 1, 2, 4)), n, d, h * b)
    res_scalar = Lux.batched_matmul(attn_p, v_p; lhs_contracting_dim=2, rhs_contracting_dim=1)
    res_scalar = reshape(permutedims(res_scalar, (2, 3, 1, 4)), d, h, n, b)

    # Point result (global to local)
    v_pts_p = reshape(permutedims(v_pts, (4, 1, 2, 3, 5)), n, 3 * nv, h * b)
    res_pts_global = Lux.batched_matmul(attn_p, v_pts_p; lhs_contracting_dim=2, rhs_contracting_dim=1)
    res_pts_global = reshape(permutedims(res_pts_global, (2, 3, 1, 4)), 3, nv, h, n, b)
    res_pts_local = apply_rigid_transform_inverse(r, res_pts_global)

    # Pair result
    z_p = reshape(z, size(z, 1), n * n, b)
    # This needs special handling because z is [c_z, n, n, b]
    # @tullio res_pair[c_idx, h, i, b] := attn[i, j, h, b] * z[c_idx, i, j, b]
    # attn_p2: [h, i, j, b] -> [h, i, n*n, b]? No.
    # We can do it by reshaping attn to [h*i, j, b] and z to [c_z, j, i, b]? 
    # Actually, let's keep it simple:
    res_pair = zeros(eltype(z), size(z, 1), h, n, b)
    for h_idx in 1:h
        attn_h = reshape(attn[:, :, h_idx, :], n, n, b) # [i, j, b]
        # Contract over j
        res_pair[:, h_idx, :, :] = Lux.batched_matmul(z, attn_h; lhs_contracting_dim=3, rhs_contracting_dim=2)
    end

    # 6. Concatenate and Project
    # Simplified concatenation for now
    res_scalar = reshape(res_scalar, h * d, n, b)
    res_pts = reshape(res_pts_local, 3 * nv * h, n, b)
    res_pts_norm = reshape(sqrt.(sum(res_pts_local .^ 2, dims=1)), nv * h, n, b)
    res_pair = reshape(res_pair, size(z, 1) * h, n, b) # This might be wrong, need to check dim

    # ... concat ...
    # Placeholder for final projection
    out = s # Dummy

    return out, (linear_q=st_q, linear_k=st_k, linear_v=st_v, linear_q_pts=st_qp, linear_k_pts=st_kp, linear_v_pts=st_vp, linear_z=st_z, linear_out=st.linear_out)
end

struct BackboneUpdate <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::Lux.Dense
end

function BackboneUpdate(config::NamedTuple)
    return BackboneUpdate(Lux.Dense(config.c_s, 6; init_weight=Lux.zeros32, init_bias=Lux.zeros32))
end

function (m::BackboneUpdate)(s, ps, st)
    # s: [c_s, n, b]
    # out: [6, n, b]
    out, st_lin = m.linear(s, ps.linear, st.linear)

    # split into translation and rotation update
    # t: [3, n, b], q: [3, n, b]
    t = out[1:3, :, :]
    q = out[4:6, :, :]

    # Convert q to rotation matrix update
    # quat = [1, q1, q2, q3] normalized
    # But AF2 uses a simplified update: 
    # R_update = [1, -q3, q2; q3, 1, -q1; -q2, q1, 1] / sqrt(1 + q1^2 + q2^2 + q3^2)
    # Actually, it's a full quaternion to SO(3) mapping.

    # ... SO(3) from quat ...
    # Simplified for now (linearized SO(3) update)
    # R_update: [3, 3, n, b]

    return (R=nothing, T=t), st_lin
end

struct MultiRigidSidechain <: Lux.AbstractLuxContainerLayer{(:linear_in, :linear_out)}
    linear_in::Lux.Dense
    linear_out::Lux.Dense
end

function MultiRigidSidechain(config::NamedTuple)
    return MultiRigidSidechain(
        Lux.Dense(config.c_s, config.c_s),
        Lux.Dense(config.c_s, 7 * 2) # 7 angles * (sin, cos)
    )
end

function (m::MultiRigidSidechain)(s, ps, st)
    x, st_in = m.linear_in(s, ps.linear_in, st.linear_in)
    x = relu.(x)
    y, st_out = m.linear_out(x, ps.linear_out, st.linear_out)

    # y: [14, n, b] -> [2, 7, n, b]
    y = reshape(y, 2, 7, size(y, 2), size(y, 3))

    # Normalize to unit circle
    y = y ./ (sqrt.(sum(y .^ 2, dims=1)) .+ 1f-8)

    return y, (linear_in=st_in, linear_out=st_out)
end

struct StructureModule <: Lux.AbstractLuxContainerLayer{(:initial_linear, :ipa, :backbone_update, :sidechain_head)}
    num_layers::Int
    initial_linear::Lux.Dense
    ipa::InvariantPointAttention
    backbone_update::BackboneUpdate
    sidechain_head::MultiRigidSidechain
end

function StructureModule(config::NamedTuple)
    return StructureModule(
        8, # num_layers
        Lux.Dense(config.c_z, config.c_s),
        InvariantPointAttention(config),
        BackboneUpdate(config),
        MultiRigidSidechain(config)
    )
end

function (m::StructureModule)(z, ps, st)
    # z: [c_z, N_res, N_res, batch]

    # Initialize s from diagonal of z
    # s_initial = linear(z_ii)
    # z_diag: [c_z, n, b]
    n, b = size(z, 2), size(z, 4)
    z_diag = zeros(eltype(z), size(z, 1), n, b)
    for i in 1:n
        z_diag[:, i, :] = z[:, i, i, :]
    end

    s, st_init = m.initial_linear(z_diag, ps.initial_linear, st.initial_linear)

    # Initialize frames as identity
    R = zeros(eltype(s), 3, 3, n, b)
    for i in 1:n, j in 1:b
        R[1, 1, i, j] = R[2, 2, i, j] = R[3, 3, i, j] = 1.0
    end
    T = zeros(eltype(s), 3, n, b)
    r = (R=R, T=T)

    # Iterative loop
    st_ipa = []
    st_bu = []
    for i in 1:m.num_layers
        # IPA
        s_new, st_i = m.ipa(s, z, r, nothing, ps.ipa, st.ipa)
        s = s .+ s_new
        push!(st_ipa, st_i)

        # Backbone Update
        update, st_b = m.backbone_update(s, ps.backbone_update, st.backbone_update)
        # Compose frames: r = r * update
        # r = compose_frames(r, update)
        # T = T + R * update.T
        # R = R * update.R
        push!(st_bu, st_b)
    end

    # Sidechain torsion angles
    torsions, st_sc = m.sidechain_head(s, ps.sidechain_head, st.sidechain_head)

    return (s=s, frames=r, torsions=torsions), (initial_linear=st_init, ipa=Tuple(st_ipa), backbone_update=Tuple(st_bu), sidechain_head=st_sc)
end

# Rigid transform helpers
function apply_rigid_transform(r, pts)
    # r.R: [3, 3, n, b], r.T: [3, n, b]
    # pts: [3, p, h, n, b]
    c, p, h, n, b = size(pts)

    pts_reshaped = reshape(pts, 3, p * h, n * b)
    R_reshaped = reshape(r.R, 3, 3, n * b)

    out = Lux.batched_matmul(R_reshaped, pts_reshaped)
    out = reshape(out, 3, p, h, n, b) .+ reshape(r.T, 3, 1, 1, n, b)

    return out
end

function apply_rigid_transform_inverse(r, pts)
    # R^T * (pts - T)
    c, p, h, n, b = size(pts)

    pts_centered = pts .- reshape(r.T, 3, 1, 1, n, b)
    pts_reshaped = reshape(pts_centered, 3, p * h, n * b)
    R_reshaped = reshape(r.R, 3, 3, n * b)

    out = Lux.batched_matmul(R_reshaped, pts_reshaped; lhs_contracting_dim=1)
    out = reshape(out, 3, p, h, n, b)

    return out
end
