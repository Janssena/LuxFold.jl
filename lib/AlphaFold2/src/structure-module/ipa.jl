"""
    PointProjection(chn_in, num_points, no_heads; use_bias=true, is_multimer=false)

Project a per-residue single representation `[chn_in, N, B]` into a set of points
in a residue's local frame, then transform them into the global frame via a
`Rigid`. Matches `openfold.model.structure_module.PointProjection`.

# Arguments
- `chn_in`: Input channel dimension.
- `num_points`: Number of points to predict per head.
- `no_heads`: Number of heads.

# Keyword Arguments
- `use_bias`: bias config for the internal `Dense`.
- `is_multimer`: `true` toggles the multimer layout (different memory order of
  `(h, p, xyz)` in the linear output). Defaults to `false`.

# Inputs
- `s`: `[chn_in, N, B]` activations.
- `r::Rigid`: per-residue frames.

# Returns
- `pts_global`: `[3, num_points, no_heads, N, B]` — points in the global frame.
- `st`: Updated state.
"""
struct PointProjection{M,L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
    num_points::Int
    no_heads::Int
    is_multimer::M
end

function PointProjection(
    chn_in::Int, num_points::Int, no_heads::Int;
    use_bias=true, is_multimer=false,
)
    use_bias = resolve_defaults(use_bias, (:linear,))
    linear = Lux.Dense(chn_in => no_heads * num_points * 3; use_bias=use_bias.linear)
    return PointProjection(linear, num_points, no_heads, static(is_multimer))
end

function (l::PointProjection)(s::AbstractArray, r::Rigid, ps, st)
    H, P = l.no_heads, l.num_points
    proj, st_lin = l.linear(s, ps.linear, st.linear)   # [3*H*P, N, B]
    N, B = size(proj, 2), size(proj, 3)

    # The Python forward splits the last dim into 3 chunks (x|y|z) of size H*P
    # (monomer) or stacks them per-head (multimer). In Julia channel-first this
    # means the leading dim layout is the SAME — three contiguous blocks of
    # H*P elements. We reshape to put xyz as a leading dim.
    pts_local = _point_proj_reshape(l.is_multimer, proj, H, P, N, B)
    pts_global = apply(r, pts_local)
    return pts_global, merge(st, (; linear=st_lin))
end

@inline function _point_proj_reshape(::False, proj::AbstractArray{T,3}, H::Int, P::Int,
                                     N::Int, B::Int) where T
    # monomer: leading dim = [x_{HP}; y_{HP}; z_{HP}] (3 chunks of H*P).
    # Reshape to [H*P, 3, N, B] then permute xyz to leading dim,
    # then reshape to [3, P, H, N, B] matching openfold view(..., H, P, 3).
    tmp = reshape(proj, H * P, 3, N, B)
    tmp = permutedims(tmp, (2, 1, 3, 4))           # [3, H*P, N, B]
    return reshape(tmp, 3, P, H, N, B)
end

@inline function _point_proj_reshape(::True, proj::AbstractArray{T,3}, H::Int, P::Int,
                                     N::Int, B::Int) where T
    # multimer: Python views [B, N, H*P*3] as [B, N, H, P*3] (H outer), then
    # splits the inner P*3 dim into 3 chunks of P → stack → [B, N, H, P, 3].
    # In Julia channel-first this is: leading dim [P*3*H] viewed as [P*3, H, N, B],
    # then dim 1 split into 3 chunks → reshape [P, 3, H, N, B] → permute.
    tmp = reshape(proj, P, 3, H, N, B)
    return permutedims(tmp, (2, 1, 3, 4, 5))       # [3, P, H, N, B]
end

# NamedTuple-friendly dispatch
(l::PointProjection)(inputs::NamedTuple, ps, st) = l(inputs.s, inputs.r, ps, st)


"""
    HeadWeights(no_heads)

Wrapper layer holding a single trainable parameter `w` of length `no_heads`,
initialised so that `softplus.(w) ≈ 1` (i.e. `w = log(exp(1) - 1)`). Matches
`openfold.model.structure_module.InvariantPointAttention.head_weights`.

Calling `hw(ps, st)` returns `(ps.w, st)`.
"""
struct HeadWeights <: Lux.AbstractLuxLayer
    no_heads::Int
end

function Lux.initialparameters(::Random.AbstractRNG, l::HeadWeights)
    val = log(exp(one(Float32)) - one(Float32))
    return (; w = val .* ones(Float32, l.no_heads))
end

Lux.initialstates(::Random.AbstractRNG, ::HeadWeights) = NamedTuple()

(l::HeadWeights)(ps, st) = (ps.w, st)


"""
    InvariantPointAttention(chn_s, chn_z, chn_hidden, no_heads, no_qk_points, no_v_points;
                            eps=1f-8, is_multimer=false, use_bias=true)

Algorithm 22 — IPA from AlphaFold2/Multimer. The monomer flavour uses a
joint `linear_kv` and a joint `linear_kv_pts.linear`; the multimer flavour
uses separate K/V projections and separate K/V point projections.

Attention masking uses `Lux.scaled_dot_product_attention` with the native
mask parameter, which sets masked logits to `typemin(T)`. This causes NaN
outputs in Float16 for query positions where all keys are masked
(`softmax(-Inf, …, -Inf) = 0/0`). Use Float32 or higher for reliable results.

# Arguments
- `chn_s`: Single representation channel dimension.
- `chn_z`: Pair representation channel dimension.
- `chn_hidden`: Per-head scalar channel dimension.
- `no_heads`: Number of attention heads.
- `no_qk_points`: Number of query/key points per head.
- `no_v_points`: Number of value points per head.

# Keyword Arguments
- `eps`: Numerical stabilisation for point norms (default `1f-8`).
- `is_multimer`: `true` selects separate K/V projections (multimer); `false`
  the joint KV layout (monomer). Defaults to `false`.
- `use_bias`: bias config.

# Inputs
- `s`: `[chn_s, N, B]` single representation.
- `z`: `[chn_z, N, N, B]` pair representation.
- `r::Rigid`: per-residue frames.
- `mask`: `[N, B]` `AbstractArray{Bool}`, or `nothing` for no masking.

# Returns
- `s_out`: `[chn_s, N, B]` IPA single update.
- `st`: Updated state.
"""
struct InvariantPointAttention{
    M, LQ, LK, LV, LKV, LQP, LKP, LVP, LKVP, LB, HW, LO, EF,
} <: Lux.AbstractLuxContainerLayer{(
    :linear_q, :linear_k, :linear_v, :linear_kv,
    :linear_q_pts, :linear_k_pts, :linear_v_pts, :linear_kv_pts,
    :linear_b, :head_weights, :linear_out,
)}
    is_multimer::M    # StaticBool: separate K/V (multimer) vs joint KV (monomer)?
    linear_q::LQ
    linear_k::LK
    linear_v::LV
    linear_kv::LKV
    linear_q_pts::LQP
    linear_k_pts::LKP
    linear_v_pts::LVP
    linear_kv_pts::LKVP
    linear_b::LB
    head_weights::HW
    linear_out::LO
    chn_hidden::Int
    no_heads::Int
    no_qk_points::Int
    no_v_points::Int
    eps::EF
end

function InvariantPointAttention(
    chn_s::Int, chn_z::Int, chn_hidden::Int, no_heads::Int,
    no_qk_points::Int, no_v_points::Int;
    eps=1f-8, is_multimer=false, use_bias=true,
)
    is_multimer = static(is_multimer)
    ub = resolve_defaults(use_bias, (
        :linear_q, :linear_k, :linear_v, :linear_kv,
        :linear_q_pts, :linear_k_pts, :linear_v_pts, :linear_kv_pts,
        :linear_b, :linear_out,
    ))

    hc = chn_hidden * no_heads

    if dynamic(is_multimer)
        # Multimer: bias-free Q, separate K/V (also bias-free); separate point projections.
        linear_q = Lux.Dense(chn_s => hc; use_bias=false)
        linear_k = Lux.Dense(chn_s => hc; use_bias=false)
        linear_v = Lux.Dense(chn_s => hc; use_bias=false)
        linear_kv = Lux.NoOpLayer()
        linear_q_pts  = PointProjection(chn_s, no_qk_points, no_heads;
                                        use_bias=ub.linear_q_pts, is_multimer=true)
        linear_k_pts  = PointProjection(chn_s, no_qk_points, no_heads;
                                        use_bias=ub.linear_k_pts, is_multimer=true)
        linear_v_pts  = PointProjection(chn_s, no_v_points,  no_heads;
                                        use_bias=ub.linear_v_pts, is_multimer=true)
        linear_kv_pts = Lux.NoOpLayer()
    else
        # Monomer (AF2): biased Q, joint KV; joint KV-points projection.
        linear_q  = Lux.Dense(chn_s => hc;        use_bias=ub.linear_q)
        linear_k  = Lux.NoOpLayer()
        linear_v  = Lux.NoOpLayer()
        linear_kv = Lux.Dense(chn_s => 2 * hc;   use_bias=ub.linear_kv)
        linear_q_pts  = PointProjection(chn_s, no_qk_points, no_heads;
                                        use_bias=ub.linear_q_pts, is_multimer=false)
        linear_k_pts  = Lux.NoOpLayer()
        linear_v_pts  = Lux.NoOpLayer()
        linear_kv_pts = PointProjection(chn_s, no_qk_points + no_v_points, no_heads;
                                        use_bias=ub.linear_kv_pts, is_multimer=false)
    end

    linear_b   = Lux.Dense(chn_z => no_heads; use_bias=ub.linear_b)
    head_weights = HeadWeights(no_heads)
    concat_in  = no_heads * (chn_z + chn_hidden + no_v_points * 4)
    linear_out = Lux.Dense(concat_in => chn_s; use_bias=ub.linear_out)

    return InvariantPointAttention(
        is_multimer,
        linear_q, linear_k, linear_v, linear_kv,
        linear_q_pts, linear_k_pts, linear_v_pts, linear_kv_pts,
        linear_b, head_weights, linear_out,
        chn_hidden, no_heads, no_qk_points, no_v_points,
        eps,
    )
end

# NamedTuple- and mask-optional dispatches ----------------------------------------
(l::InvariantPointAttention)(inputs::NamedTuple, ps, st) =
    l(inputs.s, inputs.z, inputs.r, get(inputs, :mask, nothing), ps, st)

(l::InvariantPointAttention)(s::AbstractArray, z::AbstractArray, r::Rigid, ps, st) =
    l(s, z, r, nothing, ps, st)

# === Forward =====================================================================
#
# Monomer and multimer share this single forward; they differ *only* in how the
# scalar `q/k/v` and the point `q_pts/k_pts/v_pts` are projected, which is handled
# by `_prep_qkv` dispatching on `InvariantPointAttention{False}` (joint KV) vs
# `{True}` (separate K/V). Everything from the pair bias onward is identical.
function (l::InvariantPointAttention)(
    s::AbstractArray{T,3}, z::AbstractArray{T,4}, r::Rigid,
    mask::Union{Nothing,AbstractArray{Bool}}, ps, st,
) where T
    c, H = l.chn_hidden, l.no_heads
    P_qk, P_v = l.no_qk_points, l.no_v_points
    N, B = size(s, 2), size(s, 3)

    # ---- scalar + point Q/K/V (layout differs monomer vs multimer) -----------
    (q, k, v, q_pts, k_pts, v_pts), st_proj = _prep_qkv(l, s, r, ps, st)
    # q/k/v       : [c, H, N, B]
    # q_pts/k_pts : [3, P_qk, H, N, B]; v_pts : [3, P_v, H, N, B]

    # ---- pair bias -----------------------------------------------------------
    b_flat, st_b = l.linear_b(z, ps.linear_b, st.linear_b)                  # [H, N_q, N_k, B]

    # ---- head weights --------------------------------------------------------
    hw, st_hw = l.head_weights(ps.head_weights, st.head_weights)             # [H]
    head_w = @. Lux.softplus(hw) * T(sqrt(1.0 / (3 * (P_qk * 9 / 2))))       # [H]

    # ---- point attention score -----------------------------------------------
    q_pts_e = reshape(q_pts, 3, P_qk, H, N, 1, B)
    k_pts_e = reshape(k_pts, 3, P_qk, H, 1, N, B)
    pt_diff = q_pts_e .- k_pts_e                                             # [3, P_qk, H, N_q, N_k, B]
    pt_sq   = sum(abs2, pt_diff; dims=1)                                     # [1, P_qk, H, N_q, N_k, B]
    pt_sum  = dropdims(sum(pt_sq; dims=2); dims=(1, 2))                      # [H, N_q, N_k, B]
    # Per spec, head weighting applies before the per-point sum. Since (a+b)*w
    # equals a*w + b*w, applying the scalar `head_w` per-head after the point-sum
    # is numerically equivalent.
    a_pt = pt_sum .* reshape(head_w, H, 1, 1, 1) .* T(-0.5)                 # [H, N_q, N_k, B]

    # ---- square mask: both query and key must be valid (or `nothing`) -------
    mask_sdpa = _ipa_square_mask(mask, N, B)                                 # [N_k, N_q, 1, B] | nothing

    # ---- combined SDPA bias: (point score + pair bias) * sqrt(1/3) ----------
    # Permuted to [N_k, N_q, H, B] to match SDPA's logit layout.
    # The scalar Q·K component is handled by SDPA itself at scale sqrt(3c).
    bias_sdpa = permutedims(
        a_pt .+ b_flat .* T(sqrt(1.0 / 3)),
        (3, 2, 1, 4),
    )                                                                         # [N_k, N_q, H, B]

    # ---- SDPA: q, k, v [c, H, N, B]; scale = sqrt(3c) so Q·K is / sqrt(3c) -
    o_scalar, attn_weights = Lux.scaled_dot_product_attention(
        q, k, v;
        head_dim=1, token_dim=3,
        scale=T(sqrt(3 * c)),
        bias=bias_sdpa,
        mask=mask_sdpa,
    )
    # o_scalar:     [c, H, N, B]
    # attn_weights: [N_k, N_q, H, B]

    o = reshape(o_scalar, c * H, N, B)                                       # [c*H, N, B]

    # [N_k, N_q, H, B] → [N_q, N_k, H, B] for the matmuls below
    attnP = permutedims(attn_weights, (2, 1, 3, 4))                          # [N_q, N_k, H, B]

    # ---- o_pt ----------------------------------------------------------------
    vP_pts    = permutedims(v_pts, (4, 1, 2, 3, 5))                          # [N_k, 3, P_v, H, B]
    vP_pts    = reshape(vP_pts, N, 3 * P_v, H, B)                            # [N_k, 3*P_v, H, B]
    o_pt_flat = Lux.batched_matmul(attnP, vP_pts)                            # [N_q, 3*P_v, H, B]
    o_pt_flat = reshape(o_pt_flat, N, 3, P_v, H, B)
    o_pt_global = permutedims(o_pt_flat, (2, 3, 4, 1, 5))                    # [3, P_v, H, N, B]

    o_pt_local = invert_apply(r, o_pt_global)                           # [3, P_v, H, N, B]

    o_pt_sqsum = dropdims(sum(abs2, o_pt_local; dims=1); dims=1)             # [P_v, H, N, B]
    o_pt_norm  = sqrt.(o_pt_sqsum .+ T(l.eps))
    o_pt_norm  = reshape(o_pt_norm, P_v * H, N, B)

    o_pt_x = reshape(@view(o_pt_local[1, :, :, :, :]), P_v * H, N, B)
    o_pt_y = reshape(@view(o_pt_local[2, :, :, :, :]), P_v * H, N, B)
    o_pt_z = reshape(@view(o_pt_local[3, :, :, :, :]), P_v * H, N, B)

    # ---- o_pair --------------------------------------------------------------
    Cz = size(z, 1)
    attn_zP = permutedims(attn_weights, (3, 1, 2, 4))                        # [H, N_k, N_q, B]
    zP      = permutedims(z, (3, 1, 2, 4))                                   # [N_k, Cz, N_q, B]
    o_pair  = Lux.batched_matmul(attn_zP, zP)                                # [H, Cz, N_q, B]
    o_pair  = permutedims(o_pair, (2, 1, 3, 4))                              # [Cz, H, N_q, B]
    o_pair  = reshape(o_pair, Cz * H, N, B)                                  # [Cz*H, N, B]

    # ---- concat + linear_out -------------------------------------------------
    out_in = vcat(o, o_pt_x, o_pt_y, o_pt_z, o_pt_norm, o_pair)
    s_out, st_out = l.linear_out(out_in, ps.linear_out, st.linear_out)

    st_new = merge(st, st_proj, (; linear_b=st_b, head_weights=st_hw, linear_out=st_out))
    return s_out, st_new
end

# === Q/K/V projection (the only monomer/multimer difference) =====================

# Monomer: scalar K/V come from a single joint `linear_kv` (split in two); the
# point K/V come from a single joint `linear_kv_pts` (split along the point dim).
function _prep_qkv(l::InvariantPointAttention{False}, s::AbstractArray, r::Rigid, ps, st)
    C = l.chn_hidden
    P_qk, P_v = l.no_qk_points, l.no_v_points
    N, B = size(s, 2), size(s, 3)
    H = l.no_heads

    q_flat, st_q   = l.linear_q(s, ps.linear_q, st.linear_q)        # [H*C, N, B]
    kv_flat, st_kv = l.linear_kv(s, ps.linear_kv, st.linear_kv)     # [2*H*C, N, B]

    q  = reshape(q_flat, C, H, N, B)                                # [C, H, N, B]
    kv = reshape(kv_flat, 2 * C, H, N, B)
    k  = @view kv[1:C, :, :, :]                                     # [C, H, N, B]
    v  = @view kv[C+1:2C, :, :, :]                                  # [C, H, N, B]

    q_pts, st_qp = l.linear_q_pts(s, r, ps.linear_q_pts, st.linear_q_pts)
    kv_pts, st_kvp = l.linear_kv_pts(s, r, ps.linear_kv_pts, st.linear_kv_pts)
    # q_pts : [3, P_qk, H, N, B], kv_pts : [3, P_qk+P_v, H, N, B]
    k_pts = @view kv_pts[:, 1:P_qk, :, :, :]                                # [3, P_qk, H, N, B]
    v_pts = @view kv_pts[:, P_qk+1:P_qk+P_v, :, :, :]                       # [3, P_v,  H, N, B]

    st_proj = (; linear_q=st_q, linear_kv=st_kv,
                 linear_q_pts=st_qp, linear_kv_pts=st_kvp)
    return (q, k, v, q_pts, k_pts, v_pts), st_proj
end

# Multimer: scalar K/V come from separate `linear_k`/`linear_v`; the point K/V
# come from separate `linear_k_pts`/`linear_v_pts`.
function _prep_qkv(l::InvariantPointAttention{True}, s::AbstractArray, r::Rigid, ps, st)
    C = l.chn_hidden
    N, B = size(s, 2), size(s, 3)
    H = l.no_heads

    q_flat, st_q = l.linear_q(s, ps.linear_q, st.linear_q)                 # [H*c, N, B]
    k_flat, st_k = l.linear_k(s, ps.linear_k, st.linear_k)                 # [H*c, N, B]
    v_flat, st_v = l.linear_v(s, ps.linear_v, st.linear_v)                 # [H*c, N, B]

    q = reshape(q_flat, C, H, N, B)                                         # [c, H, N, B]
    k = reshape(k_flat, C, H, N, B)
    v = reshape(v_flat, C, H, N, B)

    q_pts, st_qp = l.linear_q_pts(s, r, ps.linear_q_pts, st.linear_q_pts)  # [3, P_qk, H, N, B]
    k_pts, st_kp = l.linear_k_pts(s, r, ps.linear_k_pts, st.linear_k_pts)  # [3, P_qk, H, N, B]
    v_pts, st_vp = l.linear_v_pts(s, r, ps.linear_v_pts, st.linear_v_pts)  # [3, P_v,  H, N, B]

    st_proj = (; linear_q=st_q, linear_k=st_k, linear_v=st_v,
                 linear_q_pts=st_qp, linear_k_pts=st_kp, linear_v_pts=st_vp)
    return (q, k, v, q_pts, k_pts, v_pts), st_proj
end


_ipa_square_mask(::Nothing, N::Int, B::Int) = nothing

# `[N, B]` residue mask → square `[N_k, N_q, 1, B]` mask (both query and key must
# be valid), matching openfold's `mask.unsqueeze(-1) * mask.unsqueeze(-2)`. The
# result broadcasts against the SDPA logit shape `[N_k, N_q, H, B]`.
function _ipa_square_mask(mask::AbstractArray{Bool}, N::Int, B::Int)
    # Precondition: every batch element must have at least one valid residue.
    # In production AF2 inference this is always true — the mask marks actual
    # residue positions and an all-zero column would mean an empty sequence.
    # Violating this produces NaN outputs (softmax of all -Inf logits).
    # TODO: Use any.(eachcol(mask)) here
    @assert all(any(@view mask[:, b]) for b in 1:B) "IPA mask: every batch element must have at least one valid (non-zero) residue position"
    return reshape(mask, N, 1, 1, B) .& reshape(mask, 1, N, 1, B)
end
