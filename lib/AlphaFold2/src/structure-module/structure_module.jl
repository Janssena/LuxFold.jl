"""
    StructureModuleFold(chn_s, chn_z, chn_ipa, chn_resnet, no_heads_ipa, no_qk_points, no_v_points;
                        no_transition_layers=1, no_resnet_blocks=2, no_angles=7,
                        epsilon=1f-8, is_multimer=static(false), use_bias=true)

One iteration of the Structure Module inner loop — matches the loop body of
`openfold.model.structure_module.StructureModule._forward_monomer`.

Sub-layers are **shared** across all `no_blocks` iterations: `StructureModule` calls
this block `no_blocks` times with the **same** `ps` / `st`.

# Arguments
- `chn_s`: Single representation channel dimension.
- `chn_z`: Pair representation channel dimension.
- `chn_ipa`: IPA hidden channel dimension.
- `chn_resnet`: Angle-resnet hidden channel dimension.
- `no_heads_ipa`: Number of IPA heads.
- `no_qk_points`: Number of query/key points per IPA head.
- `no_v_points`: Number of value points per IPA head.

# Keyword Arguments
- `no_transition_layers`: Layers inside `StructureModuleTransition` (default: 1).
- `no_resnet_blocks`: Blocks inside `AngleResnet` (default: 2).
- `no_angles`: Number of torsion angles predicted (default: 7).
- `epsilon`: L2-norm epsilon in angle resnet and IPA (default: `1f-8`).
- `layernorm_epsilon`: LayerNorm epsilon for `layer_norm_ipa` (default: `1f-5`).
- `is_multimer`: `static(true)` enables multimer IPA layout.
- `use_bias`: bias config; resolved over
  `(:ipa, :layer_norm_ipa, :transition, :backbone_update, :angle_resnet)`.

# Inputs (NamedTuple)
- `s`: `[chn_s, N, B]` — single representation.
- `z`: `[chn_z, N, N, B]` — pair representation (passed through unchanged).
- `r::Rigid`: per-residue frames.
- `mask`: `[N, B]` `AbstractArray{Bool}`, or `nothing` for no masking.
- `s_init`: `[chn_s, N, B]` — initial single representation (captured before `linear_in`
  in `StructureModule`; fed to the angle resnet at every block).

# Returns
- A NamedTuple extending the inputs with updated `s`, `r`, `unnorm_angles`, `angles`.
- `st`: Updated state.
"""
struct StructureModuleFold{IPA, LN, TR, BB, AR} <:
    Lux.AbstractLuxContainerLayer{(:ipa, :layer_norm_ipa, :transition, :backbone_update, :angle_resnet)}
    ipa::IPA
    layer_norm_ipa::LN
    transition::TR
    backbone_update::BB
    angle_resnet::AR
end

function StructureModuleFold(
    chn_s::Int, chn_z::Int, chn_ipa::Int, chn_resnet::Int,
    no_heads_ipa::Int, no_qk_points::Int, no_v_points::Int;
    no_transition_layers::Int=1,
    no_resnet_blocks::Int=2,
    no_angles::Int=7,
    epsilon=1f-8, layernorm_epsilon::Real=1f-5,
    is_multimer=static(false),
    use_bias=true,
)
    ub = resolve_defaults(use_bias, (
        :ipa, :layer_norm_ipa, :transition, :backbone_update, :angle_resnet
        )
    )

    ipa = InvariantPointAttention(
        chn_s, chn_z, chn_ipa, no_heads_ipa, no_qk_points, no_v_points;
        eps=epsilon, is_multimer=is_multimer, use_bias=ub.ipa,
    )
    layer_norm_ipa = ub.layer_norm_ipa ?
        Lux.LayerNorm((chn_s, 1); dims=1, epsilon=layernorm_epsilon) :
        LayerNormNoBias((chn_s, 1); dims=1, epsilon=layernorm_epsilon)
    transition = StructureModuleTransition(
        chn_s; num_layers=no_transition_layers, use_bias=ub.transition,
    )
    backbone_update = BackboneUpdate(chn_s; use_bias=ub.backbone_update)
    angle_resnet = AngleResnet(
        chn_s, chn_resnet;
        no_blocks=no_resnet_blocks, no_angles=no_angles,
        epsilon=epsilon, use_bias=ub.angle_resnet,
    )

    return StructureModuleFold(ipa, layer_norm_ipa, transition, backbone_update, angle_resnet)
end

"""
    apply_sequence_mask!(x, mask)

Zero positions where `mask` is false (or zero) in-place.
`x` has shape `[C, N, B]`; `mask` has shape `[N, B]` (Bool or numeric).

**Use `ifelse`, never multiplication** — `NaN * 0 = NaN` in IEEE 754, so multiplication
cannot clear NaN values at masked positions. `ifelse(false, NaN, 0) = 0` is correct.
"""
apply_sequence_mask!(x, ::Nothing) = nothing

function apply_sequence_mask!(x::AbstractArray{T}, mask::AbstractArray{Bool}) where T
    _zero = zero(T)
    mask_reshaped = reshape(mask, 1, size(mask)...)           # [1, N, B]
    @. x = ifelse(mask_reshaped, x, _zero)
    return nothing
end

function (l::StructureModuleFold)(inputs::NamedTuple, ps, st)
    (; s, z, r, mask, s_init) = inputs

    # Three sequence-mask gates prevent NaN from propagating across blocks and outputs.
    # See apply_sequence_mask! for why ifelse is required instead of multiplication.
    #
    #  (a) Before IPA: Julia uses typemin(T) masking → softmax(-Inf,...,-Inf) = NaN for
    #      masked queries. Without gating, NaN s feeds k_pts in the next block, poisoning
    #      valid queries via bias_sdpa.
    #
    #  (b) After layer_norm_ipa: re-zero so transition/backbone/resnet/angle_resnet see
    #      finite inputs at masked positions — states, single, and angles are NaN-free.
    #
    #  (c) Before frame composition: zero the backbone update so r stays at identity at
    #      masked positions — a NaN r would corrupt q_pts/k_pts for ALL residues.

    # 1. IPA residual: s ← s + IPA(s, z, r, mask)
    apply_sequence_mask!(s, mask)                               # (a)
    ipa_out, st_ipa = l.ipa(s, z, r, mask, ps.ipa, st.ipa)
    s = s .+ ipa_out

    # 2. LayerNorm after IPA (openfold: layer_norm_ipa)
    s, st_ln = l.layer_norm_ipa(s, ps.layer_norm_ipa, st.layer_norm_ipa)
    apply_sequence_mask!(s, mask)                               # (b)

    # 3. Transition (no external residual — internal residuals live inside)
    s, st_tr = l.transition(s, ps.transition, st.transition)

    # 4. BackboneUpdate → compose rigid frame
    update, st_bb = l.backbone_update(s, ps.backbone_update, st.backbone_update)
    apply_sequence_mask!(update, mask)                          # (c)
    r = compose_q_update_vec(r, update)

    # 5. AngleResnet (uses current s and initial s_init)
    ar_out, st_ar = l.angle_resnet(s, s_init, ps.angle_resnet, st.angle_resnet)
    unnorm_angles = ar_out.unnormalized_angles
    angles = ar_out.angles

    new_st = merge(st, (;
        ipa=st_ipa, layer_norm_ipa=st_ln, transition=st_tr,
        backbone_update=st_bb, angle_resnet=st_ar,
    ))
    # Pass through all inputs, overwriting updated fields; add per-block angle outputs.
    new_inputs = merge(inputs, (; s, r, unnorm_angles, angles))
    return new_inputs, new_st
end


# ==============================================================================

"""
    StructureModule(chn_s, chn_z; chn_ipa=16, chn_resnet=128, no_heads_ipa=12,
                   no_qk_points=4, no_v_points=8, no_blocks=8,
                   no_transition_layers=1, no_resnet_blocks=2, no_angles=7,
                   trans_scale_factor=10f0, epsilon=1f-8,
                   is_multimer=static(false), use_bias=true)

Full Structure Module — Algorithm 20/23 from AlphaFold2. Matches
`openfold.model.structure_module.StructureModule._forward_monomer`.

The inner `fold` block (`ipa`, `layer_norm_ipa`, `transition`, `backbone_update`,
`angle_resnet`) is called `no_blocks` times with **shared weights** — the same
design as openfold.

Dropout is omitted (inference-only design, consistent with the rest of LuxFold.jl).
The `stop_rot_gradient` call present in openfold is also omitted (no-op at inference).

# Arguments
- `chn_s`: Single representation channel dimension.
- `chn_z`: Pair representation channel dimension.

# Keyword Arguments
- `chn_ipa`: IPA hidden channel dimension (default: 16).
- `chn_resnet`: Angle-resnet hidden channel dimension (default: 128).
- `no_heads_ipa`: Number of IPA heads (default: 12).
- `no_qk_points`: Number of query/key points per head (default: 4).
- `no_v_points`: Number of value points per head (default: 8).
- `no_blocks`: Number of times the fold block is applied (default: 8).
- `no_transition_layers`: Layers inside `StructureModuleTransition` (default: 1).
- `no_resnet_blocks`: Blocks inside `AngleResnet` (default: 2).
- `no_angles`: Number of torsion angles predicted (default: 7).
- `trans_scale_factor`: Scale applied to translations before output (default: `10f0`).
- `epsilon`: L2-norm epsilon forwarded to IPA and AngleResnet (default: `1f-8`).
- `layernorm_epsilon`: LayerNorm epsilon for `layer_norm_s`, `layer_norm_z`, and `layer_norm_ipa` (default: `1f-5`).
- `is_multimer`, `use_bias`: forwarded to sub-layers.

# Inputs
- `s`: `[chn_s, N, B]` — single representation (from Evoformer).
- `z`: `[chn_z, N, N, B]` — pair representation (from Evoformer).
- `mask`: `[N, B]` `AbstractArray{Bool}`, or `nothing` for no masking.

# Returns
- A NamedTuple:
  - `frames`: `[7, N, B, no_blocks]` — per-block scaled rigid frames (quat + trans).
  - `unnormalized_angles`: `[2, no_angles, N, B, no_blocks]`
  - `angles`: `[2, no_angles, N, B, no_blocks]` — L2-normalised
  - `states`: `[chn_s, N, B, no_blocks]` — single representation per block.
  - `single`: `[chn_s, N, B]` — final single representation (last block).
- `st`: Updated state.
"""
struct StructureModule{LNS, LNZ, LIN, FOLD} <:
    Lux.AbstractLuxContainerLayer{(:layer_norm_s, :layer_norm_z, :linear_in, :fold)}
    layer_norm_s::LNS
    layer_norm_z::LNZ
    linear_in::LIN
    fold::FOLD
    no_blocks::Int
    trans_scale_factor::Float32
end

function StructureModule(
    chn_s::Int, chn_z::Int;
    chn_ipa::Int=16,
    chn_resnet::Int=128,
    no_heads_ipa::Int=12,
    no_qk_points::Int=4,
    no_v_points::Int=8,
    no_blocks::Int=8,
    no_transition_layers::Int=1,
    no_resnet_blocks::Int=2,
    no_angles::Int=7,
    trans_scale_factor::Real=10f0,
    epsilon=1f-8, layernorm_epsilon::Real=1f-5,
    is_multimer=static(false),
    use_bias=true,
)
    ub = resolve_defaults(use_bias, (
        :layer_norm_s, :layer_norm_z, :linear_in, :fold
        )
    )

    layer_norm_s = if ub.layer_norm_s
        Lux.LayerNorm((chn_s, 1); dims=1, epsilon=layernorm_epsilon)
    else
        LayerNormNoBias((chn_s, 1); dims=1, epsilon=layernorm_epsilon)
    end
    
    layer_norm_z = if ub.layer_norm_z
        Lux.LayerNorm((chn_z, 1, 1); dims=1, epsilon=layernorm_epsilon)
    else
        LayerNormNoBias((chn_z, 1, 1); dims=1, epsilon=layernorm_epsilon)
    end

    linear_in = Lux.Dense(chn_s => chn_s; use_bias=ub.linear_in)
    fold = StructureModuleFold(
        chn_s, chn_z, chn_ipa, chn_resnet, no_heads_ipa, no_qk_points, no_v_points;
        no_transition_layers=no_transition_layers,
        no_resnet_blocks=no_resnet_blocks,
        no_angles=no_angles,
        epsilon=epsilon, layernorm_epsilon,
        is_multimer=is_multimer,
        use_bias=ub.fold,
    )

    return StructureModule(layer_norm_s, layer_norm_z, linear_in, fold, no_blocks, trans_scale_factor)
end

function (l::StructureModule)(
    s::AbstractArray{T, 3}, z::AbstractArray{T, 4},
    mask::Union{Nothing, AbstractArray{Bool}},
    ps, st,
) where T
    N, B = size(s, 2), size(s, 3)
    no_angles = l.fold.angle_resnet.no_angles

    # 1. Normalise s, capture s_init (before linear_in), project s
    s, st_lns = l.layer_norm_s(s, ps.layer_norm_s, st.layer_norm_s)
    s_init = s
    s, st_lin = l.linear_in(s, ps.linear_in, st.linear_in)

    # 2. Normalise z
    z, st_lnz = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)

    # 3. Identity rigid frames (device-matching: uses same device/storage as s)
    r = rigid_identity(s, N, B)

    # 4. Iterate fold (shared weights) — collect per-block outputs.
    # Pre-seed fold_state with dummy angle outputs so the NamedTuple type is
    # identical on every iteration (type-stability across the loop).
    init_angles = similar(s, T, 2, no_angles, N, B); fill!(init_angles, zero(T))
    fold_state = (; s, z, r, mask, s_init,
                    unnorm_angles=init_angles, angles=init_angles)
    st_fold = st.fold

    # Pre-allocate output collectors (device-matching)
    frames_out = similar(s, T, 7, N, B, l.no_blocks)
    unnorm_angles_out = similar(s, T, 2, no_angles, N, B, l.no_blocks)
    angles_out = similar(s, T, 2, no_angles, N, B, l.no_blocks)
    states_out = similar(s, T, size(s, 1), N, B, l.no_blocks)

    for i in 1:l.no_blocks
        fold_state, st_fold = l.fold(fold_state, ps.fold, st_fold)

        # Scaled frames → 7-vector (quat + trans)
        scaled_r = scale_translation(fold_state.r, T(l.trans_scale_factor))
        frames_out[:, :, :, i] .= to_tensor_7(scaled_r)
        unnorm_angles_out[:, :, :, :, i] .= fold_state.unnorm_angles
        angles_out[:, :, :, :, i] .= fold_state.angles
        states_out[:, :, :, i] .= fold_state.s
    end

    new_st = merge(st, (;
        layer_norm_s=st_lns, layer_norm_z=st_lnz, linear_in=st_lin, fold=st_fold,
    ))

    outputs = (;
        frames = frames_out,
        unnormalized_angles = unnorm_angles_out,
        angles = angles_out,
        states = states_out,
        single = fold_state.s,
    )

    return outputs, new_st
end

# NamedTuple-friendly dispatch
(l::StructureModule)(inputs::NamedTuple, ps, st) = l(
    inputs.s, 
    inputs.z, 
    get(inputs, :mask, nothing), 
    ps, st
)

# ==============================================================================
# Canonical config factories
# ==============================================================================

StructureModule(s::Symbol; kwargs...) = StructureModule(static(s); kwargs...)
# Monomer: uses all constructor defaults (trans_scale_factor=10f0, is_multimer=static(false))
StructureModule(::StaticSymbol{:monomer}; kwargs...) = StructureModule(384, 128; kwargs...)
# Multimer: trans_scale_factor=20f0, is_multimer=static(true)
StructureModule(::StaticSymbol{:multimer}; kwargs...) =
    StructureModule(384, 128; merge((trans_scale_factor=20f0, is_multimer=static(true)), kw)...)
