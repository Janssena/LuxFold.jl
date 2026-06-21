# alphafold.jl — Top-level AlphaFold2 model
#
# Reference implementation: openfold/model/model.py `AlphaFold`
#
# Implements the recycling loop as a standard Lux (model, ps, st) forward pass.
# Recycling state (m_1_prev, z_prev, x_prev) lives in st.recycling_state and is
# threaded functionally through each iteration.
# Residue constants (default_frames, group_idx, lit_positions, atom_mask) live in
# st.residue_constants; gpu_device()(ps, st) moves them to the target device.
#
# Monomer-only. Multimer (is_multimer=true) throws NotImplementedError.


"""
    AlphaFold(; chn_msa=256, chn_z=128, chn_s=384, ...)

Top-level AlphaFold2 model. Wires all sub-modules together and manages the recycling loop.

# Keyword Arguments
- `chn_msa`: MSA channel dimension (default: 256)
- `chn_z`: Pair channel dimension (default: 128)
- `chn_s`: Single channel dimension (default: 384)
- `chn_target_feat`: Target feature channels (default: 22)
- `chn_msa_feat`: MSA feature channels (default: 49)
- `chn_extra_msa`: Extra MSA embedding channels (default: 64)
- `chn_extra_msa_feat`: Extra MSA feature channels (default: 25)
- `use_templates`: Enable template embedder (default: true)
- `use_extra_msa`: Enable extra MSA stack (default: true)
- `relpos_k`: Relative position encoding window (default: 32)

Call via `predict(model, feats, ps, st; num_recycles=3)` or the Lux functor
`model(feats, ps, st)` (uses `num_recycles=3`).

# Inputs
`feats::NamedTuple` with at minimum:
- `target_feat [chn_target_feat, N, B]`, `residue_index [N, B]`, `msa_feat [chn_msa_feat, N, S_c, B]`
- `seq_mask [N, B]`, `msa_mask [N, S_c, B]`, `aatype [N, B]` (1-based)
- `atom37_atom_exists [37, N, B]` Bool
- If `use_extra_msa`: `extra_msa [N, S_extra, B]`, `extra_msa_deletion_value`, `extra_msa_has_deletion`, `extra_msa_mask`

# Returns
`(outputs::NamedTuple, st_new)` where `outputs` contains `m`, `z`, `s`, `sm`,
`final_atom_positions [3, 37, N, B]`, `final_atom_mask`, `final_affine_tensor [7, N, B]`,
`lddt_logits`, `plddt`, `distogram_logits`, `masked_msa_logits`,
`experimentally_resolved_logits`, `tm_logits`, `num_recycles`.
"""
struct AlphaFold{IE,RE,TE,MEE,MES,EVO,SM,AH,UT,UE,IM} <:
    Lux.AbstractLuxContainerLayer{(:input_embedder, :recycling_embedder,
                                   :template_embedder, :extra_msa_embedder,
                                   :extra_msa_stack, :evoformer, :structure_module,
                                   :aux_heads)}
    input_embedder::IE
    recycling_embedder::RE
    template_embedder::TE       # Lux.NoOpLayer() when use_templates=static(false)
    extra_msa_embedder::MEE     # Lux.NoOpLayer() when use_extra_msa=static(false)
    extra_msa_stack::MES        # Lux.NoOpLayer() when use_extra_msa=static(false)
    evoformer::EVO
    structure_module::SM
    aux_heads::AH
    use_templates::UT           # StaticBool — NOT in container layer tuple
    use_extra_msa::UE           # StaticBool — NOT in container layer tuple
    is_multimer::IM             # StaticBool — NOT in container layer tuple
    # num_recycles is NOT a struct field — it is a runtime kwarg on predict()
end

function AlphaFold(;
    chn_msa::Int       = 256,
    chn_z::Int         = 128,
    chn_s::Int         = 384,
    chn_target_feat::Int = 22,
    chn_msa_feat::Int  = 49,
    chn_extra_msa::Int = 64,
    chn_extra_msa_feat::Int = 25,
    use_templates::Bool  = true,
    use_extra_msa::Bool  = true,
    relpos_k::Int      = 32,
    # Multimer / variant kwargs
    is_multimer::Bool              = false,
    sm_trans_scale_factor::Float32 = 10f0,
    chn_msa_out::Int               = 23,   # monomer=23, multimer=22
    # EvoformerStack kwargs
    evo_no_blocks::Int         = 48,
    # ExtraMSAStack kwargs
    extra_msa_no_blocks::Int   = 4,
    # StructureModule kwargs
    sm_no_blocks::Int          = 8,
    sm_c_ipa::Int              = 16,
    sm_no_heads_ipa::Int       = 12,
    sm_no_qk_points::Int       = 4,
    sm_no_v_points::Int        = 8,
    sm_no_resnet_blocks::Int   = 2,
    sm_c_resnet::Int           = 128,
    # TemplateEmbedder kwargs (AF2 monomer defaults)
    chn_templ_in_pair::Int     = 88,
    chn_templ_in_angles::Int   = 57,
    chn_templ::Int             = 64,
    chn_hidden_tri_att::Int    = 16,
    chn_hidden_tri_mul::Int    = 64,
    no_templ_blocks::Int       = 2,
    no_heads_tri::Int          = 4,
    pair_transition_n::Int     = 2,
    chn_hidden_pt_att::Int     = 16,   # canonical value is 16 (was incorrectly 64)
    no_heads_pt_att::Int       = 4,
)
    input_embedder     = InputEmbedder(chn_target_feat, chn_msa_feat, chn_z, chn_msa, relpos_k; is_multimer)
    recycling_embedder = RecyclingEmbedder(chn_msa, chn_z)

    template_embedder = use_templates ?
        TemplateEmbedder(
            chn_templ_in_pair, chn_templ_in_angles, chn_templ, chn_z, chn_msa,
            chn_hidden_tri_att, chn_hidden_tri_mul, no_templ_blocks,
            no_heads_tri, pair_transition_n, chn_hidden_pt_att, no_heads_pt_att,
        ) :
        Lux.NoOpLayer()

    extra_msa_embedder, extra_msa_stack = if use_extra_msa
        ExtraMSAEmbedder(chn_extra_msa_feat, chn_extra_msa),
        ExtraMSAStack(chn_extra_msa, chn_z; no_blocks=extra_msa_no_blocks)
    else
        Lux.NoOpLayer(), Lux.NoOpLayer()
    end

    evoformer = EvoformerStack(chn_msa, chn_z, chn_s; no_blocks=evo_no_blocks)
    structure_module = StructureModule(chn_s, chn_z;
        no_blocks=sm_no_blocks, chn_ipa=sm_c_ipa,
        no_heads_ipa=sm_no_heads_ipa, no_qk_points=sm_no_qk_points,
        no_v_points=sm_no_v_points, no_resnet_blocks=sm_no_resnet_blocks,
        chn_resnet=sm_c_resnet, trans_scale_factor=sm_trans_scale_factor,
        is_multimer=static(is_multimer),
    )
    aux_heads = AuxiliaryHeads(chn_s, chn_z, chn_msa; chn_out_msa=chn_msa_out)

    return AlphaFold(
        input_embedder, recycling_embedder,
        template_embedder, extra_msa_embedder, extra_msa_stack,
        evoformer, structure_module, aux_heads,
        static(use_templates), static(use_extra_msa), static(is_multimer),
    )
end

# ==============================================================================
# Model registry and symbol dispatch
# ==============================================================================

"""
    AF2_MODEL_REGISTRY

Mapping from official model-weight symbols to architecture and `.npz` filename.
Used by `AlphaFold(s::Symbol)` and `af2_weight_filename`.
"""
const AF2_MODEL_REGISTRY = Dict{Symbol, @NamedTuple{architecture::Symbol, filename::String, use_templates::Bool}}(
    :model_1     => (; architecture=:monomer, filename="params_model_1.npz",     use_templates=true),
    :model_2     => (; architecture=:monomer, filename="params_model_2.npz",     use_templates=true),
    :model_3     => (; architecture=:monomer, filename="params_model_3.npz",     use_templates=false),
    :model_4     => (; architecture=:monomer, filename="params_model_4.npz",     use_templates=false),
    :model_5     => (; architecture=:monomer, filename="params_model_5.npz",     use_templates=false),
    :model_1_ptm => (; architecture=:monomer, filename="params_model_1_ptm.npz", use_templates=true),
    :model_2_ptm => (; architecture=:monomer, filename="params_model_2_ptm.npz", use_templates=true),
    :model_3_ptm => (; architecture=:monomer, filename="params_model_3_ptm.npz", use_templates=false),
    :model_4_ptm => (; architecture=:monomer, filename="params_model_4_ptm.npz", use_templates=false),
    :model_5_ptm => (; architecture=:monomer, filename="params_model_5_ptm.npz", use_templates=false),
    :model_1_multimer_v1 => (; architecture=:multimer, filename="params_model_1_multimer_v1.npz", use_templates=true),
    :model_2_multimer_v1 => (; architecture=:multimer, filename="params_model_2_multimer_v1.npz", use_templates=true),
    :model_3_multimer_v1 => (; architecture=:multimer, filename="params_model_3_multimer_v1.npz", use_templates=true),
    :model_4_multimer_v1 => (; architecture=:multimer, filename="params_model_4_multimer_v1.npz", use_templates=true),
    :model_5_multimer_v1 => (; architecture=:multimer, filename="params_model_5_multimer_v1.npz", use_templates=true),
    :model_1_multimer_v2 => (; architecture=:multimer, filename="params_model_1_multimer_v2.npz", use_templates=true),
    :model_2_multimer_v2 => (; architecture=:multimer, filename="params_model_2_multimer_v2.npz", use_templates=true),
    :model_3_multimer_v2 => (; architecture=:multimer, filename="params_model_3_multimer_v2.npz", use_templates=true),
    :model_4_multimer_v2 => (; architecture=:multimer, filename="params_model_4_multimer_v2.npz", use_templates=true),
    :model_5_multimer_v2 => (; architecture=:multimer, filename="params_model_5_multimer_v2.npz", use_templates=true),
    :model_1_multimer_v3 => (; architecture=:multimer, filename="params_model_1_multimer_v3.npz", use_templates=true),
    :model_2_multimer_v3 => (; architecture=:multimer, filename="params_model_2_multimer_v3.npz", use_templates=true),
    :model_3_multimer_v3 => (; architecture=:multimer, filename="params_model_3_multimer_v3.npz", use_templates=true),
    :model_4_multimer_v3 => (; architecture=:multimer, filename="params_model_4_multimer_v3.npz", use_templates=true),
    :model_5_multimer_v3 => (; architecture=:multimer, filename="params_model_5_multimer_v3.npz", use_templates=true),
)

"""
    af2_weight_filename(variant::Symbol) -> String

Return the `.npz` filename for a registered AlphaFold2 model symbol.
Does not require PyCall — pure metadata lookup.

# Example
```julia
af2_weight_filename(:model_1)   # "params_model_1.npz"
```
"""
function af2_weight_filename(variant::Symbol)
    haskey(AF2_MODEL_REGISTRY, variant) ||
        throw(ArgumentError(
            "Unknown model: $(repr(variant)). " *
            "Known models: $(sort(collect(keys(AF2_MODEL_REGISTRY))))"
        ))
    return AF2_MODEL_REGISTRY[variant].filename
end

"""
    AlphaFold(s::Symbol; kwargs...)

Two-tier dispatch:
- **Model-weight symbols** (`:model_1`..`:model_5`, `:model_1_ptm`, `:model_N_multimer_vM`, …)
  look up `AF2_MODEL_REGISTRY` → route to the correct architecture.
- **Architecture symbols** (`:monomer`, `:multimer`) construct the canonical sub-layer config.

All `kwargs` are forwarded as sub-layer object overrides:
```julia
AlphaFold(:monomer)                                         # canonical monomer
AlphaFold(:model_1)                                         # same architecture, tagged for weight loading
AlphaFold(:monomer; evoformer = EvoformerStack(:monomer; no_blocks=2))  # small evoformer
```
For fine-grained scalar overrides use `AlphaFold(; ...)` directly.
"""
function AlphaFold(s::Symbol; kwargs...)
    if haskey(AF2_MODEL_REGISTRY, s)
        entry = AF2_MODEL_REGISTRY[s]
        # Registry sets use_templates; user kwargs take precedence
        return AlphaFold(
            static(entry.architecture); 
            merge(kwargs, (use_templates=entry.use_templates, ))...
        )
    end
    return AlphaFold(static(s); kwargs...)
end

# Fallback error for unsupported StaticSymbols
AlphaFold(s::Static.StaticSymbol; kwargs...) =
    throw(ArgumentError(
        "Unknown AlphaFold variant $(Static.known(s)). " *
        "Architecture symbols: :monomer, :multimer. " *
        "Model symbols: $(sort(collect(keys(AF2_MODEL_REGISTRY))))"
    ))

function AlphaFold(v::Static.StaticSymbol{:monomer};
                   use_templates::Bool=true, use_extra_msa::Bool=true, kwargs...)
    defaults = (;
        input_embedder     = InputEmbedder(v),
        recycling_embedder = RecyclingEmbedder(v),
        template_embedder  = use_templates ? TemplateEmbedder(v) : Lux.NoOpLayer(),
        extra_msa_embedder = use_extra_msa ? ExtraMSAEmbedder(v) : Lux.NoOpLayer(),
        extra_msa_stack    = use_extra_msa ? ExtraMSAStack(v) : Lux.NoOpLayer(),
        evoformer          = EvoformerStack(v),
        structure_module   = StructureModule(v),
        aux_heads          = AuxiliaryHeads(v),
    )
    merged = merge(defaults, kwargs)
    return AlphaFold(
        merged.input_embedder, merged.recycling_embedder,
        merged.template_embedder, merged.extra_msa_embedder, merged.extra_msa_stack,
        merged.evoformer, merged.structure_module, merged.aux_heads,
        static(use_templates), static(use_extra_msa), static(false),
    )
end

function AlphaFold(v::Static.StaticSymbol{:multimer};
                   use_templates::Bool=true, use_extra_msa::Bool=true, kwargs...)
    defaults = (;
        input_embedder     = InputEmbedder(v),
        recycling_embedder = RecyclingEmbedder(v),
        template_embedder  = use_templates ? TemplateEmbedder(v) : Lux.NoOpLayer(),
        extra_msa_embedder = use_extra_msa ? ExtraMSAEmbedder(v) : Lux.NoOpLayer(),
        extra_msa_stack    = use_extra_msa ? ExtraMSAStack(v) : Lux.NoOpLayer(),
        evoformer          = EvoformerStack(v),
        structure_module   = StructureModule(v),
        aux_heads          = AuxiliaryHeads(v),
    )
    merged = merge(defaults, kwargs)
    return AlphaFold(
        merged.input_embedder, merged.recycling_embedder,
        merged.template_embedder, merged.extra_msa_embedder, merged.extra_msa_stack,
        merged.evoformer, merged.structure_module, merged.aux_heads,
        static(use_templates), static(use_extra_msa), static(true),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::AlphaFold)
    return (
        input_embedder     = Lux.initialstates(rng, model.input_embedder),
        recycling_embedder = Lux.initialstates(rng, model.recycling_embedder),
        template_embedder  = Lux.initialstates(rng, model.template_embedder),
        extra_msa_embedder = Lux.initialstates(rng, model.extra_msa_embedder),
        extra_msa_stack    = Lux.initialstates(rng, model.extra_msa_stack),
        evoformer          = Lux.initialstates(rng, model.evoformer),
        structure_module   = Lux.initialstates(rng, model.structure_module),
        aux_heads          = Lux.initialstates(rng, model.aux_heads),
        recycling_state    = (; m_1_prev=nothing, z_prev=nothing, x_prev=nothing),
        residue_constants  = (;
            default_frames = restype_rigid_group_default_frame,
            group_idx      = restype_atom14_to_rigid_group,
            lit_positions  = restype_atom14_rigid_group_positions,
            atom_mask      = restype_atom14_mask,
        ),
    )
end


# ==============================================================================
# Template and extra-MSA helpers (static dispatch)
# ==============================================================================

# Active template path — concat template_pair into z, optionally extend m + msa_mask
function _embed_templates(model, feats, m, z, msa_mask, pair_mask, ps, st, ::Static.True)
    templ_in = merge(feats, (; z, pair_mask))
    templ_out, st_te = model.template_embedder(templ_in, ps.template_embedder, st.template_embedder)
    z_new = z .+ templ_out.template_pair
    m_new, msa_mask_new = if !isnothing(templ_out.template_single)
        t_single = templ_out.template_single             # [C_m, N, N_templ, B]
        t_mask   = feats.template_torsion_angles_mask[3, :, :, :]  # [N, N_templ, B]
        cat(m, t_single; dims=3), cat(msa_mask, t_mask; dims=2)
    else
        m, msa_mask
    end
    return m_new, z_new, msa_mask_new, st_te
end

# No-op template path
function _embed_templates(model, feats, m, z, msa_mask, pair_mask, ps, st, ::Static.False)
    return m, z, msa_mask, st.template_embedder
end

# Active extra-MSA path
function _run_extra_msa_stack(model, feats, z, pair_mask, ps, st, ::Static.True)
    extra_feat       = build_extra_msa_feat(feats.extra_msa, feats.extra_msa_deletion_value,
                                             feats.extra_msa_has_deletion)
    a, st_embe       = model.extra_msa_embedder(extra_feat, ps.extra_msa_embedder,
                                                 st.extra_msa_embedder)
    z_new, st_emsa   = model.extra_msa_stack(a, z, Bool.(feats.extra_msa_mask), pair_mask,
                                              ps.extra_msa_stack, st.extra_msa_stack)
    return z_new, st_embe, st_emsa
end

# No-op extra-MSA path
function _run_extra_msa_stack(model, feats, z, pair_mask, ps, st, ::Static.False)
    return z, st.extra_msa_embedder, st.extra_msa_stack
end


# ==============================================================================
# iteration() — one recycling step
# ==============================================================================

"""
    iteration(model::AlphaFold, feats::NamedTuple, ps, st)

Perform one recycling iteration: embed → evoformer → structure module → heads.

Returns `(outputs::NamedTuple, st_new)`. The updated recycling state is in
`st_new.recycling_state`. On the first call, recycling state is zero-initialized
lazily from the embedding dimensions and element type.
"""
function iteration(model::AlphaFold, feats::NamedTuple, ps, st)
    N, B = size(feats.seq_mask)

    # 1. Pair mask [N, N, B] Bool — all downstream layers require AbstractArray{Bool}
    seq_mask_bool = Bool.(feats.seq_mask)
    pair_mask = reshape(seq_mask_bool, N, 1, B) .& reshape(seq_mask_bool, 1, N, B)

    # 2. Input embedder → m [C_m, N, S, B], z [C_z, N, N, B]
    emb, st_ie = model.input_embedder(feats, ps.input_embedder, st.input_embedder)

    # 3. Extract / zero-initialize recycling state
    m_1_prev = isnothing(st.recycling_state.m_1_prev) ?
        zeros(eltype(emb.m), size(emb.m, 1), size(emb.m, 2), size(emb.m, 4)) :
        st.recycling_state.m_1_prev
    z_prev = isnothing(st.recycling_state.z_prev) ?
        zeros(eltype(emb.z), size(emb.z)) :
        st.recycling_state.z_prev
    x_prev = isnothing(st.recycling_state.x_prev) ?
        zeros(eltype(emb.m), 3, 37, size(emb.m, 2), size(emb.m, 4)) :
        st.recycling_state.x_prev

    # 4. Pseudo-beta from previous atom positions (no mask — matches Python)
    pseudo_beta_x_prev = pseudo_beta_fn(feats.aatype, x_prev)  # [3, N, B]

    # 5. Recycling embedder → re.m [C_m, N, B], re.z [C_z, N, N, B]
    re, st_re = model.recycling_embedder(
        (; x=pseudo_beta_x_prev, m=m_1_prev, z=z_prev),
        ps.recycling_embedder, st.recycling_embedder,
    )

    # 6. Add recycling embeddings: update first MSA row, add to pair
    C_m, _, S, _ = size(emb.m)
    m = cat(
        emb.m[:, :, 1:1, :] .+ reshape(re.m, C_m, N, 1, B),
        emb.m[:, :, 2:end, :];
        dims=3,
    )
    z = emb.z .+ re.z

    # 7. Template embedding (static dispatch)
    # Ensure msa_mask is Bool for all downstream layers
    msa_mask_bool = Bool.(feats.msa_mask)
    m, z, msa_mask, st_te = _embed_templates(
        model, feats, m, z, msa_mask_bool, pair_mask, ps, st, model.use_templates,
    )

    # 8. Extra MSA stack (static dispatch)
    z, st_embe, st_emsa = _run_extra_msa_stack(
        model, feats, z, pair_mask, ps, st, model.use_extra_msa,
    )

    # 9. Evoformer → evo.m, evo.z, evo.s
    evo, st_evo = model.evoformer(m, z, msa_mask, pair_mask, ps.evoformer, st.evoformer)

    # 10. Structure module → frames [7, N, B, no_blocks], angles, states, single
    sm_out, st_sm = model.structure_module(
        (; s=evo.s, z=evo.z, mask=seq_mask_bool),
        ps.structure_module, st.structure_module,
    )

    # 11. Atom positions from structure module output
    rc     = st.residue_constants
    r_last = Rigid(sm_out.frames[:, :, :, end])        # [7,N,B] → Rigid
    all_frames = torsion_angles_to_frames(
        r_last, sm_out.angles[:, :, :, :, end], feats.aatype, rc.default_frames,
    )
    atom14_pos = frames_and_literature_positions_to_atom14_pos(
        all_frames, feats.aatype, rc.group_idx, rc.lit_positions, rc.atom_mask,
    )
    final_atom_positions = atom14_to_atom37(atom14_pos, feats.aatype)  # [3, 37, N, B]

    # 12. Final outputs from feats
    final_atom_mask    = feats.atom37_atom_exists      # [37, N, B] Bool
    final_affine_tensor = sm_out.frames[:, :, :, end]  # [7, N, B]

    # 13. Auxiliary heads
    head_out, st_ah = model.aux_heads(
        (; s=evo.s, z=evo.z, m=evo.m),
        ps.aux_heads, st.aux_heads,
    )

    # 14. New recycling state
    new_recycling_state = (;
        m_1_prev = evo.m[:, :, 1, :],      # [C_m, N, B]
        z_prev   = evo.z,                   # [C_z, N, N, B]
        x_prev   = final_atom_positions,    # [3, 37, N, B]
    )

    # 15. Assemble outputs
    outputs = merge(
        (;
            m                   = evo.m,
            z                   = evo.z,
            s                   = evo.s,
            sm                  = sm_out,
            final_atom_positions,
            final_atom_mask,
            final_affine_tensor,
        ),
        head_out,
    )

    st_new = merge(st, (;
        input_embedder     = st_ie,
        recycling_embedder = st_re,
        template_embedder  = st_te,
        extra_msa_embedder = st_embe,
        extra_msa_stack    = st_emsa,
        evoformer          = st_evo,
        structure_module   = st_sm,
        aux_heads          = st_ah,
        recycling_state    = new_recycling_state,
    ))

    return outputs, st_new
end


# ==============================================================================
# tolerance_reached() — Cα RMSD convergence check
# ==============================================================================

"""
    tolerance_reached(prev_pos, next_pos, seq_mask; threshold=-1.0)

Return `true` if the mean Cα pairwise distance change between iterations is ≤ `threshold`.

If `threshold < 0` (default), always returns `false` (early stop disabled).

- `prev_pos`, `next_pos`: `[3, 37, N, B]` atom37 positions
- `seq_mask`: `[N, B]` sequence mask
"""
function tolerance_reached(prev_pos, next_pos, seq_mask; threshold=-1.0)
    threshold < 0 && return false

    ca_idx  = atom_order["CA"] + 1
    N, B    = size(seq_mask)
    prev_ca = prev_pos[:, ca_idx, :, :]    # [3, N, B]
    next_ca = next_pos[:, ca_idx, :, :]

    # Pairwise Euclidean distances [N, N, B]
    function _pdist(x)
        d = reshape(x, 3, N, 1, B) .- reshape(x, 3, 1, N, B)
        dropdims(sqrt.(sum(d .^ 2; dims=1)); dims=1)
    end

    sq_diff = (_pdist(prev_ca) .- _pdist(next_ca)) .^ 2    # [N, N, B]
    mask_2d = reshape(seq_mask, N, 1, B) .* reshape(seq_mask, 1, N, B)
    denom   = max(sum(mask_2d), 1)
    diff    = sqrt(sum(sq_diff .* mask_2d) / denom + 1f-8)

    return diff <= threshold
end


# ==============================================================================
# predict() — recycling loop + Lux functor
# ==============================================================================

"""
    predict(model::AlphaFold, feats::NamedTuple, ps, st;
            num_recycles=3, early_stop_threshold=-1.0)

Run the AlphaFold recycling loop. Calls `iteration()` exactly `num_recycles + 1` times
(one initial pass + `num_recycles` recycle steps).

`num_recycles` is a runtime kwarg — it is NOT a struct field on `AlphaFold`.
Use `num_recycles=0` for a single forward pass (no recycling).

Returns `(outputs, st_new)` where `outputs.num_recycles` is the number of additional
recycles actually performed (may be fewer if early-stop triggered).
"""
function predict(model::AlphaFold, feats::NamedTuple, ps, st;
                 num_recycles::Int=3, early_stop_threshold::Real=-1.0)
    outputs = nothing
    n_recycles = 0

    for i in 1:(num_recycles + 1)
        prev_x  = isnothing(st.recycling_state.x_prev) ? nothing : st.recycling_state.x_prev
        outputs, st = iteration(model, feats, ps, st)
        next_x  = st.recycling_state.x_prev

        # Early-stop check after each iteration except the last
        if i < num_recycles + 1
            n_recycles = i
            if !isnothing(prev_x) &&
                tolerance_reached(prev_x, next_x, feats.seq_mask; threshold=early_stop_threshold)
                break
            end
        else
            n_recycles = num_recycles
        end
    end

    outputs = merge(outputs, (; num_recycles=n_recycles))
    return outputs, st
end

# Thin Lux functor — delegates to predict() with default num_recycles=3
(model::AlphaFold)(feats::NamedTuple, ps, st) = predict(model, feats, ps, st)
