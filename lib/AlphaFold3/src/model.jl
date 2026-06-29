# Top-level AlphaFold3 model.
#
# NOTE: the design (af3-full-model, task 1.1) calls for `src/alphafold3.jl`, but on a
# case-insensitive filesystem (macOS) that path collides with the module file `AlphaFold3.jl`.
# The top-level model therefore lives here in `model.jl` (already on the include path).


"""
    AlphaFold3Model(; chn_s_input, chn_s, chn_z, chn_m, chn_token_atom, chn_atom, chn_atom_pair, …)

Top-level AlphaFold3 model (openfold-3 `OpenFold3`). A `Lux.AbstractLuxContainerLayer` that
owns every trunk + structure + head sub-module and implements the full inference pipeline:

    run_trunk (pairformer recycling)  →  SampleDiffusion (structure)  →  AuxiliaryHeadsAllAtom

# Sub-layers
- `input_embedder` (`InputEmbedderAllAtom`) — token single/pair init from atom + token features
- `template_embedder` (`TemplateEmbedderAllAtom`) — adds template pair features to `z`
- `msa_embedder` (`MSAModuleEmbedder`) — builds the MSA representation `m`
- `msa_module` (`MSAModuleStack`) — refines `z` with `m`
- `pairformer` (`PairFormerStack`) — refines `(s, z)`
- `layer_norm_z` / `linear_z`, `layer_norm_s` / `linear_s` — recycling projections
- `sample_diffusion` (`SampleDiffusion`, owns the `DiffusionModule`) — predicts atom positions
- `auxiliary_heads` (`AuxiliaryHeadsAllAtom`) — distogram + confidence heads

> Note: the design lists `diffusion_module` and `sample_diffusion` as separate fields, but
> `SampleDiffusion` already owns a `DiffusionModule`; holding both as Lux fields would duplicate
> the diffusion parameters. We hold `sample_diffusion` only (its `.diffusion_module` is the
> single parameter owner).

# Inputs
- `batch::NamedTuple` — token features (`restype`, `profile`, `deletion_mean`, `token_bonds`,
  relpos ids, `token_mask`), atom features (`ref_*`, `atom_mask`, `num_atoms_per_token`,
  `atom_to_token_index`, `start_atom_index`), MSA (`msa`, `has_deletion`, `deletion_value`,
  `msa_mask`), and template (`template_*`) features.

# Returns
- `(; atom_positions, s, z, plddt, pae, pde, distogram, experimentally_resolved)`, `st`
"""
struct AlphaFold3Model{IE,TE,ME,MM,PF,LNZ,LZ,LNS,LS,SD,AH} <:
       Lux.AbstractLuxContainerLayer{(:input_embedder, :template_embedder, :msa_embedder,
                                      :msa_module, :pairformer, :layer_norm_z, :linear_z,
                                      :layer_norm_s, :linear_s, :sample_diffusion,
                                      :auxiliary_heads)}
    input_embedder::IE
    template_embedder::TE
    msa_embedder::ME
    msa_module::MM
    pairformer::PF
    layer_norm_z::LNZ
    linear_z::LZ
    layer_norm_s::LNS
    linear_s::LS
    sample_diffusion::SD
    auxiliary_heads::AH
end

function AlphaFold3Model(;
    # core channel dims
    chn_s_input::Int, chn_s::Int, chn_z::Int, chn_m::Int,
    # atom / token dims (shared by input-embedder atom encoder and diffusion atom modules)
    chn_atom::Int, chn_atom_pair::Int, chn_token_atom::Int,
    chn_atom_hidden::Int, atom_no_heads::Int, atom_no_blocks::Int, atom_n_transition::Int,
    n_query::Int, n_key::Int, chn_ref_element::Int=128, chn_ref_name_chars::Int=256,
    # MSA module
    chn_hidden_msa_att::Int, chn_hidden_opm::Int, chn_hidden_mul_msa::Int, chn_hidden_pair_att_msa::Int,
    no_heads_msa::Int, no_heads_pair_msa::Int, transition_n_msa::Int, msa_no_blocks::Int,
    chn_m_feats::Int=34,
    # template module
    chn_t::Int, templ_no_blocks::Int, templ_no_heads::Int,
    chn_hidden_tri_mul::Int, chn_hidden_tri_att::Int, templ_transition_n::Int,
    # pairformer trunk
    chn_hidden_pair_bias::Int, no_heads_pair_bias::Int, chn_hidden_mul::Int,
    chn_hidden_pair_att::Int, no_heads_pair::Int, transition_n::Int, pairformer_no_blocks::Int,
    # diffusion
    chn_token_diff::Int, chn_fourier_emb::Int=256, diff_no_heads::Int, diff_no_blocks::Int,
    diff_n_transition::Int, sigma_data=16f0,
    gamma_0=0.8, gamma_min=1.0, noise_scale=1.003, step_scale=1.5,
    # confidence heads
    max_atoms_per_token::Int, conf_no_blocks::Int=pairformer_no_blocks,
    no_dist_bins::Int=64, no_pae_bins::Int=64, no_pde_bins::Int=64,
    no_lddt_bins::Int=50, no_er_bins::Int=2, conf_no_bin_dist::Int=15,
    # relpos
    max_relative_idx::Int=32, max_relative_chain::Int=2, num_relpos_dims::Int=139,
)
    # ---- input embedder (atom encoder without noisy positions) ----
    input_atom_enc = AtomAttentionEncoder(;
        chn_atom=chn_atom, chn_atom_pair=chn_atom_pair, chn_token=chn_token_atom,
        chn_hidden=chn_atom_hidden, no_heads=atom_no_heads, no_blocks=atom_no_blocks,
        n_transition=atom_n_transition, n_query=n_query, n_key=n_key, add_noisy_pos=false,
        chn_ref_element=chn_ref_element, chn_ref_name_chars=chn_ref_name_chars,
    )
    input_embedder = InputEmbedderAllAtom(input_atom_enc; chn_s_input, chn_s, chn_z,
                                          max_relative_idx, max_relative_chain, num_relpos_dims)

    # ---- template embedder ----
    template_embedder = TemplateEmbedderAllAtom(; chn_in=chn_z, chn_t, chn_z,
                                                no_blocks=templ_no_blocks, no_heads=templ_no_heads,
                                                chn_hidden_tri_mul, chn_hidden_tri_att,
                                                transition_n=templ_transition_n)

    # ---- MSA embedder + module ----
    msa_embedder = MSAModuleEmbedder(; chn_m, chn_s_input, chn_m_feats)
    msa_module = MSAModuleStack(;
        chn_m, chn_z, chn_hidden_msa_att, chn_hidden_opm, chn_hidden_mul=chn_hidden_mul_msa,
        chn_hidden_pair_att=chn_hidden_pair_att_msa, no_heads_msa, no_heads_pair=no_heads_pair_msa,
        transition_n=transition_n_msa, no_blocks=msa_no_blocks,
    )

    # ---- pairformer trunk ----
    pairformer = PairFormerStack(;
        chn_s, chn_z, chn_hidden_pair_bias, no_heads_pair_bias, chn_hidden_mul,
        chn_hidden_pair_att, no_heads_pair, transition_n, no_blocks=pairformer_no_blocks,
    )

    # ---- recycling projections ----
    layer_norm_z = Lux.LayerNorm((chn_z, 1, 1); dims=1)
    linear_z     = Lux.Dense(chn_z => chn_z; use_bias=false)
    layer_norm_s = Lux.LayerNorm((chn_s, 1); dims=1)
    linear_s     = Lux.Dense(chn_s => chn_s; use_bias=false)

    # ---- diffusion (conditioning + atom enc/dec + transformer) ----
    diffusion_conditioning = DiffusionConditioning(;
        chn_s_input, chn_s, chn_z, chn_fourier_emb,
        max_relative_idx, max_relative_chain, sigma_data, num_relpos_dims,
    )
    diff_atom_enc = AtomAttentionEncoder(;
        chn_atom=chn_atom, chn_atom_pair=chn_atom_pair, chn_token=chn_token_diff,
        chn_hidden=chn_atom_hidden, no_heads=atom_no_heads, no_blocks=atom_no_blocks,
        n_transition=atom_n_transition, n_query=n_query, n_key=n_key, add_noisy_pos=true,
        chn_single=chn_s, chn_pair=chn_z, chn_ref_element=chn_ref_element, chn_ref_name_chars=chn_ref_name_chars,
    )
    diff_atom_dec = AtomAttentionDecoder(;
        chn_atom=chn_atom, chn_atom_pair=chn_atom_pair, chn_token=chn_token_diff,
        chn_hidden=chn_atom_hidden, no_heads=atom_no_heads, no_blocks=atom_no_blocks,
        n_transition=atom_n_transition, n_query=n_query, n_key=n_key,
    )
    diff_transformer = DiffusionTransformer(;
        chn_a=chn_token_diff, chn_cond=chn_s, chn_pair=chn_z, chn_hidden=chn_atom_hidden,
        no_heads=diff_no_heads, no_blocks=diff_no_blocks, n_transition=diff_n_transition,
        use_ada_layer_norm=true,
    )
    diffusion_module = DiffusionModule(;
        diffusion_conditioning, atom_attn_enc=diff_atom_enc, atom_attn_dec=diff_atom_dec,
        diffusion_transformer=diff_transformer, chn_s, chn_token=chn_token_diff, sigma_data,
    )
    sample_diffusion = SampleDiffusion(diffusion_module; gamma_0, gamma_min, noise_scale, step_scale)

    # ---- auxiliary heads ----
    conf_pairformer = PairFormerStack(;
        chn_s, chn_z, chn_hidden_pair_bias, no_heads_pair_bias, chn_hidden_mul,
        chn_hidden_pair_att, no_heads_pair, transition_n, no_blocks=conf_no_blocks,
    )
    auxiliary_heads = AuxiliaryHeadsAllAtom(;
        pairformer_embedding = PairformerEmbedding(; chn_s_input=chn_s, chn_z, pairformer_stack=conf_pairformer,
                                                   no_bin=conf_no_bin_dist),
        distogram_head               = DistogramHead(chn_z; no_bins=no_dist_bins),
        pae_head                     = PredictedAlignedErrorHead(chn_z; no_bins=no_pae_bins),
        pde_head                     = PredictedDistanceErrorHead(chn_z; no_bins=no_pde_bins),
        plddt_head                   = PerResidueLDDTAllAtom(chn_s, max_atoms_per_token; no_bins=no_lddt_bins),
        experimentally_resolved_head = ExperimentallyResolvedHeadAllAtom(chn_s, max_atoms_per_token; no_bins=no_er_bins),
    )

    return AlphaFold3Model(
        input_embedder, template_embedder, msa_embedder, msa_module, pairformer,
        layer_norm_z, linear_z, layer_norm_s, linear_s, sample_diffusion, auxiliary_heads,
    )
end

# Custom state: the auto-derived sublayer states plus a FROZEN master `rng`. Diffusion sampling
# draws its per-sample RNGs from `Lux.replicate(st.rng)` (never mutating the stored copy), so the
# whole model forward is reproducible — same `st` ⇒ same samples — while `num_samples > 1` still
# yields distinct structures (each sample gets a different derived seed). `Lux.replicate(rng)` is
# the rng consumer here, satisfying the rng-first convention at the construction boundary.
function Lux.initialstates(rng::Random.AbstractRNG, m::AlphaFold3Model)
    master = Lux.replicate(rng)   # snapshot of the seed for diffusion sampling
    return (;
        input_embedder    = Lux.initialstates(rng, m.input_embedder),
        template_embedder = Lux.initialstates(rng, m.template_embedder),
        msa_embedder      = Lux.initialstates(rng, m.msa_embedder),
        msa_module        = Lux.initialstates(rng, m.msa_module),
        pairformer        = Lux.initialstates(rng, m.pairformer),
        layer_norm_z      = Lux.initialstates(rng, m.layer_norm_z),
        linear_z          = Lux.initialstates(rng, m.linear_z),
        layer_norm_s      = Lux.initialstates(rng, m.layer_norm_s),
        linear_s          = Lux.initialstates(rng, m.linear_s),
        sample_diffusion  = Lux.initialstates(rng, m.sample_diffusion),
        auxiliary_heads   = Lux.initialstates(rng, m.auxiliary_heads),
        rng               = master,
    )
end

"""
    run_trunk(model::AlphaFold3Model, batch, ps, st; num_recycles=3) -> ((; s, z, s_input), st)

Pairformer recycling loop (openfold-3 `OpenFold3.run_trunk`). The input embedder runs **once**;
`s`/`z` are the recycled state (zero-initialised), iterated `num_recycles + 1` times. Each cycle:

    z = z_init + linear_z(layer_norm_z(z));  z += template_embedder(batch, z)
    m  = msa_embedder(batch, s_input);        z  = msa_module(m, z)
    s  = s_init + linear_s(layer_norm_s(s));  s, z = pairformer(s, z)

Returns `((s, z, s_input), st)` for the diffusion + head stages.
"""
function run_trunk(model::AlphaFold3Model, batch::NamedTuple, ps, st; num_recycles::Int=3)
    emb, st_ie = model.input_embedder(batch, ps.input_embedder, st.input_embedder)
    s_input, s_init, z_init = emb.s_input, emb.s, emb.z

    s = zero(s_init)
    z = zero(z_init)

    token_mask = batch.token_mask                       # [N, B] Bool
    N, B = size(token_mask)
    pair_mask = reshape(token_mask, N, 1, B) .& reshape(token_mask, 1, N, B)

    st_lnz = st.layer_norm_z; st_lz = st.linear_z
    st_lns = st.layer_norm_s; st_ls = st.linear_s
    st_te  = st.template_embedder; st_me = st.msa_embedder
    st_mm  = st.msa_module; st_pf = st.pairformer

    for _ in 1:(num_recycles + 1)
        z_recycle, st_lnz = model.layer_norm_z(z, ps.layer_norm_z, st_lnz)
        z_recycle, st_lz  = model.linear_z(z_recycle, ps.linear_z, st_lz)
        z = z_init .+ z_recycle

        z_templ, st_te = model.template_embedder(batch, z, pair_mask, ps.template_embedder, st_te)
        z = z .+ z_templ

        msa, st_me = model.msa_embedder(batch, s_input, ps.msa_embedder, st_me)
        z, st_mm = model.msa_module(msa.msa, z, msa.msa_mask, pair_mask, ps.msa_module, st_mm)

        s_recycle, st_lns = model.layer_norm_s(s, ps.layer_norm_s, st_lns)
        s_recycle, st_ls  = model.linear_s(s_recycle, ps.linear_s, st_ls)
        s = s_init .+ s_recycle

        pf, st_pf = model.pairformer(s, z, token_mask, pair_mask, ps.pairformer, st_pf)
        s, z = pf.s, pf.z
    end

    st_new = merge(st, (input_embedder=st_ie, template_embedder=st_te, msa_embedder=st_me,
                        msa_module=st_mm, pairformer=st_pf, layer_norm_z=st_lnz, linear_z=st_lz,
                        layer_norm_s=st_lns, linear_s=st_ls))
    return (; s, z, s_input), st_new
end

"""
    (model::AlphaFold3Model)(batch, ps, st; num_recycles=3, num_samples=1,
                        num_rollout_steps=200)

Full AF3 forward pass: trunk recycling → diffusion sampling → auxiliary heads. Returns the
output NamedTuple `(; atom_positions [3, N_atom, num_samples, B], s, z, plddt, pae, pde,
distogram, experimentally_resolved)` and updated `st`.

Diffusion randomness is sourced from the frozen master `st.rng`: each of the `num_samples`
samples gets a distinct derived seed, drawn from `Lux.replicate(st.rng)` so the stored rng is
never mutated. The forward is therefore reproducible (same `st` ⇒ same samples) while distinct
samples remain distinct.
"""
function (model::AlphaFold3Model)(batch::NamedTuple, ps, st;
                             num_recycles::Int=3, num_samples::Int=1,
                             num_rollout_steps::Int=200)
    (; s, z, s_input), st1 = run_trunk(model, batch, ps, st; num_recycles)

    # Pass all `T`-typed kwargs together so `af3_noise_schedule`'s `where {T}` binds consistently.
    T = eltype(s)
    schedule = af3_noise_schedule(num_rollout_steps;
                                  sigma_data=T(16f0), s_max=T(160f0), s_min=T(4f-4), p=7)

    # Per-sample RNGs derived from a replicate of the frozen master (st.rng is left untouched).
    sample_rng = Lux.replicate(st.rng)
    st_sd = st1.sample_diffusion
    xl_first, st_sd = model.sample_diffusion(Random.Xoshiro(rand(sample_rng, UInt64)), batch, schedule,
                                             s_input, s, z, ps.sample_diffusion, st_sd)   # [3, N_atom, B]
    N_atom = size(xl_first, 2); B = size(xl_first, 3)
    xl = reshape(xl_first, 3, N_atom, 1, B)
    for _ in 2:num_samples
        xl_sample, st_sd = model.sample_diffusion(Random.Xoshiro(rand(sample_rng, UInt64)), batch, schedule,
                                                  s_input, s, z, ps.sample_diffusion, st_sd)
        xl = cat(xl, reshape(xl_sample, 3, N_atom, 1, B); dims=3)
    end

    # Heads use the first sample's predicted positions for representative-atom extraction.
    batch_h = merge(batch, (; atom_positions_predicted = xl_first))
    heads, st_h = model.auxiliary_heads(s, z, batch_h, ps.auxiliary_heads, st1.auxiliary_heads)

    out = (;
        atom_positions = xl,                       # [3, N_atom, num_samples, B]
        s, z,
        plddt                   = heads.plddt,
        pae                     = heads.pae,
        pde                     = heads.pde,
        distogram               = heads.distogram,
        experimentally_resolved = heads.experimentally_resolved,
    )
    st_new = merge(st1, (sample_diffusion=st_sd, auxiliary_heads=st_h))
    return out, st_new
end
