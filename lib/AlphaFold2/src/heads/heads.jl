"""
    PerResidueLDDTCaPredictor(chn_s, chn_hidden, no_bins; use_bias=true)

Predicts per-residue pLDDT confidence logits from the single representation `s`.

Architecture: LayerNorm → Dense(chn_s ⇒ chn_hidden, relu) → Dense(chn_hidden ⇒ chn_hidden, relu) → Dense(chn_hidden ⇒ no_bins)

# Arguments
- `chn_s`: Single representation channel dimension
- `chn_hidden`: Hidden dimension for the MLP
- `no_bins`: Number of pLDDT bins

# Keyword Arguments
- `use_bias`: Bool or NamedTuple for bias control (default: true)
- `epsilon`: LayerNorm epsilon (default: `1f-5`)

# Inputs
- `s`: Single representation `[chn_s, N, B]`

# Returns
- `logits`: pLDDT logits `[no_bins, N, B]`
- `st`: Updated state
"""
struct PerResidueLDDTCaPredictor{LN, L1, L2, L3} <:
    Lux.AbstractLuxContainerLayer{(:layer_norm, :linear_1, :linear_2, :linear_3)}
    layer_norm::LN
    linear_1::L1
    linear_2::L2
    linear_3::L3
end

function PerResidueLDDTCaPredictor(chn_s, chn_hidden, no_bins; use_bias=true, epsilon::Real=1f-5)
    use_bias = resolve_defaults(use_bias, (:layer_norm, :linear_1, :linear_2, :linear_3))
    return PerResidueLDDTCaPredictor(
        Lux.LayerNorm((chn_s, 1); dims=1, epsilon),
        Lux.Dense(chn_s => chn_hidden, Lux.relu; use_bias=use_bias.linear_1),
        Lux.Dense(chn_hidden => chn_hidden, Lux.relu; use_bias=use_bias.linear_2),
        Lux.Dense(chn_hidden => no_bins; use_bias=use_bias.linear_3),
    )
end

function (l::PerResidueLDDTCaPredictor)(s, ps, st)
    x, st_ln = l.layer_norm(s, ps.layer_norm, st.layer_norm)
    x, st_l1 = l.linear_1(x, ps.linear_1, st.linear_1)
    x, st_l2 = l.linear_2(x, ps.linear_2, st.linear_2)
    x, st_l3 = l.linear_3(x, ps.linear_3, st.linear_3)
    return x, merge(st, (; layer_norm=st_ln, linear_1=st_l1, linear_2=st_l2, linear_3=st_l3))
end

"""
    DistogramHead(chn_z, no_bins; use_bias=true)

Predicts pairwise distance distribution logits from the pair representation `z`.
Output is symmetrized: `logits + permutedims(logits, (1, 3, 2, 4))`.

# Arguments
- `chn_z`: Pair representation channel dimension
- `no_bins`: Number of distance bins

# Keyword Arguments
- `use_bias`: Bool for linear bias (default: true)

# Inputs
- `z`: Pair representation `[chn_z, N, N, B]`

# Returns
- `logits`: Symmetrized distance logits `[no_bins, N, N, B]`
- `st`: Updated state
"""
struct DistogramHead{L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
end

DistogramHead(chn_z, no_bins; use_bias=true) = DistogramHead(Lux.Dense(chn_z => no_bins; use_bias))

function (l::DistogramHead)(z, ps, st)
    logits, st_linear = l.linear(z, ps.linear, st.linear)
    return logits .+ permutedims(logits, (1, 3, 2, 4)), merge(st, (; linear=st_linear))
end

"""
    TMScoreHead(chn_z, no_bins; use_bias=true)

Predicts TM-score logits from the pair representation `z`.
Single linear projection with no activation or symmetry.

# Arguments
- `chn_z`: Pair representation channel dimension
- `no_bins`: Number of TM-score bins

# Keyword Arguments
- `use_bias`: Bool for linear bias (default: true)

# Inputs
- `z`: Pair representation `[chn_z, N, N, B]`

# Returns
- `logits`: TM-score logits `[no_bins, N, N, B]`
- `st`: Updated state
"""
struct TMScoreHead{L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
end

TMScoreHead(chn_z, no_bins; use_bias=true) = TMScoreHead(Lux.Dense(chn_z => no_bins; use_bias))

function (l::TMScoreHead)(z, ps, st)
    logits, st_linear = l.linear(z, ps.linear, st.linear)
    return logits, merge(st, (; linear=st_linear))
end

"""
    MaskedMSAHead(chn_m, chn_out; use_bias=true)

Predicts MSA reconstruction logits from the MSA representation `m`.
Architecture: Dense(chn_m ⇒ chn_out).

# Arguments
- `chn_m`: MSA representation channel dimension
- `chn_out`: Output vocabulary size (23 for monomer, 22 for multimer)

# Keyword Arguments
- `use_bias`: Bool for linear bias (default: true)

# Inputs
- `m`: MSA representation `[chn_m, N, S, B]`

# Returns
- `logits`: Raw logits `[chn_out, N, S, B]`
- `st`: Updated state
"""
struct MaskedMSAHead{L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
end

MaskedMSAHead(chn_m, chn_out; use_bias=true) = 
    MaskedMSAHead(Lux.Dense(chn_m => chn_out; use_bias))

function (l::MaskedMSAHead)(m, ps, st)
    logits, st_linear = l.linear(m, ps.linear, st.linear)
    return logits, merge(st, (; linear=st_linear))
end

"""
    ExperimentallyResolvedHead(chn_s, chn_out; use_bias=true)

Predicts per-atom experimental resolvability logits from the single representation `s`.
Single linear projection with no activation.

# Arguments
- `chn_s`: Single representation channel dimension
- `chn_out`: Number of atom types (37 for standard atom37)

# Keyword Arguments
- `use_bias`: Bool for linear bias (default: true)

# Inputs
- `s`: Single representation `[chn_s, N, B]`

# Returns
- `logits`: Per-atom resolvability logits `[chn_out, N, B]`
- `st`: Updated state
"""
struct ExperimentallyResolvedHead{L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
end

ExperimentallyResolvedHead(chn_s, chn_out; use_bias=true) = 
    ExperimentallyResolvedHead(Lux.Dense(chn_s => chn_out; use_bias))

function (l::ExperimentallyResolvedHead)(s, ps, st)
    logits, st_linear = l.linear(s, ps.linear, st.linear)
    return logits, merge(st, (; linear=st_linear))
end

"""
    AuxiliaryHeads(chn_s, chn_z, chn_m; kwargs...)

Container layer that wires all 5 auxiliary heads together and computes
inference-time post-processing (pLDDT scoring, and optionally pTM/iPTM/PAE).

# Arguments
- `chn_s`: Single representation channel dimension
- `chn_z`: Pair representation channel dimension
- `chn_m`: MSA representation channel dimension

# Keyword Arguments
- `no_bins_lddt`: Number of pLDDT bins (default: 50)
- `no_bins_dist`: Number of distogram bins (default: 64)
- `no_bins_tm`: Number of TM-score bins (default: 64)
- `chn_hidden_lddt`: Hidden dim for pLDDT (default: 128)
- `chn_out_msa`: MSA output vocab size (default: 23)
- `chn_out_exp`: Atom types for exp. resolved (default: 37)
- `use_bias`: Bool or NamedTuple for bias control (default: true)

# Inputs (NamedTuple)
- `s`: Single representation `[chn_s, N, B]`
- `z`: Pair representation `[chn_z, N, N, B]`
- `m`: MSA representation `[chn_m, N, S, B]`

# Returns (NamedTuple)
- `lddt_logits`: Per-residue pLDDT logits `[no_bins_lddt, N, B]`
- `plddt`: Per-residue pLDDT scores (scaled to 0-100) `[N, B]`
- `distogram_logits`: Symmetric pairwise distance logits `[no_bins_dist, N, N, B]`
- `tm_logits`: TM-score logits `[no_bins_tm, N, N, B]`
- `masked_msa_logits`: MSA reconstruction log-probabilities `[chn_out_msa, N, S, B]`
- `experimentally_resolved_logits`: Per-atom resolvability logits `[chn_out_exp, N, B]`
- `st`: Updated state
"""
struct AuxiliaryHeads{PLDDT, DIST, TM, MSA, EXP} <: Lux.AbstractLuxContainerLayer{(
    :plddt, :distogram, :tm, :masked_msa, :experimentally_resolved
)}
    plddt::PLDDT
    distogram::DIST
    tm::TM
    masked_msa::MSA
    experimentally_resolved::EXP
end

function AuxiliaryHeads(
    chn_s, chn_z, chn_m;
    no_bins_lddt=50, no_bins_dist=64, no_bins_tm=64,
    chn_hidden_lddt=128, chn_out_msa=23, chn_out_exp=37,
    use_bias=true,
)
    use_bias = resolve_defaults(use_bias, (:plddt, :distogram, :tm, :masked_msa, :experimentally_resolved))
    return AuxiliaryHeads(
        PerResidueLDDTCaPredictor(chn_s, chn_hidden_lddt, no_bins_lddt; use_bias=use_bias.plddt),
        DistogramHead(chn_z, no_bins_dist; use_bias=use_bias.distogram),
        TMScoreHead(chn_z, no_bins_tm; use_bias=use_bias.tm),
        MaskedMSAHead(chn_m, chn_out_msa; use_bias=use_bias.masked_msa),
        ExperimentallyResolvedHead(chn_s, chn_out_exp; use_bias=use_bias.experimentally_resolved),
    )
end

(l::AuxiliaryHeads)(inputs::NamedTuple, ps, st) = l(
    inputs.s,
    inputs.z,
    inputs.m,
    ps, st
)

function (l::AuxiliaryHeads)(s, z, m, ps, st)
    lddt_logits, st_plddt = l.plddt(s, ps.plddt, st.plddt)
    dist_logits, st_dist = l.distogram(z, ps.distogram, st.distogram)
    tm_logits, st_tm = l.tm(z, ps.tm, st.tm)
    msa_logits, st_msa = l.masked_msa(m, ps.masked_msa, st.masked_msa)
    exp_logits, st_exp = l.experimentally_resolved(s, ps.experimentally_resolved, st.experimentally_resolved)

    plddt_scores = compute_plddt(lddt_logits)

    output = (;
        lddt_logits, plddt=plddt_scores,
        distogram_logits=dist_logits,
        tm_logits=tm_logits,
        masked_msa_logits=msa_logits,
        experimentally_resolved_logits=exp_logits,
    )

    st_out = merge(st, (;
        plddt=st_plddt, distogram=st_dist,
        tm=st_tm,
        masked_msa=st_msa, experimentally_resolved=st_exp,
    ))

    return output, st_out
end

# ==============================================================================
# Canonical config factories
# ==============================================================================

AuxiliaryHeads(s::Symbol; kwargs...) = AuxiliaryHeads(static(s); kwargs...)
# Monomer: chn_out_msa=23 (default), multimer: chn_out_msa=22
AuxiliaryHeads(::StaticSymbol{:monomer}; kwargs...) = AuxiliaryHeads(384, 128, 256; kwargs...)
AuxiliaryHeads(::StaticSymbol{:multimer}; kwargs...) =
    AuxiliaryHeads(384, 128, 256; merge((chn_out_msa=22,), kw)...)
