
"""
    DistogramHead(chn_z; no_bins=64)

Distogram head (AF3 §4.4) — 1:1 port of
`openfold3.core.model.heads.prediction_heads.DistogramHead`.
Projects the **trunk** pair rep `z_trunk` to `no_bins` symmetrised distance bins:

    logits = linear(z) + linear(permutedims(z, (1,3,2,4)))

**Important**: always called on `z_trunk` (before the confidence pairformer), not `z_conf`.

# Arguments
- `chn_z`: Pair embedding channel dimension
- `no_bins`: Number of distance bins (default 64)

# Inputs
- `z [chn_z, N, N, B]`

# Returns
- `logits [no_bins, N, N, B]`, `st`
"""
struct DistogramHead{L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
end

DistogramHead(chn_z::Int; no_bins::Int=64) =
    DistogramHead(Lux.Dense(chn_z => no_bins; use_bias=false))

function (l::DistogramHead)(z, ps, st)
    logits, st_linear = l.linear(z, ps.linear, st.linear)
    # Symmetrize (matches Python: logits + logits.transpose(-2, -3))
    return logits .+ permutedims(logits, (1, 3, 2, 4)), merge(st, (; linear=st_linear))
end

"""
    PredictedAlignedErrorHead(chn_z; no_bins=64)

PAE head (AF3 Alg 31 line 5) — asymmetric projection of `z_conf` →
`[no_bins, N, N, B]`. A 1:1 port of
`openfold3.core.model.heads.prediction_heads.PredictedAlignedErrorHead`.

# Inputs
- `z [chn_z, N, N, B]` — post-confidence-pairformer pair rep

# Returns
- `logits [no_bins, N, N, B]`, `st`
"""
struct PredictedAlignedErrorHead{LN,L} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :linear)}
    layer_norm::LN
    linear::L
end

PredictedAlignedErrorHead(chn_z::Int; no_bins::Int=64) =
    PredictedAlignedErrorHead(
        Lux.LayerNorm((chn_z, 1, 1); dims=1),
        Lux.Dense(chn_z => no_bins; use_bias=false),
    )

function (l::PredictedAlignedErrorHead)(z, ps, st)
    z_norm, st_ln = l.layer_norm(z, ps.layer_norm, st.layer_norm)
    logits, st_linear = l.linear(z_norm, ps.linear, st.linear)
    return logits, merge(st, (; layer_norm=st_ln, linear=st_linear))
end

"""
    PredictedDistanceErrorHead(chn_z; no_bins=64)

PDE head (AF3 Alg 31 line 6) — symmetrised projection of `z_conf` →
`[no_bins, N, N, B]`. A 1:1 port of
`openfold3.core.model.heads.prediction_heads.PredictedDistanceErrorHead`.

# Inputs
- `z [chn_z, N, N, B]` — post-confidence-pairformer pair rep

# Returns
- `logits [no_bins, N, N, B]`, `st`
"""
struct PredictedDistanceErrorHead{LN,L} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :linear)}
    layer_norm::LN
    linear::L
end

PredictedDistanceErrorHead(chn_z::Int; no_bins::Int=64) =
    PredictedDistanceErrorHead(
        Lux.LayerNorm((chn_z, 1, 1); dims=1),
        Lux.Dense(chn_z => no_bins; use_bias=false),
    )

function (l::PredictedDistanceErrorHead)(z, ps, st)
    z_norm, st_ln = l.layer_norm(z, ps.layer_norm, st.layer_norm)
    logits, st_linear = l.linear(z_norm, ps.linear, st.linear)
    # Symmetrize (matches Python: logits + logits.transpose(-2, -3))
    return logits .+ permutedims(logits, (1, 3, 2, 4)), merge(st, (; layer_norm=st_ln, linear=st_linear))
end

"""
    PerResidueLDDTAllAtom(chn_s, max_atoms_per_token; no_bins=50)

pLDDT head (AF3 Alg 31 line 7) — per-atom confidence from the single rep. A 1:1
port of `openfold3.core.model.heads.prediction_heads.PerResidueLDDTAllAtom`.

Architecture:
1. `layer_norm(s_conf)` — normalise `[chn_s, N_token, B]`
2. `linear(...)` → `[max_atoms * no_bins, N_token, B]`
3. Reshape → `[no_bins, max_atoms * N_token, B]`
4. `max_atom_per_token_masked_select` → `[no_bins, N_atom, B]`

# Arguments
- `chn_s`: Single embedding channel dimension
- `max_atoms_per_token`: Maximum atoms per token (padding dim)
- `no_bins`: Number of pLDDT bins (default 50)

# Inputs
- `s [chn_s, N_token, B]`
- `max_atom_per_token_mask [max_atoms * N_token, B]` — Bool mask of valid atom slots

# Returns
- `logits [no_bins, N_atom, B]`, `st`
"""
struct PerResidueLDDTAllAtom{LN,L} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :linear)}
    max_atoms_per_token::Int
    no_bins::Int
    layer_norm::LN
    linear::L
end

function PerResidueLDDTAllAtom(chn_s::Int, max_atoms_per_token::Int; no_bins::Int=50)
    PerResidueLDDTAllAtom(
        max_atoms_per_token, no_bins,
        Lux.LayerNorm((chn_s, 1); dims=1),
        Lux.Dense(chn_s => max_atoms_per_token * no_bins; use_bias=false),
    )
end

(l::PerResidueLDDTAllAtom)(inputs::NamedTuple, ps, st) =
    l(inputs.s, inputs.max_atom_per_token_mask, ps, st)

function (l::PerResidueLDDTAllAtom)(s, max_atom_per_token_mask, ps, st)
    _, N_token, B = size(s)
    s_norm, st_ln = l.layer_norm(s, ps.layer_norm, st.layer_norm)
    logits, st_linear = l.linear(s_norm, ps.linear, st.linear)       # [max_atoms*no_bins, N_token, B]
    # Reshape to [no_bins, max_atoms*N_token, B] — see CONTEXT.md §token–atom layout note.
    # In column-major: [max_atoms*no_bins, N_token, B] → [no_bins, max_atoms*N_token, B]
    # matches Python's row-major [B, N_token, max_atoms*c_out] → [B, N_token*max_atoms, c_out].
    logits = reshape(logits, l.no_bins, l.max_atoms_per_token * N_token, B)
    logits = max_atom_per_token_masked_select(logits, max_atom_per_token_mask)
    return logits, merge(st, (; layer_norm=st_ln, linear=st_linear))
end

"""
    ExperimentallyResolvedHeadAllAtom(chn_s, max_atoms_per_token; no_bins=2)

Experimentally-resolved head (AF3 §4.3.3) — same architecture as
`PerResidueLDDTAllAtom` but with `no_bins=2` (resolved / not resolved). A 1:1
port of `openfold3.core.model.heads.prediction_heads.ExperimentallyResolvedHeadAllAtom`.

# Inputs
- `s [chn_s, N_token, B]`
- `max_atom_per_token_mask [max_atoms * N_token, B]`

# Returns
- `logits [2, N_atom, B]`, `st`
"""
struct ExperimentallyResolvedHeadAllAtom{LN,L} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :linear)}
    max_atoms_per_token::Int
    no_bins::Int
    layer_norm::LN
    linear::L
end

function ExperimentallyResolvedHeadAllAtom(chn_s::Int, max_atoms_per_token::Int; no_bins::Int=2)
    ExperimentallyResolvedHeadAllAtom(
        max_atoms_per_token, no_bins,
        Lux.LayerNorm((chn_s, 1); dims=1),
        Lux.Dense(chn_s => max_atoms_per_token * no_bins; use_bias=false),
    )
end

# NamedTuple dispatch → positional forward (mask is required, not optional)
(l::ExperimentallyResolvedHeadAllAtom)(inputs::NamedTuple, ps, st) =
    l(inputs.s, inputs.max_atom_per_token_mask, ps, st)

function (l::ExperimentallyResolvedHeadAllAtom)(s, max_atom_per_token_mask, ps, st)
    _, N_token, B = size(s)
    s_norm, st_ln = l.layer_norm(s, ps.layer_norm, st.layer_norm)
    logits, st_linear = l.linear(s_norm, ps.linear, st.linear)       # [max_atoms*no_bins, N_token, B]
    logits = reshape(logits, l.no_bins, l.max_atoms_per_token * N_token, B)
    logits = max_atom_per_token_masked_select(logits, max_atom_per_token_mask)
    return logits, merge(st, (; layer_norm=st_ln, linear=st_linear))
end

# ─── PairformerEmbedding ──────────────────────────────────────────────────────
"""
    PairformerEmbedding(; chn_s_input, chn_z, pairformer_stack,
                        min_bin=3.25, max_bin=20.75, no_bin=15, inf=1f8)

Confidence Pairformer embedding (AF3 Alg 31 lines 1–6) — 1:1 port of
`openfold3.core.model.heads.prediction_heads.PairformerEmbedding`. Adds single-rep
outer-product projections and distance-binned representative-atom pair features to `z`,
then runs a `PairFormerStack` to produce `(s_conf, z_conf)`.

**Sub-layers**:
- `linear_i`, `linear_j` (`Dense(chn_s_input => chn_z; use_bias=false)`) — outer-product projections
- `linear_distance` (`Dense(no_bin => chn_z; use_bias=false)`) — distance one-hot projection
- `pairformer_stack` (`PairFormerStack`)

**Distance bins**: `linspace(min_bin, max_bin, no_bin)` Å; element k is active when
`sq_dist > bins[k]² AND sq_dist < bins[k+1]²` (strict bounds; last bin: `bins[k]² < ∞`).

# Inputs
- `s_input [chn_s_input, N, B]` — InputFeatureEmbedder output (used for z projection)
- `s [chn_s, N, B]` — trunk single rep (initial single state for pairformer stack)
- `z [chn_z, N, N, B]` — trunk pair rep
- `x_pred [3, N, B]` — representative atom coords per token
- `single_mask [N, B] Bool` or `nothing`
- `pair_mask [N, N, B] Bool` or `nothing`

# Returns
- `(; s [chn_s, N, B], z [chn_z, N, N, B])`, `st`
"""
struct PairformerEmbedding{LI,LJ,LD,PF} <:
       Lux.AbstractLuxContainerLayer{(:linear_i, :linear_j, :linear_distance, :pairformer_stack)}
    min_bin::Float64
    max_bin::Float64
    no_bin::Int
    inf::Float64
    linear_i::LI
    linear_j::LJ
    linear_distance::LD
    pairformer_stack::PF
end

function PairformerEmbedding(;
    chn_s_input::Int, chn_z::Int,
    pairformer_stack::PairFormerStack,
    min_bin::Real=3.25f0, max_bin::Real=20.75f0, no_bin::Int=15,
    inf::Real=1f8,
)
    PairformerEmbedding(
        Float64(min_bin), Float64(max_bin), no_bin, Float64(inf),
        Lux.Dense(chn_s_input => chn_z; use_bias=false),
        Lux.Dense(chn_s_input => chn_z; use_bias=false),
        Lux.Dense(no_bin => chn_z; use_bias=false),
        pairformer_stack,
    )
end

function _pairformer_emb_embed_z(l::PairformerEmbedding, s_input, z, x_pred, ps, st)
    T = eltype(z)
    C_z, N, _, B = size(z)

    s_input_i, st_li = l.linear_i(s_input, ps.linear_i, st.linear_i)    # [C_z, N, B]
    s_input_j, st_lj = l.linear_j(s_input, ps.linear_j, st.linear_j)    # [C_z, N, B]
    z = z .+ reshape(s_input_i, C_z, N, 1, B) .+ reshape(s_input_j, C_z, 1, N, B)

    sq_bins = abs2.(range(T(l.min_bin), T(l.max_bin), l.no_bin))
    upper = T[sq_bins[2:end]; T(l.inf)]
    sq_bins_r = reshape(sq_bins, l.no_bin, 1, 1, 1)
    upper_r = reshape(upper, l.no_bin, 1, 1, 1)

    dij_sq = reshape(
        dropdims(sum(abs2, reshape(x_pred, 3, N, 1, B) .- reshape(x_pred, 3, 1, N, B); dims=1); dims=1),
        1, N, N, B,
    )   # [1, N, N, B]
    # one_hot[k, i, j, b] = 1 iff dij_sq(i,j,b) ∈ (sq_bins[k], upper[k])
    one_hot = @. ifelse((dij_sq > sq_bins_r) & (dij_sq < upper_r), one(T), zero(T))

    dist_emb, st_ld = l.linear_distance(one_hot, ps.linear_distance, st.linear_distance)
    z = z .+ dist_emb

    return z, (linear_i=st_li, linear_j=st_lj, linear_distance=st_ld)
end

(l::PairformerEmbedding)(inputs::NamedTuple, ps, st) = l(
    inputs.s_input, inputs.s, inputs.z, inputs.x_pred,
    get(inputs, :single_mask, nothing), get(inputs, :pair_mask, nothing),
    ps, st,
)

function (l::PairformerEmbedding)(s_input, s, z, x_pred, single_mask, pair_mask, ps, st)
    z_new, st_embed = _pairformer_emb_embed_z(l, s_input, z, x_pred, ps, st)

    pf, st_pf = l.pairformer_stack(s, z_new, single_mask, pair_mask, ps.pairformer_stack, st.pairformer_stack)

    st_new = merge(merge(st, st_embed), (pairformer_stack=st_pf,))
    return (s = pf.s, z = pf.z, ), st_new
end
