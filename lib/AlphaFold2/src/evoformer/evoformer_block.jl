"""
    EvoformerBlock(chn_msa, chn_z; kwargs...)

A single iteration of the Evoformer (Algorithm 6) processing both the MSA representation
`m` and the pair representation `z`. Applies MSA row attention, column attention,
MSA transition, outer product mean, and a pair stack.

Accepts and returns `NamedTuple (; m, z, msa_mask, pair_mask)` for type-stable chaining
via `Lux.Chain` in `EvoformerStack`.

# Arguments
- `chn_msa`: MSA channel dimension (default: 256)
- `chn_z`: Pair channel dimension (default: 128)

# Keyword Arguments
- `chn_hidden_msa_att`: Per-head hidden for MSA attention (default: 32)
- `chn_hidden_opm`: Hidden dimension for OuterProductMean (default: 32)
- `no_heads_msa`: Number of MSA attention heads (default: 8)
- `no_heads_pair`: Number of pair attention heads (default: 4)
- `transition_n`: MSA transition expansion factor (default: 4)
- `chn_hidden_mul`: Hidden dimension for triangle multiplication (default: 128)
- `chn_hidden_pair_att`: Per-head hidden for pair attention (default: 32)
- `eps`: Small constant for OuterProductMean stability (default: `1f-8`) [!NOT USED]
- `opm_first`: If `static(true)`, OuterProductMean runs before MSA attention (default: `static(false)`)
- `tri_mul_first`: Operation order in `PairStack` — if `true`, multiplication before attention (default: `true`)
- `use_bias`: `Bool` or `NamedTuple` for per-sublayer bias control (default: `true`)
- `epsilon`: LayerNorm epsilon (default: `1f-5`)

# Inputs (NamedTuple form)
- `m`: MSA tensor `[chn_msa, N_res, N_seq, B]`
- `z`: Pair tensor `[chn_z, N_res, N_res, B]`
- `msa_mask`: Bool mask `[N_res, N_seq, B]` or `nothing`
- `pair_mask`: Bool mask `[N_res, N_res, B]` or `nothing`

# Returns (positional)
- `((m_out, z_out), st_new)` — tuple of updated representations and state

# Returns (NamedTuple)
- `((; m, z, msa_mask, pair_mask), st_new)` — masks pass through unchanged
"""
struct EvoformerBlock{OPM_FIRST, MAR, MAC, MT, OPM, PS} <: Lux.AbstractLuxContainerLayer{(:msa_att_row, :msa_att_col, :msa_transition, :outer_product_mean, :pair_stack)}
    opm_first::OPM_FIRST    # StaticBool: run OPM before MSA attention?
    msa_att_row::MAR
    msa_att_col::MAC
    msa_transition::MT
    outer_product_mean::OPM
    pair_stack::PS
end

function EvoformerBlock(
    chn_msa::Int=256, chn_z::Int=128;
    chn_hidden_msa_att::Int=32,
    chn_hidden_opm::Int=32,
    no_heads_msa::Int=8,
    no_heads_pair::Int=4,
    transition_n::Int=4,
    chn_hidden_mul::Int=128,
    chn_hidden_pair_att::Int=32,
    eps=1f-8, # TODO: This is a dead path for OPM
    opm_first=false,
    tri_mul_first=true,
    use_bias=true,
    epsilon::Real=1f-5
)
    use_bias = resolve_defaults(use_bias, (
        :msa_att_row, :msa_att_col, :msa_transition, :outer_product_mean, :pair_stack
        )
    )

    msa_att_row = MSARowAttentionPairBias(chn_msa, chn_z, chn_hidden_msa_att, no_heads_msa)

    msa_att_col = MSAColumnAttention(chn_msa, chn_hidden_msa_att, no_heads_msa;
        use_bias=use_bias.msa_att_col, epsilon)

    msa_transition = MSATransition(chn_msa; n=transition_n)

    # Python OuterProductMean always uses its default eps=1e-3 (not the block-level eps=1e-8).
    outer_product_mean = OuterProductMean(chn_msa, chn_z, chn_hidden_opm; project_first=true)

    pair_stack = PairStackBlock(chn_z, chn_hidden_mul, chn_hidden_pair_att, no_heads_pair, 2;
        tri_mul_first, use_bias=use_bias.pair_stack, epsilon)

    return EvoformerBlock(
        static(opm_first),
        msa_att_row, msa_att_col, msa_transition, outer_product_mean, pair_stack
    )
end

# NamedTuple dispatch: unpack → forward → repack (masks pass through unchanged)
function (l::EvoformerBlock)(inputs::NamedTuple, ps, st)
    out, st_new = l(
        inputs.m, 
        inputs.z, 
        get(inputs, :msa_mask, nothing), 
        get(inputs, :pair_mask, nothing), 
        ps, st
    )
    return merge(inputs, out), st_new
end

# opm_first = False (default): MSA attention → OPM → pair stack
function (l::EvoformerBlock{False})(m, z, msa_mask, pair_mask, ps, st)
    # MSA row attention (pair-biased)
    m_row, st_mar = l.msa_att_row((; x=m, z, mask=msa_mask), ps.msa_att_row, st.msa_att_row)
    m = m .+ m_row

    # MSA column attention
    m_col, st_mac = l.msa_att_col(m, msa_mask, ps.msa_att_col, st.msa_att_col)
    m = m .+ m_col

    # MSA transition
    m_tr, st_mt = l.msa_transition(m, msa_mask, ps.msa_transition, st.msa_transition)
    m = m .+ m_tr

    # Outer product mean (MSA → pair update)
    z_opm, st_opm = l.outer_product_mean((; m, mask=msa_mask), ps.outer_product_mean, st.outer_product_mean)
    z = z .+ z_opm

    # Pair stack (triangle ops + transition; returns accumulated z)
    z, st_ps = l.pair_stack(z, pair_mask, ps.pair_stack, st.pair_stack)

    st_new = merge(st, (;
        msa_att_row=st_mar, msa_att_col=st_mac, msa_transition=st_mt,
        outer_product_mean=st_opm, pair_stack=st_ps
    ))
    return (; m, z), st_new
end

# opm_first = True: OPM → MSA attention → pair stack
function (l::EvoformerBlock{True})(m, z, msa_mask, pair_mask, ps, st)
    # Outer product mean first
    z_opm, st_opm = l.outer_product_mean((; m, mask=msa_mask), ps.outer_product_mean, st.outer_product_mean)
    z = z .+ z_opm

    # MSA row attention
    m_row, st_mar = l.msa_att_row((; x=m, z, mask=msa_mask), ps.msa_att_row, st.msa_att_row)
    m = m .+ m_row

    # MSA column attention
    m_col, st_mac = l.msa_att_col(m, msa_mask, ps.msa_att_col, st.msa_att_col)
    m = m .+ m_col

    # MSA transition
    m_tr, st_mt = l.msa_transition(m, msa_mask, ps.msa_transition, st.msa_transition)
    m = m .+ m_tr

    # Pair stack
    z, st_ps = l.pair_stack(z, pair_mask, ps.pair_stack, st.pair_stack)

    st_new = merge(st, (;
        msa_att_row=st_mar, msa_att_col=st_mac, msa_transition=st_mt,
        outer_product_mean=st_opm, pair_stack=st_ps
    ))
    return (; m, z), st_new
end