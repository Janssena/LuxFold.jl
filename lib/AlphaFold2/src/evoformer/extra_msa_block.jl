"""
    ExtraMSABlock(chn_msa, chn_z; kwargs...)

A single iteration of the ExtraMSA stack — identical to `EvoformerBlock` except
`msa_att_col` is `MSAColumnGlobalAttention` (efficient large-MSA variant) instead
of `MSAColumnAttention`.

Default channels differ from `EvoformerBlock`: `chn_msa=64` (ExtraMSA channel `c_e`),
`chn_hidden_msa_att=8` (smaller per-head hidden for global attention).

See `EvoformerBlock` for full documentation of keyword arguments and dispatch.

# Arguments
- `chn_msa`: ExtraMSA channel dimension (default: 64)
- `chn_z`: Pair channel dimension (default: 128)
"""
struct ExtraMSABlock{OPM_FIRST, MAR, MAC, MT, OPM, PS} <: Lux.AbstractLuxContainerLayer{(:msa_att_row, :msa_att_col, :msa_transition, :outer_product_mean, :pair_stack)}
    opm_first::OPM_FIRST
    msa_att_row::MAR
    msa_att_col::MAC    # MSAColumnGlobalAttention (differs from EvoformerBlock)
    msa_transition::MT
    outer_product_mean::OPM
    pair_stack::PS
end

function ExtraMSABlock(
    chn_msa::Int=64, chn_z::Int=128;
    chn_hidden_msa_att::Int=8,       # smaller than Evoformer
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

    msa_att_col = MSAColumnGlobalAttention(chn_msa, chn_hidden_msa_att, no_heads_msa;
        use_bias=use_bias.msa_att_col, epsilon, eps=Float32(eps))

    msa_transition = MSATransition(chn_msa; n=transition_n)

    # Python OuterProductMean always uses its default eps=1e-3 (not the block-level eps=1e-8).
    outer_product_mean = OuterProductMean(chn_msa, chn_z, chn_hidden_opm; project_first=true)

    pair_stack = PairStackBlock(chn_z, chn_hidden_mul, chn_hidden_pair_att, no_heads_pair, 2;
        tri_mul_first, use_bias=use_bias.pair_stack, epsilon)

    return ExtraMSABlock(
        static(opm_first),
        msa_att_row, msa_att_col, msa_transition, outer_product_mean, pair_stack
    )
end

# NamedTuple dispatch
function (l::ExtraMSABlock)(inputs::NamedTuple, ps, st)
    out, st_new = l(
        inputs.m, 
        inputs.z, 
        get(inputs, :msa_mask, nothing), 
        get(inputs, :pair_mask, nothing), 
        ps, st
    )
    return merge(inputs, out), st_new
end

# opm_first = False (default)
function (l::ExtraMSABlock{False})(m, z, msa_mask, pair_mask, ps, st)
    m_row, st_mar = l.msa_att_row((; x=m, z, mask=msa_mask), ps.msa_att_row, st.msa_att_row)
    m = m .+ m_row

    m_col, st_mac = l.msa_att_col(m, msa_mask, ps.msa_att_col, st.msa_att_col)
    m = m .+ m_col

    m_tr, st_mt = l.msa_transition(m, msa_mask, ps.msa_transition, st.msa_transition)
    m = m .+ m_tr

    z_opm, st_opm = l.outer_product_mean((; m, mask=msa_mask), ps.outer_product_mean, st.outer_product_mean)
    z = z .+ z_opm

    z, st_ps = l.pair_stack(z, pair_mask, ps.pair_stack, st.pair_stack)

    st_new = merge(st, (;
        msa_att_row=st_mar, msa_att_col=st_mac, msa_transition=st_mt,
        outer_product_mean=st_opm, pair_stack=st_ps
    ))
    return (; m, z), st_new
end

# opm_first = True
function (l::ExtraMSABlock{True})(m, z, msa_mask, pair_mask, ps, st)
    z_opm, st_opm = l.outer_product_mean((; m, mask=msa_mask), ps.outer_product_mean, st.outer_product_mean)
    z = z .+ z_opm

    m_row, st_mar = l.msa_att_row((; x=m, z, mask=msa_mask), ps.msa_att_row, st.msa_att_row)
    m = m .+ m_row

    m_col, st_mac = l.msa_att_col(m, msa_mask, ps.msa_att_col, st.msa_att_col)
    m = m .+ m_col

    m_tr, st_mt = l.msa_transition(m, msa_mask, ps.msa_transition, st.msa_transition)
    m = m .+ m_tr

    z, st_ps = l.pair_stack(z, pair_mask, ps.pair_stack, st.pair_stack)

    st_new = merge(st, (;
        msa_att_row=st_mar, msa_att_col=st_mac, msa_transition=st_mt,
        outer_product_mean=st_opm, pair_stack=st_ps
    ))
    return (; m, z), st_new
end
