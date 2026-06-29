"""
    MSAModuleBlock(; chn_m, chn_z, chn_hidden_msa_att, chn_hidden_opm, chn_hidden_mul, chn_hidden_pair_att,
                   no_heads_msa, no_heads_pair, transition_n, opm_first=false)

One block of AF3 Algorithm 8 — 1:1 port of
`openfold3.core.model.latent.msa_module.MSAModuleBlock`. Differs from AF2's `EvoformerBlock`:
the MSA attention is `PairWeightedAveraging` (Alg 10, gated pair-weighted averaging — NOT
key/query attention), there is **no column attention**, the transitions are `SwiGLUTransition`,
and `opm_first=false` (OuterProductMean runs after the MSA transition).

Sub-layers: `msa_att_row::PairWeightedAveraging`, `msa_transition::SwiGLUTransition` (4D),
`outer_product_mean::OuterProductMean` (`project_first=true`), `pair_block::PairBlock`.

# Forward (opm_first=false)
1. `m += msa_att_row(m, z, mask=pair_mask)`
2. `m += msa_transition(m, mask=msa_mask)`
3. `z += outer_product_mean(m, mask=msa_mask)`
4. `z = pair_block(z, pair_mask)`

# Inputs
- `m`: MSA representation `[chn_m, N_token, N_seq, B]`
- `z`: pair representation `[chn_z, N_token, N_token, B]`
- `msa_mask`: `AbstractArray{Bool}` `[N_token, N_seq, B]` or `nothing`
- `pair_mask`: `AbstractArray{Bool}` `[N_token, N_token, B]` or `nothing`

# Returns
- `(; m, z)`: updated representations (block-in-stack convention; `Lux.Chain`-threaded)
- `st`: updated state
"""
struct MSAModuleBlock{OPM_FIRST<:StaticBool,MAR,MT,OPM,PB} <:
       Lux.AbstractLuxContainerLayer{(:msa_att_row, :msa_transition, :outer_product_mean, :pair_block)}
    opm_first::OPM_FIRST
    msa_att_row::MAR
    msa_transition::MT
    outer_product_mean::OPM
    pair_block::PB
end

function MSAModuleBlock(; chn_m::Int, chn_z::Int, chn_hidden_msa_att::Int, chn_hidden_opm::Int,
                        chn_hidden_mul::Int, chn_hidden_pair_att::Int, no_heads_msa::Int,
                        no_heads_pair::Int, transition_n::Int, opm_first::Bool=false)
    return MSAModuleBlock(
        static(opm_first),
        # openfold-3 msa_pair_avg_init: all linears bias-free; LayerNorms stay affine.
        PairWeightedAveraging(chn_m, chn_z, chn_hidden_msa_att, no_heads_msa;
            use_bias=(layer_norm_m=true, layer_norm_z=true, linear_v=false,
                      linear_z=false, linear_g=false, linear_out=false)),
        SwiGLUTransition(chn_m, transition_n; is_4d=true),     # MSA `m` is 4D [chn_m, N, S, B]
        # openfold-3 opm_init: linear_1/2 bias-free, linear_out biased (LuxFoldCore `use_bias`
        # controls only linear_1/2; linear_out is always biased) — matches exactly.
        OuterProductMean(chn_m, chn_z, chn_hidden_opm; project_first=true, use_bias=false),
        PairBlock(; chn_z, chn_hidden_mul, chn_hidden_pair_att, no_heads_pair, transition_n),
    )
end

# config-NamedTuple compatibility shim
MSAModuleBlock(config::NamedTuple; kwargs...) = MSAModuleBlock(;
    chn_m=config.c_m, chn_z=config.c_z, chn_hidden_msa_att=config.c_hidden_msa_att,
    chn_hidden_opm=config.c_hidden_opm, chn_hidden_mul=config.c_hidden_mul,
    chn_hidden_pair_att=config.c_hidden_pair_att, no_heads_msa=config.no_heads_msa,
    no_heads_pair=config.no_heads_pair, transition_n=config.transition_n,
    opm_first=get(config, :opm_first, false),
)

# NamedTuple dispatch (Chain threading): updates `m` and `z`; masks pass through.
function (l::MSAModuleBlock)(inputs::NamedTuple, ps, st)
    out, st_new = l(
        inputs.m, inputs.z,
        get(inputs, :msa_mask, nothing), get(inputs, :pair_mask, nothing),
        ps, st,
    )
    return merge(inputs, out), st_new
end

function _msa_pair_update(l::MSAModuleBlock, m, z, msa_mask, pair_mask, ps, st)
    m_att_row, st_row = l.msa_att_row((; m, z, mask=pair_mask), ps.msa_att_row, st.msa_att_row)
    m = m .+ m_att_row
    m_trans, st_tr = l.msa_transition(m, msa_mask, ps.msa_transition, st.msa_transition)
    m = m .+ m_trans
    return m, (msa_att_row=st_row, msa_transition=st_tr)
end

# opm_first == false (AF3 default): MSA attention → transition → OPM → pair block
function (l::MSAModuleBlock{False})(m, z, msa_mask, pair_mask, ps, st)
    m, st_msa = _msa_pair_update(l, m, z, msa_mask, pair_mask, ps, st)

    z_opm, st_opm = l.outer_product_mean(m, msa_mask, ps.outer_product_mean, st.outer_product_mean)
    z = z .+ z_opm

    z, st_pb = l.pair_block(z, pair_mask, ps.pair_block, st.pair_block)

    return (; m, z), merge(st, (; st_msa..., outer_product_mean=st_opm, pair_block=st_pb))
end

# opm_first == true: OPM → MSA attention → transition → pair block
function (l::MSAModuleBlock{True})(m, z, msa_mask, pair_mask, ps, st)
    z_opm, st_opm = l.outer_product_mean(m, msa_mask, ps.outer_product_mean, st.outer_product_mean)
    z = z .+ z_opm

    m, st_msa = _msa_pair_update(l, m, z, msa_mask, pair_mask, ps, st)

    z, st_pb = l.pair_block(z, pair_mask, ps.pair_block, st.pair_block)

    return (; m, z), merge(st, (; st_msa..., outer_product_mean=st_opm, pair_block=st_pb))
end
