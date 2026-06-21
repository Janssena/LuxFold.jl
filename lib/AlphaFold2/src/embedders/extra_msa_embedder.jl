"""
    ExtraMSAEmbedder(chn_in, chn_out; use_bias=true)
    ExtraMSAEmbedder(chn_in => chn_out; use_bias=true)

Projects extra MSA features to the ExtraMSA channel dimension via a single linear layer.

# Arguments
- `chn_in`: Input channel dimension (number of extra MSA features, e.g. 25 for monomer)
- `chn_out`: Output channel dimension (ExtraMSA embedding dimension, e.g. `c_e` = 64)

# Keyword Arguments
- `use_bias`: Whether to include bias in the linear projection (default: `true`)

# Inputs
- `x`: Extra MSA feature tensor of shape `[chn_in, N_res, N_seq, B]`

# Returns
- `y`: Extra MSA embedding of shape `[chn_out, N_res, N_seq, B]`
- `st`: Updated state
"""
struct ExtraMSAEmbedder{L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
end

ExtraMSAEmbedder(inout::Pair; kwargs...) = 
    ExtraMSAEmbedder(inout.first, inout.second; kwargs...)

ExtraMSAEmbedder(chn_in::Int, chn_out::Int; use_bias=true) = 
    ExtraMSAEmbedder(Lux.Dense(chn_in => chn_out; use_bias))

function (l::ExtraMSAEmbedder)(x, ps, st)
    y, st_linear = l.linear(x, ps.linear, st.linear)
    return y, (; linear = st_linear)
end

# ==============================================================================
# Canonical config factories
# ==============================================================================

ExtraMSAEmbedder(s::Symbol; kwargs...) = ExtraMSAEmbedder(static(s); kwargs...)
ExtraMSAEmbedder(::StaticSymbol; kwargs...) = ExtraMSAEmbedder(25, 64; kwargs...)
