"""
    ExtraMSAEmbedder(c_in, c_out; use_bias=true)
    ExtraMSAEmbedder(c_in => c_out; use_bias=true)

Projects extra MSA features to the ExtraMSA channel dimension via a single linear layer.

# Arguments
- `c_in`: Input channel dimension (number of extra MSA features, e.g. 25 for monomer)
- `c_out`: Output channel dimension (ExtraMSA embedding dimension, e.g. `c_e` = 64)

# Keyword Arguments
- `use_bias`: Whether to include bias in the linear projection (default: `true`)

# Inputs
- `x`: Extra MSA feature tensor of shape `[c_in, N_res, N_seq, B]`

# Returns
- `y`: Extra MSA embedding of shape `[c_out, N_res, N_seq, B]`
- `st`: Updated state
"""
struct ExtraMSAEmbedder{L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
end

ExtraMSAEmbedder(inout::Pair; kwargs...) = 
    ExtraMSAEmbedder(inout.first, inout.second; kwargs...)

ExtraMSAEmbedder(c_in::Int, c_out::Int; use_bias=true) = 
    ExtraMSAEmbedder(Lux.Dense(c_in => c_out; use_bias))

(l::ExtraMSAEmbedder)(x, ps, st) = l.linear(x, ps.linear, st.linear)
