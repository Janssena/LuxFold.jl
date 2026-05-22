struct ExtraMSAEmbedder{L} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
end

ExtraMSAEmbedder(inout::Pair; kwargs...) = 
    ExtraMSAEmbedder(inout.first, inout.second; kwargs...)

function ExtraMSAEmbedder(c_in::Int, c_out::Int; use_bias=true)
    return ExtraMSAEmbedder(Lux.Dense(c_in => c_out; use_bias))
end

function (l::ExtraMSAEmbedder)(x, ps, st)
    return l.linear(x, ps.linear, st.linear)
end
