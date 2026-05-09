"""
    AdaLN(chn_a, chn_s; rank=3, epsilon=1f-5, affine=(layer_norm_a=false, layer_norm_s=true), use_bias=false)
    AdaLN(chn_a => chn_s; kwargs...)

Adaptive Layer Normalization (AdaLN) layer. Normalizes the input `a` and then applies a 
scale and shift derived from the conditioning input `s`.

# Arguments
- `chn_a`: Number of channels in the input `a`.
- `chn_s`: Number of channels in the conditioning signal `s`.

# Keyword Arguments
- `rank`: The rank of the input tensors. Typically 3 for [C, N, B] or 4 for [C, N, S, B].
- `epsilon`: A small constant for numerical stability in LayerNorm.
- `affine`: A `NamedTuple` or `Bool` specifying which internal LayerNorms should have 
  affine transformations. Defaults to `(layer_norm_a=false, layer_norm_s=true)`.
- `use_bias`: A `NamedTuple` or `Bool` specifying which internal layers should use bias. 
  Defaults to `false`.

# Inputs
- `a`: The input array to be normalized. Expected shape: `[chn_a, N, (S, ) B]`.
- `s`: The conditioning signal. Expected shape: `[chn_s, N, (S, ) B]`.

# Returns
- `y`: The modulated output. Shape matches `a`.
- `st`: Updated state.
"""
struct AdaLN{LNA,LNS,S,G} <: Lux.AbstractLuxContainerLayer{(:layer_norm_a,:layer_norm_s,:shift,:gate)}
    layer_norm_a::LNA
    layer_norm_s::LNS
    shift::S
    gate::G
end

AdaLN(a_s::Pair; kwargs...) = AdaLN(a_s.first, a_s.second; kwargs...)

function AdaLN(chn_a::Int, chn_s::Int; rank::Int=3, epsilon=1f-5, affine=(layer_norm_a = false, layer_norm_s = true), use_bias=false)
    @assert rank > 1 "rank should be greater than 1."
    use_bias = resolve_defaults(use_bias, (:layer_norm_a,:layer_norm_s,:gate,:shift))
    affine = resolve_defaults(affine, (:layer_norm_a,:layer_norm_s,))
    
    shape_a = (chn_a, ntuple(one, rank-2)...)
    layer_norm_a = if affine.layer_norm_a && !use_bias.layer_norm_a
        LayerNormNoBias(shape_a; dims=1, epsilon)
    else
        Lux.LayerNorm(shape_a; dims=1, affine=affine.layer_norm_a, epsilon)
    end
    shape_s = (chn_s, ntuple(one, rank-2)...)
    layer_norm_s = if affine.layer_norm_s && !use_bias.layer_norm_s
        LayerNormNoBias(shape_s; dims=1, epsilon)
    else
        Lux.LayerNorm(shape_s; dims=1, affine=affine.layer_norm_s, epsilon)
    end
    
    return AdaLN(
        layer_norm_a, 
        layer_norm_s, 
        Lux.Dense(chn_s => chn_a; use_bias=use_bias.shift), # shift
        Lux.Dense(chn_s => chn_a, Lux.sigmoid; use_bias=use_bias.gate), # gate
    )
end

(l::AdaLN)(inputs::NamedTuple, ps, st) = l(
    inputs.a,
    inputs.s,
    ps, st
)

function (l::AdaLN)(a::AbstractArray, s::AbstractArray, ps, st)
    a, layer_norm_a = l.layer_norm_a(a, ps.layer_norm_a, st.layer_norm_a)
    s, layer_norm_s = l.layer_norm_s(s, ps.layer_norm_s, st.layer_norm_s)
    g, gate = l.gate(s, ps.gate, st.gate)
    sh, shift = l.shift(s, ps.shift, st.shift)
    @. a = g * a + sh
    return a, (; layer_norm_a, layer_norm_s, gate, shift) 
end