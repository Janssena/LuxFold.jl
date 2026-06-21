"""
    RecyclingEmbedder(chn_msa, chn_z; min_bin=3.25, max_bin=20.75, no_bins=15, use_bias=true, affine=(layer_norm_m=true, layer_norm_z=true))

Embeds the output of a previous model iteration for recycling (Algorithm 32).
Layer-norms the MSA and pair embeddings, then adds a linear projection of
squared-distance binned Cβ coordinates to the pair update.

# Arguments
- `chn_msa`: Channel dimension of the MSA embedding
- `chn_z`: Channel dimension of the pair embedding

# Keyword Arguments
- `min_bin`: Smallest distogram bin in Å (default: 3.25)
- `max_bin`: Largest distogram bin in Å (default: 20.75)
- `no_bins`: Number of distogram bins (default: 15)
- `use_bias`: Bool or NamedTuple for bias resolution, passed to `resolve_defaults`
- `affine`: NamedTuple or Bool controlling LayerNorm affine parameters

# Inputs
- `m`: MSA embedding tensor of shape `[chn_msa, N, B]`
- `z`: Pair embedding tensor of shape `[chn_z, N, N, B]`
- `x`: Cβ (pseudo-beta) position tensor of shape `[3, N, B]`

# Returns
- `m_update`: MSA update tensor of shape `[chn_msa, N, B]` (pure normalization)
- `z_update`: Pair update tensor of shape `[chn_z, N, N, B]`
- `st`: Updated state
"""
struct RecyclingEmbedder{LN1,LN2,L} <: Lux.AbstractLuxContainerLayer{(:layer_norm_m, :layer_norm_z, :linear)}
    layer_norm_m::LN1
    layer_norm_z::LN2
    linear::L
    min_bin::Float32
    max_bin::Float32
    no_bins::Int
end

function RecyclingEmbedder(chn_msa::Int, chn_z::Int; min_bin=3.25f0, max_bin=20.75f0, no_bins=15, use_bias=true, affine=true)
    use_bias = resolve_defaults(use_bias, (:layer_norm_m, :layer_norm_z, :linear))
    affine = resolve_defaults(affine, (:layer_norm_m, :layer_norm_z))

    layer_norm_m = if affine.layer_norm_m && !use_bias.layer_norm_m
        LayerNormNoBias((chn_msa, 1); dims=1)
    else
        Lux.LayerNorm((chn_msa, 1); dims=1, affine=affine.layer_norm_m)
    end

    layer_norm_z = if affine.layer_norm_z && !use_bias.layer_norm_z
        LayerNormNoBias((chn_z, 1, 1); dims=1)
    else
        Lux.LayerNorm((chn_z, 1, 1); dims=1, affine=affine.layer_norm_z)
    end

    return RecyclingEmbedder(
        layer_norm_m, layer_norm_z,
        Lux.Dense(no_bins => chn_z; use_bias=use_bias.linear),
        Float32(min_bin), Float32(max_bin), no_bins,
    )
end

Lux.initialstates(rng::Random.AbstractRNG, l::RecyclingEmbedder) = (
    layer_norm_m = Lux.initialstates(rng, l.layer_norm_m),
    layer_norm_z = Lux.initialstates(rng, l.layer_norm_z),
    linear = Lux.initialstates(rng, l.linear),
    squared_bins = [abs2.(range(l.min_bin, l.max_bin; length=l.no_bins)); Inf32],
)

(l::RecyclingEmbedder)(inputs::NamedTuple, ps, st) = l(
    inputs.x,
    inputs.m,
    inputs.z,
    ps, st
)

function (l::RecyclingEmbedder)(x::AbstractArray{T}, m, z, ps, st) where T
    num_bins = length(st.squared_bins)-1
    _, N, B = size(x)

    m_update, st_ln_m = l.layer_norm_m(m, ps.layer_norm_m, st.layer_norm_m)
    z_ln, st_ln_z = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)

    sq_norm = sum(abs2, x; dims=1)
    gram = Lux.batched_matmul(permutedims(x, (2, 1, 3)), x)
    sq_norm_perm = permutedims(sq_norm, (2, 1, 3))
    d_sq = @. sq_norm + sq_norm_perm - 2 * gram

    lower = view(st.squared_bins, 1:num_bins)
    upper = view(st.squared_bins, 2:num_bins+1)

    d_sq_r = reshape(d_sq, 1, N, N, B)
    bins = @. T((d_sq_r > lower) & (d_sq_r < upper))

    z_bin, st_linear = l.linear(bins, ps.linear, st.linear)
    z_update = z_ln .+ z_bin

    st_out = merge(st, (; layer_norm_m=st_ln_m, layer_norm_z=st_ln_z, linear=st_linear))
    return (m = m_update, z = z_update), st_out
end

# ==============================================================================
# Canonical config factories
# ==============================================================================

RecyclingEmbedder(s::Symbol; kwargs...) = 
    RecyclingEmbedder(static(s); kwargs...)

RecyclingEmbedder(::StaticSymbol; kwargs...) = 
    RecyclingEmbedder(256, 128; kwargs...)
