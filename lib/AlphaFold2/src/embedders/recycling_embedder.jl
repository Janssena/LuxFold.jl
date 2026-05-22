"""
    RecyclingEmbedder(c_m, c_z; min_bin=3.25, max_bin=20.75, no_bins=15, use_bias=true, affine=(layer_norm_m=true, layer_norm_z=true))

Embeds the output of a previous model iteration for recycling (Algorithm 32).
Layer-norms the MSA and pair embeddings, then adds a linear projection of
squared-distance binned Cβ coordinates to the pair update.

# Arguments
- `c_m`: Channel dimension of the MSA embedding
- `c_z`: Channel dimension of the pair embedding

# Keyword Arguments
- `min_bin`: Smallest distogram bin in Å (default: 3.25)
- `max_bin`: Largest distogram bin in Å (default: 20.75)
- `no_bins`: Number of distogram bins (default: 15)
- `use_bias`: Bool or NamedTuple for bias resolution, passed to `resolve_defaults`
- `affine`: NamedTuple or Bool controlling LayerNorm affine parameters

# Inputs
- `m`: MSA embedding tensor of shape `[c_m, N, B]`
- `z`: Pair embedding tensor of shape `[c_z, N, N, B]`
- `x`: Cβ (pseudo-beta) position tensor of shape `[3, N, B]`

# Returns
- `m_update`: MSA update tensor of shape `[c_m, N, B]` (pure normalization)
- `z_update`: Pair update tensor of shape `[c_z, N, N, B]`
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

function RecyclingEmbedder(c_m::Int, c_z::Int; min_bin=3.25f0, max_bin=20.75f0, no_bins=15, use_bias=true, affine=(layer_norm_m=true, layer_norm_z=true))
    use_bias = resolve_defaults(use_bias, (:layer_norm_m, :layer_norm_z, :linear))
    affine = resolve_defaults(affine, (:layer_norm_m, :layer_norm_z))

    layer_norm_m = if affine.layer_norm_m && !use_bias.layer_norm_m
        LayerNormNoBias((c_m, 1); dims=1)
    else
        Lux.LayerNorm((c_m, 1); dims=1, affine=affine.layer_norm_m)
    end

    layer_norm_z = if affine.layer_norm_z && !use_bias.layer_norm_z
        LayerNormNoBias((c_z, 1, 1); dims=1)
    else
        Lux.LayerNorm((c_z, 1, 1); dims=1, affine=affine.layer_norm_z)
    end

    return RecyclingEmbedder(
        layer_norm_m, layer_norm_z,
        Lux.Dense(no_bins => c_z; use_bias=use_bias.linear),
        Float32(min_bin), Float32(max_bin), no_bins,
    )
end

function (l::RecyclingEmbedder)(m, z, x, ps, st)
    T = eltype(ps.linear.weight)
    N, B = size(x, 2), size(x, 3)

    m_update, st_ln_m = l.layer_norm_m(m, ps.layer_norm_m, st.layer_norm_m)
    z_ln, st_ln_z = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)

    sq_norm = sum(T.(x) .^ 2; dims=1)
    gram = Lux.batched_matmul(permutedims(T.(x), (2, 1, 3)), T.(x))
    d_sq = sq_norm .+ permutedims(sq_norm, (2, 1, 3)) .- 2 .* gram

    bin_edges = T.(range(l.min_bin, l.max_bin; length=l.no_bins))
    squared_bins = bin_edges .^ 2
    upper = vcat(squared_bins[2:end], T[floatmax(T)])

    d_sq_r = reshape(d_sq, 1, N, N, B)
    lower_r = reshape(squared_bins, l.no_bins, 1, 1, 1)
    upper_r = reshape(upper, l.no_bins, 1, 1, 1)
    bins = T.((d_sq_r .> lower_r) .* (d_sq_r .< upper_r))

    z_bin, st_linear = l.linear(bins, ps.linear, st.linear)
    z_update = z_ln .+ z_bin

    st_out = merge(st, (; layer_norm_m=st_ln_m, layer_norm_z=st_ln_z, linear=st_linear))
    return (m_update, z_update), st_out
end
