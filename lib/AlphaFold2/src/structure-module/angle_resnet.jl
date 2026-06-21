"""
    AngleResnetBlock(chn_hidden; use_bias=true)

Single residual block of the angle resnet — matches
`openfold.model.structure_module.AngleResnetBlock`.

# Arguments
- `chn_hidden`: Hidden channel dimension.

# Keyword Arguments
- `use_bias`: `Bool` / `NamedTuple` controlling biases on `:linear_1` and `:linear_2`.

# Inputs
- `s`: `[chn_hidden, N, B]`

# Returns
- `s_out`: `[chn_hidden, N, B]` — `s_initial + linear_2(relu(linear_1(relu(s_initial))))`
- `st`: Updated state.
"""
struct AngleResnetBlock{L1,L2} <: Lux.AbstractLuxContainerLayer{(:linear_1, :linear_2)}
    linear_1::L1
    linear_2::L2
end

function AngleResnetBlock(chn_hidden::Int; use_bias=true)
    use_bias = resolve_defaults(use_bias, (:linear_1, :linear_2))
    return AngleResnetBlock(
        Lux.Dense(chn_hidden => chn_hidden, Lux.relu; use_bias=use_bias.linear_1),
        Lux.Dense(chn_hidden => chn_hidden; use_bias=use_bias.linear_2),
    )
end

function (l::AngleResnetBlock)(s::AbstractArray, ps, st)
    s_initial = s
    s = Lux.relu.(s)
    s, st1 = l.linear_1(s, ps.linear_1, st.linear_1)
    s, st2 = l.linear_2(s, ps.linear_2, st.linear_2)
    return s .+ s_initial, merge(st, (; linear_1=st1, linear_2=st2))
end

# Chain interoperability: Lux.Chain unpacks the first return as input to next layer.


"""
    AngleResnet(chn_s, chn_hidden; no_blocks=2, no_angles=7, epsilon=1f-8, use_bias=true)

Angle resnet from Algorithm 20 (lines 11-14). Matches
`openfold.model.structure_module.AngleResnet`.

The current-`s` and initial-`s_init` projections are combined by **addition** (not
concatenation). Note: relu is applied *before* each projection, matching the
openfold source comment about diverging from the supplement pseudocode.

# Arguments
- `chn_s`: Single representation channel dimension.
- `chn_hidden`: Hidden channel dimension inside the resnet.

# Keyword Arguments
- `no_blocks`: Number of `AngleResnetBlock`s (openfold AF2 default: 2).
- `no_angles`: Number of torsion angles to predict (default 7).
- `epsilon`: L2-normalisation epsilon (default `1f-8`).
- `use_bias`: bias config; resolved over `:linear_in`, `:linear_initial`, `:blocks`, `:linear_out`.

# Inputs
- `s`: `[chn_s, N, B]` — current single representation
- `s_init`: `[chn_s, N, B]` — initial single representation (captured before `linear_in` in
  `StructureModule`)

# Returns
- `(; unnormalized_angles, angles)`: NamedTuple where
  - `unnormalized_angles`: `[2, no_angles, N, B]`
  - `angles`: `[2, no_angles, N, B]` — L2-normalised
- `st`: Updated state.
"""
struct AngleResnet{LIN,LINI,BL,LOUT,E} <: Lux.AbstractLuxContainerLayer{
    (:linear_in, :linear_initial, :blocks, :linear_out)
}
    linear_in::LIN
    linear_initial::LINI
    blocks::BL
    linear_out::LOUT
    no_angles::Int
    epsilon::E
end

function AngleResnet(
    chn_s::Int, chn_hidden::Int;
    no_blocks::Int=2, no_angles::Int=7, epsilon=1f-8, use_bias=true,
)
    use_bias = resolve_defaults(use_bias, (:linear_in, :linear_initial, :blocks, :linear_out))
    block_use_bias = use_bias.blocks
    blocks = Lux.Chain( # TODO: Do we want block_i naming here?
        ntuple(i -> AngleResnetBlock(chn_hidden; use_bias=block_use_bias), no_blocks)...
    )
    return AngleResnet(
        Lux.Dense(chn_s => chn_hidden; use_bias=use_bias.linear_in),
        Lux.Dense(chn_s => chn_hidden; use_bias=use_bias.linear_initial),
        blocks,
        Lux.Dense(chn_hidden => no_angles * 2; use_bias=use_bias.linear_out),
        no_angles,
        epsilon,
    )
end

function (l::AngleResnet)(s::AbstractArray{T}, s_init::AbstractArray{T}, ps, st) where T
    # Two-path projection (relu-first, addition)
    s_init_a = Lux.relu.(s_init)
    s_init_p, st_init = l.linear_initial(s_init_a, ps.linear_initial, st.linear_initial)
    s_a = Lux.relu.(s)
    s_p, st_in = l.linear_in(s_a, ps.linear_in, st.linear_in)
    h = s_p .+ s_init_p

    # Residual blocks
    h, st_blocks = l.blocks(h, ps.blocks, st.blocks)

    # Final relu + project + reshape
    h = Lux.relu.(h)
    h, st_out = l.linear_out(h, ps.linear_out, st.linear_out)   # [2*no_angles, N, B]
    N = size(h, 2); B = size(h, 3)
    h = reshape(h, 2, l.no_angles, N, B)
    unnormalized = h

    # L2-normalize the sin/cos pair (dim 1) per angle, per residue, per batch.
    eps_T = T(l.epsilon)
    h2_sum = sum(abs2, h; dims=1)
    angles = @. h / sqrt(max(h2_sum, eps_T))

    new_st = merge(st, (;
        linear_in=st_in, linear_initial=st_init, blocks=st_blocks, linear_out=st_out,
    ))
    return (; unnormalized_angles=unnormalized, angles), new_st
end

# NamedTuple-friendly dispatch
(l::AngleResnet)(inputs::NamedTuple, ps, st) = l(
    inputs.s, 
    inputs.s_init, 
    ps, st
)
