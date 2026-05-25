# ===  LuxTriangleAttention overloads for 5D template tensors  ===
#
# LuxTriangleAttention's existing overloads handle 4D features [C,N,N,B] and 3D Bool masks
# [N,N,B]. Template processing adds a N_templ dimension, requiring:
#
#   1. ending_permute for 5D <:Real  — overrides the existing (unused) LuxTriangleAttention
#      method that swaps dims 3,4.  Here we swap dims 2,3 for [C,Ni,Nj,N_templ,B].
#   2. ending_permute for 4D Bool    — new overload, no conflict (LuxTriangleAttention only
#      has Bool,3).
#   3. prep_mask for 4D Bool         — new overload; reshapes to 6D for logit broadcasting.
#   4. prep_bias for 5D bias + 5D x  — new overload; permutes/reshapes for logit broadcasting.
#   5. _batched_matmul for 5D a, b   — TriMulCore batched matmul; batch over (1,4,5), contract
#      over dim 3 (outgoing) or dim 2 (incoming).
#   6. _permute_mult_out for 5D y    — [Ni,Nj,H,N_templ,B] → [H,Ni,Nj,N_templ,B].
#
# Dispatch convention inherited from LuxTriangleAttention:
#   <:Real → feature tensor,  Bool → mask
#
# See openspec/TEMPLATE_EMBEDDERS.md for full derivation of all shapes.

import LuxTriangleAttention: ending_permute, prep_mask, prep_bias, _batched_matmul, _permute_mult_out

# Override unused 5D <:Real method: [C, Ni, Nj, N_templ, B] → swap dims 2,3
LuxTriangleAttention.ending_permute(x::AbstractArray{<:Real,5}) =
    permutedims(x, (1, 3, 2, 4, 5))

# New Bool,4 method: [Ni, Nj, N_templ, B] → swap dims 1,2
LuxTriangleAttention.ending_permute(mask::AbstractArray{Bool,4}) =
    permutedims(mask, (2, 1, 3, 4))

# prep_mask: 4D Bool (possibly permuted) → 6D for logits [Na, Na, H, Nb, N_templ, B]
function LuxTriangleAttention.prep_mask(mask::AbstractArray{Bool,4})
    Na, Nb, N_templ, B = size(mask)
    return reshape(mask, Na, 1, 1, Nb, N_templ, B)
end

# prep_bias: [H, Na, Nb, N_templ, B] + 5D x_norm → [Na, Nb, H, 1, N_templ, B]
# Mirrors the 4D overload: [H,Nq,Nk,B] → [Nq,Nk,H,1,B].
function LuxTriangleAttention.prep_bias(
    bias::AbstractArray{T,5}, ::AbstractArray{<:Any,5}, ::StaticSymbol{:qk}
) where T
    H, Na, Nb, N_templ, B = size(bias)
    return reshape(permutedims(bias, (2, 3, 1, 4, 5)), Na, Nb, H, 1, N_templ, B)
end

# _batched_matmul: 5D overloads for TriMulCore with [C, Ni, Nj, N_templ, B] tensors.
# Batch over dims (1=C, 4=N_templ, 5=B); contract over dim 3 (Nj) or dim 2 (Ni).
LuxTriangleAttention._batched_matmul(::True, a::AbstractArray{<:Any,5}, b::AbstractArray{<:Any,5}) =
    Lux.batched_matmul(a, b;
        lhs_contracting_dim=3, rhs_contracting_dim=3,
        lhs_batching_dims=(1, 4, 5), rhs_batching_dims=(1, 4, 5))

LuxTriangleAttention._batched_matmul(::False, a::AbstractArray{<:Any,5}, b::AbstractArray{<:Any,5}) =
    Lux.batched_matmul(a, b;
        lhs_contracting_dim=2, rhs_contracting_dim=2,
        lhs_batching_dims=(1, 4, 5), rhs_batching_dims=(1, 4, 5))

# _permute_mult_out: 5D output [Ni, Nj, H, N_templ, B] → [H, Ni, Nj, N_templ, B]
LuxTriangleAttention._permute_mult_out(y::AbstractArray{<:Any,5}) =
    permutedims(y, (3, 1, 2, 4, 5))

# =============================================================================

"""
    TemplatePairStackBlock(chn_templ, chn_hidden_tri_att, chn_hidden_tri_mul, no_heads, pair_transition_n; ...)

A single block of the template pair stack (Algorithm 16).
Applies triangular attention (starting/ending), triangular multiplication (outgoing/incoming),
and a pair transition. Operates natively on 5D tensors — all templates in parallel, no loop.

# Arguments
- `chn_templ`: Template pair embedding channel dimension
- `chn_hidden_tri_att`: Head dimension for triangle attention
- `chn_hidden_tri_mul`: Hidden dimension for triangle multiplication
- `no_heads`: Number of attention heads
- `pair_transition_n`: Expansion factor for the pair transition MLP

# Keyword Arguments
- `tri_mul_first`: If `true`, triangle multiplications run before attentions (default: `false`)
- `use_bias`: `Bool` or `NamedTuple` for per-sublayer bias control (default: `true`)
- `epsilon`: LayerNorm epsilon (default: `1f-5`)

# Inputs
- `z`: Template pair embedding `[chn_templ, N_res, N_res, N_templ, B]`
- `mask`: Pair mask `[N_res, N_res, N_templ, B]` (Bool)

# Returns
- `z`: Updated tensor (same shape as input)
- `st`: Updated state
"""
struct TemplatePairStackBlock{TRI_MUL_FIRST, TAS, TAE, TMO, TMI, PT} <: Lux.AbstractLuxContainerLayer{(:tri_att_start, :tri_att_end, :tri_mul_out, :tri_mul_in, :pair_transition)}
    tri_mul_first::TRI_MUL_FIRST   # StaticBool — FIRST so TemplatePairStackBlock{True} dispatches cleanly
    tri_att_start::TAS
    tri_att_end::TAE
    tri_mul_out::TMO
    tri_mul_in::TMI
    pair_transition::PT
end

# Map a top-level Bool use_bias to the openfold-compatible per-sublayer settings for
# TriangleAttention. Openfold sets bias=False on the triangle-bias linear projection
# and on QKV, but True on the gate and output projections.
_tri_att_use_bias(b::Bool) =
    b ? (layer_norm=true, linear=false, mha=(qkv=false, gate=true, out=true)) : false
_tri_att_use_bias(b) = b   # NamedTuple overrides → pass through

function TemplatePairStackBlock(
    chn_templ::Int, chn_hidden_tri_att::Int, chn_hidden_tri_mul::Int,
    no_heads::Int, pair_transition_n::Int;
    tri_mul_first=false,
    use_bias=true,
    epsilon=1f-5
)
    use_bias = resolve_defaults(
        use_bias, (:tri_att_start, :tri_att_end, :tri_mul_out, :tri_mul_in, :pair_transition)
    )
    tri_att_start = TriangleAttention(
        chn_templ, chn_hidden_tri_att, no_heads;
        is_starting=static(true), rank=5,
        use_bias=_tri_att_use_bias(use_bias.tri_att_start), layernorm_eps=epsilon
    )
    tri_att_end = TriangleAttention(
        chn_templ, chn_hidden_tri_att, no_heads;
        is_starting=static(false), rank=5,
        use_bias=_tri_att_use_bias(use_bias.tri_att_end), layernorm_eps=epsilon
    )
    tri_mul_out = TriangleMultiplication(
        chn_templ, chn_hidden_tri_mul;
        is_outgoing=static(true), rank=5, use_bias=use_bias.tri_mul_out, layernorm_eps=epsilon
    )
    tri_mul_in = TriangleMultiplication(
        chn_templ, chn_hidden_tri_mul;
        is_outgoing=static(false), rank=5, use_bias=use_bias.tri_mul_in, layernorm_eps=epsilon
    )
    pair_transition = Transition(
        chn_templ; n=pair_transition_n, rank=5, use_bias=use_bias.pair_transition
    )
    return TemplatePairStackBlock(
        static(tri_mul_first),
        tri_att_start, tri_att_end, tri_mul_out, tri_mul_in, pair_transition
    )
end

# NamedTuple dispatch — lets Lux.Chain thread (; z, mask) through the stack.
# The mask is passed through unchanged; only z is updated.
(l::TemplatePairStackBlock)(inputs::NamedTuple, ps, st) = l(
    inputs.z, 
    get(inputs, :mask, nothing), 
    ps, st
)

# tri_mul_first = False: attention → multiplication (default AlphaFold2 order)
function (l::TemplatePairStackBlock{False})(z, mask, ps, st)
    u, tri_att_start = l.tri_att_start(z, mask, ps.tri_att_start, st.tri_att_start)
    z = z .+ u
    u, tri_att_end = l.tri_att_end(z, mask, ps.tri_att_end, st.tri_att_end)
    z = z .+ u
    u, tri_mul_out = l.tri_mul_out(z, mask, ps.tri_mul_out, st.tri_mul_out)
    z = z .+ u
    u, tri_mul_in = l.tri_mul_in(z, mask, ps.tri_mul_in, st.tri_mul_in)
    z = z .+ u
    u, pair_transition = l.pair_transition(z, mask, ps.pair_transition, st.pair_transition)
    z = z .+ u

    st_new = merge(st, (;
        tri_att_start, tri_att_end, tri_mul_out, tri_mul_in, pair_transition
    ))
    return (; z, mask, ), st_new
end

# tri_mul_first = True: multiplication → attention
function (l::TemplatePairStackBlock{True})(z, mask, ps, st)
    u, tri_mul_out = l.tri_mul_out(z, mask, ps.tri_mul_out, st.tri_mul_out)
    z = z .+ u
    u, tri_mul_in = l.tri_mul_in(z, mask, ps.tri_mul_in, st.tri_mul_in)
    z = z .+ u
    u, tri_att_start = l.tri_att_start(z, mask, ps.tri_att_start, st.tri_att_start)
    z = z .+ u
    u, tri_att_end = l.tri_att_end(z, mask, ps.tri_att_end, st.tri_att_end)
    z = z .+ u
    u, pair_transition  = l.pair_transition(z, mask, ps.pair_transition, st.pair_transition)
    z = z .+ u

    st_new = merge(st, (;
        tri_att_start, tri_att_end, tri_mul_out, tri_mul_in, pair_transition
    ))

    return (; z, mask, ), st_new
end

# =============================================================================

"""
    TemplatePairStack(chn_templ, chn_hidden_tri_att, chn_hidden_tri_mul, no_blocks, no_heads, pair_transition_n; ...)

Sequential stack of `TemplatePairStackBlock`s followed by a final `LayerNorm` (Algorithm 16).
All `N_templ` templates are processed in parallel throughout.

Each block accepts and returns `(; z, mask)` so the stack is implemented as a `Lux.Chain`,
which provides type-stable threading without a manual loop.

# Arguments
- `chn_templ`: Template pair embedding channel dimension
- `chn_hidden_tri_att`: Head dimension for triangle attention
- `chn_hidden_tri_mul`: Hidden dimension for triangle multiplication
- `no_blocks`: Number of blocks in the stack
- `no_heads`: Number of attention heads
- `pair_transition_n`: Expansion factor for the pair transition MLP

# Keyword Arguments
- `tri_mul_first`: Operation order within each block (default: `false`)
- `use_bias`: `Bool` or `NamedTuple` for bias control (default: `true`)
- `epsilon`: LayerNorm epsilon (default: `1f-5`)

# Inputs
- `t`: Template pair embedding `[chn_templ, N_res, N_res, N_templ, B]`
- `mask`: Pair mask `[N_res, N_res, N_templ, B]` (Bool), or `nothing`

# Returns
- `t`: Normalized template pair embedding (same shape)
- `st`: Updated state
"""
struct TemplatePairStack{B, LN} <: Lux.AbstractLuxContainerLayer{(:blocks, :layer_norm)}
    blocks::B    # Lux.Chain of TemplatePairStackBlocks
    layer_norm::LN
end

function TemplatePairStack(
    chn_templ::Int, chn_hidden_tri_att::Int, chn_hidden_tri_mul::Int,
    no_blocks::Int, no_heads::Int, pair_transition_n::Int;
    tri_mul_first=false,
    use_bias=true,
    epsilon=1f-5
)
    use_bias = resolve_defaults(use_bias, (:blocks, :layer_norm))

    block_nt = NamedTuple{Tuple(Symbol("block_$i") for i in 1:no_blocks)}(
        ntuple(no_blocks) do _
            TemplatePairStackBlock(
                chn_templ, chn_hidden_tri_att, chn_hidden_tri_mul, no_heads, pair_transition_n;
                tri_mul_first, use_bias=use_bias.blocks, epsilon
            )
        end
    )
    blocks = Lux.Chain(block_nt)

    layer_norm = if use_bias.layer_norm
        Lux.LayerNorm((chn_templ, 1, 1, 1); dims=1, epsilon)
    else
        LayerNormNoBias((chn_templ, 1, 1, 1); dims=1, epsilon)
    end

    return TemplatePairStack(blocks, layer_norm)
end

(l::TemplatePairStack)(z, mask, ps, st) = l((; z, mask), ps, st)

# Lux.Chain threads (; z, mask) through each block type-stably.
function (l::TemplatePairStack)(inputs::NamedTuple, ps, st)
    outputs, st_blocks = l.blocks(
        (z = inputs.z, mask = get(inputs, :mask, nothing)), 
        ps.blocks, st.blocks
    )
    z, st_ln = l.layer_norm(outputs.z, ps.layer_norm, st.layer_norm)
    return z, merge(st, (; blocks=st_blocks, layer_norm=st_ln))
end
