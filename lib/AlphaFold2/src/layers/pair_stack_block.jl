"""
    PairStackBlock(chn_pair, chn_hidden_tri_mul, chn_hidden_tri_att, no_heads, transition_n; ...)

Generic pair-representation update block used in both `EvoformerBlock` and `ExtraMSABlock`.
Applies triangle multiplications, triangle attentions, and a pair transition with residual
connections. Operates on rank-4 tensors `[C_z, N, N, B]`.

When called with a rank-5 tensor `[C_z, N, N, N_templ, B]` (template use-case), the block
automatically merges the `N_templ` and `B` dimensions, runs the standard 4D forward, then
restores the 5D shape — provided the block was constructed with `rank=4`.

The default operation order (`tri_mul_first=true`, Evoformer convention) is
multiplication → attention, matching the Python `PairStack` in `evoformer.py`.

# Arguments
- `chn_pair`: Pair channel dimension
- `chn_hidden_tri_mul`: Hidden dimension for triangle multiplication
- `chn_hidden_tri_att`: Hidden dimension for triangle attention
- `no_heads`: Number of triangle attention heads
- `transition_n`: Expansion factor for the pair transition MLP

# Keyword Arguments
- `tri_mul_first`: If `true` (default), triangle multiplications precede attentions
- `rank`: Tensor rank the block is configured for (default: `4`)
- `use_bias`: `Bool` or `NamedTuple` per-sublayer bias control (default: `true`)
- `epsilon`: LayerNorm epsilon (default: `1f-5`)

# Inputs
- `z`: Pair embedding `[chn_pair, N, N, B]`; or `[chn_pair, N, N, N_templ, B]` when rank=4
  (5D input is reshaped to 4D internally)
- `mask`: Bool mask matching spatial+batch dims, or `nothing`

# Returns
- `z`: Updated tensor (same shape as input, including accumulated residuals)
- `st`: Updated state
"""
struct PairStackBlock{TRI_MUL_FIRST, RANK, TMO, TMI, TAS, TAE, PT} <: Lux.AbstractLuxContainerLayer{(:tri_mul_out, :tri_mul_in, :tri_att_start, :tri_att_end, :pair_transition)}
    tri_mul_first::TRI_MUL_FIRST
    rank::RANK
    tri_mul_out::TMO
    tri_mul_in::TMI
    tri_att_start::TAS
    tri_att_end::TAE
    pair_transition::PT
end

# =============================================================================
# Mask helpers for 5D → 4D reshape (used by TemplatePairStack and 5D dispatch below)
# =============================================================================

_expand_pair_mask(::Nothing, args...) = nothing

# [N, N, B] → insert singleton template dim, tile N_templ times, flatten to [N, N, N_templ*B].
# Uses reshape+repeat rather than repeat directly to preserve column-major ordering:
# reshape(z, C,N,N, N_templ*B) iterates templates fastest (dim 4 then dim 5),
# so the mask must do the same — template dim inserted before B.
function _expand_pair_mask(mask::AbstractArray{Bool,3}, Ni, Nj, N_templ, B)
    return reshape(repeat(reshape(mask, Ni, Nj, 1, B), 1, 1, N_templ, 1), Ni, Nj, N_templ * B)
end

# [N, N, N_templ, B] → flatten last two dims (per-template mask already in correct shape)
function _expand_pair_mask(mask::AbstractArray{Bool,4}, Ni, Nj, N_templ, B)
    return reshape(mask, Ni, Nj, N_templ * B)
end

# =============================================================================
# 5D dispatch for rank=4 blocks (template tensors: N_templ merged into B)
# =============================================================================
#
# When a rank=4 PairStackBlock receives 5D input [C, N, N, N_templ, B], it merges
# N_templ*B → 4D, runs the standard forward, then restores 5D on return.
# This is a zero-cost reshape (Julia column-major: N_templ and B are contiguous).
#
# Two explicit methods (one per tri_mul_first variant) are required to avoid dispatch
# ambiguity: (l::PairStackBlock{True})(z, ...) is more specific on l than a generic
# PairStackBlock, while (z::AbstractArray{T,5}) is more specific on z. Having both
# type params fully pinned makes these methods unambiguously most specific.

# TriangleAttention bias helper: copied from (and consistent with) TemplatePairStackBlock.
# OpenFold convention: layer_norm=true, linear (pair→heads bias) =false, QKV=false, gate+out=true.
_tri_att_use_bias(b::Bool) =
    b ? (layer_norm=true, linear=false, mha=(qkv=false, gate=true, out=true)) : false
_tri_att_use_bias(b) = b

function PairStackBlock(
    chn_pair::Int, chn_hidden_tri_mul::Int, chn_hidden_tri_att::Int,
    no_heads::Int, transition_n::Int;
    tri_mul_first=true,
    rank=4,
    use_bias=true,
    epsilon::Real=1f-5
)
    use_bias = resolve_defaults(
        use_bias, (:tri_att_start, :tri_att_end, :tri_mul_out, :tri_mul_in, :pair_transition)
    )

    tri_att_start = TriangleAttention(
        chn_pair, chn_hidden_tri_att, no_heads;
        is_starting=true, rank,
        use_bias=_tri_att_use_bias(use_bias.tri_att_start), layernorm_eps=epsilon
    )
    tri_att_end = TriangleAttention(
        chn_pair, chn_hidden_tri_att, no_heads;
        is_starting=false, rank,
        use_bias=_tri_att_use_bias(use_bias.tri_att_end), layernorm_eps=epsilon
    )
    tri_mul_out = TriangleMultiplication(
        chn_pair, chn_hidden_tri_mul;
        is_outgoing=true, rank, use_bias=use_bias.tri_mul_out, layernorm_eps=epsilon
    )
    tri_mul_in = TriangleMultiplication(
        chn_pair, chn_hidden_tri_mul;
        is_outgoing=false, rank, use_bias=use_bias.tri_mul_in, layernorm_eps=epsilon
    )
    pair_transition = Transition(
        chn_pair; n=transition_n, rank, use_bias=use_bias.pair_transition
    )

    return PairStackBlock(
        static(tri_mul_first),
        static(rank),
        tri_mul_out, tri_mul_in, tri_att_start, tri_att_end, pair_transition
    )
end

# NamedTuple dispatch — flexible key handling for both Evoformer (; z, pair_mask)
# and template (; z, mask) conventions. Returns merge(inputs, (; z=z_out)) so all
# input keys (including the mask key) pass through unchanged.
function (l::PairStackBlock)(inputs::NamedTuple, ps, st)
    z_out, st_new = l(
        inputs.z, 
        get(inputs, :pair_mask, get(inputs, :mask, nothing)), 
        ps, st
    )
    return merge(inputs, (; z=z_out, )), st_new
end

# TODO: If we want a StaticInt{5} version, we need overloads for prep_mask, prep_bias, etc.
function (l::PairStackBlock{<:StaticBool, StaticInt{4}})(z::AbstractArray{T,5}, mask, ps, st) where T
    C, Ni, Nj, N_templ, B = size(z)
    z4d_out, st_out = l(
        reshape(z, C, Ni, Nj, N_templ * B), # [C, N, N, T*B]
        _expand_pair_mask(mask, Ni, Nj, N_templ, B),  # [N, N, T*B] or nothing
        ps, st
    )
    return reshape(z4d_out, C, Ni, Nj, N_templ, B), st_out
end

# tri_mul_first = True (default Evoformer order): multiplication → attention
function (l::PairStackBlock{True})(z::AbstractArray{T,4}, mask, ps, st) where T
    u, st_tmo = l.tri_mul_out(z, mask, ps.tri_mul_out, st.tri_mul_out)
    z_update = z .+ u

    u, st_tmi = l.tri_mul_in(z_update, mask, ps.tri_mul_in, st.tri_mul_in)
    z_update = z_update .+ u
    
    u, st_tas = l.tri_att_start(z_update, mask, ps.tri_att_start, st.tri_att_start)
    z_update = z_update .+ u
    
    u, st_tae = l.tri_att_end(z_update, mask, ps.tri_att_end, st.tri_att_end)
    z_update = z_update .+ u
    
    u, st_pt  = l.pair_transition(z_update, mask, ps.pair_transition, st.pair_transition)
    z_update = z_update .+ u

    return z_update, merge(st, (;
        tri_mul_out=st_tmo, tri_mul_in=st_tmi,
        tri_att_start=st_tas, tri_att_end=st_tae, pair_transition=st_pt
    ))
end

# tri_mul_first = False: attention → multiplication (TemplatePairStack order)
function (l::PairStackBlock{False})(z::AbstractArray{T, 4}, mask, ps, st) where T
    u, st_tas = l.tri_att_start(z, mask, ps.tri_att_start, st.tri_att_start)
    z_update = z .+ u
    
    u, st_tae = l.tri_att_end(z_update, mask, ps.tri_att_end, st.tri_att_end)
    z_update = z_update .+ u
    
    u, st_tmo = l.tri_mul_out(z_update, mask, ps.tri_mul_out, st.tri_mul_out)
    z_update = z_update .+ u
    
    u, st_tmi = l.tri_mul_in(z_update, mask, ps.tri_mul_in, st.tri_mul_in)
    z_update = z_update .+ u
    
    u, st_pt  = l.pair_transition(z_update, mask, ps.pair_transition, st.pair_transition)
    z_update = z_update .+ u

    return z_update, merge(st, (;
        tri_att_start=st_tas, tri_att_end=st_tae,
        tri_mul_out=st_tmo, tri_mul_in=st_tmi, pair_transition=st_pt
    ))
end
