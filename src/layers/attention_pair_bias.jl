"""
    AttentionPairBias(layer_norm_in, layer_norm_z, linear_z, mha, linear_out)

Attention layer with pair bias support, specializing for global or local attention via static dispatch.

# Arguments
- `ln_a`: Input LayerNorm or AdaLN.
- `ln_z`: Pair representation LayerNorm.
- `mha`: MultiHeadAttention layer. Should be one of TriAttnCore or AttnCore.
- `head_dim`: Dimension of each attention head.
- `num_heads`: Number of attention heads.
- `block_size`: Size of spatial blocks for local attention.
"""
struct AttentionPairBias{LNI,LNZ,LZ,MHA,LO} <: Lux.AbstractLuxContainerLayer{(:layer_norm_in,:layer_norm_z,:linear_z,:mha,:linear_out)}
    layer_norm_in::LNI
    layer_norm_z::LNZ
    linear_z::LZ
    mha::MHA
    linear_out::LO
end

# c_q: int,         # CHN_IN
# c_k: int,         # CHN_IN
# c_v: int,         # CHN_IN
# c_s: int,         # CHN_COND
# c_z: int,         # CHN_PAIR
# c_hidden: int,    # HEAD_DIM
# no_heads: int,    # NUM_HEADS

function AttentionPairBias(
    chn_in::Int,
    chn_z::Int,
    head_dim::Int,
    num_heads::Int;
    chn_cond::Union{Nothing, Int} = nothing, # if isInt, then use AdaLN 
    use_gate::Bool = true,
    use_bias=false,
    affine=true,
    kwargs...
)
    affine = resolve_defaults(affine, (:layer_norm_in, :layer_norm_z))
    use_bias = resolve_defaults(use_bias, (:layer_norm_in, :layer_norm_z, :linear_z, :mha, :linear_out))

    if isnothing(chn_cond)
        shape = (chn_in, 1)
        layer_norm_in = if !affine.layer_norm_in || use_bias.layer_norm_in 
            Lux.LayerNorm(shape; dims=1, affine=affine.layer_norm_in)
        else
            LayerNormNoBias(shape; dims=1)
        end
        linear_out = Lux.NoOpLayer()
    else
        layer_norm_in = AdaLN(chn_in => chn_cond; affine=affine.layer_norm_in, rank=3, use_bias=use_bias.layer_norm_in)
        linear_out = Lux.Dense(chn_cond => chn_in, Lux.sigmoid; use_bias=use_bias.linear_out)
    end

    layer_norm_z = if !affine.layer_norm_z || use_bias.layer_norm_z
        Lux.LayerNorm((chn_z, 1, 1); dims=1, affine=affine.layer_norm_z)
    else
        LayerNormNoBias((chn_z, 1, 1); dims=1)
    end

    return AttentionPairBias(
        layer_norm_in,
        layer_norm_z,
        Lux.Dense(chn_z => num_heads; use_bias=use_bias.linear_z),
        Attention(chn_in, head_dim, num_heads; use_gate, use_bias=use_bias.mha, kwargs...),
        linear_out
    )
end

(l::AttentionPairBias)(inputs::NamedTuple, ps, st) = l(
    inputs.x,
    inputs.z,
    get(inputs, :cond, nothing),
    get(inputs, :mask, nothing),
    ps, st
)

(l::AttentionPairBias)(x, z, ps, st) = l(x, z, nothing, nothing, ps, st)
(l::AttentionPairBias)(x, z, ::Nothing, ps, st) = l(x, z, nothing, nothing, ps, st)

(l::AttentionPairBias)(x, z, mask::AbstractArray{Bool}, ps, st) = 
    l(x, z, nothing, mask, ps, st)

(l::AttentionPairBias)(x, z, cond::AbstractArray{<:AbstractFloat}, ps, st) = 
    l(x, z, cond, nothing, ps, st)

function (l::AttentionPairBias)(x, z, ::Nothing, mask, ps, st)
    x, layer_norm_in = l.layer_norm_in(x, ps.layer_norm_in, st.layer_norm_in)
    
    z, layer_norm_z = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)
    bias, linear_z = l.linear_z(z, ps.linear_z, st.linear_z)

    attn, mha = l.mha(x, bias, mask, ps.mha, st.mha)

    return attn, merge(st, (; layer_norm_z, linear_z, layer_norm_in, mha))
end

function (l::AttentionPairBias)(x, z, cond::AbstractArray, mask, ps, st)
    x, layer_norm_in = l.layer_norm_in(x, cond, ps.layer_norm_in, st.layer_norm_in)

    z, layer_norm_z = l.layer_norm_z(z, ps.layer_norm_z, st.layer_norm_z)
    bias, linear_z = l.linear_z(z, ps.linear_z, st.linear_z)

    (attn, scores), mha = l.mha(x, bias, mask, ps.mha, st.mha)

    g, linear_out = l.linear_out(cond, ps.linear_out, st.linear_out)
    
    y = @. g * attn
    
    return (y, scores), (; layer_norm_z, linear_z, layer_norm_in, mha, linear_out)
end


# struct CrossAttentionPairBias{} <: Lux.AbstractLuxContainerLayer{(:todo)} 
# end

# """
#     AttentionPairBias(is_local, ln_a, ln_z, to_qkv, to_bias, to_gate, to_out, n_heads, block_size, head_dim)

# Attention layer with pair bias support, specializing for global or local attention via static dispatch.

# # Arguments
# - `is_local`: StaticBool, whether to use sequence-local (blocked) attention.
# - `ln_a`: Input LayerNorm or AdaLN.
# - `ln_z`: Pair representation LayerNorm.
# - `to_qkv`: Linear projection for Q, K, V.
# - `to_bias`: Linear projection for pair bias (d_z -> n_heads).
# - `to_gate`: Linear projection for gating.
# - `to_out`: Final output projection.
# - `n_heads`: Number of attention heads.
# - `block_size`: Size of spatial blocks for local attention.
# - `head_dim`: Dimension of each attention head.
# """
# struct AttentionPairBias{L, LN_A, LN_Z, TO_QKV, TO_BIAS, TO_GATE, TO_OUT} <: Lux.AbstractLuxContainerLayer{(:ln_a, :ln_z, :to_qkv, :to_bias, :to_gate, :to_out)}
#     is_local::L
#     ln_a::LN_A
#     ln_z::LN_Z
#     to_qkv::TO_QKV
#     to_bias::TO_BIAS
#     to_gate::TO_GATE
#     to_out::TO_OUT
#     n_heads::Int
#     block_size::Int
#     head_dim::Int
# end

# function AttentionPairBias(
#     c_s::Int,
#     c_z::Int,
#     n_heads::Int;
#     is_local::StaticBool = False(),
#     block_size::Int = 256,
#     head_dim::Int = 32,
#     c_cond::Union{Nothing, Int} = nothing, # Conditioning dimension for AdaLN
#     gating::Bool = true
# )
#     # Projections
#     to_qkv = Lux.Dense(c_s => 3 * n_heads * head_dim; use_bias = false)
#     ln_z = Lux.LayerNorm((c_z, 1, 1); dims=1)
#     to_bias = Lux.Dense(c_z => n_heads; use_bias = false)
#     to_gate = gating ? Lux.Dense(c_s => n_heads * head_dim; use_bias = false) : Lux.NoOpLayer()
#     to_out = Lux.Dense(n_heads * head_dim => c_s; use_bias = false)

#     # Normalization
#     if !isnothing(c_cond)
#         ln_a = AdaLN(c_cond, c_s)
#     else
#         ln_a = Lux.LayerNorm((c_s, 1); dims=1)
#     end

#     return AttentionPairBias(
#         is_local,
#         ln_a,
#         ln_z,
#         to_qkv,
#         to_bias,
#         to_gate,
#         to_out,
#         n_heads,
#         block_size,
#         head_dim
#     )
# end

# # Input handling signatures
# (l::AttentionPairBias)(x::AbstractArray, ps, st) = l(x, nothing, nothing, ps, st)
# (l::AttentionPairBias)(x::AbstractArray, pair::Union{Nothing, AbstractArray}, ps, st) = l(x, pair, nothing, nothing, ps, st)
# (l::AttentionPairBias)(x::AbstractArray, pair::Union{Nothing, AbstractArray}, mask::Union{Nothing, AbstractArray}, ps, st) = l(x, pair, mask, nothing, ps, st)

# (l::AttentionPairBias)(inputs::NamedTuple, ps, st) = l(
#     inputs.x,
#     get(inputs, :pair, nothing),
#     get(inputs, :mask, nothing),
#     get(inputs, :cond, nothing),
#     ps, st
# )

# # Core implementation
# function (l::AttentionPairBias)(x, pair, mask, cond, ps, st)
#     # 1. Prep inputs (Normalization and Blocking)
#     x_proc, st_ln_a = _prep_attention_input(l.is_local, l.ln_a, x, cond, l.block_size, ps.ln_a, st.ln_a)
    
#     # 2. Bias and Mask preparation
#     b, st_idx_z, st_bias = _prep_attention_bias(l.is_local, l.ln_z, l.to_bias, pair, l.block_size, ps.ln_z, ps.to_bias, st.ln_z, st.to_bias)
#     mask_proc = _prep_attention_mask(l.is_local, mask, l.block_size)

#     # 3. QKV Projections
#     # [c_s, N, B] -> [3 * n_heads * head_dim, N, B]
#     qkv, st_qkv = l.to_qkv(x_proc, ps.to_qkv, st.to_qkv)
    
#     curr_N = size(x_proc, 2)
#     # Target Layout: [head_dim, 3, N, heads, Batch]
#     # This leads to scores [N, N, heads, Batch]
#     qkv = reshape(qkv, l.head_dim, 3, curr_N, l.n_heads, :)
    
#     # Split Q, K, V
#     # [head_dim, N, heads, Batch]
#     q = view(qkv, :, 1, :, :, :)
#     k = view(qkv, :, 2, :, :, :)
#     v = view(qkv, :, 3, :, :, :)

#     # 4. Biased Attention
#     attn, st_attn = LuxLib.scaled_dot_product_attention(q, k, v; bias=b, mask=mask_proc)

#     # 5. Gating
#     if !isnothing(l.to_gate)
#         g, st_to_gate = l.to_gate(x_proc, ps.to_gate, st.to_gate)
#         # Reshape to match attn [D, N, H, B_total]
#         g = Lux.sigmoid.(reshape(g, l.head_dim, size(g, 2), l.n_heads, :))
#         attn = attn .* g
#     else
#         st_to_gate = NamedTuple()
#     end

#     # 6. Final Output Projection
#     # [head_dim, N, heads, B_total] -> [head_dim * heads, N, B_total]
#     attn_flat = reshape(attn, l.head_dim * l.n_heads, size(attn, 2), :)
#     out, st_out = l.to_out(attn_flat, ps.to_out, st.to_out)

#     # 7. Post-process (Unblocking)
#     final_out = _post_attention_output(l.is_local, out, size(x))

#     return final_out, (
#         ln_a = st_ln_a,
#         ln_z = st_idx_z,
#         to_qkv = st_qkv,
#         to_bias = st_bias,
#         to_gate = st_to_gate,
#         to_out = st_out
#     )
# end

# # Internal Helpers
# function _prep_attention_input(::StaticBool{true}, ln, x, cond, bs, ps, st)
#     x_norm, st_ln = _apply_norm(ln, x, cond, ps, st)
#     return block_array(x_norm, bs), st_ln
# end

# function _prep_attention_input(::StaticBool{false}, ln, x, cond, bs, ps, st)
#     x_norm, st_ln = _apply_norm(ln, x, cond, ps, st)
#     return x_norm, st_ln
# end

# # Helper to apply either LayerNorm or AdaLN
# _apply_norm(ln::Lux.LayerNorm, x, cond, ps, st) = ln(x, ps, st)
# _apply_norm(ln::AdaLN, x, cond, ps, st) = ln(x, cond, ps, st)
# _apply_norm(::Nothing, x, cond, ps, st) = (x, NamedTuple())

# # Bias prep for pair rep
# function _prep_attention_bias(::StaticBool{false}, ln_z, to_bias, pair, bs, ps_ln, ps_linear, st_ln, st_linear)
#     isnothing(pair) && return nothing, st_ln, st_linear
#     z, st_ln_new = ln_z(pair, ps_ln, st_ln)
#     b, st_linear_new = to_bias(z, ps_linear, st_linear)
#     # b is [heads, N, N, Batch]. Permute to [N, N, heads, Batch] for LuxLib
#     return collect(permutedims(b, (2, 3, 1, 4))), st_ln_new, st_linear_new
# end

# function _prep_attention_bias(::StaticBool{true}, ln_z, to_bias, pair, bs, ps_ln, ps_linear, st_ln, st_linear)
#     isnothing(pair) && return nothing, st_ln, st_linear
#     # pair is [c_z, N, N, Batch].
#     # In local attention, we need biased attention on blocks.
#     # AF3 CrossAttentionPairBias uses blocks of the bias.
#     # For now, we block the pair rep spatially.
#     # This is a simplification: we block [c_z, bs, nb, bs, nb, B] 
#     # and then take the diagonal blocks [c_z, bs, bs, nb, B].
#     C_z, N, _, B = size(pair)
#     nb = div(N, bs)
    
#     # Reshape and take diagonal (local) blocks
#     # [C, bs, nb, bs, nb, B]
#     pair_blocked = reshape(pair, C_z, bs, nb, bs, nb, B)
#     # Extract diagonal blocks: [C, bs, bs, nb, B]
#     # We use a view to avoid copy if possible, but step is needed
#     # b_local[c, i, j, k, b] = pair_blocked[c, i, k, j, k, b]
#     # For simplicity, we'll do the projection first on the full pair 
#     # or block it properly.
    
#     z, st_ln_new = ln_z(pair, ps_ln, st_ln)
#     b, st_linear_new = to_bias(z, ps_linear, st_linear)
#     # b is [H, N, N, Batch]. Block it to [H, bs, nb, bs, nb, B]
#     n_heads = size(b, 1)
#     b_blocked = reshape(b, n_heads, bs, nb, bs, nb, B)
#     # Take diagonal: b_local is [H, bs, bs, nb, B]
#     # We fold nb and B: [H, bs, bs, nb * B]
#     # Then permute to [bs, bs, H, nb * B]
    
#     # Extract diagonal blocks (lazy implementation for now)
#     b_diag = zeros(eltype(b), size(b, 1), bs, bs, nb, B)
#     for k in 1:nb
#         b_diag[:, :, :, k, :] .= b_blocked[:, :, k, :, k, :]
#     end
#     b_final = reshape(b_diag, n_heads, bs, bs, :)
#     # Permute to [bs, bs, n_heads, Batch_total]
#     return collect(permutedims(b_final, (2, 3, 1, 4))), st_ln_new, st_linear_new
# end

# function _post_attention_output(::StaticBool{true}, x, orig_size)
#     return unblock_array(x, orig_size[2])
# end

# function _post_attention_output(::StaticBool{false}, x, orig_size)
#     return x
# end

# # Mask prep
# function _prep_attention_mask(::StaticBool{false}, mask, bs)
#     isnothing(mask) && return nothing
#     # mask is [N, N, B] or [1, N, N, B] or similar.
#     # Scores are [N, N, H, B].
#     if ndims(mask) == 3
#         # [N, N, Batch] -> [N, N, 1, Batch]
#         return reshape(mask, size(mask, 1), size(mask, 2), 1, size(mask, 3))
#     elseif ndims(mask) == 4
#         # Check if head dim is at 1 or 3
#         if size(mask, 1) < size(mask, 2) # Probably [H, N, N, B]
#             return collect(permutedims(mask, (2, 3, 1, 4)))
#         end
#     end
#     return mask
# end

# function _prep_attention_mask(::StaticBool{true}, mask, bs)
#     isnothing(mask) && return nothing
#     # mask is usually [N] or [N, N, B]
#     # For local attention, we need to block it if it's 2D/3D or slice it if 1D.
#     # Boltz2 uses [N] masks mostly. 1D mask [N] -> [bs, nb, Batch]
#     if ndims(mask) == 1
#         return block_array(mask, bs)
#     elseif ndims(mask) >= 2
#         # Local blocking logic for 2D/3D masks (diagonal blocks)
#         # Similar to bias blocking.
#         return _block_mask_diagonal(mask, bs)
#     end
#     return mask
# end

# function _block_mask_diagonal(mask, bs)
#     # mask is [N, N, Batch]
#     N, _, B = size(mask)
#     nb = div(N, bs)
#     m_blocked = reshape(mask, bs, nb, bs, nb, B)
#     m_diag = zeros(eltype(mask), bs, bs, nb, B)
#     for k in 1:nb
#         m_diag[:, :, k, :] .= m_blocked[:, k, :, k, :]
#     end
#     # Mask should be [bs, bs, 1, Batch_total] or [bs, bs, H, Batch_total]
#     return reshape(m_diag, bs, bs, 1, :)
# end
