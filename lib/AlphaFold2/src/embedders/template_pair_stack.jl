# =============================================================================
# TemplatePairStackBlock and TemplatePairStack
# =============================================================================
#
# Template tensors are 5D: [C, N, N, N_templ, B].
#
# TemplatePairStackBlock is a PairStackBlock(rank=4) — all sub-layers are configured
# for 4D input. The 5D dispatch on PairStackBlock{True/False, StaticInt{4}} (defined
# in pair_stack_block.jl) handles standalone calls with 5D input by merging N_templ*B.
#
# TemplatePairStack does the N_templ*B merge ONCE at the stack level before the Chain,
# so blocks receive plain 4D tensors throughout. After the chain the 5D shape is
# restored before the final LayerNorm (whose parameters match the 5D layout).
#
# Mask conventions (_expand_pair_mask is defined in pair_stack_block.jl):
#   Bool,3  [N, N, B]          — shared mask, replicated across templates
#   Bool,4  [N, N, N_templ, B] — per-template mask

# =============================================================================

"""
    TemplatePairStackBlock(chn_templ, chn_hidden_tri_att, chn_hidden_tri_mul, no_heads, pair_transition_n; ...)

Factory function that constructs a `PairStack` configured for rank-5 template tensors
(Algorithm 16). Operates natively on 5D tensors `[chn_templ, N_res, N_res, N_templ, B]`
— all templates are processed in parallel.

The N_templ and B dimensions are merged to 4D before each block's forward pass, then
restored — a zero-cost reshape since all triangle operations are batch-independent.

Default operation order is `tri_mul_first=false` (attention-first), matching the
original AlphaFold2 template pair stack.

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
- Updated `(; z, mask)` NamedTuple (mask unchanged)
- `st`: Updated state
"""
function TemplatePairStackBlock(
    chn_templ::Int, chn_hidden_tri_att::Int, chn_hidden_tri_mul::Int,
    no_heads::Int, pair_transition_n::Int;
    tri_mul_first=false,
    use_bias=true,
    epsilon=1f-5
)
    return PairStackBlock(
        chn_templ, chn_hidden_tri_mul, chn_hidden_tri_att, no_heads, pair_transition_n;
        tri_mul_first, rank=4, use_bias, epsilon
    )
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
- `mask`: Pair mask — `[N_res, N_res, B]` (shared across templates),
  `[N_res, N_res, N_templ, B]` (per-template), or `nothing`

# Returns
- `t`: Normalized template pair embedding (same shape as input)
- `st`: Updated state
"""
struct TemplatePairStack{B, LN} <: Lux.AbstractLuxContainerLayer{(:blocks, :layer_norm)}
    blocks::B
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

(l::TemplatePairStack)(inputs::NamedTuple, ps, st) = l(
    inputs.z, 
    get(inputs, :mask, nothing), 
    ps, st
)

# The N_templ*B merge is done once here so each block receives plain 4D tensors —
# no repeated reshape overhead inside every block. After the chain outputs are
# reshaped back to 5D before LayerNorm so the layer-norm parameters (shape
# [C, 1, 1, 1], matching 5D data) are applied at the correct rank.
function (l::TemplatePairStack)(z, mask, ps, st)
    C, Ni, Nj, N_templ, B = size(z)

    z4d = reshape(z, C, Ni, Nj, N_templ * B)
    mask4d = _expand_pair_mask(mask, Ni, Nj, N_templ, B)

    outputs, st_blocks = l.blocks((z = z4d, mask = mask4d), ps.blocks, st.blocks)

    z5d = reshape(outputs.z, C, Ni, Nj, N_templ, B)
    z_out, st_ln = l.layer_norm(z5d, ps.layer_norm, st.layer_norm)
    return z_out, merge(st, (; blocks=st_blocks, layer_norm=st_ln))
end
