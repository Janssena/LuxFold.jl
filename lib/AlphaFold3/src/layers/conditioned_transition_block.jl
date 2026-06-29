
"""
    ConditionedTransitionBlock(chn_a, chn_s, n_transition)

AdaLN-conditioned SwiGLU transition block (part of AF3 Algorithm 23). Applies
`AdaLN(chn_a ← chn_s)`, then a shared unfused `SwiGLU`, gated by a sigmoid linear on `s`.

# Arguments
- `chn_a`: Channel dim of the activation `a`
- `chn_s`: Channel dim of the single conditioning `s`
- `n_transition`: Width multiplier for the SwiGLU hidden dim (`chn_a * n_transition`)

# Inputs
- `a`: Activation `[chn_a, ..., B]`
- `s`: Single conditioning `[chn_s, ..., B]`
- `mask`: `AbstractArray{Bool}` (same spatial dims as `a`) or `nothing`

# Returns
- `a_new`: Updated activation `[chn_a, ..., B]`
- `st`: Updated state
"""
struct ConditionedTransitionBlock{LN,G,LG,LO} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :swiglu, :linear_g, :linear_out)}
    layer_norm::LN
    swiglu::G
    linear_g::LG
    linear_out::LO
end

function ConditionedTransitionBlock(chn_a::Int, chn_s::Int, n_transition::Int)
    return ConditionedTransitionBlock(
        AdaLN(chn_a, chn_s; use_bias=(false, (gate=true, shift=false))),
        SwiGLU(chn_a => n_transition * chn_a; fused=false, use_bias=false),  # shared SwiGLU (unfused)
        Lux.Dense(chn_s => chn_a, sigmoid; use_bias=true),         # linear_g
        Lux.Dense(n_transition * chn_a => chn_a; use_bias=false),  # linear_out
    )
end

# config-NamedTuple compatibility shim
ConditionedTransitionBlock(config::NamedTuple) =
    ConditionedTransitionBlock(config.c_a, config.c_s, config.n_transition)

# NamedTuple-input dispatch → positional forward
(l::ConditionedTransitionBlock)(inputs::NamedTuple, ps, st) =
    l(inputs.a, inputs.s, get(inputs, :mask, nothing), ps, st)

function (l::ConditionedTransitionBlock)(a, s, mask, ps, st)
    a_norm, st_ln = l.layer_norm(a, s, ps.layer_norm, st.layer_norm)
    a_swiglu, st_swiglu = l.swiglu(a_norm, ps.swiglu, st.swiglu)

    a_gate, st_g = l.linear_g(s, ps.linear_g, st.linear_g)
    a_proj, st_out = l.linear_out(a_swiglu, ps.linear_out, st.linear_out)

    a_new = a_gate .* a_proj
    apply_transition_mask!(a_new, mask)   # Bool mask, ifelse, in-place (no-op if nothing)

    return a_new, merge(st, (; layer_norm=st_ln, swiglu=st_swiglu, linear_g=st_g, linear_out=st_out))
end
