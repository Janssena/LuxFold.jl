"""
    is_pynone(x) -> Bool

Is `x` Python's `None`?

Use this, never `isnothing`, on anything that came from Python. PyCall converted `None` to Julia
`nothing` on attribute access, so guards were written `!isnothing(py.bias)`. PythonCall hands back
a `Py` wrapping `None`, for which `isnothing` is ALWAYS false — the guard silently inverts, and a
layer that is absent in Python gets treated as present. It is exported because the model suites
need it too: AlphaFold3's MSA stack has a block whose attention is `None` under `opm_first=true`.
"""
is_pynone(x) = pyis(x, pybuiltins.None)
const _isnone = is_pynone

function copy_jl_ps_to_py!(py, jl::AbstractArray{T}; swap_batch_dim=false) where T
    if isnothing(py)
        @error "copy_jl_ps_to_py! received nothing for py. jl keys: $(keys(jl))"
        return nothing
    end
    @assert pyisinstance(py, pyimport("torch").nn.Parameter) "Passed Python object is not a torch.nn.Parameter (got $(pytype(py)))"
    @assert pyconvert(Tuple, py.shape) == size(jl) "Shape of py $(py.shape) and jl $(size(jl)) do not match."
    py.data = to_py(jl; swap_batch_dim)

    return nothing
end

function sync_dense!(py::Py, jl::NamedTuple)
    @assert pyhasattr(py, "weight") "Python object (Linear) does not have weight attribute."
    py_has_bias = pyhasattr(py, "bias") && !_isnone(py.bias)
    jl_has_bias = :bias ∈ keys(jl)
    @assert py_has_bias == jl_has_bias "Python object (Linear) and NamedTuple have non-matching bias attributes (py = $(py_has_bias), jl = $(jl_has_bias))."

    copy_jl_ps_to_py!(py.weight, jl.weight)
    if :bias ∈ keys(jl) && (pyhasattr(py, "bias") && !_isnone(py.bias))
        copy_jl_ps_to_py!(py.bias, jl.bias)
    end

    return nothing
end

function sync_layernorm!(py::Py, jl::NamedTuple)
    py_has_weight = pyhasattr(py, "weight") && !_isnone(py.weight)
    jl_has_weight = :scale ∈ keys(jl)
    @assert py_has_weight == jl_has_weight "Python object (LayerNorm) and NamedTuple have non-matching weight attributes (py = $(py_has_weight), jl = $(jl_has_weight))."

    py_has_bias = (pyhasattr(py, "bias") && !_isnone(py.bias)) || (pyhasattr(py, "offset") && !_isnone(py.offset))
    jl_has_bias = :bias ∈ keys(jl)
    @assert py_has_bias == jl_has_bias "Python object (LayerNorm) and NamedTuple have non-matching bias attributes (py = $(py_has_bias), jl = $(jl_has_bias))."

    if jl_has_weight
        copy_jl_ps_to_py!(py.weight, vec(jl.scale))
    end
    if jl_has_bias
        copy_jl_ps_to_py!(py.bias, vec(jl.bias))
    end

    return nothing
end


function sync_glu!(py::Py, jl::NamedTuple; ref=(linear=:linear_z, gate=:linear_g))
    @assert pyhasattr(py, String(ref.linear)) "Python object does not have the referenced linear attribute ($(ref.linear))."
    @assert pyhasattr(py, String(ref.gate)) "Python object does not have the referenced gate attribute ($(ref.gate))."
    @assert (pyhasattr(getproperty(py, ref.linear), "bias") && !_isnone(getproperty(py, ref.linear).bias)) == (:bias ∈ keys(jl.linear)) "Python object linear and NamedTuple have non-matching bias attributes."
    gate_should_have_bias_keys = isempty(jl.gate) ? (:bias ∈ keys(jl.linear)) : (:bias ∈ keys(jl.gate))
    @assert (pyhasattr(getproperty(py, ref.gate), "bias") && !_isnone(getproperty(py, ref.gate).bias)) == gate_should_have_bias_keys "Python object gate and NamedTuple have non-matching bias attributes."

    jl_unfused = _unfuse(jl)
    sync_dense!(getproperty(py, ref.linear), jl_unfused.linear)
    sync_dense!(getproperty(py, ref.gate), jl_unfused.gate)

    return nothing
end

function _unfuse(jl::NamedTuple{(:linear, :gate)})
    if !isempty(jl.gate)
        return jl
    end

    w = jl.linear.weight
    chn = size(w, 1) ÷ 2

    ps = (
        linear=(weight=view(w, 1:chn, :),),
        gate=(weight=view(w, chn+1:2*chn, :),),
    )

    if :bias ∈ keys(jl.linear)
        b = jl.linear.bias
        ps = (
            linear=merge(ps.linear, (bias=view(b, 1:chn),)),
            gate=merge(ps.gate, (bias=view(b, chn+1:2*chn),)),
        )
    end

    return ps
end

sync_af3_adaln!(args...) =
    sync_adaln!(args...; ref=(layer_norm_a=:layer_norm_a, layer_norm_s=:layer_norm_s, shift=:linear_s, gate=:linear_g))

sync_boltz2_adaln!(args...) =
    sync_adaln!(args...; ref=(layer_norm_a=:a_norm, layer_norm_s=:s_norm, shift=:s_bias, gate=:s_scale))

function sync_adaln!(py::Py, jl::NamedTuple; ref::NamedTuple)
    sync_layernorm!(getproperty(py, ref.layer_norm_a), jl.layer_norm_a)
    sync_layernorm!(getproperty(py, ref.layer_norm_s), jl.layer_norm_s)
    sync_dense!(getproperty(py, ref.shift), jl.shift)
    sync_dense!(getproperty(py, ref.gate), jl.gate)

    return nothing
end

function sync_af3_attention!(py::Py, ps::NamedTuple)
    if :weight ∈ keys(ps.qkv) # is_fused
        # Julia qkv is fused: [3*C_hidden*H, C_in]
        # Python has linear_q, linear_k, linear_v
        # Split Julia weight
        W = ps.qkv.weight
        C_h = size(W, 1) ÷ 3

        copy_jl_ps_to_py!(py.linear_q.weight, view(W, 1:C_h, :))
        copy_jl_ps_to_py!(py.linear_k.weight, view(W, C_h+1:2*C_h, :))
        copy_jl_ps_to_py!(py.linear_v.weight, view(W, 2*C_h+1:3*C_h, :))

        if :bias ∈ keys(ps.qkv)
            B = ps.qkv.bias
            copy_jl_ps_to_py!(py.linear_q.bias, view(B, 1:C_h))
            copy_jl_ps_to_py!(py.linear_k.bias, view(B, C_h+1:2*C_h))
            copy_jl_ps_to_py!(py.linear_v.bias, view(B, 2*C_h+1:3*C_h))
        end
    elseif :q in keys(ps.qkv) && :kv in keys(ps.qkv)
        # Version where kv is fused.
        sync_dense!(py.linear_q, ps.qkv.q)

        W_kv = ps.qkv.kv.weight
        C_out_kv = size(W_kv, 1) ÷ 2

        copy_jl_ps_to_py!(py.linear_k.weight, view(W_kv, 1:C_out_kv, :))
        copy_jl_ps_to_py!(py.linear_v.weight, view(W_kv, C_out_kv+1:2*C_out_kv, :))

        if :bias ∈ keys(ps.qkv.kv) && (pyhasattr(py.linear_k, "bias") && !_isnone(py.linear_k.bias))
            B_kv = ps.qkv.kv.bias
            copy_jl_ps_to_py!(py.linear_k.bias, view(B_kv, 1:C_out_kv))
            copy_jl_ps_to_py!(py.linear_v.bias, view(B_kv, C_out_kv+1:2*C_out_kv))
        end
    else # unfused
        sync_dense!(py.linear_q, ps.qkv.q)
        sync_dense!(py.linear_k, ps.qkv.k)
        sync_dense!(py.linear_v, ps.qkv.v)
    end
    sync_dense!(py.linear_o, ps.out)

    if !isempty(ps.gate)
        sync_dense!(py.linear_g, ps.gate)
    end

    return nothing
end

function sync_af3_attention_pair_bias!(py::Py, ps::NamedTuple)
    if isempty(ps.linear_out)
        sync_layernorm!(py.layer_norm_a, ps.layer_norm_in)
    else
        sync_af3_adaln!(py.layer_norm_a, ps.layer_norm_in)
        sync_dense!(py.linear_ada_out, ps.linear_out)
    end
    # Empty when the caller normalises `z` outside the block (AF3's DiffusionTransformer owns a
    # single `layer_norm_z` for the whole stack). openfold-3's `DiffusionAttentionPairBias` has no
    # `layer_norm_z` attribute at all in that case, so it must not be touched — same guard idiom
    # as `ps.linear_out` above.
    isempty(ps.layer_norm_z) || sync_layernorm!(py.layer_norm_z, ps.layer_norm_z)
    sync_dense!(py.linear_z, ps.linear_z)

    sync_af3_attention!(py.mha, ps.mha)

    return nothing
end

sync_af3_opm!(args...) =
    sync_opm!(args...; ref=(layer_norm=:layer_norm, linear_1=:linear_1, linear_2=:linear_2, linear_out=:linear_out))

# openfold's AF2 `OuterProductMean` uses the same attribute names as openfold-3's, so the AF3 ref
# serves both. Named separately because the two are different classes in different packages, and a
# future divergence should be a one-line change here rather than a puzzle at the call site.
sync_af2_opm!(args...) = sync_af3_opm!(args...)

sync_boltz2_opm!(args...) =
    sync_opm!(args...; ref=(layer_norm=:norm, linear_1=:proj_a, linear_2=:proj_b, linear_out=:proj_o))

function sync_opm!(py::Py, jl::NamedTuple; ref::NamedTuple)
    sync_layernorm!(getproperty(py, ref.layer_norm), jl.layer_norm)
    sync_dense!(getproperty(py, ref.linear_1), jl.linear1)
    sync_dense!(getproperty(py, ref.linear_2), jl.linear2)
    sync_dense!(getproperty(py, ref.linear_out), jl.linear_out)

    return nothing
end

sync_opm!(args...) =
    sync_pwa!(args...; ref=(layer_norm_m=:layer_norm_m, layer_norm_z=:layer_norm_z, linear_z=:linear_z, linear_v=:linear_v, linear_g=:linear_g, linear_out=:linear_o))

sync_af3_pwa!(args...) =
    sync_pwa!(args...; ref=(layer_norm_m=:layer_norm_m, layer_norm_z=:layer_norm_z,
        linear_z=:linear_z, linear_v=:linear_v, linear_g=:linear_g, linear_out=:linear_o))

# Attribute names are the REAL `boltz.model.layers.pair_averaging.PairWeightedAveraging`'s
# (norm_m/norm_z/proj_*), not the hand-copied reference's (m_norm/z_norm/*_proj) this used to
# target. `linear_v` is upstream's `proj_m`: the value projection is taken from `m`.
sync_boltz2_pwa!(args...) =
    sync_pwa!(args...; ref=(layer_norm_m=:norm_m, layer_norm_z=:norm_z, linear_z=:proj_z,
        linear_v=:proj_m, linear_g=:proj_g, linear_out=:proj_o))

function sync_pwa!(py::Py, jl::NamedTuple; ref::NamedTuple)
    sync_layernorm!(getproperty(py, ref.layer_norm_m), jl.layer_norm_m)
    sync_layernorm!(getproperty(py, ref.layer_norm_z), jl.layer_norm_z)
    sync_dense!(getproperty(py, ref.linear_z), jl.linear_z)
    sync_dense!(getproperty(py, ref.linear_v), jl.linear_v)
    sync_dense!(getproperty(py, ref.linear_g), jl.linear_g)
    sync_dense!(getproperty(py, ref.linear_out), jl.linear_out)

    return nothing
end

function sync_af3_msa_row_attention_with_pair_bias!(py::Py, ps::NamedTuple)
    sync_layernorm!(py.layer_norm_m, ps.layer_norm_in)
    sync_layernorm!(py.layer_norm_z, ps.layer_norm_z)
    sync_dense!(py.linear_z, ps.linear_z)
    sync_af3_attention!(py.mha, ps.mha)
    return nothing
end

function sync_af3_cross_attention_pair_bias!(py::Py, ps::NamedTuple)
    if isempty(ps.linear_out)
        sync_layernorm!(py.layer_norm_a_q, ps.layer_norm_a_q)
        sync_layernorm!(py.layer_norm_a_k, ps.layer_norm_a_k)
    else
        sync_af3_adaln!(py.layer_norm_a_q, ps.layer_norm_a_q)
        sync_af3_adaln!(py.layer_norm_a_k, ps.layer_norm_a_k)
        sync_dense!(py.linear_ada_out, ps.linear_out)
    end
    sync_dense!(py.linear_z, ps.linear_z)
    sync_af3_attention!(py.mha, ps.mha)
    return nothing
end

function sync_boltz2_dense!(py::Py, jl::NamedTuple)
    copy_jl_ps_to_py!(py.weight, jl.weight)
    if :bias ∈ keys(jl) && (pyhasattr(py, "bias") && !_isnone(py.bias))
        copy_jl_ps_to_py!(py.bias, jl.bias)
    end
    return nothing
end

function sync_boltz2_attention!(py::Py, ps::NamedTuple)
    if :weight ∈ keys(ps.qkv) # is_fused
        throw(ErrorException("Not implemented."))
    else
        sync_boltz2_dense!(py.proj_q, ps.qkv.q)
        sync_boltz2_dense!(py.proj_k, ps.qkv.k)
        sync_boltz2_dense!(py.proj_v, ps.qkv.v)
    end
    sync_boltz2_dense!(py.proj_o, ps.out)

    if !isempty(ps.gate)
        sync_boltz2_dense!(py.proj_g, ps.gate)
    end

    return nothing
end

function sync_boltz2_attention_pair_bias!(py::Py, ps::NamedTuple)
    # Boltz2 AttentionPairBias reference doesn't have layer_norm_in inside the module.
    # We sync only the parts that exist in the reference.
    # `proj_z` is Sequential(LayerNorm, Linear, Rearrange) — 0-based, Python's own indices.
    # (These read [1]/[2] under PyCall, which subtracted 1; leaving them would have handed the
    # LINEAR to `sync_layernorm!`.)
    sync_layernorm!(py.proj_z[0], ps.layer_norm_z)
    sync_boltz2_dense!(py.proj_z[1], ps.linear_z)


    sync_boltz2_attention!(py, ps.mha)

    return nothing
end

function sync_triangle_attention!(py_att::Py, jl_ps::NamedTuple)
    sync_layernorm!(py_att.layer_norm, jl_ps.layer_norm)
    sync_dense!(py_att.linear, jl_ps.linear)
    sync_af3_attention!(py_att.mha, jl_ps.mha)
end

function sync_triangle_multiplication!(py_mul::Py, jl_ps::NamedTuple)
    sync_layernorm!(py_mul.layer_norm_in, jl_ps.layer_norm)

    if isempty(jl_ps.core.glu_ab.gate)
        # Fused GLU: weight is [2*C_hidden, C_in]; split into projection and gate halves
        W = jl_ps.core.glu_ab.linear.weight
        C_out = size(W, 1) ÷ 2
        ps_p = (; weight=view(W, 1:C_out, :))
        ps_g = (; weight=view(W, C_out+1:2*C_out, :))
        if :bias ∈ keys(jl_ps.core.glu_ab.linear)
            b = jl_ps.core.glu_ab.linear.bias
            ps_p = merge(ps_p, (; bias=view(b, 1:C_out)))
            ps_g = merge(ps_g, (; bias=view(b, C_out+1:2*C_out)))
        end
        sync_dense!(py_mul.linear_ab_p, ps_p)
        sync_dense!(py_mul.linear_ab_g, ps_g)
    else
        sync_dense!(py_mul.linear_ab_p, jl_ps.core.glu_ab.linear)
        sync_dense!(py_mul.linear_ab_g, jl_ps.core.glu_ab.gate)
    end

    sync_layernorm!(py_mul.layer_norm_out, jl_ps.core.layer_norm_out)
    sync_dense!(py_mul.linear_z,          jl_ps.core.glu_out.linear)
    sync_dense!(py_mul.linear_g,          jl_ps.core.glu_out.gate)
end