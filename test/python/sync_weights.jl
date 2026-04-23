function copy_jl_ps_to_py!(py::PyObject, jl::AbstractArray{T}; swap_batch_dim=false) where T 
    @assert py"type"(py) == torch.nn.Parameter "Passed PyObject is not a torch.nn.Parameter"
    @assert py.shape == size(jl) "Shape of py $(py.shape) and jl $(size(jl)) do not match."
    py.data = to_py(jl; swap_batch_dim)
    
    return nothing
end

function sync_dense!(py::PyObject, jl::NamedTuple)
    @assert py"hasattr"(py, "weight") "PyObject (Linear) does not have weight attribute."
    py_has_bias = py"hasattr"(py, "bias") && !isnothing(py.bias)
    jl_has_bias = :bias ∈ keys(jl)
    @assert py_has_bias == jl_has_bias "PyObject (Linear) and NamedTuple have non-matching bias attributes (py = $(py_has_bias), jl = $(jl_has_bias))."

    copy_jl_ps_to_py!(py.weight, jl.weight)
    if :bias ∈ keys(jl) && (py"hasattr"(py, "bias") && !isnothing(py.bias))
        copy_jl_ps_to_py!(py.bias, jl.bias)
    end

    return nothing
end

function sync_layernorm!(py::PyObject, jl::NamedTuple)
    py_has_weight = py"hasattr"(py, "weight") && !isnothing(py.weight)
    jl_has_weight = :scale ∈ keys(jl)
    @assert py_has_weight == jl_has_weight "PyObject (LayerNorm) and NamedTuple have non-matching weight attributes (py = $(py_has_weight), jl = $(jl_has_weight))."
    py_has_bias = py"hasattr"(py, "bias") && !isnothing(py.bias)
    jl_has_bias = :bias ∈ keys(jl)
    @assert py_has_bias == jl_has_bias "PyObject (LayerNorm) and NamedTuple have non-matching bias attributes (py = $(py_has_bias), jl = $(jl_has_bias))."

    if :scale ∈ keys(jl)
        copy_jl_ps_to_py!(py.weight, vec(jl.scale))
    end
    if :bias ∈ keys(jl)
        copy_jl_ps_to_py!(py.bias, vec(jl.bias))
    end

    return nothing
end


function sync_glu!(py::PyObject, jl::NamedTuple; ref=(linear = :linear_z, gate = :linear_g))
    @assert py"hasattr"(py, ref.linear) "PyObject does not have the referenced linear attribute ($(ref.linear))."
    @assert py"hasattr"(py, ref.gate) "PyObject does not have the referenced gate attribute ($(ref.gate))." 
    @assert (py"hasattr"(py[ref.linear], "bias") && !isnothing(py[ref.linear].bias)) == (:bias ∈ keys(jl.linear)) "PyObject linear and NamedTuple have non-matching bias attributes."
    gate_should_have_bias_keys = isempty(jl.gate) ? (:bias ∈ keys(jl.linear)) : (:bias ∈ keys(jl.gate))
    @assert (py"hasattr"(py[ref.gate], "bias") && !isnothing(py[ref.gate].bias)) == gate_should_have_bias_keys "PyObject gate and NamedTuple have non-matching bias attributes."

    jl_unfused = _unfuse(jl)
    sync_dense!(py[ref.linear], jl_unfused.linear)
    sync_dense!(py[ref.gate], jl_unfused.gate)

    return nothing
end

function _unfuse(jl::NamedTuple{(:linear, :gate)})
    if !isempty(jl.gate)
        return jl
    end

    w = jl.linear.weight
    chn = size(w, 1) ÷ 2

    ps = (
        linear = (weight = view(w, 1:chn, :), ),
        gate = (weight = view(w, chn+1:2*chn, :), ),
    )

    if :bias ∈ keys(jl.linear)
        b = jl.linear.bias
        ps = (
            linear = merge(ps.linear, (bias = view(b, 1:chn), )),
            gate = merge(ps.gate, (bias = view(b, chn+1:2*chn), )),
        )
    end

    return ps
end

sync_af3_adaln!(args...) = 
    sync_adaln!(args...; ref=(layer_norm_a = :layer_norm_a, layer_norm_s = :layer_norm_s, shift = :linear_s, gate = :linear_g))

sync_boltz2_adaln!(args...) = 
    sync_adaln!(args...; ref=(layer_norm_a = :a_norm, layer_norm_s = :s_norm, shift = :s_bias, gate = :s_scale))

function sync_adaln!(py::PyObject, jl::NamedTuple; ref::NamedTuple)
    sync_layernorm!(py[ref.layer_norm_a], jl.layer_norm_a)
    sync_layernorm!(py[ref.layer_norm_s], jl.layer_norm_s)
    sync_dense!(py[ref.shift], jl.shift)
    sync_dense!(py[ref.gate], jl.gate)

    return nothing
end

function sync_af3_attention!(py::PyObject, ps::NamedTuple)
    if :weight ∈ keys(ps.qkv) # is_fused
        throw(ErrorException("Not implemented."))
    else
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

function sync_af3_attention_pair_bias!(py::PyObject, ps::NamedTuple)   
    if isempty(ps.linear_out)
        sync_layernorm!(py.layer_norm_a, ps.layer_norm_in)
    else
        sync_af3_adaln!(py.layer_norm_a, ps.layer_norm_in)
        sync_dense!(py.linear_ada_out, ps.linear_out)    
    end
    sync_layernorm!(py.layer_norm_z, ps.layer_norm_z)
    sync_dense!(py.linear_z, ps.linear_z)

    sync_af3_attention!(py.mha, ps.mha)

    return nothing
end

sync_af3_opm!(args...) = 
    sync_opm!(args...; ref=(layer_norm = :layer_norm, linear_1 = :linear_1, linear_2 = :linear_2, linear_out = :linear_out))

sync_af2_opm!(args...) = sync_af3_opm!(args...)

sync_boltz2_opm!(args...) = 
    sync_opm!(args...; ref=(layer_norm = :norm, linear_1 = :proj_a, linear_2 = :proj_b, linear_out = :proj_o))

function sync_opm!(py::PyObject, jl::NamedTuple; ref::NamedTuple)
    sync_layernorm!(py[ref.layer_norm], jl.layer_norm)
    sync_dense!(py[ref.linear_1], jl.linear1)
    sync_dense!(py[ref.linear_2], jl.linear2)
    sync_dense!(py[ref.linear_out], jl.linear_out)

    return nothing
end

sync_af3_pwa!(args...) = 
    sync_pwa!(args...; ref=(layer_norm_m = :layer_norm_m, layer_norm_z = :layer_norm_z, linear_z = :linear_z, linear_v = :linear_v, linear_g = :linear_g, linear_out = :linear_o))

sync_boltz2_pwa!(args...) = 
    sync_pwa!(args...; ref=(layer_norm_m = :m_norm, layer_norm_z = :z_norm, linear_z = :z_proj, linear_v = :v_proj, linear_g = :g_proj, linear_out = :o_proj))

function sync_pwa!(py::PyObject, jl::NamedTuple; ref::NamedTuple)
    sync_layernorm!(py[ref.layer_norm_m], jl.layer_norm_m)
    sync_layernorm!(py[ref.layer_norm_z], jl.layer_norm_z)
    sync_dense!(py[ref.linear_z], jl.linear_z)
    sync_dense!(py[ref.linear_v], jl.linear_v)
    sync_dense!(py[ref.linear_g], jl.linear_g)
    sync_dense!(py[ref.linear_out], jl.linear_out)

    return nothing
end