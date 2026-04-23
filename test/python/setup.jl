import Pkg

python_path = abspath(joinpath(@__DIR__, "..", "..", ".venv", "bin", "python"))
ENV["PYTHON"] = python_path
Pkg.build("PyCall")

using PyCall

const torch = pyimport("torch")

py_dtype(::Type{Float64}) = torch.float64
py_dtype(::Type{Float32}) = torch.float32
py_dtype(::Type{Float16}) = torch.float16
py_dtype(::Type{Int32}) = torch.int32
py_dtype(::Type{Int64}) = torch.int64
py_dtype(::Type{Bool}) = torch.bool

_swap_batch_dim(x::AbstractVector) = x
_swap_batch_dim(x::AbstractArray{T, N}) where {T,N} = permutedims(x, (N, 2:N-1..., 1))

function to_py(x::AbstractArray{T}; swap_batch_dim=false, device="cpu") where T
    x_py = swap_batch_dim ? _swap_batch_dim(x) : x

    return torch.from_numpy(collect(x_py)).to(py_dtype(T)).to(device)
end

function to_jl(x::PyObject; device="cpu", swap_batch_dim=false)
    x_jl = device == "cpu" ? x.detach().cpu() : x.detach().gpu()
    x_jl = x_jl.numpy()
    return swap_batch_dim ? _swap_batch_dim(x_jl) : x_jl
end

function copy_jl_ps_to_py!(py::PyObject, jl::AbstractArray{T}; swap_batch_dim=false) where T 
    @assert py"type"(py) == torch.nn.Parameter "Passed PyObject is not a torch.nn.Parameter"
    @assert py.shape == size(jl) "Shape of py $(py.shape) and jl $(size(jl)) do not match."
    py.data = to_py(jl; swap_batch_dim)
    
    return nothing
end

function sync_dense!(py::PyObject, jl::NamedTuple)
    @assert py"hasattr"(py, "weight") "PyObject (Linear) does not have weight attribute."
    @assert (py"hasattr"(py, "bias") && !isnothing(py.bias)) == (:bias ∈ keys(jl)) "PyObject (Linear) and NamedTuple have non-matching bias attributes."

    copy_jl_ps_to_py!(py.weight, jl.weight)
    if :bias ∈ keys(jl) && (py"hasattr"(py, "bias") && !isnothing(py.bias))
        copy_jl_ps_to_py!(py.bias, jl.bias)
    end

    return nothing
end

function sync_layernorm!(py::PyObject, jl::NamedTuple)
    @assert (py"hasattr"(py, "weight") && !isnothing(py.weight)) == (:scale ∈ keys(jl)) "PyObject (LayerNorm) and NamedTuple have non-matching weight attributes."
    @assert (py"hasattr"(py, "bias") && !isnothing(py.bias)) == (:bias ∈ keys(jl)) "PyObject (LayerNorm) and NamedTuple have non-matching bias attributes."

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