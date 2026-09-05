convert_types(::Type{Float64}) = Lux.f64
convert_types(::Type{Float32}) = Lux.f32
convert_types(::Type{Float16}) = Lux.f16

py_dtype(::Type{Float64}) = pyimport("torch").float64
py_dtype(::Type{Float32}) = pyimport("torch").float32
py_dtype(::Type{Float16}) = pyimport("torch").float16
py_dtype(::Type{Int32}) = pyimport("torch").int32
py_dtype(::Type{Int64}) = pyimport("torch").int64
py_dtype(::Type{Bool}) = pyimport("torch").bool

_swap_batch_dim(x::AbstractVector) = x
_swap_batch_dim(x::AbstractArray{T, N}) where {T,N} = permutedims(x, (N, 2:N-1..., 1))

# `np.asarray` explicitly, where PyCall converted a Julia array implicitly. Verified to produce a
# byte-identical tensor to the PyCall path in BOTH `swap_batch_dim` modes: shape and flattened
# element order match, and `to_jl(to_py(x)) == x` round-trips either way. So no call site's
# `swap_batch_dim` argument changes meaning — in particular `swap_batch_dim = false` still does
# not transpose.
function to_py(x::AbstractArray{T}; swap_batch_dim=false, device="cpu") where T
    x_py = swap_batch_dim ? _swap_batch_dim(x) : x

    np = pyimport("numpy")
    return pyimport("torch").from_numpy(np.asarray(collect(x_py))).to(py_dtype(T)).to(device)
end

# `pyconvert` explicitly, where PyCall converted the numpy return implicitly. Without it this
# returns a `Py` and every downstream `≈` silently compares Python objects rather than numbers.
function to_jl(x::Py; device="cpu", swap_batch_dim=false)
    x_py = device == "cpu" ? x.detach().cpu() : x.detach().gpu()
    x_jl = pyconvert(Array, x_py.numpy())
    return swap_batch_dim ? _swap_batch_dim(x_jl) : x_jl
end