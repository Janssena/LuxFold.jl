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

# include weight syncing code
include("sync_weights.jl");