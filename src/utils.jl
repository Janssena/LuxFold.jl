resolve_defaults(defaults::NamedTuple, ::NTuple{N,Symbol}) where N = defaults

resolve_defaults(default::Bool, keys::NTuple{N,Symbol}) where N =
    NamedTuple{keys}(ntuple(_ -> default, N))

function resolve_defaults(defaults::Tuple{Bool,<:NamedTuple}, _keys::NTuple{N,Symbol}) where N
    value, nt = defaults
    default_keys = filter(!∈(keys(nt)), _keys)
    defaults = NamedTuple{default_keys}(ntuple(_ -> value, length(default_keys)))
    return merge(nt, defaults)
end

"""
    pad_and_block(x, block_size; dims=2, pad_val=zero(eltype(x)))

Pads the tensor `x` along dimension `dims` to a multiple of `block_size` and reshapes 
it to add a block dimension. The resulting shape will have `block_size` at `dims` 
and the number of blocks at `dims+1`.
"""
function pad_and_block(x::AbstractArray{T,N}, block_size::Int; dims::Int=2, pad_val=zero(T)) where {T,N}
    sz = size(x)
    n = sz[dims]
    n_blocks = ceil(Int, n / block_size)
    n_padded = n_blocks * block_size

    if n_padded == n
        x_padded = x
    else
        pad_sz = collect(sz)
        pad_sz[dims] = n_padded - n
        padding = fill!(similar(x, pad_sz...), pad_val)
        x_padded = cat(x, padding; dims=dims)
    end

    new_sz = (sz[1:dims-1]..., block_size, n_blocks, sz[dims+1:end]...)
    return reshape(x_padded, new_sz)
end

function pad_and_block(x::AbstractArray{T,N}, block_sizes::NTuple{M,Int}; dims::NTuple{M,Int}, pad_val=zero(T)) where {T,N,M}
    res = x
    # We apply blocking sequentially. 
    # Note that each blocking adds a dimension, so we need to adjust dims for subsequent calls.
    # However, if we block dims from highest to lowest, the lower dims indices don't change.
    # But here dims are usually (1, 2).
    # If we block dim 1: [n1, n2, ...] -> [b1, nb1, n2, ...]
    # Then dim 2 is now index 3.

    sorted_dims = sort(collect(dims); rev=true)
    for d in sorted_dims
        idx = findfirst(==(d), dims)
        res = pad_and_block(res, block_sizes[idx]; dims=d, pad_val=pad_val)
    end
    return res
end

"""
    unblock_and_slice(x, original_n; dims=2)

Inverse of `pad_and_block`. Flattens dimensions `dims` and `dims+1` and slices 
the resulting dimension `dims` to `original_n`.
"""
function unblock_and_slice(x::AbstractArray{T,N}, original_n::Int; dims::Int=2) where {T,N}
    sz = size(x)
    n_padded = sz[dims] * sz[dims+1]

    # Flatten dims and dims+1
    new_sz = (sz[1:dims-1]..., n_padded, sz[dims+2:end]...)
    x_flat = reshape(x, new_sz)

    # Slice. We use selectdim for stability.
    return collect(selectdim(x_flat, dims, 1:original_n))
end