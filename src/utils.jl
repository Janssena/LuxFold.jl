resolve_defaults(defaults::NamedTuple, ::NTuple{N,Symbol}) where N = defaults

resolve_defaults(default::Bool, keys::NTuple{N,Symbol}) where N = 
    NamedTuple{keys}(ntuple(_ -> default, N))

function resolve_defaults(defaults::Tuple{Bool, <:NamedTuple}, _keys::NTuple{N,Symbol}) where N
    value, nt = defaults
    default_keys = filter(!∈(keys(nt)), _keys)
    defaults = NamedTuple{default_keys}(ntuple(_ -> value, length(default_keys)))
    return merge(nt, defaults)
end

function block_array(x::AbstractArray{T, 3}, block_size::Int) where T
    C, N, B = size(x)
    num_blocks = ceil(Int, N / block_size)
    pad_len = num_blocks * block_size - N
    
    x = pad_array(x, pad_len)
    
    return reshape(x, C, block_size, num_blocks, B)
end

function unblock_array(x::AbstractArray{T, 4}, N::Int) where T
    C, block_size, num_blocks, B = size(x)
    x = reshape(x, C, block_size * num_blocks, B)
    return x[:, 1:N, :]
end

function pad_array(x::AbstractArray{T, 3}, pad_len::Int) where T
    if pad_len > 0
        C, _, B = size(x)
        padding = zeros(T, C, pad_len, B)
        return cat(x, padding; dims=2)
    else 
        return x
    end
end