function dropout_rowwise(x::AbstractArray{T, 4}, p, training) where T
    if !training || p == 0
        return x
    end
    # x: [C, S, N, B] -> dropout rows (dimension S)
    C, S, N, B = size(x)
    mask = rand(T, 1, S, 1, B) .> p
    return x .* mask ./ (1 - p)
end

function dropout_columnwise(x::AbstractArray{T, 4}, p, training) where T
    if !training || p == 0
        return x
    end
    # x: [C, S, N, B] -> dropout columns (dimension N)
    C, S, N, B = size(x)
    mask = rand(T, 1, 1, N, B) .> p
    return x .* mask ./ (1 - p)
end

# Helper to mimic Python's getattr with default
function getattr(obj, sym, default)
    return hasfield(typeof(obj), sym) ? getfield(obj, sym) : default
end
