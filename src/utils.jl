resolve_defaults(defaults::NamedTuple, ::NTuple{N,Symbol}) where N = defaults

resolve_defaults(default::Bool, keys::NTuple{N,Symbol}) where N =
    NamedTuple{keys}(ntuple(_ -> default, N))

function resolve_defaults(defaults::Tuple{Bool,<:NamedTuple}, _keys::NTuple{N,Symbol}) where N
    value, nt = defaults
    default_keys = filter(!∈(keys(nt)), _keys)
    defaults = NamedTuple{default_keys}(ntuple(_ -> value, length(default_keys)))
    return merge(nt, defaults)
end