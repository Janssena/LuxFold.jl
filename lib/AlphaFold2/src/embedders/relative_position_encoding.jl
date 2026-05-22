struct RelativePositionEncoding{L,UM} <: Lux.AbstractLuxContainerLayer{(:linear,)}
    linear::L
    relpos_k::Int
    max_relative_chain::Int
    is_multimer::UM
end

function RelativePositionEncoding(c_z::Int, relpos_k::Int=32; is_multimer=false, max_relative_chain=2, use_bias=true)
    no_bins = is_multimer ? (2*relpos_k+2) + 1 + (2*max_relative_chain+2) : 2*relpos_k+1
    return RelativePositionEncoding(
        Lux.Dense(no_bins => c_z; use_bias),
        relpos_k,
        max_relative_chain,
        static(is_multimer),
    )
end

function (l::RelativePositionEncoding{<:Any, <:Static.False})(ri, ps, st)
    N, B = size(ri)
    T = eltype(ps.linear.weight)
    k = l.relpos_k
    no_bins = 2k + 1

    d = T.(reshape(ri, N, 1, B) .- reshape(ri, 1, N, B))
    offset = Int.(clamp.(d, -k, k)) .+ (k + 1)

    oh = zeros(T, no_bins, N, N, B)
    for i in CartesianIndices(offset)
        oh[offset[i], i] = one(T)
    end

    y, st_linear = l.linear(oh, ps.linear, st.linear)
    return y, merge(st, (; linear=st_linear))
end

function (l::RelativePositionEncoding{<:Any, <:Static.True})(ri, asym_id, entity_id, sym_id, ps, st)
    N, B = size(ri)
    k = l.relpos_k
    mc = l.max_relative_chain
    T = eltype(ps.linear.weight)

    n_bins_off = 2k + 2
    n_bins_sym = 2mc + 2

    d = T.(reshape(ri, N, 1, B) .- reshape(ri, 1, N, B))
    offset = clamp.(d, -k, k) .+ k
    asym_same = reshape(asym_id, N, 1, B) .== reshape(asym_id, 1, N, B)
    offset_bin = Int.(round.(ifelse.(asym_same, offset, T(2k + 1)))) .+ 1

    oh_offset = zeros(T, n_bins_off, N, N, B)
    for i in CartesianIndices(offset_bin)
        oh_offset[offset_bin[i], i] = one(T)
    end

    entity_same = T.(reshape(entity_id, N, 1, B) .== reshape(entity_id, 1, N, B))
    entity_same = reshape(entity_same, 1, N, N, B)

    sym_diff = T.(reshape(sym_id, N, 1, B) .- reshape(sym_id, 1, N, B))
    sym_clipped = clamp.(sym_diff, -mc, mc) .+ mc
    entity_same_bool = reshape(entity_id, N, 1, B) .== reshape(entity_id, 1, N, B)
    sym_bin = Int.(round.(ifelse.(entity_same_bool, sym_clipped, T(2mc + 1)))) .+ 1

    oh_sym = zeros(T, n_bins_sym, N, N, B)
    for i in CartesianIndices(sym_bin)
        oh_sym[sym_bin[i], i] = one(T)
    end

    cat_feat = vcat(oh_offset, entity_same, oh_sym)

    y, st_linear = l.linear(cat_feat, ps.linear, st.linear)
    return y, merge(st, (; linear=st_linear))
end
