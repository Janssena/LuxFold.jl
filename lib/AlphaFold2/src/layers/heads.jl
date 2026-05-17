struct DistogramHead <: Lux.AbstractExplicitContainerLayer{(:linear,)}
    linear::Lux.Dense
end

function DistogramHead(config::NamedTuple)
    return DistogramHead(Lux.Dense(config.c_z, 64))
end

function (m::DistogramHead)(z, ps, st)
    z_sym = (z .+ permutedims(z, (1, 3, 2, 4))) ./ 2
    out, st_lin = m.linear(z_sym, ps.linear, st.linear)
    return out, (linear=st_lin,)
end

struct PLDDTHead <: Lux.AbstractExplicitContainerLayer{(:linear,)}
    linear::Lux.Dense
end

function PLDDTHead(config::NamedTuple)
    return PLDDTHead(Lux.Dense(config.c_s, 50)) # 50 bins
end

function (m::PLDDTHead)(s, ps, st)
    out, st_lin = m.linear(s, ps.linear, st.linear)
    return out, (linear=st_lin,)
end

struct MaskedMSAHead <: Lux.AbstractExplicitContainerLayer{(:linear,)}
    linear::Lux.Dense
end

function MaskedMSAHead(config::NamedTuple)
    return MaskedMSAHead(Lux.Dense(config.c_m, 23)) # 23 amino acid types
end

function (m::MaskedMSAHead)(m_rep, ps, st)
    out, st_lin = m.linear(m_rep, ps.linear, st.linear)
    return out, (linear=st_lin,)
end

struct ExperimentallyResolvedHead <: Lux.AbstractExplicitContainerLayer{(:linear,)}
    linear::Lux.Dense
end

function ExperimentallyResolvedHead(config::NamedTuple)
    return ExperimentallyResolvedHead(Lux.Dense(config.c_s, 37)) # 37 atom types
end

function (m::ExperimentallyResolvedHead)(s, ps, st)
    out, st_lin = m.linear(s, ps.linear, st.linear)
    return out, (linear=st_lin,)
end
