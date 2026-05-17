struct MSATransition <: Lux.AbstractExplicitContainerLayer{(:layer_norm, :linear_up, :linear_down)}
    layer_norm::Lux.LayerNorm
    linear_up::Lux.Dense
    linear_down::Lux.Dense
end

function MSATransition(config::NamedTuple; c_in=nothing, n=nothing)
    c_in = isnothing(c_in) ? config.c_m : c_in
    n = isnothing(n) ? config.msa_transition_n : n
    return MSATransition(
        Lux.LayerNorm((c_in,)),
        Lux.Dense(c_in, n * c_in),
        Lux.Dense(n * c_in, c_in)
    )
end

function (m::MSATransition)(x, ps, st)
    x_norm, st_ln = m.layer_norm(x, ps.layer_norm, st.layer_norm)
    x_up, st_up = m.linear_up(x_norm, ps.linear_up, st.linear_up)
    x_down, st_down = m.linear_down(relu.(x_up), ps.linear_down, st.linear_down)
    return x_down, (layer_norm=st_ln, linear_up=st_up, linear_down=st_down)
end
