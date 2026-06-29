
"""
    af3_noise_schedule(no_rollout_steps; sigma_data=16f0, s_max=160f0, s_min=4f-4, p=7)

AF3 noise schedule — 1:1 port of `openfold3...diffusion_module.create_noise_schedule`.
Returns a descending `Vector` of `no_rollout_steps + 1` σ values:

    t = (0,1,…,no_rollout_steps) / no_rollout_steps
    σ = sigma_data * (s_max^(1/p) + t*(s_min^(1/p) - s_max^(1/p)))^p
"""
function af3_noise_schedule(no_rollout_steps::Int; sigma_data::T=16f0, s_max::T=160f0, s_min::T=T(4f-4), p::Int=7) where {T<:Real}
    ts = T.(0:no_rollout_steps) ./ T(no_rollout_steps)
    inv_p = one(T) / T(p)
    return @. sigma_data * (s_max^inv_p + ts * (s_min^inv_p - s_max^inv_p)) ^ p
end

"""
    SampleDiffusion(diffusion_module; gamma_0, gamma_min, noise_scale, step_scale)

Iterative EDM denoising/churn sampler (AF3) — 1:1 port of
`openfold3...diffusion_module.SampleDiffusion`, as a manual loop over an `af3_noise_schedule`
(no `StochasticDiffEq` dependency).
"""
struct SampleDiffusion{DM} <: Lux.AbstractLuxContainerLayer{(:diffusion_module,)}
    diffusion_module::DM
    gamma_0::Float32
    gamma_min::Float32
    noise_scale::Float32
    step_scale::Float32
end

SampleDiffusion(diffusion_module; gamma_0, gamma_min, noise_scale, step_scale) = SampleDiffusion(
    diffusion_module, 
    Float32(gamma_0), 
    Float32(gamma_min),
    Float32(noise_scale), 
    Float32(step_scale)
)

"""
    (l::SampleDiffusion)(rng, batch, noise_schedule, s_input, s_trunk, z_trunk, ps, st)

`rng::AbstractRNG` is the mandatory first positional argument (RNG convention). Sampling always
conditions on the trunk reps `s_trunk`/`z_trunk` (matching AF3 inference).
Returns `(xl [3, N_atom, B], st)`.
"""
function (l::SampleDiffusion)(rng::Random.AbstractRNG, batch::NamedTuple, noise_schedule, s_input, s_trunk::AbstractArray{T}, z_trunk, ps, st) where T
    atom_mask = batch.atom_mask                              # [N_atom, B]
    N_atom, B = size(atom_mask)
    σ = T.(noise_schedule)
    # TODO: These should probably live in the state of the layer:
    gamma_0, gamma_min = T(l.gamma_0), T(l.gamma_min)
    noise_scale, step_scale = T(l.noise_scale), T(l.step_scale)

    xl = σ[1] .* randn(rng, T, 3, N_atom, B)                 # init noise
    st_dm = st.diffusion_module
    for τ in 1:(length(σ) - 1)
        c_tau = σ[τ + 1]
        xl = centre_random_augmentation(rng, xl, atom_mask)
        γ  = c_tau > gamma_min ? gamma_0 : zero(T)
        t  = σ[τ] * (γ + one(T))                             # churned-up noise level (scalar)
        ε  = noise_scale .* sqrt(t^2 - σ[τ]^2) .* randn(rng, T, size(xl)...)
        xl_noisy = xl .+ ε
        t_vec = similar(xl, T, B); t_vec .= t
        xl_denoised, st_dm = l.diffusion_module(
            batch, xl_noisy, t_vec, s_input, s_trunk, z_trunk,
            ps.diffusion_module, st_dm,
        )
        delta = (xl_noisy .- xl_denoised) ./ t               # note: xl_noisy, per Python
        dt = c_tau - t
        xl = @. xl_noisy + step_scale * dt * delta
    end
    return xl, merge(st, (diffusion_module=st_dm,))
end
