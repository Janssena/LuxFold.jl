const PyCreateNoiseSchedule = pyimport("openfold3.core.model.structure.diffusion_module").create_noise_schedule
const _torch_sd = pyimport("torch")
using AlphaFold3: DiffusionTransformer

@testset "af3_noise_schedule" begin
    n = 20
    σd, smax, smin, p = 16.0, 160.0, 4e-4, 7
    sched_jl = af3_noise_schedule(n; sigma_data=σd, s_max=smax, s_min=smin, p=p)
    sched_py = PyCreateNoiseSchedule(no_rollout_steps=n, sigma_data=σd, s_max=smax, s_min=smin,
                                     p=p, dtype=_torch_sd.float64, device="cpu")

    @test length(sched_jl) == n + 1
    @test issorted(sched_jl; rev=true)                 # descending
    @test sched_jl[1] ≈ σd * smax
    @test sched_jl ≈ to_jl(sched_py)                   # Python parity (Float64)
end

@testset "SampleDiffusion" begin
    rng = Random.Xoshiro(42)
    N_token, N_atom, B = 4, 16, 1
    nq, nk = 4, 8
    apt = N_atom ÷ N_token
    ca, cap, ch = 8, 4, 4
    no_heads, no_blocks, n_trans = 2, 2, 2
    cre, crnc = 128, 256
    c_s_input, c_s, c_z, c_token, c_fourier = 12, 16, 6, 12, 32
    max_idx, max_chain = 32, 2
    num_relpos = 2 * (2max_idx + 2) + 1 + (2max_chain + 2)
    sigma_data = 16f0
    T = Float32

    # --- build a small DiffusionModule (sub-layers parity-validated elsewhere) ---
    dm = DiffusionModule(;
        diffusion_conditioning=DiffusionConditioning(; chn_s_input=c_s_input, chn_s=c_s, chn_z=c_z, chn_fourier_emb=c_fourier,
            max_relative_idx=max_idx, max_relative_chain=max_chain, sigma_data, num_relpos_dims=num_relpos),
        atom_attn_enc=AtomAttentionEncoder(; chn_atom=ca, chn_atom_pair=cap, chn_token=c_token,
            chn_hidden=ch, no_heads, no_blocks, n_transition=n_trans, n_query=nq, n_key=nk,
            add_noisy_pos=true, chn_single=c_s, chn_pair=c_z, chn_ref_element=cre, chn_ref_name_chars=crnc),
        atom_attn_dec=AtomAttentionDecoder(; chn_atom=ca, chn_atom_pair=cap, chn_token=c_token,
            chn_hidden=ch, no_heads, no_blocks, n_transition=n_trans, n_query=nq, n_key=nk),
        diffusion_transformer=DiffusionTransformer(; chn_a=c_token, chn_cond=c_s, chn_pair=c_z,
            chn_hidden=ch, no_heads, no_blocks, n_transition=n_trans, use_ada_layer_norm=true),
        chn_s=c_s, chn_token=c_token, sigma_data)

    # gamma_0 = 0 ⇒ no churn noise (ε ∝ sqrt(t² - σ²) = 0), so the only randomness is the
    # init noise + centre_random_augmentation — keeps the one-step check tractable.
    s = SampleDiffusion(dm; gamma_0=0f0, gamma_min=0f0, noise_scale=1f0, step_scale=1.5f0)
    ps, st = Lux.setup(rng, s) |> convert_types(T)

    # batch + trunk inputs
    half = N_token ÷ 2
    batch = (;
        ref_pos=randn(rng, T, 3, N_atom, B), ref_charge=randn(rng, T, 1, N_atom, B),
        ref_mask=rand(rng, Bool, 1, N_atom, B), ref_element=randn(rng, T, cre, N_atom, B),
        ref_atom_name_chars=randn(rng, T, crnc, N_atom, B), ref_space_uid=rand(rng, 1:5, 1, N_atom, B),
        atom_mask=trues(N_atom, B), token_mask=trues(N_token, B),
        num_atoms_per_token=fill(apt, N_token, B),
        atom_to_token_index=repeat(reduce(vcat, [fill(t, apt) for t in 1:N_token]), 1, B),
        residue_index=repeat(reshape(collect(1:N_token), N_token, 1), 1, B),
        token_index=repeat(reshape(collect(1:N_token), N_token, 1), 1, B),
        asym_id=repeat(reshape([fill(1, half); fill(2, half)], N_token, 1), 1, B),
        sym_id=repeat(reshape([fill(1, half); fill(2, half)], N_token, 1), 1, B),
        entity_id=ones(Int, N_token, B))
    si_input  = randn(rng, T, c_s_input, N_token, B)
    si_trunk  = randn(rng, T, c_s, N_token, B)
    zij_trunk = randn(rng, T, c_z, N_token, N_token, B)
    schedule  = af3_noise_schedule(3; sigma_data=sigma_data, s_max=160f0, s_min=4f-4, p=7)

    @testset "output shape" begin
        xl, _ = s(Random.Xoshiro(7), batch, schedule, si_input, si_trunk, zij_trunk, ps, st)
        @test size(xl) == (3, N_atom, B)
    end

    @testset "reproducibility (same rng ⇒ identical)" begin
        a, _ = s(Random.Xoshiro(9), batch, schedule, si_input, si_trunk, zij_trunk, ps, st)
        b, _ = s(Random.Xoshiro(9), batch, schedule, si_input, si_trunk, zij_trunk, ps, st)
        @test a == b
    end

    @testset "one-step churn/Euler arithmetic" begin
        # Replicate the single loop iteration manually with an identical rng draw sequence;
        # the struct's loop must reproduce it. (DiffusionModule parity ⇒ Python equivalence.)
        sched1 = [σ for σ in schedule[1:2]]   # 1 step
        xl_struct, _ = s(Random.Xoshiro(123), batch, sched1, si_input, si_trunk, zij_trunk, ps, st)

        rng2 = Random.Xoshiro(123)
        xl = sched1[1] .* randn(rng2, T, 3, N_atom, B)
        xl = centre_random_augmentation(rng2, xl, batch.atom_mask)
        t  = sched1[1]                                   # γ=0 ⇒ t = σ0
        ε  = 1f0 .* sqrt(t^2 - sched1[1]^2) .* randn(rng2, T, size(xl)...)   # = 0 (draw kept in sync)
        xl_noisy = xl .+ ε
        denoised, _ = dm(batch, xl_noisy, fill(t, B), si_input, si_trunk, zij_trunk,
                         ps.diffusion_module, st.diffusion_module)
        delta  = (xl_noisy .- denoised) ./ t
        xl_man = xl_noisy .+ 1.5f0 .* (sched1[2] - t) .* delta

        @test xl_struct ≈ xl_man
    end
end
