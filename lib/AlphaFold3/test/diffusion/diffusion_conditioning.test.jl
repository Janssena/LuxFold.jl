const PyDiffusionConditioning = pyimport("openfold3.core.model.layers.diffusion_conditioning").DiffusionConditioning

# --- sync helpers (AF3 style; SwiGLUTransition uses the shared unfused SwiGLU) ---
function sync_swiglu_transition!(py, ps)
    sync_layernorm!(py.layer_norm, ps.layer_norm)         # regular LayerNorm (scale + offset)
    sync_dense!(py.swiglu.linear_a, ps.swiglu.gate)       # swish branch
    sync_dense!(py.swiglu.linear_b, ps.swiglu.linear)     # value branch
    sync_dense!(py.linear_out, ps.linear_out)
    return nothing
end

function sync_diffusion_conditioning!(py, ps)
    sync_layernorm!(py.layer_norm_z, ps.layer_norm_z)     # offset-free
    sync_dense!(py.linear_z, ps.linear_z)
    for (i, blk) in enumerate(py.transition_z)
        sync_swiglu_transition!(blk, ps.transition_z[i])
    end
    sync_layernorm!(py.layer_norm_s, ps.layer_norm_s)
    sync_dense!(py.linear_s, ps.linear_s)
    sync_layernorm!(py.layer_norm_n, ps.layer_norm_n)
    sync_dense!(py.linear_n, ps.linear_n)
    for (i, blk) in enumerate(py.transition_s)
        sync_swiglu_transition!(blk, ps.transition_s[i])
    end
    return nothing
end

@testset "DiffusionConditioning" begin
    rng = Random.Xoshiro(42)
    N, B = 16, 2
    c_s_input, c_s, c_z, c_fourier = 12, 16, 6, 32
    max_idx, max_chain = 32, 2
    num_relpos = 2 * (2max_idx + 2) + 1 + (2max_chain + 2)   # 139
    sigma_data = 16f0

    function _batch(rng, T, token_mask)
        half = N ÷ 2
        residue_index = repeat(reshape(collect(1:N), N, 1), 1, B)
        token_index   = copy(residue_index)
        asym_id       = repeat(reshape([fill(1, half); fill(2, half)], N, 1), 1, B)
        sym_id        = copy(asym_id)
        entity_id     = ones(Int, N, B)
        batch_jl = (; residue_index, token_index, asym_id, sym_id, entity_id, token_mask)
        batch_py = Dict(
            "residue_index" => to_py(residue_index; swap_batch_dim=true),
            "token_index"   => to_py(token_index; swap_batch_dim=true),
            "asym_id"       => to_py(asym_id; swap_batch_dim=true),
            "sym_id"        => to_py(sym_id; swap_batch_dim=true),
            "entity_id"     => to_py(entity_id; swap_batch_dim=true),
            "token_mask"    => to_py(T.(token_mask); swap_batch_dim=true),
        )
        return batch_jl, batch_py
    end

    rand_mask = rand(rng, Bool, N, B); rand_mask[1, :] .= true
    mask_cfg = (("No mask", trues(N, B)), ("Random mask", rand_mask))

    for T in [Float64, Float32, Float16], (mask_name, token_mask) in mask_cfg
        @testset "$T, $mask_name" begin
            jl = DiffusionConditioning(; chn_s_input=c_s_input, chn_s=c_s, chn_z=c_z, chn_fourier_emb=c_fourier, max_relative_idx=max_idx, max_relative_chain=max_chain, sigma_data, num_relpos_dims=num_relpos)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            py = PyDiffusionConditioning(c_s_input, c_s, c_z, c_fourier, max_idx, max_chain, Float64(sigma_data))
            py.to(py_dtype(T))   # convert params + buffers (FourierEmbedding w/b) to T
            sync_diffusion_conditioning!(py, ps)
            # FourierEmbedding w/b are non-trainable STATE — sync them into st
            st = merge(st, (fourier_emb = (w = T.(to_jl(py.fourier_emb.w)),
                                           b = T.(to_jl(py.fourier_emb.b))),))

            t = rand(rng, T, B) .+ T(0.5)                # noise level σ > 0
            s_input = randn(rng, T, c_s_input, N, B)
            s_trunk = randn(rng, T, c_s, N, B)
            z_trunk = randn(rng, T, c_z, N, N, B)

            batch_jl, batch_py = _batch(rng, T, token_mask)

            out_jl, _ = jl(batch_jl, t, s_input, s_trunk, z_trunk, ps, st)
            si_jl, zij_jl = out_jl.s, out_jl.z
            si_py, zij_py = py(
                batch=batch_py, t=to_py(t; swap_batch_dim=true),
                si_input=to_py(s_input; swap_batch_dim=true),
                si_trunk=to_py(s_trunk; swap_batch_dim=true),
                zij_trunk=to_py(z_trunk; swap_batch_dim=true),
                use_conditioning=true,
            )

            atol = T == Float16 ? 2f-2 : (T == Float32 ? 1f-4 : 1e-9)
            @test isapprox(si_jl, to_jl(si_py; swap_batch_dim=true); atol=atol, rtol=atol)
            @test isapprox(zij_jl, to_jl(zij_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    # Unconditioned path: Julia `nothing` trunk reps ⇔ Python `use_conditioning=false` (zeros).
    @testset "Unconditioned (nothing trunk ⇔ use_conditioning=false)" begin
        T = Float64
        jl = DiffusionConditioning(; chn_s_input=c_s_input, chn_s=c_s, chn_z=c_z, chn_fourier_emb=c_fourier, max_relative_idx=max_idx, max_relative_chain=max_chain, sigma_data, num_relpos_dims=num_relpos)
        ps, st = Lux.setup(rng, jl) |> convert_types(T)
        py = PyDiffusionConditioning(c_s_input, c_s, c_z, c_fourier, max_idx, max_chain, Float64(sigma_data))
        py.to(py_dtype(T)); sync_diffusion_conditioning!(py, ps)
        st = merge(st, (fourier_emb = (w = T.(to_jl(py.fourier_emb.w)),
                                       b = T.(to_jl(py.fourier_emb.b))),))

        t       = rand(rng, T, B) .+ T(0.5)
        s_input = randn(rng, T, c_s_input, N, B)
        s_trunk = randn(rng, T, c_s, N, B)            # passed to Python (zeroed there); Julia uses nothing
        z_trunk = randn(rng, T, c_z, N, N, B)
        batch_jl, batch_py = _batch(rng, T, trues(N, B))

        out_jl, _ = jl(batch_jl, t, s_input, nothing, nothing, ps, st)
        si_py, zij_py = py(batch=batch_py, t=to_py(t; swap_batch_dim=true),
                           si_input=to_py(s_input; swap_batch_dim=true),
                           si_trunk=to_py(s_trunk; swap_batch_dim=true),
                           zij_trunk=to_py(z_trunk; swap_batch_dim=true),
                           use_conditioning=false)
        @test isapprox(out_jl.s, to_jl(si_py; swap_batch_dim=true); atol=1e-9, rtol=1e-9)
        @test isapprox(out_jl.z, to_jl(zij_py; swap_batch_dim=true); atol=1e-9, rtol=1e-9)
    end
end
