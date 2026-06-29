const PyDiffusionModule = pyimport("openfold3.core.model.structure.diffusion_module").DiffusionModule
const _mlc_dm = pyimport("ml_collections")
using AlphaFold3: DiffusionTransformer
# sync_diffusion_conditioning! / sync_swiglu_transition! come from diffusion_conditioning.test.jl
isdefined(@__MODULE__, :sync_diffusion_conditioning!) || include("diffusion_conditioning.test.jl")

function sync_diffusion_module!(py, ps)
    sync_diffusion_conditioning!(py.diffusion_conditioning, ps.diffusion_conditioning)
    sync_atom_attention_encoder!(py.atom_attn_enc, ps.atom_attn_enc; add_noisy_pos=true)
    sync_diffusion_transformer!(py.diffusion_transformer, ps.diffusion_transformer; cross_attention=false)
    sync_atom_attention_decoder!(py.atom_attn_dec, ps.atom_attn_dec)
    sync_layernorm!(py.layer_norm_s, ps.layer_norm_s)
    sync_dense!(py.linear_s, ps.linear_s)
    sync_layernorm!(py.layer_norm_a, ps.layer_norm_a)
    return nothing
end

@testset "DiffusionModule" begin
    rng = Random.Xoshiro(42)
    N_token, N_atom, B = 8, 32, 1
    nq, nk = 4, 8
    N_blocks = cld(N_atom, nq)
    apt = N_atom ÷ N_token
    ca, cap, ch = 8, 4, 4
    no_heads, no_blocks, n_trans = 2, 2, 2
    cre, crnc = 128, 256
    c_s_input, c_s, c_z, c_token, c_fourier = 12, 16, 6, 12, 32
    max_idx, max_chain = 32, 2
    num_relpos = 2 * (2max_idx + 2) + 1 + (2max_chain + 2)   # 139
    sigma_data = 16f0

    _c_atom_ref() = _mlc_dm.ConfigDict(Dict("element" => cre, "name_chars" => crnc))

    _py_config() = _mlc_dm.ConfigDict(Dict(
        "diffusion_module" => _mlc_dm.ConfigDict(Dict(
            "c_s" => c_s, "c_token" => c_token, "sigma_data" => Float64(sigma_data))),
        "diffusion_conditioning" => Dict(
            "c_s_input" => c_s_input, "c_s" => c_s, "c_z" => c_z, "c_fourier_emb" => c_fourier,
            "max_relative_idx" => max_idx, "max_relative_chain" => max_chain,
            "sigma_data" => Float64(sigma_data), "seed_fourier_emb" => 42),
        "atom_attn_enc" => Dict(
            "c_atom_ref" => _c_atom_ref(), "c_atom" => ca, "c_atom_pair" => cap,
            "c_token" => c_token, "c_hidden" => ch, "no_heads" => no_heads, "no_blocks" => no_blocks,
            "n_transition" => n_trans, "n_query" => nq, "n_key" => nk, "use_ada_layer_norm" => true,
            "c_s" => c_s, "c_z" => c_z),
        "diffusion_transformer" => Dict(
            "c_a" => c_token, "c_s" => c_s, "c_z" => c_z, "c_hidden" => ch, "no_heads" => no_heads,
            "no_blocks" => no_blocks, "n_transition" => n_trans, "use_ada_layer_norm" => true,
            "n_query" => nothing, "n_key" => nothing, "inf" => 1e9),
        "atom_attn_dec" => Dict(
            "c_atom" => ca, "c_atom_pair" => cap, "c_token" => c_token, "c_hidden" => ch,
            "no_heads" => no_heads, "no_blocks" => no_blocks, "n_transition" => n_trans,
            "n_query" => nq, "n_key" => nk, "use_ada_layer_norm" => true),
    ))

    function _make_batch(rng, T, atom_mask, token_mask)
        ref_pos       = randn(rng, T, 3, N_atom, B)
        ref_charge    = randn(rng, T, 1, N_atom, B)
        ref_mask      = rand(rng, Bool, 1, N_atom, B)
        ref_element   = randn(rng, T, cre, N_atom, B)
        ref_chars     = randn(rng, T, crnc, N_atom, B)
        ref_space_uid = rand(rng, 1:5, 1, N_atom, B)
        num_atoms_per_token = fill(apt, N_token, B)
        atom_to_token_index = repeat(reduce(vcat, [fill(t, apt) for t in 1:N_token]), 1, B)
        half = N_token ÷ 2
        residue_index = repeat(reshape(collect(1:N_token), N_token, 1), 1, B)
        token_index   = copy(residue_index)
        asym_id       = repeat(reshape([fill(1, half); fill(2, half)], N_token, 1), 1, B)
        sym_id        = copy(asym_id)
        entity_id     = ones(Int, N_token, B)

        batch_jl = (; ref_pos, ref_charge, ref_mask, ref_element,
                    ref_atom_name_chars=ref_chars, ref_space_uid, atom_mask, token_mask,
                    num_atoms_per_token, atom_to_token_index,
                    residue_index, token_index, asym_id, sym_id, entity_id)
        batch_py = Dict(
            "ref_pos"             => to_py(ref_pos; swap_batch_dim=true),
            "ref_charge"          => to_py(ref_charge; swap_batch_dim=true).squeeze(-1),
            "ref_mask"            => to_py(ref_mask; swap_batch_dim=true).squeeze(-1),
            "ref_element"         => to_py(ref_element; swap_batch_dim=true),
            "ref_atom_name_chars" => to_py(ref_chars; swap_batch_dim=true).reshape(B, N_atom, 4, 64),
            "ref_space_uid"       => to_py(ref_space_uid; swap_batch_dim=true).squeeze(-1),
            "atom_mask"           => to_py(T.(atom_mask); swap_batch_dim=true),
            "token_mask"          => to_py(T.(token_mask); swap_batch_dim=true),
            "num_atoms_per_token" => to_py(num_atoms_per_token; swap_batch_dim=true),
            "atom_to_token_index" => to_py(atom_to_token_index .- 1; swap_batch_dim=true),
            "residue_index"       => to_py(residue_index; swap_batch_dim=true),
            "token_index"         => to_py(token_index; swap_batch_dim=true),
            "asym_id"             => to_py(asym_id; swap_batch_dim=true),
            "sym_id"              => to_py(sym_id; swap_batch_dim=true),
            "entity_id"           => to_py(entity_id; swap_batch_dim=true),
        )
        return batch_jl, batch_py
    end

    for T in [Float64, Float32]
        @testset "Python parity ($T)" begin
            jl_cond = DiffusionConditioning(; chn_s_input=c_s_input, chn_s=c_s, chn_z=c_z, chn_fourier_emb=c_fourier, max_relative_idx=max_idx, max_relative_chain=max_chain, sigma_data, num_relpos_dims=num_relpos)
            jl_enc = AtomAttentionEncoder(; chn_atom=ca, chn_atom_pair=cap, chn_token=c_token,
                                          chn_hidden=ch, no_heads, no_blocks, n_transition=n_trans,
                                          n_query=nq, n_key=nk, add_noisy_pos=true,
                                          chn_single=c_s, chn_pair=c_z,
                                          chn_ref_element=cre, chn_ref_name_chars=crnc)
            jl_dec = AtomAttentionDecoder(; chn_atom=ca, chn_atom_pair=cap, chn_token=c_token,
                                          chn_hidden=ch, no_heads, no_blocks, n_transition=n_trans,
                                          n_query=nq, n_key=nk)
            jl_dt = DiffusionTransformer(; chn_a=c_token, chn_cond=c_s, chn_pair=c_z, chn_hidden=ch,
                                         no_heads, no_blocks, n_transition=n_trans,
                                         use_ada_layer_norm=true)
            jl = DiffusionModule(; diffusion_conditioning=jl_cond, atom_attn_enc=jl_enc,
                                 atom_attn_dec=jl_dec, diffusion_transformer=jl_dt,
                                 chn_s=c_s, chn_token=c_token, sigma_data)
            ps, st = Lux.setup(rng, jl) |> convert_types(T)

            py = PyDiffusionModule(_py_config())
            py.to(py_dtype(T))
            sync_diffusion_module!(py, ps)
            # FourierEmbedding buffers (state) → sync from the (already dtype-converted) Python
            st = merge(st, (diffusion_conditioning = merge(st.diffusion_conditioning,
                (fourier_emb = (w = T.(to_jl(py.diffusion_conditioning.fourier_emb.w)),
                                b = T.(to_jl(py.diffusion_conditioning.fourier_emb.b))),)),))

            atom_mask  = trues(N_atom, B)
            token_mask = trues(N_token, B)
            xl_noisy   = randn(rng, T, 3, N_atom, B)
            t          = rand(rng, T, B) .+ T(0.5)
            si_input   = randn(rng, T, c_s_input, N_token, B)
            si_trunk   = randn(rng, T, c_s, N_token, B)
            zij_trunk  = randn(rng, T, c_z, N_token, N_token, B)

            batch_jl, batch_py = _make_batch(rng, T, atom_mask, token_mask)

            xl_jl, _ = jl(batch_jl, xl_noisy, t, si_input, si_trunk, zij_trunk, ps, st)
            xl_py = py(
                batch=batch_py,
                xl_noisy=to_py(xl_noisy; swap_batch_dim=true),
                token_mask=to_py(T.(token_mask); swap_batch_dim=true),
                atom_mask=to_py(T.(atom_mask); swap_batch_dim=true),
                t=to_py(t; swap_batch_dim=true),
                si_input=to_py(si_input; swap_batch_dim=true),
                si_trunk=to_py(si_trunk; swap_batch_dim=true),
                zij_trunk=to_py(zij_trunk; swap_batch_dim=true),
                use_conditioning=true,
            )

            atol = T == Float32 ? 1f-3 : 1e-8
            @test isapprox(xl_jl, to_jl(xl_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end
end
