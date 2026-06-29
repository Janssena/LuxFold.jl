const PyInputEmbAllAtom = pyimport("openfold3.core.model.feature_embedders.input_embedders").InputEmbedderAllAtom
const _mlc_ie = pyimport("ml_collections")

# sync_atom_attention_encoder! / convert_types come from ../atom_module/sync_helpers.jl
# (included by embedders/runtests.jl). When running this file standalone, include it first.
isdefined(@__MODULE__, :convert_types) || include("../atom-module/sync_helpers.jl")

@testset "InputEmbedderAllAtom" begin
    rng = Random.Xoshiro(42)

    N_token, N_atom, B = 4, 16, 2
    nq, nk = 4, 8
    ca, cap, ct, ch = 8, 4, 12, 4
    no_heads, no_blocks, n_trans = 2, 2, 2
    cre, crnc = 128, 256
    apt = N_atom ÷ N_token
    n_restype, n_profile = 32, 32
    c_s_input = ct + n_restype + n_profile + 1     # 77
    c_s, c_z = 16, 6
    max_idx, max_chain = 32, 2
    num_relpos = 2 * (2max_idx + 2) + 1 + (2max_chain + 2)   # 139

    _c_atom_ref() = _mlc_ie.ConfigDict(Dict("element" => cre, "name_chars" => crnc))
    _atom_enc_cfg() = Dict(
        "c_atom_ref" => _c_atom_ref(), "c_atom" => ca, "c_atom_pair" => cap,
        "c_token" => ct, "c_hidden" => ch, "no_heads" => no_heads, "no_blocks" => no_blocks,
        "n_transition" => n_trans, "n_query" => nq, "n_key" => nk, "use_ada_layer_norm" => true,
    )

    # AF2-style random masks. Guarantee ≥1 valid atom per token so the encoder's masked
    # mean-aggregation is defined.
    function _random_masks(rng)
        atom_mask  = rand(rng, Bool, N_atom, B)
        token_mask = rand(rng, Bool, N_token, B)
        for b in 1:B, t in 1:N_token
            atom_mask[(t - 1) * apt + 1, b] = true
        end
        token_mask[1, :] .= true   # ≥1 valid token per batch; rest stay random
        return (atom_mask, token_mask)
    end

    # `masks` is `nothing` (≈ no mask → all-true) or a tuple `(atom_mask, token_mask)`.
    function _make_batch(rng, T, masks=nothing)
        # --- atom features (AtomAttentionEncoder) ---
        ref_pos       = randn(rng, T, 3, N_atom, B)
        ref_charge    = randn(rng, T, 1, N_atom, B)
        ref_mask      = rand(rng, Bool, 1, N_atom, B)
        ref_element   = randn(rng, T, cre, N_atom, B)
        ref_chars     = randn(rng, T, crnc, N_atom, B)
        ref_space_uid = rand(rng, 1:5, 1, N_atom, B)
        num_atoms_per_token = fill(apt, N_token, B)
        atom_to_token_index = repeat(reduce(vcat, [fill(t, apt) for t in 1:N_token]), 1, B)

        atom_mask, token_mask = isnothing(masks) ? (trues(N_atom, B), trues(N_token, B)) : masks

        # --- token features (input embedder) ---
        restype       = randn(rng, T, n_restype, N_token, B)
        profile       = randn(rng, T, n_profile, N_token, B)
        deletion_mean = randn(rng, T, N_token, B)
        token_bonds   = T.(rand(rng, Bool, N_token, N_token, B))

        # --- relpos integer features (two chains) ---
        half = N_token ÷ 2
        residue_index = repeat(reshape(collect(1:N_token), N_token, 1), 1, B)
        token_index   = copy(residue_index)
        asym_id       = repeat(reshape([fill(1, half); fill(2, half)], N_token, 1), 1, B)
        sym_id        = copy(asym_id)
        entity_id     = ones(Int, N_token, B)

        batch_jl = (; ref_pos, ref_charge, ref_mask, ref_element,
                    ref_atom_name_chars=ref_chars, ref_space_uid, atom_mask,
                    token_mask, num_atoms_per_token, atom_to_token_index,
                    restype, profile, deletion_mean, token_bonds,
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
            "restype"             => to_py(restype; swap_batch_dim=true),
            "profile"             => to_py(profile; swap_batch_dim=true),
            "deletion_mean"       => to_py(deletion_mean; swap_batch_dim=true),
            # [N,N,B]: swap_batch_dim transposes the inner [N,N]; pre-transpose so
            # Python receives py[b,i,j] = jl[i,j,b].
            "token_bonds"         => to_py(permutedims(token_bonds, (2, 1, 3)); swap_batch_dim=true),
            "residue_index"       => to_py(residue_index; swap_batch_dim=true),
            "token_index"         => to_py(token_index; swap_batch_dim=true),
            "asym_id"             => to_py(asym_id; swap_batch_dim=true),
            "sym_id"              => to_py(sym_id; swap_batch_dim=true),
            "entity_id"           => to_py(entity_id; swap_batch_dim=true),
        )
        return batch_jl, batch_py
    end

    mask_cfg = (
        ("No mask",     nothing),
        ("Random mask", _random_masks(rng)),
    )

    # openfold does NOT mask s_input/s/z; the Julia layer optionally does (ifelse, Bool
    # mask). Mask the Python outputs identically to compare. `nothing` → no-op (raw parity).
    _msk_single(x, ::Nothing) = x
    _msk_single(x, mask::AbstractArray{Bool}) =
        ifelse.(reshape(mask, 1, size(mask)...), x, zero(eltype(x)))
    _msk_pair(z, ::Nothing) = z
    function _msk_pair(z, mask::AbstractArray{Bool})
        N, Bd = size(mask)
        pm = reshape(reshape(mask, N, 1, Bd) .& reshape(mask, 1, N, Bd), 1, N, N, Bd)
        return ifelse.(pm, z, zero(eltype(z)))
    end

    for T in [Float64, Float32], (mask_name, masks) in mask_cfg
        @testset "Python parity ($T, $mask_name)" begin
            jl_enc = AtomAttentionEncoder(;
                chn_atom=ca, chn_atom_pair=cap, chn_token=ct, chn_hidden=ch,
                no_heads=no_heads, no_blocks=no_blocks, n_transition=n_trans,
                n_query=nq, n_key=nk, add_noisy_pos=false,
                chn_ref_element=cre, chn_ref_name_chars=crnc,
            )
            jl = InputEmbedderAllAtom(jl_enc; chn_s_input=c_s_input, chn_s=c_s, chn_z=c_z, max_relative_idx=max_idx, max_relative_chain=max_chain, num_relpos_dims=num_relpos)
            jl_ps, jl_st = Lux.setup(rng, jl) |> convert_types(T)

            py = PyInputEmbAllAtom(
                c_s_input=c_s_input, c_s=c_s, c_z=c_z,
                max_relative_idx=max_idx, max_relative_chain=max_chain,
                atom_attn_enc=_atom_enc_cfg(),
            )

            sync_atom_attention_encoder!(py.atom_attn_enc, jl_ps.atom_attn_enc; add_noisy_pos=false)
            sync_dense!(py.linear_s, jl_ps.linear_s)
            sync_dense!(py.linear_z_i, jl_ps.linear_z_i)
            sync_dense!(py.linear_z_j, jl_ps.linear_z_j)
            sync_dense!(py.linear_relpos, jl_ps.linear_relpos)
            sync_dense!(py.linear_token_bonds, jl_ps.linear_token_bonds)

            batch_jl, batch_py = _make_batch(rng, T, masks)
            out_mask = isnothing(masks) ? nothing : masks[2]   # token_mask [N_token, B] Bool
            emb_jl, _ = jl(batch_jl, out_mask, jl_ps, jl_st)
            s_input_jl, s_jl, z_jl = emb_jl.s_input, emb_jl.s, emb_jl.z
            s_input_py, s_py, z_py = py(batch=batch_py)

            @test s_input_jl ≈ _msk_single(to_jl(s_input_py; swap_batch_dim=true), out_mask)
            @test s_jl ≈ _msk_single(to_jl(s_py; swap_batch_dim=true), out_mask)
            @test z_jl ≈ _msk_pair(to_jl(z_py; swap_batch_dim=true), out_mask)
        end
    end

    # NOTE: `@inferred` is deferred for InputEmbedderAllAtom — it wraps the deep
    # AtomAttentionEncoder stack, whose @inferred was deferred in af3-atom-module
    # (the output type is concrete, but inference can't fully prove it through the
    # stack). Parity is the gate; leaf embedders (Fourier, MSA) are @inferred-tested.
end
