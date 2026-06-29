# Python weight-sync helpers + layout converters for the AF3 template module.
# Reuses sync_pair_block! (PairBlock) from the pairformer helpers.

if !@isdefined(_AF3_TEMPLATE_SYNC_LOADED)
    const _AF3_TEMPLATE_SYNC_LOADED = true

    isdefined(@__MODULE__, :sync_pair_block!) || include("../pairformer/sync_helpers.jl")

    # 5D pair tensor layout: Julia [C, Ni, Nj, T, B] <-> Python [B, T, Ni, Nj, C]
    to_py5(x) = to_py(permutedims(x, (5, 4, 2, 3, 1)); swap_batch_dim=false)
    to_jl5(y) = permutedims(to_jl(y; swap_batch_dim=false), (5, 3, 4, 2, 1))

    function sync_template_pair_embedder!(py, ps)
        for n in (:dgram_linear, :aatype_linear_1, :aatype_linear_2, :pseudo_beta_mask_linear,
                  :x_linear, :y_linear, :z_linear, :backbone_mask_linear, :linear_z)
            sync_dense!(getproperty(py, n), getproperty(ps, n))
        end
        sync_layernorm!(py.layer_norm_z, ps.layer_norm_z)
        return nothing
    end

    # TemplatePairBlock IS a PairBlock subclass (attrs tri_mul_out/in, tri_att_start/end, pair_transition);
    # the Julia wrapper nests them under `pair_block`.
    sync_template_pair_block!(py, ps) = sync_pair_block!(py, ps.pair_block)

    function sync_template_pair_stack!(py, ps)
        py_blocks = collect(py.blocks)
        for (i, name) in enumerate(keys(ps.blocks))
            sync_template_pair_block!(py_blocks[i], ps.blocks[name])
        end
        sync_layernorm!(py.layer_norm, ps.layer_norm)
        return nothing
    end

    function sync_template_embedder!(py, ps)
        sync_template_pair_embedder!(py.template_pair_embedder, ps.template_pair_embedder)
        sync_template_pair_stack!(py.template_pair_stack, ps.template_pair_stack)
        sync_dense!(py.linear_t, ps.linear_t)
        return nothing
    end

    # Matching Julia + Python template feature batches (shared by embedder tests).
    function template_batch(rng, T, N, Tpl, B)
        restype = Float32.(rand(rng, Bool, AlphaFold3.AF3_RESTYPE_NUM, N, Tpl, B))
        distogram = randn(rng, T, 39, N, N, Tpl, B)
        unit_vector = randn(rng, T, 3, N, N, Tpl, B)
        pbm = rand(rng, Bool, N, Tpl, B)
        bfm = rand(rng, Bool, N, Tpl, B)
        asym = ones(Int, N, B)

        batch_jl = (; template_restype=T.(restype), template_distogram=distogram,
                    template_unit_vector=unit_vector, template_pseudo_beta_mask=pbm,
                    template_backbone_frame_mask=bfm, asym_id=asym)
        batch_py = Dict(
            "template_restype"             => to_py(permutedims(T.(restype), (4, 3, 2, 1)); swap_batch_dim=false),
            "template_distogram"           => to_py(permutedims(distogram, (5, 4, 2, 3, 1)); swap_batch_dim=false),
            "template_unit_vector"         => to_py(permutedims(unit_vector, (5, 4, 2, 3, 1)); swap_batch_dim=false),
            "template_pseudo_beta_mask"    => to_py(permutedims(T.(pbm), (3, 2, 1)); swap_batch_dim=false),
            "template_backbone_frame_mask" => to_py(permutedims(T.(bfm), (3, 2, 1)); swap_batch_dim=false),
            "asym_id"                      => to_py(permutedims(asym, (2, 1)); swap_batch_dim=false),
        )
        return batch_jl, batch_py
    end
end
