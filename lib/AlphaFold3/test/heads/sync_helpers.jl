# Python weight-sync helpers for AF3 auxiliary heads.
# Guarded so it can be `include`d from multiple test files in one session.

if !@isdefined(_AF3_HEADS_SYNC_LOADED)
    const _AF3_HEADS_SYNC_LOADED = true

    # ── Pair heads ──────────────────────────────────────────────────────────
    # DistogramHead: only `linear` (no layer_norm).
    function sync_distogram_head!(py, ps)
        sync_dense!(py.linear, ps.linear)
        return nothing
    end

    # PAE head: layer_norm + linear.
    function sync_pae_head!(py, ps)
        sync_layernorm!(py.layer_norm, ps.layer_norm)
        sync_dense!(py.linear, ps.linear)
        return nothing
    end

    # PDE head: same architecture as PAE head.
    const sync_pde_head! = sync_pae_head!

    # ── Per-atom heads ──────────────────────────────────────────────────────
    # PerResidueLDDTAllAtom / ExperimentallyResolvedHeadAllAtom: layer_norm + linear.
    function sync_per_atom_head!(py, ps)
        sync_layernorm!(py.layer_norm, ps.layer_norm)
        sync_dense!(py.linear, ps.linear)
        return nothing
    end

    # ── PairformerEmbedding ─────────────────────────────────────────────────
    # Sync linear_i, linear_j, linear_distance, then the pairformer stack.
    # The pairformer stack sync helper is already defined in pairformer/sync_helpers.jl;
    # include it here if not already loaded.
    if !@isdefined(_AF3_PAIRFORMER_SYNC_LOADED)
        include(joinpath(@__DIR__, "..", "pairformer", "sync_helpers.jl"))
    end

    function sync_pairformer_embedding!(py, ps)
        sync_dense!(py.linear_i, ps.linear_i)
        sync_dense!(py.linear_j, ps.linear_j)
        sync_dense!(py.linear_distance, ps.linear_distance)
        for (i, blk) in enumerate(py.pairformer_stack.blocks)
            sync_pairformer_block!(blk, ps.pairformer_stack.blocks[i])
        end
        return nothing
    end
end
