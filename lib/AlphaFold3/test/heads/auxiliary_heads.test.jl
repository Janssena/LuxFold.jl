const PyPredictionHeads  = pyimport("openfold3.core.model.heads.prediction_heads")
const ConfigDict         = pyimport("ml_collections").ConfigDict

include("sync_helpers.jl")

# ────────────────────────────────────────────────────────────────────────────────
# Shared test dims
# ────────────────────────────────────────────────────────────────────────────────
const _rng_h   = Random.Xoshiro(42)
const _N_h     = 8      # N_token
const _B_h     = 1
const _c_s_h   = 16
const _c_z_h   = 16
const _no_dist_bins = 15   # distance bins for PairformerEmbedding
const _no_pae  = 8
const _no_pde  = 8
const _no_lddt = 10
const _no_er   = 2
const _max_atoms = 8   # max_atoms_per_token (N_atom = N * max_atoms)

# ────────────────────────────────────────────────────────────────────────────────
# 1. DistogramHead
# ────────────────────────────────────────────────────────────────────────────────
@testset "DistogramHead" begin
    for T in [Float64, Float32]
        @testset "$T" begin
            jl = DistogramHead(_c_z_h; no_bins=_no_pde)
            ps, st = Lux.setup(_rng_h, jl) |> convert_types(T)

            py = PyPredictionHeads.DistogramHead(_c_z_h, _no_pde)
            py.to(py_dtype(T))
            sync_distogram_head!(py, ps)

            z = randn(_rng_h, T, _c_z_h, _N_h, _N_h, _B_h)

            logits_jl, _ = jl(z, ps, st)
            logits_py = py(to_py(z; swap_batch_dim=true))

            atol = T == Float64 ? 1e-9 : 1f-4
            @test isapprox(logits_jl, to_jl(logits_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = DistogramHead(_c_z_h; no_bins=_no_pde)
        ps, st = Lux.setup(_rng_h, jl) |> convert_types(Float32)
        z = randn(_rng_h, Float32, _c_z_h, _N_h, _N_h, _B_h)
        @test_nowarn @inferred jl(z, ps, st)
    end
end

# ────────────────────────────────────────────────────────────────────────────────
# 2. PredictedAlignedErrorHead
# ────────────────────────────────────────────────────────────────────────────────
@testset "PredictedAlignedErrorHead" begin
    for T in [Float64, Float32]
        @testset "$T" begin
            jl = PredictedAlignedErrorHead(_c_z_h; no_bins=_no_pae)
            ps, st = Lux.setup(_rng_h, jl) |> convert_types(T)

            py = PyPredictionHeads.PredictedAlignedErrorHead(_c_z_h, _no_pae)
            py.to(py_dtype(T))
            sync_pae_head!(py, ps)

            z = randn(_rng_h, T, _c_z_h, _N_h, _N_h, _B_h)

            logits_jl, _ = jl(z, ps, st)
            logits_py = py(to_py(z; swap_batch_dim=true))

            atol = T == Float64 ? 1e-9 : 1f-4
            @test isapprox(logits_jl, to_jl(logits_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = PredictedAlignedErrorHead(_c_z_h; no_bins=_no_pae)
        ps, st = Lux.setup(_rng_h, jl) |> convert_types(Float32)
        z = randn(_rng_h, Float32, _c_z_h, _N_h, _N_h, _B_h)
        @test_nowarn @inferred jl(z, ps, st)
    end
end

# ────────────────────────────────────────────────────────────────────────────────
# 3. PredictedDistanceErrorHead
# ────────────────────────────────────────────────────────────────────────────────
@testset "PredictedDistanceErrorHead" begin
    for T in [Float64, Float32]
        @testset "$T" begin
            jl = PredictedDistanceErrorHead(_c_z_h; no_bins=_no_pde)
            ps, st = Lux.setup(_rng_h, jl) |> convert_types(T)

            py = PyPredictionHeads.PredictedDistanceErrorHead(_c_z_h, _no_pde)
            py.to(py_dtype(T))
            sync_pde_head!(py, ps)

            z = randn(_rng_h, T, _c_z_h, _N_h, _N_h, _B_h)

            logits_jl, _ = jl(z, ps, st)
            logits_py = py(to_py(z; swap_batch_dim=true))

            atol = T == Float64 ? 1e-9 : 1f-4
            @test isapprox(logits_jl, to_jl(logits_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = PredictedDistanceErrorHead(_c_z_h; no_bins=_no_pde)
        ps, st = Lux.setup(_rng_h, jl) |> convert_types(Float32)
        z = randn(_rng_h, Float32, _c_z_h, _N_h, _N_h, _B_h)
        @test_nowarn @inferred jl(z, ps, st)
    end
end

# ────────────────────────────────────────────────────────────────────────────────
# 4. PerResidueLDDTAllAtom
# ────────────────────────────────────────────────────────────────────────────────
@testset "PerResidueLDDTAllAtom" begin
    # Simplified test: all tokens have exactly max_atoms atoms → mask is all-true.
    # N_atom = N_token * max_atoms_per_token = 64.
    N_atom = _N_h * _max_atoms

    for T in [Float64, Float32]
        @testset "$T" begin
            jl = PerResidueLDDTAllAtom(_c_s_h, _max_atoms; no_bins=_no_lddt)
            ps, st = Lux.setup(_rng_h, jl) |> convert_types(T)

            py = PyPredictionHeads.PerResidueLDDTAllAtom(_c_s_h, _no_lddt; max_atoms_per_token=_max_atoms)
            py.to(py_dtype(T))
            sync_per_atom_head!(py, ps)

            s = randn(_rng_h, T, _c_s_h, _N_h, _B_h)
            # All atoms present: mask = ones [max_atoms * N_token, B] (Julia) / [B, max_atoms * N_token] (Python)
            mask_jl = ones(Bool, _max_atoms * _N_h, _B_h)
            mask_py = to_py(ones(T, _max_atoms * _N_h, _B_h); swap_batch_dim=true)

            logits_jl, _ = jl(s, mask_jl, ps, st)                    # [no_lddt, N_atom, B]
            logits_py = py(s=to_py(s; swap_batch_dim=true), max_atom_per_token_mask=mask_py)

            atol = T == Float64 ? 1e-9 : 1f-4
            @test isapprox(logits_jl, to_jl(logits_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = PerResidueLDDTAllAtom(_c_s_h, _max_atoms; no_bins=_no_lddt)
        ps, st = Lux.setup(_rng_h, jl) |> convert_types(Float32)
        s = randn(_rng_h, Float32, _c_s_h, _N_h, _B_h)
        mask = ones(Bool, _max_atoms * _N_h, _B_h)
        @test_nowarn @inferred jl(s, mask, ps, st)
    end
end

# ────────────────────────────────────────────────────────────────────────────────
# 5. ExperimentallyResolvedHeadAllAtom
# ────────────────────────────────────────────────────────────────────────────────
@testset "ExperimentallyResolvedHeadAllAtom" begin
    for T in [Float64, Float32]
        @testset "$T" begin
            jl = ExperimentallyResolvedHeadAllAtom(_c_s_h, _max_atoms; no_bins=_no_er)
            ps, st = Lux.setup(_rng_h, jl) |> convert_types(T)

            py = PyPredictionHeads.ExperimentallyResolvedHeadAllAtom(_c_s_h, _no_er; max_atoms_per_token=_max_atoms)
            py.to(py_dtype(T))
            sync_per_atom_head!(py, ps)

            s = randn(_rng_h, T, _c_s_h, _N_h, _B_h)
            mask_jl = ones(Bool, _max_atoms * _N_h, _B_h)
            mask_py = to_py(ones(T, _max_atoms * _N_h, _B_h); swap_batch_dim=true)

            logits_jl, _ = jl(s, mask_jl, ps, st)
            logits_py = py(s=to_py(s; swap_batch_dim=true), max_atom_per_token_mask=mask_py)

            atol = T == Float64 ? 1e-9 : 1f-4
            @test isapprox(logits_jl, to_jl(logits_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        jl = ExperimentallyResolvedHeadAllAtom(_c_s_h, _max_atoms; no_bins=_no_er)
        ps, st = Lux.setup(_rng_h, jl) |> convert_types(Float32)
        s = randn(_rng_h, Float32, _c_s_h, _N_h, _B_h)
        mask = ones(Bool, _max_atoms * _N_h, _B_h)
        @test_nowarn @inferred jl(s, mask, ps, st)
    end
end

# ────────────────────────────────────────────────────────────────────────────────
# 6. PairformerEmbedding
# ────────────────────────────────────────────────────────────────────────────────
@testset "PairformerEmbedding" begin
    # Small pairformer stack (1 block) for speed
    c_hidden_pb = 8; no_heads_pb = 2
    c_hidden_mul = 8; c_hidden_pair_att = 8; no_heads_pair = 2
    transition_n = 2; no_blocks = 1
    min_bin = 3.25f0; max_bin = 20.75f0; no_bin = 15; inf = 1f8

    for T in [Float64, Float32]
        @testset "$T" begin
            pf_stack = PairFormerStack(;
                chn_s=_c_s_h, chn_z=_c_z_h,
                chn_hidden_pair_bias=c_hidden_pb, no_heads_pair_bias=no_heads_pb,
                chn_hidden_mul=c_hidden_mul, chn_hidden_pair_att=c_hidden_pair_att, no_heads_pair, transition_n, no_blocks,
            )
            jl = PairformerEmbedding(;
                chn_s_input=_c_s_h, chn_z=_c_z_h, pairformer_stack=pf_stack,
                min_bin, max_bin, no_bin, inf,
            )
            ps, st = Lux.setup(_rng_h, jl) |> convert_types(T)

            # Build Python PairformerEmbedding via ConfigDict
            pf_cfg = ConfigDict(Dict(
                "c_s"                    => _c_s_h,
                "c_z"                    => _c_z_h,
                "c_hidden_pair_bias"     => c_hidden_pb,
                "no_heads_pair_bias"     => no_heads_pb,
                "c_hidden_mul"           => c_hidden_mul,
                "c_hidden_pair_att"      => c_hidden_pair_att,
                "no_heads_pair"          => no_heads_pair,
                "no_blocks"              => no_blocks,
                "transition_type"        => "swiglu",
                "transition_n"           => transition_n,
                "pair_dropout"           => 0.0,
                "fuse_projection_weights"=> false,
                "blocks_per_ckpt"        => nothing,
                "inf"                    => Float64(inf),
            ))
            py = PyPredictionHeads.PairformerEmbedding(
                pairformer=pf_cfg,
                c_s_input=_c_s_h, c_z=_c_z_h,
                min_bin=Float64(min_bin), max_bin=Float64(max_bin),
                no_bin=no_bin, inf=Float64(inf),
            )
            py.to(py_dtype(T))
            sync_pairformer_embedding!(py, ps)

            s    = randn(_rng_h, T, _c_s_h, _N_h, _B_h)
            z    = randn(_rng_h, T, _c_z_h, _N_h, _N_h, _B_h)
            # x_pred: representative atom coords per token [3, N_token, B]
            x_pred = randn(_rng_h, T, 3, _N_h, _B_h)

            smask = rand(_rng_h, Bool, _N_h, _B_h); smask[1, :] .= true
            pmask = rand(_rng_h, Bool, _N_h, _N_h, _B_h); pmask[1, 1, :] .= true

            out_jl, _ = jl(s, s, z, x_pred, smask, pmask, ps, st)
            s_jl, z_jl = out_jl.s, out_jl.z

            # Python: pairformer_emb takes si_input, si, zij, x_pred, single_mask, pair_mask
            # x_pred: Julia [3, N, B] → swap → Python [B, N, 3]
            # single_mask [N, B] → swap → [B, N]; pair_mask [N, N, B] → permutedims → [B, N, N]
            s_py, z_py = py.pairformer_emb(
                si_input = to_py(s; swap_batch_dim=true),
                si       = to_py(s; swap_batch_dim=true),
                zij      = to_py(z; swap_batch_dim=true),
                x_pred   = to_py(x_pred; swap_batch_dim=true),
                single_mask = to_py(T.(smask); swap_batch_dim=true),
                pair_mask   = to_py(permutedims(T.(pmask), (3, 1, 2)); swap_batch_dim=false),
            )

            atol = T == Float64 ? 1e-5 : 1f-3
            @test isapprox(s_jl, to_jl(s_py; swap_batch_dim=true); atol=atol, rtol=atol)
            @test isapprox(z_jl, to_jl(z_py; swap_batch_dim=true); atol=atol, rtol=atol)
        end
    end

    @testset "Type-stability" begin
        pf_stack = PairFormerStack(;
            chn_s=_c_s_h, chn_z=_c_z_h,
            chn_hidden_pair_bias=8, no_heads_pair_bias=2,
            chn_hidden_mul=8, chn_hidden_pair_att=8, no_heads_pair=2,
            transition_n=2, no_blocks=1,
        )
        jl = PairformerEmbedding(;
            chn_s_input=_c_s_h, chn_z=_c_z_h, pairformer_stack=pf_stack,
        )
        ps, st = Lux.setup(_rng_h, jl) |> convert_types(Float32)
        s = randn(_rng_h, Float32, _c_s_h, _N_h, _B_h)
        z = randn(_rng_h, Float32, _c_z_h, _N_h, _N_h, _B_h)
        x_pred = randn(_rng_h, Float32, 3, _N_h, _B_h)
        @test_nowarn @inferred jl(s, s, z, x_pred, nothing, nothing, ps, st)
    end
end

# ────────────────────────────────────────────────────────────────────────────────
# 7. AuxiliaryHeadsAllAtom — shape + order verification (no Python parity needed)
# ────────────────────────────────────────────────────────────────────────────────
@testset "AuxiliaryHeadsAllAtom" begin
    c_hidden_pb = 8; no_heads_pb = 2
    c_hidden_mul = 8; c_hidden_pair_att = 8; no_heads_pair = 2
    no_blocks = 1

    pf_stack = PairFormerStack(;
        chn_s=_c_s_h, chn_z=_c_z_h,
        chn_hidden_pair_bias=c_hidden_pb, no_heads_pair_bias=no_heads_pb,
        chn_hidden_mul=c_hidden_mul, chn_hidden_pair_att=c_hidden_pair_att, no_heads_pair, transition_n=2, no_blocks,
    )

    jl = AuxiliaryHeadsAllAtom(;
        pairformer_embedding         = PairformerEmbedding(; chn_s_input=_c_s_h, chn_z=_c_z_h, pairformer_stack=pf_stack),
        distogram_head               = DistogramHead(_c_z_h; no_bins=_no_pde),
        pae_head                     = PredictedAlignedErrorHead(_c_z_h; no_bins=_no_pae),
        pde_head                     = PredictedDistanceErrorHead(_c_z_h; no_bins=_no_pde),
        plddt_head                   = PerResidueLDDTAllAtom(_c_s_h, _max_atoms; no_bins=_no_lddt),
        experimentally_resolved_head = ExperimentallyResolvedHeadAllAtom(_c_s_h, _max_atoms; no_bins=_no_er),
    )
    ps, st = Lux.setup(_rng_h, jl) |> convert_types(Float32)

    s_trunk = randn(_rng_h, Float32, _c_s_h, _N_h, _B_h)
    z_trunk = randn(_rng_h, Float32, _c_z_h, _N_h, _N_h, _B_h)
    # Atom positions: [3, N_atom, B] with N_atom = N_token * max_atoms
    N_atom = _N_h * _max_atoms
    atom_pos = randn(_rng_h, Float32, 3, N_atom, _B_h)

    # Build a simple batch: start_atom_index maps token t → atom (t-1)*max_atoms + 1
    start_atom_index = reshape(Int.(1:_max_atoms:N_atom), _N_h, 1) .* ones(Int, 1, _B_h)
    num_atoms_per_token = fill(_max_atoms, _N_h, _B_h)
    token_mask = trues(_N_h, _B_h)
    atom_mask  = trues(N_atom, _B_h)

    batch = (;
        token_mask, atom_mask, start_atom_index,
        atom_positions_predicted = atom_pos,
        num_atoms_per_token,
    )

    out, _ = jl(s_trunk, z_trunk, batch, ps, st)

    @testset "Output shapes" begin
        @test size(out.distogram)               == (_no_pde,  _N_h, _N_h, _B_h)
        @test size(out.pae)                     == (_no_pae,  _N_h, _N_h, _B_h)
        @test size(out.pde)                     == (_no_pde,  _N_h, _N_h, _B_h)
        @test size(out.plddt)                   == (_no_lddt, N_atom, _B_h)
        @test size(out.experimentally_resolved) == (_no_er,   N_atom, _B_h)
    end

    @testset "Distogram uses z_trunk (not z_conf)" begin
        # Distogram is computed before the confidence pairformer.
        # Verify it exactly matches a direct call to distogram_head on z_trunk.
        distogram_direct, _ = jl.distogram_head(z_trunk, ps.distogram_head, st.distogram_head)
        @test out.distogram ≈ distogram_direct
    end
end
