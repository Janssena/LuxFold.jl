# Unit tests for the shared weight machinery (LuxFoldCore/src/weights.jl). Pure Julia — no network,
# no Python. Guards the get-closure copiers (AF2's loader has no automated test), the JAX reshape
# helpers, cache-dir resolution, and the sha256-verified fetch's reuse path.

import LuxFoldCore: Lux
import Random
using Test, LuxFoldCore

@testset "weights" begin
    rng = Random.Xoshiro(42)

    @testset "cache_dir resolution (override → env → default)" begin
        envname = "LUXFOLD_TEST_CACHE_XYZ"
        delete!(ENV, envname)
        cache = WeightsCache(envname, "/tmp/default-cache")
        @test cache_dir(cache) == "/tmp/default-cache"          # default

        ENV[envname] = "/tmp/from-env"
        @test cache_dir(cache) == "/tmp/from-env"               # env beats default

        set_cache_dir!(cache, "/tmp/from-override")
        @test cache_dir(cache) == "/tmp/from-override"          # override beats env

        set_cache_dir!(cache, "")                               # clear override
        @test cache_dir(cache) == "/tmp/from-env"
        delete!(ENV, envname)
        @test cache_dir(cache) == "/tmp/default-cache"
    end

    # A `get(prefix, leaf)` closure over a fake state-dict Dict, Float32-converting and returning
    # `nothing` for absent keys — the same contract AF2's `_npz_getter` provides.
    _getter(sd) = (prefix, leaf) ->
        (k = prefix * "/" * leaf; haskey(sd, k) ? Float32.(sd[k]) : nothing)
    # Stacked getter: slice the last dim (block i), like AF2's `_npz_getter(npz, i)`.
    _getter(sd, i) = (prefix, leaf) ->
        (k = prefix * "/" * leaf; haskey(sd, k) ? Float32.(selectdim(sd[k], ndims(sd[k]), i)) : nothing)

    @testset "_load_dense! (with + without bias)" begin
        for use_bias in (true, false)
            layer = Lux.Dense(3 => 2; use_bias)
            ps, _ = Lux.setup(rng, layer)
            W = rand(rng, 2, 3)
            sd = Dict{String,Any}("d/weights" => W)
            use_bias && (sd["d/bias"] = rand(rng, 2))
            LuxFoldCore._load_dense!(ps, _getter(sd), "d")
            @test ps.weight ≈ Float32.(W)
            use_bias && @test ps.bias ≈ Float32.(sd["d/bias"])
        end
    end

    @testset "_load_layernorm! (reshapes 1D vec to param shape)" begin
        layer = Lux.LayerNorm((4, 1, 1); dims=1)
        ps, _ = Lux.setup(rng, layer)
        sc, off = rand(rng, 4), rand(rng, 4)
        sd = Dict{String,Any}("ln/scale" => sc, "ln/offset" => off)
        LuxFoldCore._load_layernorm!(ps, _getter(sd), "ln")
        @test vec(ps.scale) ≈ Float32.(sc)
        @test vec(ps.bias) ≈ Float32.(off)
        @test ndims(ps.scale) ≥ 3 && size(ps.scale, 1) == 4     # reshaped to param shape, not flat
    end

    @testset "stacked getter drives the same copier" begin
        layer = Lux.Dense(3 => 2; use_bias=false)
        ps, _ = Lux.setup(rng, layer)
        stacked = rand(rng, 2, 3, 5)                            # [out, in, n_blocks]
        sd = Dict{String,Any}("d/weights" => stacked)
        LuxFoldCore._load_dense!(ps, _getter(sd, 3), "d")       # block 3
        @test ps.weight ≈ Float32.(stacked[:, :, 3])
    end

    @testset "JAX reshape helpers" begin
        C_hd, C_in, H, C_out, C_hid, C_z = 4, 5, 3, 6, 7, 8
        @test size(LuxFoldCore._mha_qkv_w(rand(C_hd, C_in, H))) == (H * C_hd, C_in)
        @test size(LuxFoldCore._mha_out_w(rand(C_out, C_hd, H))) == (C_out, H * C_hd)
        @test size(LuxFoldCore._opm_out_w(rand(C_z, C_hid, C_hid))) == (C_z, C_hid * C_hid)
        # _mha_qkv_w: [C_hd, C_in, H] → permutedims to [H, C_hd, C_in], reshape [H*C_hd, C_in]
        # (column-major over [H, C_hd]) ⇒ row for head h, dim d is (d-1)*H + h, value w[d, :, h].
        w = rand(C_hd, C_in, H)
        r = LuxFoldCore._mha_qkv_w(w)
        @test r[(3 - 1) * H + 2, :] ≈ w[3, :, 2]                # h=2, d=3
    end

    @testset "write_state_dict → read_state_dict round-trips + drives copiers" begin
        mktempdir() do d
            # `/`-separated keys like the DeepMind AF2 layout; values in Julia [out,in] layout.
            W, sc = rand(Float32, 2, 3), rand(Float32, 4)
            sd = Dict{String,Array}("m/d/weights" => W, "m/ln/scale" => sc)
            f = joinpath(d, "w.safetensors")
            write_state_dict(f, sd)
            back = read_state_dict(f)
            @test back["m/d/weights"] == W                       # shape + values preserved
            @test back["m/ln/scale"] == sc

            # A getter over the *loaded safetensors* drives the shared copiers identically to NPZ.
            getter = (prefix, leaf) ->
                (k = prefix * "/" * leaf; haskey(back, k) ? Float32.(back[k]) : nothing)
            ps_d, _ = Lux.setup(rng, Lux.Dense(3 => 2; use_bias=false))
            LuxFoldCore._load_dense!(ps_d, getter, "m/d")
            @test ps_d.weight ≈ W
            ps_ln, _ = Lux.setup(rng, Lux.LayerNorm((4, 1); dims=1))
            LuxFoldCore._load_layernorm!(ps_ln, getter, "m/ln")
            @test vec(ps_ln.scale) ≈ sc
        end
    end

    @testset "fetch_weights reuses a present, hash-matching file (no download)" begin
        mktempdir() do d
            cache = WeightsCache("LUXFOLD_TEST_CACHE_NOPE", d)   # env unset → default = d
            f = joinpath(d, "w.safetensors")
            write(f, "hello world")
            h = LuxFoldCore._sha256(f)
            entry = ModelRegistryEntry((architecture=:x, files=["w.safetensors"],
                url="http://invalid.invalid/should-not-be-hit", sha256=h,
                format=:safetensors, layout=:torch, options=(;)))
            @test fetch_weights(entry, cache) == f               # reused, url never touched
            # A hand-placed file with no pinned hash is accepted as-is.
            entry_nohash = ModelRegistryEntry((architecture=:x, files=["w.safetensors"],
                url="http://invalid.invalid", sha256="", format=:safetensors, layout=:torch, options=(;)))
            @test fetch_weights(entry_nohash, cache) == f
        end
    end

    @testset "flatten_params / load_flat_weights! round-trip" begin
        # A nested container mixing NamedTuple, Chain (→ NamedTuple), and a NoOpLayer (empty leaf).
        layer = Lux.Chain(dense = Lux.Dense(4 => 3), norm = Lux.LayerNorm((3,)),
                          skip = Lux.NoOpLayer(), attn = AttentionPairBias(8, 4, 2, 2))
        ps, _ = Lux.setup(rng, layer)

        flat = flatten_params(ps)
        @test flat isa Dict{String,Array}
        @test haskey(flat, "dense.weight") && haskey(flat, "norm.scale")
        @test !any(startswith(k, "skip") for k in keys(flat))         # NoOpLayer contributes nothing

        # Load into a *fresh* (differently-seeded) ps → every leaf must match the source.
        ps2, _ = Lux.setup(Random.Xoshiro(7), layer)
        @test ps2.dense.weight != ps.dense.weight                     # genuinely different first
        load_flat_weights!(ps2, flat)
        @test ps2.dense.weight == ps.dense.weight
        @test ps2.attn.mha.qkv.weight == ps.attn.mha.qkv.weight

        # Full loop through a real .safetensors file (write_state_dict → read_state_dict → load).
        mktempdir() do d
            path = write_state_dict(joinpath(d, "flat.safetensors"), flat)
            ps3, _ = Lux.setup(Random.Xoshiro(9), layer)
            load_flat_weights!(ps3, read_state_dict(path))
            @test ps3.norm.scale == ps.norm.scale
            @test ps3.attn.linear_z.weight == ps.attn.linear_z.weight
        end

        # A missing key is a hard error (guards silent partial loads).
        short = copy(flat); delete!(short, "dense.weight")
        ps4, _ = Lux.setup(rng, layer)
        @test_throws ArgumentError load_flat_weights!(ps4, short)
    end
end
