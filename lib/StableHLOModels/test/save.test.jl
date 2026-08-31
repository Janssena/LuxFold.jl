# Unit tests for the checkpoint format (StableHLOModels/src/save.jl).
#
# Split by cost, because a Reactant compile is not free:
#
#   "spec"          pure Julia, no compile, no device.
#   "round-trip"    save + load with `build = false` — exercises every byte the format writes and
#                   reads (spec, state, params, manifest) without compiling anything.
#   "build_hlo"     ONE small real compile, to prove the factory extension point works end to end
#                   and that a loaded model actually runs.
#   "alphafold2"    the full AF2 round-trip. ~8 minutes, so it is opt-in:
#                       LUXFOLD_TEST_AF2_CHECKPOINT=1 julia --project=lib/StableHLOModels/test ...

import StableHLOModels: Lux, Reactant
import Random
using Test, StableHLOModels, Serialization, TOML

"""
    NTDense(in => out, in_key, out_key)

A `Lux.Dense` with NamedTuple I/O, standing in for a real architecture's entry.

Defined here rather than borrowed from a model package on purpose: `save.jl` is architecture-
agnostic, and a test for it that reached into AlphaFold2 for its stand-in layer would be asserting
something about AlphaFold2 as a side effect. It is an `AbstractLuxWrapperLayer`, so `ps` is the
wrapped `Dense`'s own `(; weight, bias)` and nothing else.
"""
struct NTDense{IN,OUT,D} <: Lux.AbstractLuxWrapperLayer{:dense}
    dense::D
end

NTDense(d::Pair, in::Symbol, out::Symbol) =
    ((l = Lux.Dense(d)); NTDense{in,out,typeof(l)}(l))

function (l::NTDense{IN,OUT})(inputs::NamedTuple, ps, st) where {IN,OUT}
    y, stn = l.dense(getproperty(inputs, IN), ps, st)
    return NamedTuple{(OUT,)}((y,)), stn
end

@testset "save" begin
    rng = Random.Xoshiro(42)

    mklayer() = NTDense(4 => 3, :x, :y)

    @testset "ArraySpec drops contents, keeps shape and eltype" begin
        a = randn(rng, Float32, 2, 3, 4)
        s = ArraySpec(a)
        @test size(s) == (2, 3, 4)
        @test eltype(s) == Float32
        # Two arrays of the same shape/dtype are the SAME spec: contents cannot reach a traced
        # program, so a checkpoint must not pretend they matter.
        @test s === ArraySpec(randn(rng, Float32, 2, 3, 4))
        @test sprint(show, s) == "Float32[2×3×4]"
    end

    @testset "materialize_inputs fill values" begin
        spec = HLOBuildSpec{:unused}(mklayer(), (;
            f=zeros(Float32, 2, 2), i=zeros(Int64, 3), b=fill(false, 2), n=7, s=:tag))
        got = materialize_inputs(spec)

        @test got.f == zeros(Float32, 2, 2)
        # `one`, not `zero`: an index tensor of zeros is out of range for a 1-based gather, and the
        # constructors DO execute the prototype once to record output shapes.
        @test got.i == ones(Int64, 3)
        # `true`, not `false`: an all-false attention mask makes the probe softmax a row of -Inf.
        @test got.b == fill(true, 2)
        @test eltype(got.b) === Bool && got.b isa Array   # never a BitArray — no device rep
        # Non-arrays are the value itself, not a shape.
        @test got.n === 7 && got.s === :tag
    end

    @testset "spec rejects build-transient keywords" begin
        for k in (:dev, :verbose, :compile_options)
            @test_throws ArgumentError HLOBuildSpec{:unused}(
                mklayer(), (; x=zeros(Float32, 4, 2)); NamedTuple{(k,)}((1,))...)
        end
        @test spec_kind(HLOBuildSpec{:unused}(mklayer(), (; x=zeros(Float32, 4, 2)))) === :unused
    end

    @testset "round-trip without building" begin
        layer = mklayer()
        ps, st = Lux.setup(rng, layer)
        spec = HLOBuildSpec{:unused}(layer, (; x=zeros(Float32, 4, 2)); alpha=3, beta=false)

        dir = joinpath(mktempdir(), "m.hlomodel")
        save_hlo(dir, spec, ps, st)

        @test all(isfile(joinpath(dir, f)) for f in
                  ("manifest.toml", "spec.jls", "state.jls", "params.safetensors"))

        loaded = load_hlo(dir; build=false)
        @test loaded.model === nothing
        # `NTDense` is an `AbstractLuxWrapperLayer`, so `ps` is the wrapped Dense's own
        # `(; weight, bias)` with no extra nesting.
        @test Array(loaded.ps.weight) == ps.weight    # exact bytes, via safetensors
        @test Array(loaded.ps.bias) == ps.bias
        @test spec_kind(loaded.spec) === :unused
        @test loaded.spec.build_kwargs == (; alpha=3, beta=false)
        @test keys(loaded.spec.inputs) == (:x,)

        # `load_hlo_spec` reads the recipe alone — no weights, no compile.
        @test spec_kind(load_hlo_spec(dir)) === :unused

        m = load_hlo_manifest(dir)
        @test m["format_version"] == StableHLOModels.HLO_FORMAT_VERSION
        @test m["kind"] == "unused"
        @test m["inputs"]["x"]["eltype"] == "Float32"
        @test m["inputs"]["x"]["size"] == [4, 2]
        @test m["params"]["count"] == length(ps.weight) + length(ps.bias)
    end

    @testset "overwrite is refused without force" begin
        layer = mklayer()
        ps, st = Lux.setup(rng, layer)
        spec = HLOBuildSpec{:unused}(layer, (; x=zeros(Float32, 4, 2)))
        dir = joinpath(mktempdir(), "m.hlomodel")

        save_hlo(dir, spec, ps, st)
        @test_throws ArgumentError save_hlo(dir, spec, ps, st)
        @test save_hlo(dir, spec, ps, st; force=true) == dir
    end

    @testset "a stale format version fails before deserializing" begin
        layer = mklayer()
        ps, st = Lux.setup(rng, layer)
        dir = joinpath(mktempdir(), "m.hlomodel")
        save_hlo(dir, HLOBuildSpec{:unused}(layer, (; x=zeros(Float32, 4, 2))), ps, st)

        mpath = joinpath(dir, "manifest.toml")
        m = TOML.parsefile(mpath)
        m["format_version"] = "999"
        open(io -> TOML.print(io, m), mpath, "w")

        # The point of reading the manifest first: a plain sentence, not a Serialization stacktrace.
        @test_throws ArgumentError load_hlo(dir; build=false)
        @test_throws ArgumentError load_hlo_spec(dir)
    end

    @testset "a directory that is not a checkpoint" begin
        @test_throws ArgumentError load_hlo_manifest(mktempdir())
        @test_throws ArgumentError load_hlo_manifest(joinpath(mktempdir(), "nope"))
    end

    @testset "unknown kind names the problem" begin
        layer = mklayer()
        ps, st = Lux.setup(rng, layer)
        # `:unused` has no `build_hlo` method — the fallback must say so rather than MethodError.
        @test_throws ArgumentError build_hlo(
            HLOBuildSpec{:unused}(layer, (; x=zeros(Float32, 4, 2))), ps, st)
    end
end
