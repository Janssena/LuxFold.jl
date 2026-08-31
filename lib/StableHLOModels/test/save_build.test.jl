# The half of the checkpoint format that costs a real Reactant compile: does a LOADED model
# actually run, and does it agree with the one that was saved?
#
# `save.test.jl` covers everything reachable with `build = false`. This file compiles, so it is
# deliberately small — one `Dense` — and it doubles as the worked example of the `build_hlo`
# extension point: a third-party factory joins by adding one `HLOBuildSpec` constructor and one
# `build_hlo` method, and nothing in `save.jl` needs to know about it.

import StableHLOModels: Lux, Reactant
import Random
using Test, StableHLOModels

# ── a test-only architecture, registered exactly as a real one would be ─────────────────────────

struct TinyHLO end   # stands in for `AlphaFold2HLO`: the factory this spec kind names

StableHLOModels.HLOBuildSpec(::Type{TinyHLO}, layer, inputs::NamedTuple; kwargs...) =
    HLOBuildSpec{:tiny}(layer, inputs; kwargs...)

function StableHLOModels.build_hlo(s::HLOBuildSpec{:tiny}, ps, st;
    dev=Lux.reactant_device(), verbose::Bool=true, compile_options=Reactant.CompileOptions())

    inputs = dev(materialize_inputs(s))
    # The cotangent prototype is DERIVED from layer config, never carried in `inputs` — a `ct` is
    # not an input, and putting one there would compile it in as a constant argument the caller
    # then has to keep supplying. `AlphaFold2HLO` derives every `ct` the same way.
    ct = (; y=dev(zeros(eltype(inputs.x), s.layer.dense.out_dims, size(inputs.x, 2))))
    return CompiledHLOModule(s.layer, inputs, dev(ps), dev(st), ct;
        active=(:x,), compile_backward=s.build_kwargs.with_backward, compile_options)
end

@testset "save (build)" begin
    rng = Random.Xoshiro(42)
    dev = Lux.reactant_device()

    layer = NTDense(4 => 3, :x, :y)   # defined in save.test.jl, included first
    ps, st = Lux.setup(rng, layer)
    inputs = (; x=randn(rng, Float32, 4, 2))

    spec = HLOBuildSpec(TinyHLO, layer, inputs; with_backward=true)
    built = build_hlo(spec, ps, st)

    x_d, psd, std_ = dev(inputs.x), dev(ps), dev(st)
    ref, _ = forward(built, (; x=x_d), psd, std_)

    dir = joinpath(mktempdir(), "tiny.hlomodel")
    save_hlo(dir, spec, ps, st)

    loaded = load_hlo(dir; verbose=false)
    @test loaded.model isa CompiledHLOModule
    @test spec_kind(loaded.spec) === :tiny

    @testset "the loaded model runs and agrees exactly" begin
        got, _ = forward(loaded.model, (; x=x_d), loaded.ps, loaded.st)
        # Same weights and the same traced program, so this is bit-identical, not merely close.
        @test Array(got.y) == Array(ref.y)
    end

    @testset "the loaded model differentiates" begin
        ct = (; y=dev(randn(rng, Float32, 3, 2)))
        g_ref = backward(built, (; x=x_d), psd, std_, ct)
        g_got = backward(loaded.model, (; x=x_d), loaded.ps, loaded.st, ct)
        @test Array(g_got.dinputs.x) == Array(g_ref.dinputs.x)
    end

    @testset "build = false really skips the compile" begin
        # Not a timing assertion — just that nothing was built and the spec still came back whole.
        nb = load_hlo(dir; build=false)
        @test nb.model === nothing
        @test nb.spec.build_kwargs == (; with_backward=true)
    end
end

# ── the real thing, opt-in ──────────────────────────────────────────────────────────────────────
# ~8 minutes: a full AlphaFold2 compile, then a second one for the load. Guarded rather than
# deleted because it is the only test that exercises `_reset_scratch`, the AF2 `build_hlo` methods,
# and a checkpoint holding a genuinely large `ps`.

if get(ENV, "LUXFOLD_TEST_AF2_CHECKPOINT", "0") == "1"
    include("save_alphafold2.test.jl")
else
    @info "skipping the AlphaFold2 checkpoint round-trip; set LUXFOLD_TEST_AF2_CHECKPOINT=1 to run it"
end
