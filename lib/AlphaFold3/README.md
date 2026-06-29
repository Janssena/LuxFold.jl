# AlphaFold3.jl

Julia implementation of AlphaFold3, built on [Lux.jl](https://github.com/LuxDL/Lux.jl). A 1:1 port of [openfold-3](https://github.com/aqlaboratory/openfold) with numerical-parity tests at Float64, Float32, and Float16 against the reference PyTorch implementation.

## Implementation Status

**All model layers complete and parity-tested.** The full-model integration (`AlphaFold3Model`) is implemented but not yet parity-tested end-to-end.

### Model Architecture

| Component | Layers | Tests | Status |
|-----------|--------|-------|--------|
| **Layers** — PairBlock, SwiGLUTransition, ResidualSwiGLUBlock, ConditionedTransitionBlock | 2 files | 2 | ✅ Complete |
| **Pairformer** — PairFormerBlock, PairFormerStack | 2 | 2 | ✅ Complete |
| **MSA Module** — MSAModuleBlock, MSAModuleStack | 2 | 2 | ✅ Complete |
| **Atom Module** — RefAtomFeatureEmbedder, SequenceLocalAttentionPairBias, AtomAttentionEncoder, AtomAttentionDecoder | 3 | 3 | ✅ Complete |
| **Embedders** — FourierEmbedding, InputEmbedderAllAtom, MSAModuleEmbedder, RefAtomEmbedder, TemplatePairEmbedderAllAtom, TemplatePairBlock, TemplatePairStack, TemplateEmbedderAllAtom | 8 | 8 | ✅ Complete |
| **Diffusion** — DiffusionConditioning, DiffusionModule, DiffusionTransformer, DiffusionTransformerBlock, NoisyPositionEmbedder, SampleDiffusion | 6 | 6 | ✅ Complete |
| **Heads** — DistogramHead, PredictedAlignedErrorHead, PredictedDistanceErrorHead, PerResidueLDDTAllAtom, ExperimentallyResolvedHeadAllAtom, PairformerEmbedding, AuxiliaryHeads | 2 | 1 | ✅ Complete |
| **Utils** — relpos, geometry, atomize_utils, atom_attention_block_utils, constants | 5 | 4 | ✅ Complete |
| **Full Model** — AlphaFold3Model | 1 | — | 🚧 Integration pending |

**Totals: 31 source files, 28 test files across 8 test groups. 428 passing tests.**

### Testing Strategy

All tests follow a consistent pattern:

- **Python numerical parity** at Float64, Float32, and Float16 — every layer output is compared to the openfold-3 reference after synchronising weights via `PythonTestHelpers`
- **Type-stability** checked via `@test_nowarn @inferred` for every layer
- **Mask parameterisation** — every layer tested with `nothing` (no mask) and random `Bool` masks

Shape and bias configuration are validated implicitly through parity — if the output matches at three precisions, the construction is correct.

### Running Tests

```julia
julia --project=lib/AlphaFold3/test lib/AlphaFold3/test/runtests.jl
```

Requires the openfold-3 pixi environment at `python/openfold-3/.pixi/envs/openfold3-cpu/`.

## Key Design Decisions

- **Channel-first layout**: `[C, spatial..., B]` throughout; `to_py`/`to_jl` in `PythonTestHelpers` handle the `[B, spatial..., C]` Python convention.
- **Bool masks with `ifelse`**: masks are always `AbstractArray{Bool}`; positions are zeroed via `@. ifelse(mask_r, x, zero(T))`, never by multiplication.
- **StaticBool dispatch**: compile-time flags (e.g., `opm_first`, `add_noisy_pos`) are stored as `StaticBool` struct fields and dispatched on type — no runtime branches.
- **Input-presence dispatch**: optional conditioning inputs are `Union{Nothing, AbstractArray}`; `nothing` means "unconditioned" and is dispatched upon, rather than using a `Bool` flag.
- **NamedTuple I/O**: every layer exposes `(l)(inputs::NamedTuple, ps, st)` → positional forward; blocks in stacks return `merge(inputs, updated_fields)` so masks pass through `Lux.Chain` unchanged.
- **`block_1…block_n` naming**: all stacks build a `NamedTuple{(:block_1, …, :block_n)}` and pass it to `Lux.Chain` for type-stable `@inferred` through the stack.
- **RNG convention**: any function drawing random values takes `rng::AbstractRNG` as the mandatory first positional argument. The full model stores a frozen master RNG in state and derives per-sample seeds via `Lux.replicate`.
