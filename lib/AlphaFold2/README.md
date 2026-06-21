# AlphaFold2.jl

A pure-Julia implementation of [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2) (Jumper et al., Nature 2021), built on [Lux.jl](https://github.com/LuxDL/Lux.jl) and [LuxFoldCore](https://github.com/LuxDL/LuxFoldCore.jl). Every layer achieves numerical parity with the [OpenFold](https://github.com/aqlaboratory/openfold) PyTorch reference at Float64, Float32, and Float16 precision.

## Implementation Status

All components of the AlphaFold2 architecture are implemented and tested:

| Module | Source Files | Tests | Python Parity |
|--------|-------------|-------|---------------|
| **Embedders** (InputEmbedder, RecyclingEmbedder, ExtraMSAEmbedder, PreEmbeddingEmbedder, RelativePositionEncoding, TemplateEmbedder pipeline) | 9 | 10 test files | ✅ F64/F32/F16 |
| **Evoformer** (EvoformerBlock, EvoformerStack, ExtraMSABlock, ExtraMSAStack, column attentions) | 5 | 5 test files | ✅ F64/F32/F16 |
| **Heads** (Distogram, TMScore, MaskedMSA, ExperimentallyResolved, pLDDT, AuxiliaryHeads) | 1 (6 layers) | 1 test file | ✅ F64/F32/F16 |
| **Structure Module** (InvariantPointAttention, BackboneUpdate, AngleResnet, StructureModuleTransition, StructureModule) | 5 | 5 test files | ✅ F64/F32/F16 |
| **Shared Layers** (Transition, MSATransition, PairTransition, PairStackBlock) | 2 | 2 test files | ✅ F64/F32/F16 |
| **Utilities** (Rigid body, geometry, scoring, atom utils, feats, constants) | 7 | 5 test files | ✅ F64/F32/F16 |
| **Weight Loading** (NPZ checkpoint loader for official DeepMind weights) | 1 | — | Tested via integration |
| **Full Model** (AlphaFold struct, recycling loop) | 1 | 2 integration | Shape + type-stability |

**Total: 29 source files, 37 test files.**

## Architecture

### Module Organisation

```
src/
  AlphaFold2.jl                      # Module entry point, exports
  alphafold.jl                       # AlphaFold model, recycling loop, predict()
  embedders/
    input_embedder.jl                # Target + MSA feature embedding
    recycling_embedder.jl            # Recycling iteration embedding
    extra_msa_embedder.jl            # Extra MSA projection
    preembedding_embedder.jl         # Precomputed embedding (e.g., ESM-2)
    relative_position_encoding.jl    # Positional encoding (monomer + multimer)
    template_embedder.jl             # Full template pipeline
    template_pair_stack.jl           # Template pair processing stack
    template_pointwise_attention.jl  # Cross-attention over templates
    utils.jl                         # dgram, template angle/pair features
  evoformer/
    evoformer_block.jl               # Single Evoformer block
    evoformer_stack.jl               # Chained Evoformer blocks
    extra_msa_block.jl               # Extra MSA block (global attention)
    extra_msa_stack.jl               # Chained extra MSA blocks
    msa_attention.jl                 # Column + global column attention
  heads/
    heads.jl                         # All 5 auxiliary prediction heads
  structure-module/
    structure_module.jl              # Full structure prediction loop
    ipa.jl                           # Invariant Point Attention
    backbone_update.jl               # Backbone rigid update
    angle_resnet.jl                  # Torsion angle prediction
    structure_module_transition.jl   # Transition in structure module
  layers/
    transition.jl                    # Generic LayerNorm + MLP
    pair_stack_block.jl              # Triangle ops + attention + transition
  utils/
    constants.jl                     # Residue constants, atom mappings
    rigid_utils.jl                   # Rigid body algebra (quaternions)
    geometry.jl                      # Reference frame construction
    feats.jl                         # Feature utilities
    atom_utils.jl                    # Torsion angles, atom positions
    scoring.jl                       # pLDDT, PAE, pTM computation
    weight_loading.jl                # NPZ checkpoint loader
```

### Tensor Layout

Julia convention: `[C, spatial..., B]` (channel-first, batch-last).

| Tensor | Julia shape | Python shape (OpenFold) |
|--------|-------------|-------------------------|
| Target features | `[22, N_res, B]` | `[B, N_res, 22]` |
| MSA features | `[49, N_res, N_seq, B]` | `[B, N_seq, N_res, 49]` |
| Pair representation | `[c_z, N_res, N_res, B]` | `[B, N_res, N_res, c_z]` |
| Single representation | `[c_s, N_res, B]` | `[B, N_res, c_s]` |
| Template angles | `[57, N_res, N_templ, B]` | `[B, N_templ, N_res, 57]` |
| Template pair | `[88, N_res, N_res, N_templ, B]` | `[B, N_templ, N_res, N_res, 88]` |

## Usage

### Constructing the Model

```julia
using AlphaFold2, Lux, Random

# Default model (model_1, monomer, 8 recycling iterations)
model = AlphaFold2.AlphaFold()

# Reduced model for testing
model = AlphaFold2.AlphaFold(
    chn_msa=32,
    chn_pair=16,
    chn_single=48,
    num_blocks=2,
    num_blocks_extra=1,
    num_ipa=2,
    num_ipa_blocks=2,
    num_angle_resnet_blocks=1
)

rng = Random.Xoshiro(42)
ps, st = Lux.setup(rng, model)
```

### Running Predictions

```julia
# Single forward iteration
result, st = iteration(model, features, ps, st)

# Full prediction with recycling
result = predict(model, features, ps, st; num_recycles=3)

# Or use the model as a functor
result = predict(features, ps, st)
```

### Loading Official Weights

```julia
# Download and extract weights (downloads ~4 GB)
download_alphafold2_weights()

# Load into a model
model_ps = load_alphafold2_weights!(model_name)  # e.g., "model_1"
```

## Testing

### Running Tests

```julia
# From the root directory:
julia --project=lib/AlphaFold2/test lib/AlphaFold2/test/runtests.jl

# Or activate the test environment:
] activate lib/AlphaFold2/test
] test
```

### Test Requirements

Tests require:
- The Python virtual environment at `python/openfold/.venv/`
- The OpenFold reference implementation (`python/openfold/`)
- `PyCall` (configured to use the venv Python)

### Test Patterns

Every layer test follows this pattern:

1. **Python parity** at Float64, Float32, and Float16
2. **Type-stability** via `@test_nowarn @inferred`
3. **Mask parameterisation** — tested with `nothing` and random `Bool` masks

```julia
for (mask_name, mask_jl) in [
    ("No mask", nothing),
    ("Random mask", rand(rng, Bool, N, N, B))
]
    @testset "$mask_name" begin
        for T in [Float64, Float32, Float16]
            sync_dense!(py_linear, jl_ps.linear_1)
            y_jl, st = jl_layer(x_jl, mask_jl, jl_ps, jl_st)
            y_py = py_layer(x_py, mask_py).detach().numpy()
            @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
            @test_nowarn @inferred jl_layer(x_jl, mask_jl, jl_ps, jl_st)
        end
    end
end
```

## Dependencies

| Package | Purpose |
|---------|---------|
| [Lux.jl](https://github.com/LuxDL/Lux.jl) | Deep learning framework |
| [LuxFoldCore](https://github.com/LuxDL/LuxFoldCore.jl) | Shared layers (AdaLN, AttentionPairBias, OuterProductMean, …) |
| [LuxTriangleAttention](https://github.com/LuxDL/LuxTriangleAttention.jl) | Triangle attention (outgoing + incoming) for pair stacks |
| [Static.jl](https://github.com/JuliaCollections/Static.jl) | Compile-time boolean dispatch |
| [NPZ.jl](https://github.com/fhs/NPZ.jl) | NumPy `.npz` file reading for weight loading |

## References

- Jumper et al., *"Highly accurate protein structure prediction with AlphaFold"*, Nature 2021. [DOI: 10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- [OpenFold](https://github.com/aqlaboratory/openfold) — PyTorch reference implementation
- [AlphaFold2 Github](https://github.com/google-deepmind/alphafold) — Official DeepMind source

## License

MIT
