"""
    StableHLOModels

Compile one piece of a Lux model to StableHLO once, then run it — forward or reverse — as many
times as the architecture needs.

Two measured facts about Reactant motivate this package (see `openspec/` notes and the project
memory):

  * **Compile time, not run time, is the binding constraint.** A single-pass compile of a whole
    fold model does not finish; a compile of a few blocks costs seconds. Since a deep stack is the
    *same* block repeated, one compiled chunk serves every layer of it.
  * **Reverse mode is memory-bound.** A single-pass VJP over a deep stack keeps every block's
    activations live. Crossing the chunk boundary by hand bounds the live set to one chunk, at the
    cost of storing (or recomputing) the chunk inputs.

Three concrete types over one interface:

| type                     | what it is                                                        |
|:-------------------------|:------------------------------------------------------------------|
| [`CompiledHLOModule`](@ref) | one compiled layer, run once                                    |
| [`CompiledHLOStack`](@ref)  | an `M`-block chunk compiled once, replayed `K` times (depth `M*K`) |
| [`CompiledHLOModel`](@ref)  | a named sequence of the above, `Lux.Chain`-style                |

All three implement `forward` / `backward` / `active_keys` / `output_keys` / `ct_keys` /
`ct_zeros`, so they nest: a `CompiledHLOModel` entry may itself be any `AbstractHLOModule`.

# Where the architectures live

This package is architecture-agnostic and depends on no model package. A particular model's
assembly — the factory that builds its entries, the glue that chains them, and the `forward` /
`backward` methods dispatched on its `CompiledHLOModel{StaticSymbol{:name}}` — lives in a
**package extension of that model**: `AlphaFold2StableHLOModelsExt`,
`AlphaFold3StableHLOModelsExt`, `Boltz2StableHLOModelsExt`. Loading this package alongside the
model package is what brings them into scope.

The dependency therefore runs model → StableHLOModels, in the one direction that works: the glue
needs the model's own layers and featurisation helpers, and nothing here needs to know a model
exists.
"""
module StableHLOModels

using Enzyme: Enzyme, Reverse, Const
using LuxFoldCore: LuxFoldCore
using Lux: Lux
using Random: Random, AbstractRNG
using Reactant: Reactant, @compile
using Serialization: Serialization
using Static: Static, StaticBool, StaticSymbol, True, False, static
using TOML: TOML

export AbstractHLOModule, CompiledHLOModule, CompiledHLOStack, CompiledHLOModel
export forward, backward, value_and_gradient, compile_backward, active_keys, output_keys,
    ct_keys, ct_zeros
export entry_names, entries, entry, model_name, chunk_size, output_shapes
export PROBE_SECONDS, COMPILE_TIMES, reset_compile_times!

# Saving and loading (see `src/save.jl`)
export HLOBuildSpec, ArraySpec, build_hlo, spec_kind, materialize_inputs, versioned_packages
export save_hlo, load_hlo, load_hlo_spec, load_hlo_manifest

include("interface.jl")
include("utils.jl")
include("module.jl")
include("stack.jl")
include("model.jl")
include("save.jl")

end # module StableHLOModels
