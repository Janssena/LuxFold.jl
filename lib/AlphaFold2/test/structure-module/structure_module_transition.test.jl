const PyStructureModuleTransitionLayer =
    pyimport("openfold.model.structure_module").StructureModuleTransitionLayer
const PyStructureModuleTransition =
    pyimport("openfold.model.structure_module").StructureModuleTransition

rng = Random.Xoshiro(42)

function sync_structure_module_transition_layer!(py_layer, jl_ps)
    sync_dense!(py_layer.linear_1, jl_ps.chain.layer_1)
    sync_dense!(py_layer.linear_2, jl_ps.chain.layer_2)
    sync_dense!(py_layer.linear_3, jl_ps.chain.layer_3)
end

function sync_structure_module_transition!(py_layer, jl_ps)
    # Iterate Python's nn.ModuleList directly (avoids 0-based index arithmetic).
    # Julia keys are layer_1, layer_2, … (generic Lux.Chain naming).
    for (i, py_block) in enumerate(getproperty(py_layer, "layers"))
        sync_structure_module_transition_layer!(py_block, jl_ps.layers[Symbol("layer_$i")])
    end
    sync_layernorm!(py_layer.layer_norm, jl_ps.layer_norm)
end

@testset "StructureModuleTransitionLayer" begin
    @testset "Python parity" begin
        chn_s, N, B = 16, 5, 2

        for T in (Float64, Float32, Float16)
            @testset "$T" begin
                jl_layer = StructureModuleTransitionLayer(chn_s)
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyStructureModuleTransitionLayer(chn_s)
                sync_structure_module_transition_layer!(py_layer, jl_ps)

                x_jl = randn(rng, T, chn_s, N, B)
                x_py = to_py(x_jl; swap_batch_dim=true)

                y_jl, _ = jl_layer(x_jl, jl_ps, jl_st)
                y_py = py_layer(x_py)

                @testset "Output" begin
                    @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred jl_layer(x_jl, jl_ps, jl_st)
                end
            end
        end
    end
end

@testset "StructureModuleTransition" begin
    @testset "Python parity" begin
        chn_s, num_layers, N, B = 16, 2, 5, 2

        for T in (Float64, Float32, Float16)
            @testset "$T" begin
                jl_layer = StructureModuleTransition(chn_s; num_layers)
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                # Python signature: StructureModuleTransition(c, num_layers, dropout_rate)
                py_layer = PyStructureModuleTransition(chn_s, num_layers, 0.0)
                # Put in eval mode so dropout becomes a no-op
                py_layer.eval()
                sync_structure_module_transition!(py_layer, jl_ps)

                x_jl = randn(rng, T, chn_s, N, B)
                x_py = to_py(x_jl; swap_batch_dim=true)

                y_jl, _ = jl_layer(x_jl, jl_ps, jl_st)
                y_py = py_layer(x_py)

                @testset "Output" begin
                    @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred jl_layer(x_jl, jl_ps, jl_st)
                end
            end
        end
    end
end
