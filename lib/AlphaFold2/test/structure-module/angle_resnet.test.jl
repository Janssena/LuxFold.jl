const PyAngleResnetBlock = pyimport("openfold.model.structure_module").AngleResnetBlock
const PyAngleResnet = pyimport("openfold.model.structure_module").AngleResnet

rng = Random.Xoshiro(42)

function sync_angle_resnet_block!(py_block, jl_ps)
    sync_dense!(py_block.linear_1, jl_ps.linear_1)
    sync_dense!(py_block.linear_2, jl_ps.linear_2)
end

function sync_angle_resnet!(py_layer, jl_ps)
    sync_dense!(py_layer.linear_in,      jl_ps.linear_in)
    sync_dense!(py_layer.linear_initial, jl_ps.linear_initial)
    # Iterate Python's nn.ModuleList directly (avoids 0-based index arithmetic).
    # Julia keys are layer_1, layer_2, … (generic Lux.Chain naming).
    for (i, py_block) in enumerate(getproperty(py_layer, "layers"))
        sync_angle_resnet_block!(py_block, jl_ps.blocks[Symbol("layer_$i")])
    end
    sync_dense!(py_layer.linear_out, jl_ps.linear_out)
end

@testset "AngleResnetBlock" begin
    @testset "Python parity" begin
        chn_hidden, N, B = 16, 5, 2

        for T in (Float64, Float32, Float16)
            @testset "$T" begin
                jl_layer = AngleResnetBlock(chn_hidden)
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyAngleResnetBlock(chn_hidden)
                py_layer.eval()
                sync_angle_resnet_block!(py_layer, jl_ps)

                s_jl = randn(rng, T, chn_hidden, N, B)
                s_py = to_py(s_jl; swap_batch_dim=true)

                out_jl, _ = jl_layer(s_jl, jl_ps, jl_st)
                out_py = py_layer(s_py)

                @testset "Python parity" begin
                    @test out_jl ≈ to_jl(out_py; swap_batch_dim=true)
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred jl_layer(s_jl, jl_ps, jl_st)
                end
            end
        end
    end
end

@testset "AngleResnet" begin
    @testset "Python parity" begin
        chn_s, chn_hidden, no_blocks, no_angles, N, B = 16, 8, 2, 7, 5, 2

        for T in (Float64, Float32, Float16)
            @testset "$T" begin
                jl_layer = AngleResnet(chn_s, chn_hidden; no_blocks, no_angles, epsilon=1f-8)
                jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyAngleResnet(chn_s, chn_hidden, no_blocks, no_angles, 1e-8)
                sync_angle_resnet!(py_layer, jl_ps)

                s_jl      = randn(rng, T, chn_s, N, B)
                s_init_jl = randn(rng, T, chn_s, N, B)
                s_py      = to_py(s_jl; swap_batch_dim=true)
                s_init_py = to_py(s_init_jl; swap_batch_dim=true)

                ar_out, _ = jl_layer(s_jl, s_init_jl, jl_ps, jl_st)
                unnorm_jl  = ar_out.unnormalized_angles
                angles_jl  = ar_out.angles
                unnorm_py, angles_py = py_layer(s_py, s_init_py)

                # Python returns [B, N, no_angles, 2] — channel-last
                # Convert: numpy [B, N, no_angles, 2] -> Julia [2, no_angles, N, B]
                unnorm_py_jl = permutedims(to_jl(unnorm_py; swap_batch_dim=false), (4, 3, 2, 1))
                angles_py_jl = permutedims(to_jl(angles_py; swap_batch_dim=false), (4, 3, 2, 1))

                @testset "Unnormalized angles" begin
                    @test unnorm_jl ≈ unnorm_py_jl
                end
                @testset "Angles" begin
                    @test angles_jl ≈ angles_py_jl
                end
                @testset "Type-stability" begin
                    @test_nowarn @inferred jl_layer(s_jl, s_init_jl, jl_ps, jl_st)
                end
            end
        end
    end
end
