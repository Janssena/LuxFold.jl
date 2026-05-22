const PyInputEmbedder = pyimport("openfold.model.embedders").InputEmbedder
const PyInputEmbedderMultimer = pyimport("openfold.model.embedders").InputEmbedderMultimer

rng = Random.Xoshiro(42)

@testset "RelativePositionEncoding" begin
    @testset "Monomer" begin
        CHN_PAIR, RELPOS_K = 16, 8
        N, B = 6, 2

        @testset "Python parity" begin
            for T in [Float64, Float32, Float16]
                @testset "$T" begin
                    jl_layer = RelativePositionEncoding(CHN_PAIR, RELPOS_K)
                    jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = PyInputEmbedder(22, 49, CHN_PAIR, 256, RELPOS_K)
                    sync_dense!(py_layer.linear_relpos, jl_ps.linear)

                    ri = rand(rng, 1:100, N, B)

                    y_jl, _ = jl_layer(ri, jl_ps, jl_st)
                    ri_py = to_py(ri; swap_batch_dim=true).to(py_dtype(T))
                    y_py = py_layer.relpos(ri_py)

                    @testset "Parity" begin
                        @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                    end

                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer(ri, jl_ps, jl_st)
                    end
                end
            end
        end
    end

    @testset "Multimer" begin
        CHN_PAIR, RELPOS_K, MAX_REL_CHAIN = 16, 8, 2
        N, B = 6, 2

        @testset "Python parity" begin
            for T in [Float64, Float32, Float16]
                @testset "$T" begin
                    jl_layer = RelativePositionEncoding(CHN_PAIR, RELPOS_K; is_multimer=true, max_relative_chain=MAX_REL_CHAIN)
                    jl_ps, jl_st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    py_layer = PyInputEmbedderMultimer(22, 49, CHN_PAIR, 256, RELPOS_K; use_chain_relative=true, max_relative_chain=MAX_REL_CHAIN)
                    sync_dense!(py_layer.linear_relpos, jl_ps.linear)

                    ri = rand(rng, 1:100, N, B)
                    asym_id = rand(rng, 1:2, N, B)
                    entity_id = rand(rng, 1:2, N, B)
                    sym_id = rand(rng, 1:2, N, B)

                    ri_py = to_py(ri; swap_batch_dim=true).to(py_dtype(T))
                    asym_py = to_py(asym_id; swap_batch_dim=true).to(py_dtype(T))
                    entity_py = to_py(entity_id; swap_batch_dim=true).to(py_dtype(T))
                    sym_py = to_py(sym_id; swap_batch_dim=true).to(py_dtype(T))

                    batch = py"dict(residue_index=$ri_py.long(), asym_id=$asym_py.long(), entity_id=$entity_py.long(), sym_id=$sym_py.long())"

                    y_jl, _ = jl_layer(ri, asym_id, entity_id, sym_id, jl_ps, jl_st)
                    y_py = py_layer.relpos(batch)

                    @testset "Parity" begin
                        @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                    end

                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer(ri, asym_id, entity_id, sym_id, jl_ps, jl_st)
                    end
                end
            end
        end
    end
end
