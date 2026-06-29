const PyRelpos = pyimport("openfold3.core.utils.relpos")

@testset "relpos_complex" begin
    N, B = 8, 2
    max_idx, max_chain = 32, 2
    C_relpos = (2max_idx + 2) + (2max_idx + 2) + 1 + (2max_chain + 2)  # 139

    # Two-chain homodimer: chain A (asym 1) + chain B (asym 2), same entity.
    half = N ÷ 2
    residue_index = repeat(reshape(collect(1:N), N, 1), 1, B)
    token_index   = repeat(reshape(collect(1:N), N, 1), 1, B)
    asym_id       = repeat(reshape([fill(1, half); fill(2, half)], N, 1), 1, B)
    sym_id        = copy(asym_id)               # 1st/2nd copy of the entity
    entity_id     = ones(Int, N, B)             # homodimer → same entity

    batch = (; residue_index, token_index, asym_id, sym_id, entity_id)

    _to_pybatch(b) = Dict(string(k) => to_py(getfield(b, k); swap_batch_dim=true) for k in keys(b))

    @testset "Python parity" begin
        out_jl = relpos_complex(batch, max_idx, max_chain, Float64)
        out_py = PyRelpos.relpos_complex(_to_pybatch(batch), max_idx, max_chain)
        @test out_jl ≈ to_jl(out_py; swap_batch_dim=true)
    end

    @testset "heterodimer parity" begin
        het = (; residue_index, token_index, asym_id, sym_id,
               entity_id=repeat(reshape([fill(1, half); fill(2, half)], N, 1), 1, B))
        out_jl = relpos_complex(het, max_idx, max_chain, Float64)
        out_py = PyRelpos.relpos_complex(_to_pybatch(het), max_idx, max_chain)
        @test out_jl ≈ to_jl(out_py; swap_batch_dim=true)
    end

    @testset "shape + dtype" begin
        @test size(relpos_complex(batch, max_idx, max_chain, Float32)) == (C_relpos, N, N, B)
        @test eltype(relpos_complex(batch, max_idx, max_chain, Float32)) == Float32
        @test eltype(relpos_complex(batch, max_idx, max_chain, Float64)) == Float64
    end

    @testset "type-stability" begin
        @test_nowarn @inferred relpos_complex(batch, max_idx, max_chain, Float32)
    end
end
