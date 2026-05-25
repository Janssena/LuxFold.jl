const PyTemplatePointwiseAttention = pyimport("openfold.model.template").TemplatePointwiseAttention

function sync_template_pointwise_attention!(py_layer::PyObject, jl_ps::NamedTuple)
    sync_af3_attention!(py_layer.mha, jl_ps.mha)
end

# ===  Dimension conventions  ===
#
# Julia t:    [C_t, Ni, Nj, N_templ, B]  ←→  Python t:    [B, N_templ, Ni, Nj, C_t]
# Julia z:    [C_z, Ni, Nj, B]           ←→  Python z:    [B, Ni, Nj, C_z]
# Julia mask: [N_templ, B]               ←→  Python mask: [B, N_templ]   (Float, not Bool)
#
# t to Python:    to_py(permutedims(t_jl, (5, 4, 2, 3, 1)); swap_batch_dim=false)
# z to Python:    to_py(z_jl; swap_batch_dim=true)
# mask to Python: to_py(permutedims(Float32.(mask), (2, 1)); swap_batch_dim=false)
# output to Julia: to_jl(y_py; swap_batch_dim=true)

rng = Random.Xoshiro(42)

C_t      = 16
C_z      = 32
C_hidden = 4
no_heads = 4
N_res, N_templ, B = 6, 3, 2

# template_mask: [N_templ, B] Bool — at least one valid template per batch element
function make_template_mask(rng, N_templ, B)
    mask = rand(rng, Bool, N_templ, B)
    for b in 1:B
        if !any(mask[:, b])
            mask[rand(rng, 1:N_templ), b] = true
        end
    end
    return mask
end

@testset "TemplatePointwiseAttention" begin
    for T in [Float64, Float32, Float16]
        @testset "$T" begin
            mask_cfg = (
                ("No mask", nothing),
                ("Random mask", make_template_mask(rng, N_templ, B))
            )
            for (name, mask) in mask_cfg
                @testset "$name" begin
                    jl_layer = TemplatePointwiseAttention(C_t, C_z, C_hidden, no_heads)
                    ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                    template = randn(rng, T, C_t, N_res, N_res, N_templ, B)
                    z = randn(rng, T, C_z, N_res, N_res, B)

                    py_layer = PyTemplatePointwiseAttention(
                        c_t=C_t, c_z=C_z, c_hidden=C_hidden,
                        no_heads=no_heads, inf=1e9
                    )
                    sync_template_pointwise_attention!(py_layer, ps)

                    t_py = to_py(permutedims(template, (5, 4, 2, 3, 1)); swap_batch_dim=false)
                    z_py = to_py(z; swap_batch_dim=true)
                    mask_py = isnothing(mask) ? nothing : to_py(mask; swap_batch_dim=true).to(py_dtype(T))

                    y_jl, _ = jl_layer(template, z, mask, ps, st)
                    y_py    = py_layer(t_py, z_py, template_mask=mask_py, chunk_size=nothing)

                    @testset "Python parity" begin
                        @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                    end

                    @testset "Type-stability" begin
                        @test_nowarn @inferred jl_layer(template, z, mask, ps, st)
                    end
                end
            end
        end
    end
end
