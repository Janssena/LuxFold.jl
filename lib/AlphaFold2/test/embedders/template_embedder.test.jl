const PyTemplateSingleEmbedder = pyimport("openfold.model.embedders").TemplateSingleEmbedder
const PyTemplatePairEmbedder   = pyimport("openfold.model.embedders").TemplatePairEmbedder
const PyTemplateEmbedder       = pyimport("openfold.model.embedders").TemplateEmbedder

function sync_single_embedder!(py_emb, jl_ps)
    sync_dense!(py_emb.linear_1, jl_ps.chain.linear_1)
    sync_dense!(py_emb.linear_2, jl_ps.chain.linear_2)
end

function sync_template_pair_stack_block!(py_block::PyObject, jl_ps::NamedTuple)
    sync_triangle_attention!(py_block.tri_att_start, jl_ps.tri_att_start)
    sync_triangle_attention!(py_block.tri_att_end,   jl_ps.tri_att_end)
    sync_triangle_multiplication!(py_block.tri_mul_out, jl_ps.tri_mul_out)
    sync_triangle_multiplication!(py_block.tri_mul_in,  jl_ps.tri_mul_in)
    sync_layernorm!(py_block.pair_transition.layer_norm, jl_ps.pair_transition.layer_norm)
    sync_dense!(py_block.pair_transition.linear_1,       jl_ps.pair_transition.linear_1)
    sync_dense!(py_block.pair_transition.linear_2,       jl_ps.pair_transition.linear_2)
end

function sync_template_pair_stack!(py_stack::PyObject, jl_ps::NamedTuple)
    for (i, py_block) in enumerate(py_stack.blocks)
        name = Symbol("block_$i")
        sync_template_pair_stack_block!(py_block, jl_ps.blocks[name])
    end
    sync_layernorm!(py_stack.layer_norm, jl_ps.layer_norm)
end

function sync_template_embedder!(py_emb, jl_ps)
    sync_single_embedder!(py_emb.template_single_embedder, jl_ps.template_single_embedder)
    sync_dense!(py_emb.template_pair_embedder.linear, jl_ps.template_pair_embedder.linear)
    sync_template_pair_stack!(py_emb.template_pair_stack, jl_ps.template_pair_stack)
    sync_af3_attention!(py_emb.template_pointwise_att.mha, jl_ps.template_pointwise_att.mha)
end

function _make_py_embedder_config(;
    c_single_in, c_single_out,
    c_pair_in, c_t, c_z, c_m,
    c_hidden_tri_att, c_hidden_tri_mul,
    no_blocks, no_heads_tri, pair_transition_n,
    c_hidden_pt_att, no_heads_pt_att,
    embed_angles
)
    ml = pyimport("ml_collections")
    pydict = pyimport("builtins").dict

    return ml.ConfigDict(pydict(
        template_single_embedder = pydict(
            c_in = c_single_in, 
            c_out = c_single_out
        ),
        template_pair_embedder = pydict(
            c_in = c_pair_in,   
            c_out = c_t)
        ,
        template_pair_stack = pydict(
            c_t = c_t,
            c_hidden_tri_att = c_hidden_tri_att,
            c_hidden_tri_mul = c_hidden_tri_mul,
            no_blocks = no_blocks,
            no_heads = no_heads_tri,
            pair_transition_n = pair_transition_n,
            dropout_rate = 0.0,
            tri_mul_first = false,
            fuse_projection_weights = true,
            blocks_per_ckpt = nothing,
            inf = 1e9,
        ),
        template_pointwise_attention = pydict(
            c_t = c_t,
            c_z = c_z,
            c_hidden = c_hidden_pt_att,
            no_heads = no_heads_pt_att,
            inf = 1e9,
        ),
        inf = 1e9,
        eps = 1e-20,
        use_unit_vector = false,
        embed_angles = embed_angles,
        distogram = pydict(
            min_bin=3.25, 
            max_bin=50.75, 
            no_bins=39
        ),
    ))
end

# Generates a full set of Julia-layout template inputs for TemplateEmbedder tests.
# embed_angles=true also builds angle_feat and the raw torsion arrays.
# Diagonal of pair_mask is forced True so no attention row is fully masked (avoids NaN in Float16).
function make_template_inputs(rng, T; embed_angles=false)
    pseudo_beta      = randn(rng, T, 3, N_RES, N_TEMPL, B) .* T(10)
    pseudo_beta_mask = rand(rng, Bool, N_RES, N_TEMPL, B)
    aatype           = rand(rng, 0:21, N_RES, N_TEMPL, B)
    all_atom_pos     = randn(rng, T, 37, 3, N_RES, N_TEMPL, B) .* T(10)
    all_atom_mask    = rand(rng, Bool, 37, N_RES, N_TEMPL, B)
    all_atom_mask[1:3, :, :, :] .= true  # backbone atoms always present
    template_mask    = rand(rng, Bool, N_TEMPL, B)
    for b in 1:B; any(template_mask[:, b]) || (template_mask[1, b] = true); end

    z         = randn(rng, T, C_Z, N_RES, N_RES, B)
    pair_mask = rand(rng, Bool, N_RES, N_RES, B)
    for b in 1:B, i in 1:N_RES; pair_mask[i, i, b] = true; end

    pair_feat = build_template_pair_feat(pseudo_beta, pseudo_beta_mask, aatype, all_atom_pos, all_atom_mask)

    if embed_angles
        tsc        = randn(rng, T, 2, 7, N_RES, N_TEMPL, B)
        atsc       = randn(rng, T, 2, 7, N_RES, N_TEMPL, B)
        tmask      = rand(rng, Bool, 7, N_RES, N_TEMPL, B)
        angle_feat = build_template_angle_feat(aatype, tsc, atsc, tmask)
        return (; pair_feat, pair_mask, template_mask, z, angle_feat, _tsc=tsc, _atsc=atsc, _tmask=tmask)
    else
        return (; pair_feat, pair_mask, template_mask, z)
    end
end

# Runs the Python TemplateEmbedder forward component-by-component using Julia-layout inputs,
# bypassing py_layer(...) to avoid numerical differences from Python's own feature builder
# and from chunked attention (chunk_size=nothing disables chunking).
# Returns a NamedTuple (; template_pair, template_single) in Julia layout.
# template_single is nothing when inputs has no :angle_feat key.
function run_py_template_embedder(py_layer, inputs, T)
    # pair_feat [C, N, N, N_templ, B] → [B, N_templ, N, N, C]
    pair_feat_py = to_py(permutedims(inputs.pair_feat, (5, 4, 2, 3, 1)); swap_batch_dim=false).to(py_dtype(T))
    t_pair_py    = py_layer.template_pair_embedder(pair_feat_py)

    # pair_mask [N, N, B] → [B, 1, N, N]  (stack broadcasts over N_templ)
    pair_mask_py = to_py(permutedims(inputs.pair_mask, (3, 1, 2)); swap_batch_dim=false).unsqueeze(1).to(py_dtype(T))
    t_pair_py    = py_layer.template_pair_stack(t_pair_py, pair_mask_py, chunk_size=nothing)

    # template_mask [N_templ, B] → [B, N_templ] Float
    template_mask_py = to_py(permutedims(Float32.(inputs.template_mask), (2, 1)); swap_batch_dim=false)
    t_att_py = py_layer.template_pointwise_att(
        t_pair_py, to_py(inputs.z; swap_batch_dim=true), template_mask_py; chunk_size=nothing
    )

    # Zero out when no valid templates — mirrors Julia's step 4
    t_valid_py = template_mask_py.sum(dim=-1).gt(0).reshape(B, 1, 1, 1)
    tp_py      = to_jl(t_att_py * t_valid_py; swap_batch_dim=true)  # [C_z, N, N, B]

    ts_py = if haskey(inputs, :angle_feat)
        # angle_feat [C, N_res, N_templ, B] → [B, N_templ, N_res, C]
        angle_feat_py = to_py(permutedims(inputs.angle_feat, (4, 3, 2, 1)); swap_batch_dim=false).to(py_dtype(T))
        permutedims(to_jl(py_layer.template_single_embedder(angle_feat_py); swap_batch_dim=false), (4, 3, 2, 1))
    else
        nothing
    end

    return (; template_pair=tp_py, template_single=ts_py)
end

rng = Random.Xoshiro(42)

# Shared dims
C_PAIR_IN = 88
C_ANGLE_IN = 57
C_T = 16     # template embedding channels
C_Z = 32     # pair embedding channels
C_M = 64     # MSA/angle output channels
C_HIDDEN_TRI_ATT = 4
C_HIDDEN_TRI_MUL = 16
C_HIDDEN_PT_ATT  = 4
NO_BLOCKS = 2
NO_HEADS_TRI = 4
NO_HEADS_PT = 4
PT_N = 2      # pair transition expansion factor
N_RES, N_TEMPL, B = 6, 3, 2

@testset "Template Embedder Pipeline" begin
    @testset "TemplateSingleEmbedder" begin
        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                jl_layer = TemplateSingleEmbedder(C_ANGLE_IN, C_M)
                ps, st   = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyTemplateSingleEmbedder(c_in=C_ANGLE_IN, c_out=C_M)
                sync_single_embedder!(py_layer, ps)

                x_jl = randn(rng, T, C_ANGLE_IN, N_RES, N_TEMPL, B)
                # Python layout: [B, N_templ, N_res, C]
                x_py = to_py(permutedims(x_jl, reverse(1:4)); swap_batch_dim=false)

                y_jl, _ = jl_layer(x_jl, ps, st)
                y_py = py_layer(x_py)

                @testset "Python parity" begin
                    # Python output: [B, N_templ, N_res, C_out] → Julia [C_out, N_res, N_templ, B]
                    @test y_jl ≈ permutedims(to_jl(y_py; swap_batch_dim=false), reverse(1:4))
                end
                @testset "Type stability" begin
                    @test_nowarn @inferred jl_layer(x_jl, ps, st)
                end
            end
        end
    end

    # =========================================================================
    @testset "TemplatePairEmbedder" begin
        for T in [Float64, Float32, Float16]
            @testset "$T" begin
                jl_layer = TemplatePairEmbedder(C_PAIR_IN, C_T)
                ps, st   = Lux.setup(rng, jl_layer) |> convert_types(T)

                py_layer = PyTemplatePairEmbedder(c_in=C_PAIR_IN, c_out=C_T)
                sync_dense!(py_layer.linear, ps.linear)

                x_jl = randn(rng, T, C_PAIR_IN, N_RES, N_RES, N_TEMPL, B)
                # Python layout: [B, N_templ, N, N, C]
                x_py = to_py(permutedims(x_jl, (5, 4, 2, 3, 1)); swap_batch_dim=false)

                y_jl, _ = jl_layer(x_jl, ps, st)
                y_py     = py_layer(x_py)

                # Python output: [B, N_templ, N, N, C_out] → Julia [C_out, N, N, N_templ, B]
                y_py_jl = permutedims(to_jl(y_py; swap_batch_dim=false), (5, 3, 4, 2, 1))

                @testset "Python parity" begin
                    @test y_jl ≈ y_py_jl
                end
                @testset "Type stability" begin
                    @test_nowarn @inferred jl_layer(x_jl, ps, st)
                end
            end
        end
    end

    # =========================================================================
    @testset "TemplateEmbedder" begin
        for embed_angles in [false, true]
            @testset "embed_angles=$embed_angles" begin
                for T in [Float64, Float32, Float16]
                    @testset "$T" begin
                        jl_layer = TemplateEmbedder(
                            C_PAIR_IN, C_ANGLE_IN, C_T, C_Z, C_M,
                            C_HIDDEN_TRI_ATT, C_HIDDEN_TRI_MUL,
                            NO_BLOCKS, NO_HEADS_TRI, PT_N,
                            C_HIDDEN_PT_ATT, NO_HEADS_PT
                        )
                        ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                        py_config = _make_py_embedder_config(;
                            c_single_in=C_ANGLE_IN, c_single_out=C_M,
                            c_pair_in=C_PAIR_IN, c_t=C_T, c_z=C_Z, c_m=C_M,
                            c_hidden_tri_att=C_HIDDEN_TRI_ATT,
                            c_hidden_tri_mul=C_HIDDEN_TRI_MUL,
                            no_blocks=NO_BLOCKS, no_heads_tri=NO_HEADS_TRI,
                            pair_transition_n=PT_N,
                            c_hidden_pt_att=C_HIDDEN_PT_ATT, no_heads_pt_att=NO_HEADS_PT,
                            embed_angles=embed_angles,
                        )
                        py_layer = PyTemplateEmbedder(py_config)
                        sync_template_embedder!(py_layer, ps)

                        inputs_jl  = make_template_inputs(rng, T; embed_angles)
                        ret_jl, _  = jl_layer(inputs_jl, ps, st)
                        ret_py     = run_py_template_embedder(py_layer, inputs_jl, T)

                        @testset "Python parity (template_pair)" begin
                            @test ret_jl.template_pair ≈ ret_py.template_pair
                        end

                        if embed_angles
                            @testset "Python parity (template_single)" begin
                                @test ret_jl.template_single ≈ ret_py.template_single
                            end
                        else
                            @testset "template_single is nothing when embed_angles=false" begin
                                @test isnothing(ret_jl.template_single)
                            end
                        end

                        @testset "Type stability" begin
                            @test_nowarn @inferred jl_layer(inputs_jl, ps, st)
                        end
                    end
                end
            end
        end
    end
end
