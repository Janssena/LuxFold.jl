struct OuterProductMean{LN, L1, L2, LO} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :linear1, :linear2, :linear_out)}
    layer_norm::LN
    linear1::L1
    linear2::L2
    linear_out::LO
    eps::Float32
    use_clamp::Bool
    project_first::Bool
end

function OuterProductMean(
    c_m::Int, c_z::Int, c_hidden::Int;
    eps=1e-3, use_bias=true, use_clamp=false, project_first=false
)
    return OuterProductMean(
        Lux.LayerNorm((c_m, 1, 1); dims=1),
        Lux.Dense(c_m => c_hidden; use_bias=use_bias),
        Lux.Dense(c_m => c_hidden; use_bias=use_bias),
        Lux.Dense(c_hidden^2 => c_z),
        Float32(eps),
        use_clamp,
        project_first
    )
end

(l::OuterProductMean)(inputs::NamedTuple, ps, st) = l(
    inputs.m, 
    get(inputs, :mask, nothing),
    ps, st
)

(l::OuterProductMean)(m, ps, st) = l(m, nothing, ps, st)

function (l::OuterProductMean)(m::AbstractArray{T, 4}, mask, ps, st) where T
    C_m, N_seq, N_res, B = size(m)
    C_h = size(ps.linear1.weight, 1)

    # 1. Normalize and Project
    m_ln, st_ln = l.layer_norm(m, ps.layer_norm, st.layer_norm)
    
    a, st_l1 = l.linear1(m_ln, ps.linear1, st.linear1)
    b, st_l2 = l.linear2(m_ln, ps.linear2, st.linear2)

    # Handle mask
    if !isnothing(mask)
        # mask is [N_seq, N_res, B]. Reshape to [1, N_seq, N_res, B] is O(1)
        mask_expanded = reshape(mask, 1, N_seq, N_res, B)
        a = a .* mask_expanded 
        b = b .* mask_expanded 
    end

    # 2. Outer Product Sum (Reducing over N_seq / dim 2)
    # To flatten (C_h, N_res) into one dimension for matmul, we MUST permute 
    # because they are separated by N_seq in the [C, S, N, B] layout.
    a_flat = reshape(permutedims(a, (1, 3, 2, 4)), C_h * N_res, N_seq, B)
    b_flat = reshape(permutedims(b, (1, 3, 2, 4)), C_h * N_res, N_seq, B)
    
    # We use rhs_contracting_dim=2 to avoid permutedims(b_flat, (2, 1, 3))
    # Output is [C_h * N_res, C_h * N_res, B]
    outer = Lux.batched_matmul(a_flat, b_flat; lhs_contracting_dim=2, rhs_contracting_dim=2)
    outer = reshape(outer, C_h, N_res, C_h, N_res, B)
    
    # To match Python's row-major flattening of (C1, C2), we put C_h(2) at dim 1.
    # Permutation is required here as non-adjacent dimensions are being grouped.
    outer = permutedims(outer, (3, 1, 2, 4, 5))
    outer = reshape(outer, C_h * C_h, N_res, N_res, B)

    # 3. Normalization Calculation
    if isnothing(mask)
        norm = fill!(similar(m, 1, N_res, N_res, B), T(N_seq))
    else
        # mask is [N_seq, N_res, B]. 
        # We want sum_s mask[s, i, b] * mask[s, j, b].
        # Contract over N_seq (dim 1) for both sides to get [N_res, N_res, B]
        norm = Lux.batched_matmul(T.(mask), T.(mask); lhs_contracting_dim=1, rhs_contracting_dim=1)
        norm = reshape(norm, 1, N_res, N_res, B)
    end

    if l.use_clamp
        norm = max.(norm, one(T))
    else
        norm = norm .+ T(l.eps)
    end

    # 4. Final Projection and Normalization Application
    if l.project_first
        # AF3 style: project then divide
        y, st_out = l.linear_out(outer, ps.linear_out, st.linear_out)
        y = y ./ norm
    else
        # Boltz2 style: divide then project
        outer = outer ./ norm
        y, st_out = l.linear_out(outer, ps.linear_out, st.linear_out)
    end

    return y, (layer_norm=st_ln, linear1=st_l1, linear2=st_l2, linear_out=st_out)
end
