# Rigid body transformations and all-atom rollout logic

"""
    quat_to_rot(q)

Convert a quaternion [q0, q1, q2, q3] to a rotation matrix.
In AF2, q is often [1, q1, q2, q3] before normalization.
"""
function quat_to_rot(q::AbstractArray{T, 3}) where T
    # q: [3, n, b] (assuming q0 = 1)
    n, b = size(q, 2), size(q, 3)
    
    q1, q2, q3 = q[1, :, :], q[2, :, :], q[3, :, :]
    norm_sq = 1 .+ q1.^2 .+ q2.^2 .+ q3.^2
    
    R = zeros(T, 3, 3, n, b)
    
    # R = I + 2/norm_sq * [ -(q2^2 + q3^2), q1*q2 - q3, q1*q3 + q2;
    #                       q1*q2 + q3, -(q1^2 + q3^2), q2*q3 - q1;
    #                       q1*q3 - q2, q2*q3 + q1, -(q1^2 + q2^2) ]
    
    @. R[1, 1, :, :] = (1 - q1^2 - q2^2 - q3^2) / norm_sq
    @. R[1, 2, :, :] = 2 * (q1 * q2 - q3) / norm_sq
    @. R[1, 3, :, :] = 2 * (q1 * q3 + q2) / norm_sq
    
    @. R[2, 1, :, :] = 2 * (q1 * q2 + q3) / norm_sq
    @. R[2, 2, :, :] = (1 - q1^2 + q2^2 - q3^2) / norm_sq
    @. R[2, 3, :, :] = 2 * (q2 * q3 - q1) / norm_sq
    
    @. R[3, 1, :, :] = 2 * (q1 * q3 - q2) / norm_sq
    @. R[3, 2, :, :] = 2 * (q2 * q3 + q1) / norm_sq
    @. R[3, 3, :, :] = (1 - q1^2 - q2^2 + q3^2) / norm_sq
    
    return R
end

function compose_frames(r1, r2_update)
    # r1: (R1, T1), r2_update: (R2, T2)
    # R_out = R1 * R2, T_out = T1 + R1 * T2
    
    R1, T1 = r1.R, r1.T
    R2, T2 = r2_update.R, r2_update.T
    
    # Rotation composition
    R_out = Lux.batched_matmul(R1, R2)
    
    # Translation composition
    T_out = T1 .+ reshape(Lux.batched_matmul(R1, reshape(T2, 3, 1, size(T2, 2), size(T2, 3))), 3, size(T2, 2), size(T2, 3))
    
    return (R=R_out, T=T_out)
end

function compute_all_atom_coordinates(frames, torsions)
    # Placeholder for Algorithm 24
    # This requires the residue geometry table
    return nothing
end
