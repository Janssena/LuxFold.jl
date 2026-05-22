struct InputEmbedder{L1,L2,L3,L4,RE} <: Lux.AbstractLuxContainerLayer{(:linear_tf_z_i, :linear_tf_z_j, :linear_tf_m, :linear_msa_m, :relpos_encoding)}
    linear_tf_z_i::L1
    linear_tf_z_j::L2
    linear_tf_m::L3
    linear_msa_m::L4
    relpos_encoding::RE
end

function InputEmbedder(
    tf_dim::Int, msa_dim::Int, c_z::Int, c_m::Int, relpos_k::Int;
    is_multimer=false, use_bias=true
)
    use_bias = resolve_defaults(use_bias, (:linear_tf_z_i, :linear_tf_z_j, :linear_tf_m, :linear_msa_m))

    return InputEmbedder(
        Lux.Dense(tf_dim => c_z; use_bias=use_bias.linear_tf_z_i),
        Lux.Dense(tf_dim => c_z; use_bias=use_bias.linear_tf_z_j),
        Lux.Dense(tf_dim => c_m; use_bias=use_bias.linear_tf_m),
        Lux.Dense(msa_dim => c_m; use_bias=use_bias.linear_msa_m),
        RelativePositionEncoding(c_z, relpos_k; is_multimer),
    )
end

function (l::InputEmbedder)(target_feat, residue_index, msa_feat, ps, st)
    c_z, c_m = l.linear_tf_z_i.out_dims, l.linear_tf_m.out_dims
    N, B = size(residue_index)

    z_relpos, st_relpos = l.relpos_encoding(residue_index, ps.relpos_encoding, st.relpos_encoding)

    tf_i, st_tf_i = l.linear_tf_z_i(target_feat, ps.linear_tf_z_i, st.linear_tf_z_i)
    tf_j, st_tf_j = l.linear_tf_z_j(target_feat, ps.linear_tf_z_j, st.linear_tf_z_j)
    z = z_relpos .+ reshape(tf_i, c_z, N, 1, B) .+ reshape(tf_j, c_z, 1, N, B)

    tf_m, st_tf_m = l.linear_tf_m(target_feat, ps.linear_tf_m, st.linear_tf_m)
    msa, st_msa_m = l.linear_msa_m(msa_feat, ps.linear_msa_m, st.linear_msa_m)
    m = msa .+ reshape(tf_m, c_m, N, 1, B)

    st_out = merge(st, (;
        linear_tf_z_i=st_tf_i, linear_tf_z_j=st_tf_j,
        linear_tf_m=st_tf_m, linear_msa_m=st_msa_m,
        relpos_encoding=st_relpos,
    ))

    return (m, z), st_out
end
