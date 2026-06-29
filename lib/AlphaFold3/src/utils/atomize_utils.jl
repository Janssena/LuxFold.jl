# Utilities for atom-to-token operations

"""
    apply_atom_mask!(x, atom_mask)

Zero out masked atoms of an atom feature array `x [C, N_atom, B]` in place. Dispatches on
`atom_mask::Nothing` (no-op) vs `atom_mask::AbstractArray` ([N_atom, B]). Uses `ifelse`
so masked positions are exactly zero regardless of `x`.
"""
apply_atom_mask!(x, ::Nothing) = nothing
function apply_atom_mask!(x::AbstractArray{T}, mask::AbstractArray{Bool}) where T
    mask_b = reshape(mask, 1, size(mask)...)
    _zero  = zero(T)
    @. x   = ifelse(mask_b, x, _zero)
    return nothing
end

# Gates token counts [N_token, B] in-place by token_mask (no-op for nothing).
_apply_n_apt_mask!(n_apt, ::Nothing) = nothing
function _apply_n_apt_mask!(n_apt::AbstractArray{T}, mask::AbstractArray{Bool}) where T
    _zero = zero(T)
    @. n_apt = ifelse(mask, n_apt, _zero)
    return nothing
end

# One-hot atom→token assignment, optionally gated by atom_mask.
# Dispatching avoids allocating an N_atom×B Bool array when mask is nothing.
_atom_one_hot(T, ati, t_range, ::Nothing, N_atom, B) = T.(ati .== t_range)
function _atom_one_hot(T, ati, t_range, mask::AbstractArray{Bool}, N_atom, B)
    T.((ati .== t_range) .& reshape(mask, N_atom, 1, B))
end

# Gather atom mask validity at representative positions, gated by token_valid.
# For nothing: all atoms valid → output equals token_valid (no extra allocation).
_rep_atom_mask(::Nothing, flat_idx, N_atom, B, token_valid) = token_valid
function _rep_atom_mask(mask::AbstractArray{Bool}, flat_idx, N_atom, B, token_valid)
    reshape(mask, N_atom * B)[flat_idx] .& token_valid
end

"""
    broadcast_token_feat_to_atoms(token_mask, num_atoms_per_token, token_feat; max_num_atoms_per_token=nothing)

Broadcasts token-level features to atom-level features.
Assumes `token_feat` is `[C, N_token, B]` or `[N_token, B]`.
"""
function broadcast_token_feat_to_atoms(token_mask::Union{Nothing, AbstractArray}, num_atoms_per_token::AbstractArray, token_feat::AbstractArray; max_num_atoms_per_token::Union{Int, Nothing}=nothing)
    has_channel = ndims(token_feat) == 3
    token_feat  = has_channel ? token_feat : reshape(token_feat, 1, size(token_feat)...)
    C, N_token, B = size(token_feat)

    n_apt = isnothing(token_mask) ? Int.(num_atoms_per_token) :
            ifelse.(token_mask, Int.(num_atoms_per_token), zero(Int))

    out_feat = if !isnothing(max_num_atoms_per_token)
        # Padded layout: [C, max_apt, N_token, B] → [C, max_apt*N_token, B]
        valid = reshape(1:max_num_atoms_per_token, 1, :, 1, 1) .<=
                reshape(n_apt, 1, 1, N_token, B)                    # [1, max_apt, N_token, B]
        tf_r  = reshape(token_feat, C, 1, N_token, B)
        _zero = zero(eltype(token_feat))
        out   = @. ifelse(valid, tf_r, _zero)                        # [C, max_apt, N_token, B]
        reshape(out, C, max_num_atoms_per_token * N_token, B)
    else
        # Packed layout: build atom_to_token_index via cumsum, then flat-index gather.
        ends      = cumsum(n_apt; dims=1)                # [N_token, B]
        max_atoms = Int(maximum(ends[end, :]))

        # covered[a, t, b] = true iff token t's end index ≥ atom a (0-indexed atom)
        atom_range = reshape(1:max_atoms, :, 1, 1)       # [max_atoms, 1, 1]
        ends_3d    = reshape(ends, 1, N_token, B)         # [1, N_token, B]
        covered    = ends_3d .>= atom_range               # [max_atoms, N_token, B]

        # First token index (along dim 2) covering each atom; defaults to 1 for padding
        ati = map(ci -> ci[2], dropdims(argmax(covered; dims=2); dims=2))  # [max_atoms, B]

        # Flat-index gather: token_feat [C, N_token*B], idx [max_atoms, B]
        b_offsets = reshape((0:B-1) .* N_token, 1, B)    # [1, B]
        flat_idx  = ati .+ b_offsets                      # [max_atoms, B]
        result    = reshape(token_feat, C, N_token * B)[:, flat_idx]   # [C, max_atoms, B]

        # Zero out atoms that don't exist in shorter batch elements
        total_atoms = ends[end:end, :]                    # [1, B]
        valid_atoms = reshape(1:max_atoms, :, 1) .<= total_atoms   # [max_atoms, B]
        va_r  = reshape(valid_atoms, 1, max_atoms, B)
        _zero = zero(eltype(result))
        @. ifelse(va_r, result, _zero)
    end

    return has_channel ? out_feat : reshape(out_feat, size(out_feat, 2), size(out_feat, 3))
end

# aggregate_fn dispatch (StaticSymbol): `:mean` divides the per-token sum by the valid-atom
# count; `:sum` returns the raw sum. No runtime branch on a String.
_aggregate_reduce(::StaticSymbol{:sum}, out_feat, one_hot, T, N_token, B, eps) = out_feat
function _aggregate_reduce(::StaticSymbol{:mean}, out_feat, one_hot, T, N_token, B, eps)
    counts = reshape(dropdims(sum(one_hot; dims=1); dims=1), 1, N_token, B)
    return out_feat ./ (counts .+ T(eps))
end

"""
    aggregate_atom_feat_to_tokens(token_mask, atom_to_token_index, atom_mask, atom_feat; aggregate_fn=static(:mean), eps=1f-9)

Aggregates atom-level features to token-level features.
Assumes `atom_feat` is `[C, N_atom, B]`. `aggregate_fn` is a `StaticSymbol` (`static(:mean)`
or `static(:sum)`) dispatched at compile time.
"""
function aggregate_atom_feat_to_tokens(token_mask::AbstractArray, atom_to_token_index::AbstractArray, atom_mask::Union{Nothing, AbstractArray}, atom_feat::AbstractArray; aggregate_fn::StaticSymbol=static(:mean), eps::Float32=1f-9)
    C, N_atom, B = size(atom_feat)
    N_token      = size(token_mask, 1)
    T            = eltype(atom_feat)

    # NOTE: aggregation is gated ONLY by `atom_mask` — matching openfold, which uses
    # `token_mask` solely for its shape (`n_token`). Do NOT gate on `token_mask[t_idx]`
    # here: that would zero masked tokens that Python computes normally — a divergence
    # invisible under all-true masks, caught by the random-mask parity test.
    masked_feat = isnothing(atom_mask) ? atom_feat :
                  ifelse.(reshape(atom_mask, 1, size(atom_mask)...), atom_feat, zero(T))

    # One-hot assignment: one_hot[a, t, b] = 1 iff atom a maps to token t AND is valid.
    # _atom_one_hot dispatches on atom_mask to avoid allocating trues(N_atom, B).
    t_range = reshape(1:N_token, 1, N_token, 1)                  # [1, N_token, 1]
    ati     = reshape(Int.(atom_to_token_index), N_atom, 1, B)   # [N_atom, 1, B]
    one_hot = _atom_one_hot(T, ati, t_range, atom_mask, N_atom, B)   # [N_atom, N_token, B]

    # Sum via batched matmul: [C, N_atom, B] × [N_atom, N_token, B] → [C, N_token, B]
    out_feat = Lux.batched_matmul(masked_feat, one_hot)

    return _aggregate_reduce(aggregate_fn, out_feat, one_hot, T, N_token, B, eps)
end

"""
    max_atom_per_token_masked_select(atom_feat, max_atom_per_token_mask)

Select atoms from features padded to max atoms per token.
Assumes `atom_feat` is `[C, N_token * max_atoms_per_token, B]`.
"""
function max_atom_per_token_masked_select(atom_feat::AbstractArray, max_atom_per_token_mask::AbstractArray)
    C, N_total, B = size(atom_feat)
    valid         = Bool.(max_atom_per_token_mask)             # [N_total, B] Bool
    max_atoms     = Int(maximum(sum(valid; dims=1)))
    out_feat      = similar(atom_feat, C, max_atoms, B); out_feat .= zero(eltype(atom_feat))

    # Output positions via cumsum; 0 for masked positions
    out_pos = cumsum(valid; dims=1) .* valid                  # [N_total, B], 1-indexed

    # Batch loop is intentional: masked_select is inherently ragged across N_total.
    for b in 1:B
        v     = findall(valid[:, b])                          # atom indices that are valid
        o     = out_pos[v, b]                                 # their 1-indexed output slots
        out_feat[:, o, b] = atom_feat[:, v, b]
    end

    return out_feat
end

"""
    get_token_representative_atoms(batch, x, atom_mask)

Extracts representative atoms per token.
Returns `(rep_x, rep_atom_mask)`.
"""
function get_token_representative_atoms(batch::NamedTuple, x::AbstractArray, atom_mask::Union{Nothing, AbstractArray})
    # TODO: Full implementation requires mapping amino acid Cb, nucleotide C4/C2, etc.
    # For inference continuity without external constants dictionary, we return the
    # start atom (e.g. Ca) as a functional representative atom placeholder.
    C, N_atom, B = size(x)
    N_token       = size(batch.token_mask, 1)
    T             = eltype(x)

    idx         = Int.(batch.start_atom_index)   # [N_token, B], 1-indexed
    token_valid = batch.token_mask               # [N_token, B] Bool

    # Flat-index gather for positions: x [C, N_atom*B], idx [N_token, B]
    b_offsets = reshape((0:B-1) .* N_atom, 1, B)   # [1, B]
    flat_idx  = idx .+ b_offsets                    # [N_token, B]
    rep_x     = reshape(x, C, N_atom * B)[:, flat_idx]   # [C, N_token, B]
    tv_r  = reshape(token_valid, 1, N_token, B)
    _zero = zero(T)
    @. rep_x = ifelse(tv_r, rep_x, _zero)

    # _rep_atom_mask dispatches on atom_mask: nothing → token_valid only (no alloc).
    rep_atom_mask = _rep_atom_mask(atom_mask, flat_idx, N_atom, B, token_valid)

    return rep_x, rep_atom_mask
end

"""
    get_token_center_atoms(batch, x, atom_mask)

Extracts center atoms per token.
Returns `(center_x, center_atom_mask)`.
"""
function get_token_center_atoms(batch::NamedTuple, x::AbstractArray, atom_mask::Union{Nothing, AbstractArray})
    # Center atoms are typically Ca for amino acids, C1' for nucleotides.
    # We use the start atom as a functional placeholder here as well.
    return get_token_representative_atoms(batch, x, atom_mask)
end
