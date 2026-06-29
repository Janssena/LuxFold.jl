# Utilities for sequence-local block attention

function get_query_block_padding(n_atom::Int, n_query::Int)
    return mod(-n_atom, n_query)
end

# Shared kernel for get_block_indices — takes pre-computed true_n_atom to avoid
# materialising an all-true mask when atom_mask is nothing.
function _block_indices_kernel(N_atom::Int, B::Int, true_n_atom::AbstractVector{Int}, n_query::Int, n_key::Int)
    num_blocks = ceil(Int, N_atom / n_query)
    offset     = n_query ÷ 2

    # 0-indexed block centers: [num_blocks]
    centers     = offset .+ (0:num_blocks-1) .* n_query
    # 0-indexed key window offsets: [n_key]
    base_window = (-(n_key ÷ 2)):(n_key ÷ 2 - 1 + n_key % 2)

    # Initial gathers (0-indexed): [n_key, num_blocks]
    initial_gathers = reshape(base_window, :, 1) .+ reshape(centers, 1, :)

    max_idx = reshape(true_n_atom .- 1, 1, 1, B)   # [1, 1, B]

    # Expand initial_gathers for B dimension (block indices are batch-independent)
    ig = reshape(initial_gathers, n_key, num_blocks, 1)   # [n_key, num_blocks, 1]

    # Underflow: how much the first key index of each block is below 0 [1, num_blocks, 1]
    underflow = reshape(max.(0, -initial_gathers[1, :]), 1, num_blocks, 1)
    # Overflow: how much the last key index exceeds the per-batch max index [1, num_blocks, B]
    overflow  = max.(0, reshape(initial_gathers[end, :], 1, num_blocks, 1) .- max_idx)
    # Prioritise underflow: if a block underflows, shift right; otherwise shift left
    shift         = ifelse.(underflow .> 0, underflow, .-overflow)   # [1, num_blocks, B]
    final_gathers = ig .+ shift                                       # [n_key, num_blocks, B]

    # 1-indexed safe indices (clamped) and boolean invalid mask
    safe_indices = Int.(clamp.(final_gathers, 0, max_idx)) .+ 1      # [n_key, num_blocks, B]
    invalid_mask = (final_gathers .< 0) .| (final_gathers .> max_idx) # [n_key, num_blocks, B]

    return safe_indices, invalid_mask
end

"""
    get_block_indices(atom_mask, n_query, n_key)

Computes safe key-atom indices and invalid mask for sequence-local block attention.
Returns `(safe_indices [n_key, num_blocks, B], invalid_mask [n_key, num_blocks, B])`.
"""
function get_block_indices(atom_mask::AbstractArray, n_query::Int, n_key::Int)
    N_atom, B   = size(atom_mask)
    true_n_atom = vec(Int.(sum(atom_mask; dims=1)))   # [B]
    return _block_indices_kernel(N_atom, B, true_n_atom, n_query, n_key)
end

# No-mask variant: all N_atom atoms valid — avoids allocating trues(N_atom, B).
function get_block_indices(N_atom::Int, B::Int, n_query::Int, n_key::Int)
    true_n_atom = fill(N_atom, B)   # B-element vector, not N_atom×B
    return _block_indices_kernel(N_atom, B, true_n_atom, n_query, n_key)
end

"""
    get_pair_atom_block_mask(atom_mask, num_blocks, n_query, n_key, safe_indices, invalid_mask)

Builds a `[n_query, n_key, num_blocks, B]` pair mask for block attention.
"""
function get_pair_atom_block_mask(atom_mask::AbstractArray, num_blocks::Int, n_query::Int, n_key::Int, safe_indices::AbstractArray, invalid_mask::AbstractArray)
    # atom_mask: [N_atom, B]
    N_atom, B = size(atom_mask)

    # Query mask: pad and reshape [N_atom, B] → [n_query, num_blocks, B]
    pad_len = get_query_block_padding(N_atom, n_query)
    atom_mask_pad = similar(atom_mask, pad_len, B); atom_mask_pad .= false
    atom_mask_q = vcat(atom_mask, atom_mask_pad)
    atom_mask_q = reshape(atom_mask_q, n_query, num_blocks, B)         # [n_query, num_blocks, B]

    # Key mask: gather from atom_mask using safe_indices via flat linear indexing
    b_offsets = reshape((0:B-1) .* N_atom, 1, 1, B)                  # [1, 1, B]
    flat_idx = safe_indices .+ b_offsets                             # [n_key, num_blocks, B]
    am_flat = reshape(atom_mask, N_atom * B)
    atom_mask_k = reshape(am_flat[flat_idx], n_key, num_blocks, B)     # [n_key, num_blocks, B]
    atom_mask_k = ifelse.(invalid_mask, zero(eltype(atom_mask_k)), atom_mask_k)

    # Outer-AND pair mask (both operands Bool): [n_query, n_key, num_blocks, B]
    atom_pair_mask = reshape(atom_mask_q, n_query, 1, num_blocks, B) .&
                     reshape(atom_mask_k, 1, n_key, num_blocks, B)

    return atom_pair_mask
end

# No-mask variant: all atoms valid — query mask is 1 for non-padding slots, key mask
# is !invalid_mask. Avoids allocating trues(N_atom, B).
function get_pair_atom_block_mask(::Nothing, N_atom::Int, B::Int, num_blocks::Int, n_query::Int, n_key::Int, safe_indices::AbstractArray, invalid_mask::AbstractArray)
    pad_len = get_query_block_padding(N_atom, n_query)

    # Query: 1 for real atoms, 0 for padding slots
    q_vals      = vcat(trues(N_atom, B), falses(pad_len, B))
    atom_mask_q = reshape(q_vals, n_query, num_blocks, B)

    # Key: valid iff not in an invalid window position
    atom_mask_k = .!invalid_mask                                        # [n_key, num_blocks, B]

    return reshape(atom_mask_q, n_query, 1, num_blocks, B) .&
           reshape(atom_mask_k, 1, n_key, num_blocks, B)
end

# Combinator: dispatches get_block_indices + get_pair_atom_block_mask together,
# selecting the mask-aware or no-mask overload based on atom_mask's type.
function _get_block_info(atom_mask::AbstractArray, N_atom::Int, B::Int, n_query::Int, n_key::Int, num_blocks::Int)
    si, im = get_block_indices(atom_mask, n_query, n_key)
    pm = get_pair_atom_block_mask(atom_mask, num_blocks, n_query, n_key, si, im)
    return si, im, pm
end

function _get_block_info(::Nothing, N_atom::Int, B::Int, n_query::Int, n_key::Int, num_blocks::Int)
    si, im = get_block_indices(N_atom, B, n_query, n_key)
    pm = get_pair_atom_block_mask(nothing, N_atom, B, num_blocks, n_query, n_key, si, im)
    return si, im, pm
end

function convert_single_rep_to_blocks(atom_single::AbstractArray{T,2}, n_query::Int, n_key::Int, atom_mask::Union{Nothing, AbstractArray}) where T
    atom_single_query, atom_single_key, atom_pair_mask = convert_single_rep_to_blocks(reshape(atom_single, 1, size(atom_single)...), n_query, n_key, atom_mask)
    # restore from [1, N, num_blocks, B] -> [N, num_blocks, B]
    return reshape(atom_single_query, size(atom_single_query)[2:end]...), reshape(atom_single_key, size(atom_single_key)[2:end]...), atom_pair_mask
end

function convert_single_rep_to_blocks(atom_single::AbstractArray{T,3}, n_query::Int, n_key::Int, atom_mask::Union{Nothing, AbstractArray}) where T
    # atom_single: [C, N_atom, B] or [N_atom, B]
    C, N_atom, B = size(atom_single)

    num_blocks = ceil(Int, N_atom / n_query)
    pad_len = get_query_block_padding(N_atom, n_query)

    # Query rep: pad then reshape into blocks
    atom_single_pad = similar(atom_single, C, pad_len, B); atom_single_pad .= zero(T)
    atom_single_padded = cat(atom_single, atom_single_pad; dims=2)                            # [C, N_padded, B]
    atom_single_query = reshape(atom_single_padded, C, n_query, num_blocks, B)                # [C, n_query, num_blocks, B]

    # Block indices + pair mask: dispatch on atom_mask type (no trues allocation for nothing)
    safe_indices, invalid_mask, atom_pair_mask = _get_block_info(atom_mask, N_atom, B, n_query, n_key, num_blocks)

    # Key rep: flat-index gather from atom_single [C, N_atom*B], then mask invalid positions
    b_offsets = reshape((0:B-1) .* N_atom, 1, 1, B)                      # [1, 1, B]
    flat_idx = safe_indices .+ b_offsets                                  # [n_key, num_blocks, B]
    atom_single_flat = reshape(atom_single, C, N_atom * B)
    atom_single_key = atom_single_flat[:, flat_idx]                       # [C, n_key, num_blocks, B]
    valid_k = reshape(.!invalid_mask, 1, n_key, num_blocks, B)
    _zero = zero(T)
    @. atom_single_key = ifelse(valid_k, atom_single_key, _zero)

    return atom_single_query, atom_single_key, atom_pair_mask
end

function convert_pair_rep_to_blocks(batch::NamedTuple, z_trunk::AbstractArray, n_query::Int, n_key::Int)
    # z_trunk: [C_z, N_token, N_token, B]
    # batch.atom_to_token_index: [N_atom, B] (1-indexed token indices)
    # batch.atom_mask: [N_atom, B] or nothing
    atom_to_token_index = batch.atom_to_token_index
    atom_mask           = batch.atom_mask

    C_z, N_token, _, B = size(z_trunk)
    N_atom              = size(atom_to_token_index, 1)
    num_blocks          = ceil(Int, N_atom / n_query)
    pad_len             = get_query_block_padding(N_atom, n_query)

    # Q indices: pad atom_to_token_index and reshape into query blocks [n_query, num_blocks, B]
    ati_pad = similar(atom_to_token_index, pad_len, B); ati_pad .= zero(eltype(atom_to_token_index))
    ati_padded = cat(atom_to_token_index, ati_pad; dims=1)
    q_indices  = Int.(reshape(ati_padded, n_query, num_blocks, B))

    # Block indices: dispatch on atom_mask type (no trues allocation for nothing)
    safe_indices, invalid_mask, atom_pair_mask = _get_block_info(atom_mask, N_atom, B, n_query, n_key, num_blocks)

    # K indices: map atom safe-indices → token indices
    b_offsets_atom = reshape((0:B-1) .* N_atom, 1, 1, B)
    flat_atom_idx  = safe_indices .+ b_offsets_atom            # [n_key, num_blocks, B]
    ati_flat       = reshape(atom_to_token_index, N_atom * B)
    k_indices      = Int.(reshape(ati_flat[flat_atom_idx], n_key, num_blocks, B))
    @. k_indices = ifelse(invalid_mask, 0, k_indices)         # zero out invalid atoms

    # Gather pair features: atom_pair[c, q, k, nb, b] = z_trunk[c, q_indices[q,nb,b], k_indices[k,nb,b], b]
    # Column-major flat index into z_trunk [C_z, N_token, N_token, B]:
    #   flat = (b-1)*N_token^2 + (k_idx-1)*N_token + q_idx   (both 1-indexed)
    # Clamp to ≥1 to avoid OOB for padding positions (q/k_indices == 0 for pad atoms)
    q_safe = reshape(max.(1, q_indices), n_query, 1, num_blocks, B)  # [n_query, 1, num_blocks, B]
    k_safe = reshape(max.(1, k_indices), 1, n_key, num_blocks, B)    # [1, n_key, num_blocks, B]

    b_off         = reshape((0:B-1) .* N_token^2, 1, 1, 1, B)
    ki_off        = (k_safe .- 1) .* N_token
    flat_pair_idx = b_off .+ ki_off .+ q_safe                        # [n_query, n_key, num_blocks, B]

    zij_flat  = reshape(z_trunk, C_z, N_token * N_token * B)
    atom_pair = zij_flat[:, flat_pair_idx]                            # [C_z, n_query, n_key, num_blocks, B]

    # Apply pair mask (zeros padding atoms and invalid key positions)
    pm_r    = reshape(atom_pair_mask, 1, n_query, n_key, num_blocks, B)
    _zero_p = zero(eltype(atom_pair))
    @. atom_pair = ifelse(pm_r, atom_pair, _zero_p)

    return atom_pair
end
