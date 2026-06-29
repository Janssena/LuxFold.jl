
"""
    AuxiliaryHeadsAllAtom(; pairformer_embedding, distogram_head,
                          pae_head, pde_head, plddt_head, experimentally_resolved_head)

AF3 Algorithm 31 auxiliary heads — 1:1 port of
`openfold3.core.model.heads.head_modules.AuxiliaryHeadsAllAtom`.

Orchestrates six sub-layers:
1. `distogram_head` — called on `z_trunk` **before** the confidence pairformer
2. `pairformer_embedding` — confidence pairformer producing `(s_conf, z_conf)`
3. `pae_head` — asymmetric projection of `z_conf`
4. `pde_head` — symmetric projection of `z_conf`
5. `plddt_head` — per-atom pLDDT from `s_conf` via token→atom expansion
6. `experimentally_resolved_head` — per-atom binary from `s_conf`

# Inputs
- `s_trunk [chn_s, N_token, B]`, `z_trunk [chn_z, N_token, N_token, B]`
- `batch::NamedTuple` with:
  - `token_mask [N_token, B] Bool`
  - `atom_mask [N_atom, B] Bool`
  - `atom_positions_predicted [3, N_atom, B]` — predicted atom positions
  - `num_atoms_per_token [N_token, B]`

# Returns
- `(; distogram, pae, pde, plddt, experimentally_resolved)`, `st`
"""
struct AuxiliaryHeadsAllAtom{PE,DH,PAE,PDE,PL,ER} <: Lux.AbstractLuxContainerLayer{(
        :pairformer_embedding, :distogram_head,:pae_head, :pde_head, :plddt_head, 
        :experimentally_resolved_head
    )}
    pairformer_embedding::PE
    distogram_head::DH
    pae_head::PAE
    pde_head::PDE
    plddt_head::PL
    experimentally_resolved_head::ER
end

function AuxiliaryHeadsAllAtom(;
    pairformer_embedding::PairformerEmbedding,
    distogram_head::DistogramHead,
    pae_head::PredictedAlignedErrorHead,
    pde_head::PredictedDistanceErrorHead,
    plddt_head::PerResidueLDDTAllAtom,
    experimentally_resolved_head::ExperimentallyResolvedHeadAllAtom,
)
    AuxiliaryHeadsAllAtom(
        pairformer_embedding, distogram_head,
        pae_head, pde_head, plddt_head, experimentally_resolved_head,
    )
end

(l::AuxiliaryHeadsAllAtom)(inputs::NamedTuple, ps, st) =
    l(inputs.s_trunk, inputs.z_trunk, inputs.batch, ps, st)

function (l::AuxiliaryHeadsAllAtom)(s_trunk, z_trunk, batch::NamedTuple, ps, st)
    token_mask = batch.token_mask    # [N_token, B] Bool
    atom_mask  = batch.atom_mask     # [N_atom, B] Bool
    N_token, B = size(token_mask)

    distogram_logits, st_dist = l.distogram_head(
        z_trunk, ps.distogram_head, st.distogram_head,
    )

    # Extract representative atom coords per token (Cα / C1' / first atom for ligands)
    x_rep, rep_mask = get_token_representative_atoms(
        batch, batch.atom_positions_predicted, atom_mask,
    )   # x_rep [3, N_token, B], rep_mask [N_token, B] Bool

    pair_mask = reshape(token_mask, N_token, 1, B) .& reshape(token_mask, 1, N_token, B)

    pf_out, st_pf = l.pairformer_embedding(
        s_trunk, s_trunk, z_trunk, x_rep, rep_mask, pair_mask,
        ps.pairformer_embedding, st.pairformer_embedding,
    )
    s_conf, z_conf = pf_out.s, pf_out.z

    max_atom_per_token_mask = broadcast_token_feat_to_atoms(
        token_mask, batch.num_atoms_per_token, token_mask;
        max_num_atoms_per_token=l.plddt_head.max_atoms_per_token,
    )   # [max_atoms * N_token, B]

    plddt_logits, st_plddt = l.plddt_head(
        s_conf, max_atom_per_token_mask, ps.plddt_head, st.plddt_head,
    )
    er_logits, st_er = l.experimentally_resolved_head(
        s_conf, max_atom_per_token_mask,
        ps.experimentally_resolved_head, st.experimentally_resolved_head,
    )

    pae_logits, st_pae = l.pae_head(z_conf, ps.pae_head, st.pae_head)
    pde_logits, st_pde = l.pde_head(z_conf, ps.pde_head, st.pde_head)

    out = (;
        distogram = distogram_logits,
        pae = pae_logits,
        pde = pde_logits,
        plddt = plddt_logits,
        experimentally_resolved = er_logits,
    )
    st_new = merge(st, (;
        pairformer_embedding = st_pf,
        distogram_head = st_dist,
        pae_head = st_pae,
        pde_head = st_pde,
        plddt_head = st_plddt,
        experimentally_resolved_head = st_er,
    ))
    return out, st_new
end
