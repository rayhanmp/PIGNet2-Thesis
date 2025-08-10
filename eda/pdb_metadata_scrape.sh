#!/usr/bin/env bash
set -euo pipefail

# deps: curl, jq
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required. Install jq and retry." >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required. Install curl and retry." >&2
  exit 1
fi

usage() {
  cat >&2 <<'EOF'
Usage:
  pdb_metadata_scrape.sh PDBID [PDBID ...]
  echo -e "1ADO\n6LU7" | pdb_metadata_scrape.sh

Outputs CSV with:
pdb_id,title,method,resolution_A,r_work,r_free,chains,residues_per_chain,total_residues,
deposited_atom_count,total_atoms_all,protein_atoms,ligand_atoms,solvent_atoms,
space_group,molecular_weight_kDa,polymer_mw_per_chain_kDa,deposition_date,release_date
EOF
}

# If no args and no stdin, show usage
if [[ $# -eq 0 ]] && [[ -t 0 ]]; then
  usage
  exit 1
fi

# Read IDs from args or stdin
ids=()
if [[ $# -gt 0 ]]; then
  ids+=("$@")
else
  # read from stdin
  while IFS= read -r line; do
    line="${line//[[:space:]]/}"
    [[ -n "$line" ]] && ids+=("$line")
  done
fi

# CSV header
echo "pdb_id,title,method,resolution_A,r_work,r_free,chains,residues_per_chain,total_residues,deposited_atom_count,total_atoms_all,protein_atoms,ligand_atoms,solvent_atoms,space_group,molecular_weight_kDa,polymer_mw_per_chain_kDa,deposition_date,release_date"

for id in "${ids[@]}"; do
  # Normalise to uppercase and strip spaces
  pid="$(echo -n "$id" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"

  # Fetch JSON
  if ! json="$(curl -sSf "https://data.rcsb.org/rest/v1/core/entry/${pid}")"; then
    # Emit a row with just the ID and blanks if it fails
    echo "\"${pid}\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\""
    continue
  fi

  # Extract fields and emit CSV via jq (handles proper CSV quoting)
  echo "$json" | jq -r '[
    .rcsb_id,                                                # pdb_id
    .struct.title,                                           # title
    (try .exptl[0].method),                                  # method
    (try .rcsb_entry_info.resolution_combined[0]             # resolution Å
      // .refine[0].ls_dres_high),
    (try .refine[0].ls_rfactor_rwork),                       # r_work
    (try .refine[0].ls_rfactor_rfree),                       # r_free
    (try .rcsb_entry_info.deposited_polymer_entity_instance_count),  # chains
    (try .rcsb_entry_info.polymer_monomer_count_maximum),    # residues_per_chain
    (try .rcsb_entry_info.deposited_polymer_monomer_count),  # total_residues
    (try .rcsb_entry_info.deposited_atom_count),             # deposited_atom_count (non-H typically)
    (try .refine_hist[0].number_atoms_total),                # total_atoms_all (incl. H if present)
    (try .refine_hist[0].pdbx_number_atoms_protein),         # protein_atoms
    (try .refine_hist[0].pdbx_number_atoms_ligand),          # ligand_atoms
    (try .refine_hist[0].number_atoms_solvent),              # solvent_atoms
    (try .symmetry.space_group_name_hm),                     # space_group
    (try .rcsb_entry_info.molecular_weight),                 # molecular_weight_kDa (whole model)
    (try .rcsb_entry_info.polymer_molecular_weight_maximum), # polymer_mw_per_chain_kDa
    (try .rcsb_accession_info.deposit_date),                 # deposition_date
    (try .rcsb_accession_info.initial_release_date)          # release_date
  ] | @csv'
  # throttle requests to avoid rate limiting
  sleep 0.2
done