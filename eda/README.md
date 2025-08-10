# PDB metadata to CSV
A small Bash script `eda/pdb_metadata_scrape.sh` that reads PDB IDs (from args or stdin) and writes a CSV with basic entry metadata fetched from the RCSB PDB API.

## Requirements

- `bash`
- `curl`
- `jq`

## Quick start

- Generic shape (as requested):

```bash
cat <train-keys>.txt | ./pdb_metadata_scrape.sh > <train-info>.csv
```

- Pass IDs as arguments:

```bash
./eda/pdb_metadata_scrape.sh 1ADO 6LU7 > info.csv
```

## Output columns

The script prints a header followed by one row per PDB ID. Header:

```text
pdb_id,title,method,resolution_A,r_work,r_free,chains,residues_per_chain,total_residues,deposited_atom_count,total_atoms_all,protein_atoms,ligand_atoms,solvent_atoms,space_group,molecular_weight_kDa,polymer_mw_per_chain_kDa,deposition_date,release_date
```

Notes:

- If an ID cannot be fetched, a row with the ID and empty fields is emitted.

