from pathlib import Path
import subprocess

# Root directory
root_dir = Path("PDBbind_v2020_refined/refined-set")

# Path to the list file
list_file = Path("pdbbind_keys.txt")

# Read all PDB IDs in that list
with list_file.open("r") as f:
    pdb_ids = [line.strip() for line in f if line.strip()]

for pdb_id in pdb_ids:
    folder = root_dir / pdb_id
    protein_file = folder / f"{pdb_id}_protein.pdb"
    ligand_file = folder / f"{pdb_id}_ligand.sdf"

    if protein_file.exists() and ligand_file.exists():
        print(f"[INFO] Processing {pdb_id}")
        subprocess.run([
            "./generate_data.py",
            "-p", str(protein_file),
            "-l", str(ligand_file)
        ])
    else:
        print(f"[WARN] Skipping {pdb_id}: missing_")
