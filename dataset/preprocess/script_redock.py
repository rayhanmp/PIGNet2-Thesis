from pathlib import Path
import subprocess
import os

root_dir = Path("redock_dataset")
save_dir = Path("data")

this_dir = Path(__file__).parent.resolve()
generate_data_py = (this_dir / "generate_data.py").resolve()

list_file = Path("keys.txt")

keys_map = None
if list_file.exists():
    with list_file.open("r") as f:
        raw_ids = [line.strip() for line in f if line.strip()]
    pdb_ids = sorted({rid.split("_", 1)[0] for rid in raw_ids})
    # Build per-PDB ligand filters
    keys_map = {}
    for rid in raw_ids:
        pdb = rid.split("_", 1)[0]
        keys_map.setdefault(pdb, set()).add(rid)
else:
    pdb_ids = [p.name for p in root_dir.iterdir() if p.is_dir()]

for pdb_id in pdb_ids:
    folder = root_dir / pdb_id
    if not folder.is_dir():
        continue

    protein_file = folder / f"{pdb_id}_protein.pdb"
    pqr_file = folder / f"{pdb_id}.pqr"

    # Determine ligands to process. If keys_map provided, restrict to listed stems
    if keys_map and pdb_id in keys_map:
        ligand_files = []
        for stem in sorted(keys_map[pdb_id]):
            path = folder / f"{stem}.mol2"
            if path.exists():
                ligand_files.append(path)
            else:
                print(f"[WARN] Missing ligand file for key '{stem}': {path}")
    else:
        # Collect all ligand mol2 files for this PDB ID (e.g., 5d3h_014.mol2)
        ligand_files = sorted(folder.glob(f"{pdb_id}_*.mol2"))

    if protein_file.exists() and pqr_file.exists() and ligand_files:
        print(f"[INFO] Processing {pdb_id} with {len(ligand_files)} ligands")
        for ligand_file in ligand_files:
            print(f"  - ligand: {ligand_file.name}")
            subprocess.run([
                os.fspath(generate_data_py),
                "-p", str(protein_file),
                "-l", str(ligand_file),
                "--pqr_file", str(pqr_file),
                "-s", str(save_dir),
            ])
    else:
        missing = []
        if not protein_file.exists():
            missing.append("protein_pdb")
        if not pqr_file.exists():
            missing.append("protein_pqr")
        if not ligand_files:
            missing.append("ligand_mol2")
        print(f"[WARN] Skipping {pdb_id}: missing {', '.join(missing)}")


