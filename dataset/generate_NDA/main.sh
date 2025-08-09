#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Forced-diversity re-docking NDA (PDBbind layout)
# - Input tree:
#     $PDBBIND_DIR/$PDB/${PDB}_protein.pdb
#     $PDBBIND_DIR/$PDB/${PDB}_ligand.mol2   (cognate ligand)
# - keys_list.txt: one PDB ID per line (case-insensitive OK)
# - Output:
#     <out_dir>/<PDB>/{kept/,poses/,manifest.tsv,pdbqt_manifest.tsv,logs/}
#     <out_dir>/ALL_manifest.tsv    (global index of kept PDBQT poses)
#
# Dependencies on PATH: smina, obabel, DockRMSD, python3, awk, sed
# ------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 -k keys_list.txt -d PDBBIND_DIR -o out_dir [options]

Options:
  --seeds N         Number of random seeds (default: 40)
  --box S           Cube box size in Å (default: 36)
  --minrmsd X       smina --min_rmsd_filter (Å) (default: 2.5)
  --mode M          'random' (randomize_only), 'minimize', or 'both' (default: random)
  --rmsd_cut X      Keep poses with RMSD > X Å (default: 4.0)
  --ph PH           Protonation pH for Open Babel (default: 7.4)

Example:
  PDBBIND_DIR=/data/PDBbind \\
  $0 -k keys_list.txt -o out_ndA -d "\$PDBBIND_DIR" --seeds 60 --box 42 --mode both
EOF
  exit 1
}

# Defaults
SEEDS=40
BOX=36
MINRMSD=2.5
MODE="random"   # random | minimize | both
CUT=4.0
PH=7.4

# Parse args
[[ $# -lt 1 ]] && usage
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k) KEYS="$2"; shift 2;;
    -d) PDBBIND_DIR="$2"; shift 2;;
    -o) OUTDIR="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --box) BOX="$2"; shift 2;;
    --minrmsd) MINRMSD="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --rmsd_cut) CUT="$2"; shift 2;;
    --ph) PH="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

[[ -z "${KEYS:-}" || -z "${PDBBIND_DIR:-}" || -z "${OUTDIR:-}" ]] && usage

# Dep checks
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1" >&2; exit 2; }; }
for b in smina obabel DockRMSD python3 awk sed; do need "$b"; done

mkdir -p "$OUTDIR"
GLOBAL="$OUTDIR/ALL_manifest.tsv"
echo -e "pdb\tpose_pdbqt\tRMSD_A" > "$GLOBAL"

process_one() {
  local PDB_ID="$1"
  local id_lc="${PDB_ID,,}"      # 4lzs
  local id_uc="${PDB_ID^^}"      # 4LZS

  # Prefer lowercase path (how PDBbind ships), fallback to uppercase if someone renamed
  local D="$PDBBIND_DIR/$id_lc"
  [[ -d "$D" ]] || D="$PDBBIND_DIR/$id_uc"

  # Accept either .mol2 or .sdf ligand
  local REC_PDB="$D/${id_lc}_protein.pdb"
  [[ -f "$REC_PDB" ]] || REC_PDB="$D/${id_uc}_protein.pdb"

  local LIG_MOL2="$D/${id_lc}_ligand.mol2"
  local LIG_SDF="$D/${id_lc}_ligand.sdf"
  [[ -f "$LIG_MOL2" ]] || LIG_MOL2="$D/${id_uc}_ligand.mol2"
  [[ -f "$LIG_SDF"  ]] || LIG_SDF="$D/${id_uc}_ligand.sdf"

  if [[ ! -f "$REC_PDB" ]]; then
    echo "[skip] $PDB_ID (no protein pdb at $REC_PDB)" >&2
    return 0
  fi
  if [[ -f "$LIG_MOL2" ]]; then
    local LIG="$LIG_MOL2"
    local LIG_FMT="mol2"
  elif [[ -f "$LIG_SDF" ]]; then
    local LIG="$LIG_SDF"
    local LIG_FMT="sdf"
  else
    echo "[skip] $PDB_ID (no ligand .mol2 or .sdf in $D)" >&2
    return 0
  fi

  local TOUT="$OUTDIR/${id_lc}"
  mkdir -p "$TOUT"/{logs,poses,kept}
  echo "[run] $PDB_ID → $TOUT"

  # Prep receptor
  obabel "$REC_PDB" -opdbqt -O "$TOUT/receptor.pdbqt" -p "$PH" -xh >"$TOUT/logs/obabel_receptor.log" 2>&1

  # Prep ligand: pick first record, then to PDBQT
  if [[ "$LIG_FMT" == "mol2" ]]; then
    obabel "$LIG" -omol2 -O "$TOUT/ligand_one.mol2" -f 1 -l 1 >"$TOUT/logs/obabel_lig_pick.log" 2>&1
    obabel "$TOUT/ligand_one.mol2" -opdbqt -O "$TOUT/crystal_ligand.pdbqt" -p "$PH" -xh >"$TOUT/logs/obabel_lig_pdbqt.log" 2>&1
  else
    obabel "$LIG" -osdf -O "$TOUT/ligand_one.sdf" -f 1 -l 1 >"$TOUT/logs/obabel_lig_pick.log" 2>&1
    obabel "$TOUT/ligand_one.sdf" -opdbqt -O "$TOUT/crystal_ligand.pdbqt" -p "$PH" -xh >"$TOUT/logs/obabel_lig_pdbqt.log" 2>&1
  fi

  # Box centre (protein centroid)
read CX CY CZ <<< "$(python3 helpers.py centroid "$TOUT/receptor.pdbqt")"

  # Generate multi-model PDBQT with forced diversity
  : > "$TOUT/redock_multimodel.pdbqt"

  if [[ "$MODE" == "random" || "$MODE" == "both" ]]; then
    # Random rigid-body placements (no minimisation) → guaranteed high RMSD variety
    for s in $(seq 1 "$SEEDS"); do
      smina -r "$TOUT/receptor.pdbqt" -l "$TOUT/crystal_ligand.pdbqt" \
            --center_x "$CX" --center_y "$CY" --center_z "$CZ" \
            --size_x "$BOX" --size_y "$BOX" --size_z "$BOX" \
            --num_modes 1 \
            --randomize_only \
            --seed "$s" \
            -o "$TOUT/tmp_random_${s}.pdbqt" >> "$TOUT/logs/smina_random.log" 2>&1 || true
      cat "$TOUT/tmp_random_${s}.pdbqt" >> "$TOUT/redock_multimodel.pdbqt"
      rm -f "$TOUT/tmp_random_${s}.pdbqt"
    done
  fi

  if [[ "$MODE" == "minimize" || "$MODE" == "both" ]]; then
    # Stochastic docking with diversity & big box
    for s in $(seq 1 "$SEEDS"); do
      smina -r "$TOUT/receptor.pdbqt" -l "$TOUT/crystal_ligand.pdbqt" \
            --center_x "$CX" --center_y "$CY" --center_z "$CZ" \
            --size_x "$BOX" --size_y "$BOX" --size_z "$BOX" \
            --exhaustiveness 16 \
            --num_modes 4 \
            --energy_range 999 \
            --min_rmsd_filter "$MINRMSD" \
            --seed "$s" \
            --minimize \
            -o "$TOUT/tmp_min_${s}.pdbqt" >> "$TOUT/logs/smina_min.log" 2>&1 || true
      cat "$TOUT/tmp_min_${s}.pdbqt" >> "$TOUT/redock_multimodel.pdbqt"
      rm -f "$TOUT/tmp_min_${s}.pdbqt"
    done
  fi

  # Split to per-pose files
  local nposes
  nposes=$(python3 helpers.py split-models "$TOUT/redock_multimodel.pdbqt" "$TOUT/poses")
  echo "[info] $PDB_ID: split $nposes poses"

  # Convert to MOL2 for DockRMSD
  obabel "$TOUT/crystal_ligand.pdbqt" -omol2 -O "$TOUT/poses/crystal_ligand.mol2" >"$TOUT/logs/obabel_ref_mol2.log" 2>&1
  for f in "$TOUT"/poses/pose_*.pdbqt; do
    obabel "$f" -omol2 -O "${f%.pdbqt}.mol2" >>"$TOUT/logs/obabel_pose_mol2.log" 2>&1
  done

  # Score & keep > CUT
  echo -e "pose\tRMSD_A" > "$TOUT/manifest.tsv"
  for m in "$TOUT"/poses/pose_*.mol2; do
    out=$(DockRMSD "$TOUT/poses/crystal_ligand.mol2" "$m" 2>&1 || true)
    rmsd=$(printf "%s\n" "$out" | awk -F': ' '/Calculated Docking RMSD/{print $2}')
    if [[ -z "$rmsd" ]]; then
      # hydrogen-policy fallback: no-H vs no-H to avoid atom mismatch
      obabel "$TOUT/poses/crystal_ligand.mol2" -omol2 -O "$TOUT/poses/crystal_noH.mol2" -d >/dev/null 2>&1 || true
      obabel "$m" -omol2 -O "${m%.mol2}_noH.mol2" -d >/dev/null 2>&1 || true
      out=$(DockRMSD "$TOUT/poses/crystal_noH.mol2" "${m%.mol2}_noH.mol2" 2>&1 || true)
      rmsd=$(printf "%s\n" "$out" | awk -F': ' '/Calculated Docking RMSD/{print $2}')
      rm -f "$TOUT/poses/crystal_noH.mol2" "${m%.mol2}_noH.mol2"
    fi
    if [[ -n "${rmsd:-}" ]] && awk -v x="$rmsd" -v c="$CUT" 'BEGIN{exit !(x>c)}'; then
      cp "${m%.mol2}.pdbqt" "$TOUT/kept/"
      printf "%s\t%.3f\n" "$(basename "$m")" "$rmsd" >> "$TOUT/manifest.tsv"
    fi
  done

  # Build PDBQT manifest (paths you’ll actually use downstream)
  {
    echo -e "pose_pdbqt\tRMSD_A"
    awk -v P="$TOUT/poses" 'NR>1{gsub(/\.mol2$/, ".pdbqt", $1); print P "/" $1 "\t" $2}' "$TOUT/manifest.tsv"
  } > "$TOUT/pdbqt_manifest.tsv"

  # Append to global index (kept only)
  awk -v K="$TOUT/kept" -v ID="$PDB_ID" '
    NR>1{
      p=$1; sub(/\.mol2$/, ".pdbqt", p);
      printf "%s\t%s/%s\t%s\n", ID, K, p, $2
    }' "$TOUT/manifest.tsv" >> "$GLOBAL"

  echo "[done] $PDB_ID → kept: $(ls "$TOUT/kept" | wc -l)"
}

# Main loop
while IFS=$'\n' read -r raw || [[ -n "${raw}" ]]; do
  PDB_ID=$(echo "$raw" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
  [[ -z "$PDB_ID" || "$PDB_ID" =~ ^# ]] && continue
  process_one "$PDB_ID"
done < "$KEYS"

echo "[summary] Global manifest: $GLOBAL"