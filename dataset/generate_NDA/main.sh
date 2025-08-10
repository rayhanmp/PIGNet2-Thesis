#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Forced-diversity re-docking NDA
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
  --seeds N         Number of random seeds (default: 80)
  --minrmsd X       smina --min_rmsd_filter (Å) (default: 2.5)
  --mode M          'random' (randomize_only), 'minimize', or 'both' (default: random)
  --rmsd_low X      Keep poses with RMSD > X Å (default: 4.0)
  --rmsd_high X     Keep poses with RMSD < X Å (default: 10.0)

Example:
  PDBBIND_DIR=/data/PDBbind \\
  $0 -k keys_list.txt -o out_ndA -d "\$PDBBIND_DIR" --seeds 60 --box 42 --mode both
EOF
  exit 1
}

ts() { date +%s; }

# Defaults
SEEDS=80
MINRMSD=2.5
MODE="random"   # random | minimize | both
RMSD_LOW=4.0
RMSD_HIGH=10.0

# Parse args
[[ $# -lt 1 ]] && usage
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k) KEYS="$2"; shift 2;;
    -d) PDBBIND_DIR="$2"; shift 2;;
    -o) OUTDIR="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --minrmsd) MINRMSD="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --rmsd_low) RMSD_LOW="$2"; shift 2;;
    --rmsd_high) RMSD_HIGH="$2"; shift 2;;
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

# Helper: split multi-model PDBQT into poses/pose_###.pdbqt
split_models() {
  local in="$1" outdir="$2"
  python3 - "$in" "$outdir" <<'PY'
import sys
from pathlib import Path
src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
t = src.read_text()
outdir.mkdir(parents=True, exist_ok=True)
parts, chunk = [], []
for line in t.splitlines(keepends=True):
    if line.startswith("MODEL "):
        if chunk:
            parts.append("".join(chunk))
        chunk = [line]
    else:
        chunk.append(line)
if chunk:
    parts.append("".join(chunk))
for i, part in enumerate(parts, 1):
    (outdir / f"pose_{i:03d}.pdbqt").write_text(part)
print(len(parts))
PY
}

process_one() {
  t0=$(ts)
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

    # Prep receptor + ligand in parallel (cached; safe waits)
  rec_done="$TOUT/receptor.pdbqt"
  lig_done="$TOUT/crystal_ligand.pdbqt"

  if [[ -f "$rec_done" && -f "$lig_done" ]]; then
    echo "[prep] cached → skip"
  else
    # receptor in background
    {
      set +e
      # requires 'reduce' in PATH; fallback to your old line if not found
      t_rec0=$(date +%s)
      if command -v reduce >/dev/null; then
        reduce -BUILD "$REC_PDB" > "$TOUT/receptor_H.pdb" 2> "$TOUT/logs/reduce_receptor.log"
        rc_reduce=$?
        # If wrote no file → mark special exit code 90 for "skip"
        if [[ ! -s "$TOUT/receptor_H.pdb" ]]; then
          echo $(( $(date +%s) - t_rec0 )) > "$TOUT/logs/t_receptor_prep.sec"
          echo "[skip] Reduce failed; empty output" >&2
          exit 90
        fi
        echo "[info] reduce ok" >&2
        obabel "$TOUT/receptor_H.pdb" -opdbqt -O "$TOUT/receptor.pdbqt" -xh >"$TOUT/logs/obabel_receptor.log" 2>&1
        rc=$?
        echo $(( $(date +%s) - t_rec0 )) > "$TOUT/logs/t_receptor_prep.sec"
        exit $rc
      else
        echo "[warn] Reduce not available, fallback to obabel" >&2
        obabel "$REC_PDB" -opdbqt -O "$TOUT/receptor.pdbqt" -xh >"$TOUT/logs/obabel_receptor.log" 2>&1
        rc=$?
        echo $(( $(date +%s) - t_rec0 )) > "$TOUT/logs/t_receptor_prep.sec"
        exit $rc
      fi
    } & pid_rec=$!

    # ligand (pick first record, then PDBQT) in background
    if [[ "$LIG_FMT" == "mol2" ]]; then
      {
        set -e
        t_lig0=$(date +%s)
        obabel "$LIG" -omol2 -O "$TOUT/ligand_one.mol2" -f 1 -l 1 \
          >"$TOUT/logs/obabel_lig_pick.log" 2>&1
        obabel "$TOUT/ligand_one.mol2" -opdbqt -O "$TOUT/crystal_ligand.pdbqt" -xh \
          >"$TOUT/logs/obabel_lig_pdbqt.log" 2>&1
        rc=$?
        echo $(( $(date +%s) - t_lig0 )) > "$TOUT/logs/t_ligand_prep.sec"
        exit $rc
      } & pid_lig=$!
    else
      {
        set -e
        t_lig0=$(date +%s)
        obabel "$LIG" -osdf -O "$TOUT/ligand_one.sdf" -f 1 -l 1 \
          >"$TOUT/logs/obabel_lig_pick.log" 2>&1
        obabel "$TOUT/ligand_one.sdf" -opdbqt -O "$TOUT/crystal_ligand.pdbqt" -xh \
          >"$TOUT/logs/obabel_lig_pdbqt.log" 2>&1
        rc=$?
        echo $(( $(date +%s) - t_lig0 )) > "$TOUT/logs/t_ligand_prep.sec"
        exit $rc
      } & pid_lig=$!
    fi

    # Receptor: capture exit code to distinguish "skip" vs "fail"
    wait "$pid_rec"; rc_rec=$?
    # Always wait on ligand to avoid zombies
    wait "$pid_lig" || { echo "[prep] ligand prep failed"; return 0; }

    if [[ $rc_rec -eq 90 ]]; then
      echo "[skip] $PDB_ID (Reduce-based receptor prep failed)"
      return 0  # graceful skip of this PDB
    elif [[ $rc_rec -ne 0 ]]; then
      echo "[prep] receptor prep failed (rc=$rc_rec)"
      return 0
    fi
  fi

# per-stage prep timings (seconds)
  rec_t=$(cat "$TOUT/logs/t_receptor_prep.sec" 2>/dev/null || echo -1)
  lig_t=$(cat "$TOUT/logs/t_ligand_prep.sec" 2>/dev/null || echo -1)
  echo "[t] prep_receptor: ${rec_t} s  prep_ligand: ${lig_t} s"

  # Use the crystal ligand to autobox the *true* binding site
  # Generate multi-model PDBQT with forced diversity
  : > "$TOUT/redock_multimodel.pdbqt"

  echo "[t] prep: $(( $(ts)-t0 )) s"; t1=$(ts)

  if [[ "$MODE" == "random" || "$MODE" == "both" ]]; then
    # Random rigid-body placements (no minimisation) → guaranteed high RMSD variety
    for s in $(seq 1 "$SEEDS"); do
      smina -r "$TOUT/receptor.pdbqt" -l "$TOUT/crystal_ligand.pdbqt" \
            --autobox_ligand "$TOUT/crystal_ligand.pdbqt" \
            --autobox_add 14 \
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
            --autobox_ligand "$TOUT/crystal_ligand.pdbqt" \
            --autobox_add 14 \
            --exhaustiveness 6 \
            --num_modes 2 \
            --energy_range 999 \
            --min_rmsd_filter "$MINRMSD" \
            --seed "$s" \
            --minimize \
            -o "$TOUT/tmp_min_${s}.pdbqt" >> "$TOUT/logs/smina_min.log" 2>&1 || true
      cat "$TOUT/tmp_min_${s}.pdbqt" >> "$TOUT/redock_multimodel.pdbqt"
      rm -f "$TOUT/tmp_min_${s}.pdbqt"
    done
  fi

  echo "[t] docking: $(( $(ts)-t1 )) s"; t2=$(ts)

  # Split to per-pose files
  local nposes
  nposes=$(split_models "$TOUT/redock_multimodel.pdbqt" "$TOUT/poses")
  echo "[info] $PDB_ID: split $nposes poses"

  # Convert to MOL2 for DockRMSD (NO --gen3d)
  obabel "$TOUT/crystal_ligand.pdbqt" -omol2 -O "$TOUT/poses/crystal_ligand.mol2" >"$TOUT/logs/obabel_ref_mol2.log" 2>&1
  for f in "$TOUT"/poses/pose_*.pdbqt; do
    obabel "$f" -omol2 -O "${f%.pdbqt}.mol2" >>"$TOUT/logs/obabel_pose_mol2.log" 2>&1
  done

  echo "[t] obabel:  $(( $(ts)-t2 )) s"; t3=$(ts)

  # Score & keep within [RMSD_LOW, RMSD_HIGH]
  echo -e "pose\tRMSD_A" > "$TOUT/manifest.tsv"
  for m in "$TOUT"/poses/pose_*.mol2; do
    out=$(DockRMSD "$TOUT/poses/crystal_ligand.mol2" "$m" 2>&1 || true)
    rmsd=$(printf "%s\n" "$out" | awk -F': ' '/Calculated Docking RMSD/{print $2}')
    if [[ -z "$rmsd" ]]; then
      # hydrogen-policy fallback: no-H vs no-H
      obabel "$TOUT/poses/crystal_ligand.mol2" -omol2 -O "$TOUT/poses/crystal_noH.mol2" -d >/dev/null 2>&1 || true
      obabel "$m" -omol2 -O "${m%.mol2}_noH.mol2" -d >/dev/null 2>&1 || true
      out=$(DockRMSD "$TOUT/poses/crystal_noH.mol2" "${m%.mol2}_noH.mol2" 2>&1 || true)
      rmsd=$(printf "%s\n" "$out" | awk -F': ' '/Calculated Docking RMSD/{print $2}')
      rm -f "$TOUT/poses/crystal_noH.mol2" "${m%.mol2}_noH.mol2"
    fi
    if [[ -n "${rmsd:-}" ]] && awk -v x="$rmsd" -v a="$RMSD_LOW" -v b="$RMSD_HIGH" 'BEGIN{exit !(x>=a && x<=b)}'; then
      cp "${m%.mol2}.pdbqt" "$TOUT/kept/"
      printf "%s\t%.3f\n" "$(basename "$m")" "$rmsd" >> "$TOUT/manifest.tsv"
    fi
  done

  echo "[t] rmsd:    $(( $(ts)-t3 )) s"

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

  # Remove receptor.pdbqt to save storage space
  rm -f "$TOUT/receptor.pdbqt"

  echo "[done] $PDB_ID → kept: $(ls "$TOUT/kept" | wc -l)"
}

# Main loop
while IFS=$'\n' read -r raw || [[ -n "${raw}" ]]; do
  PDB_ID=$(echo "$raw" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
  [[ -z "$PDB_ID" || "$PDB_ID" =~ ^# ]] && continue
  if ! process_one "$PDB_ID"; then
    echo "[warn] process_one failed unexpectedly for $PDB_ID — continuing"
    continue
  fi
done < "$KEYS"

echo "[summary] Global manifest: $GLOBAL"