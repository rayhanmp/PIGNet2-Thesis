#!/bin/bash

untar() {
  file="${1}"
  dir="$(basename "${file}" .tar.xz | rev | cut -d_ -f1 | rev)"

  if [[ "$file" == *"PDBbind-v2020"* ]]; then
    tar_dir="${PWD}/PDBbind-v2020/${dir}"
  else
    tar_dir="${PWD}/Benchmark/${dir}"
  fi

  echo "[INFO] Extracting ${file} to ${tar_dir}..."
  mkdir -p "${tar_dir}"

  if command -v pixz >/dev/null 2>&1; then
    echo "[INFO] Using pixz"
    pixz -d < "$file" | tar --no-same-owner -x -C "${tar_dir}" || {
      echo "[ERROR] Extraction failed for ${file} with pixz"
      exit 1
    }
  else
    echo "[WARN] Falling back to plain tar (no pixz)"
    tar --no-same-owner -xf "${file}" -C "${tar_dir}" || {
      echo "[ERROR] Fallback extraction failed for ${file}"
      exit 1
    }
  fi

  if [ ! -e "${tar_dir}/data" ]; then
    if [ -d "${tar_dir}/data_5_sdf" ]; then
      ln -s "data_5_sdf" "${tar_dir}/data"
    elif [ -d "${tar_dir}/data_5" ]; then
      ln -s "data_5" "${tar_dir}/data"
    fi
  fi
}

mkdir -p PDBbind-v2020 Benchmark

untar tarfiles/PDBbind-v2020_scoring.tar.xz
untar tarfiles/PDBbind-v2020_docking.tar.xz
untar tarfiles/PDBbind-v2020_cross.tar.xz
untar tarfiles/PDBbind-v2020_random.tar.xz
untar tarfiles/PDBbind-v2020_pda.tar.xz

untar tarfiles/CASF-2016_scoring.tar.xz
untar tarfiles/CASF-2016_docking.tar.xz
untar tarfiles/CASF-2016_screening.tar.xz
untar tarfiles/DUD-E.tar.xz
untar tarfiles/derivative.tar.xz
