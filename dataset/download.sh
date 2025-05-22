#!/bin/bash

download() {
  file=${1}
  url="https://thesis.rayhan.id/${file}"
  out_path="tarfiles/${file}"
  if [ ! -f "$out_path" ]; then
    aria2c -x 16 -s 16 -k 1M -o "${file}" -d tarfiles "${url}"
  fi
}

mkdir -p tarfiles

# Training
download PDBbind-v2020_scoring.tar.xz
download PDBbind-v2020_docking.tar.xz
download PDBbind-v2020_cross.tar.xz
download PDBbind-v2020_random.tar.xz
download PDBbind-v2020_pda.tar.xz

# Benchmark
download CASF-2016_scoring.tar.xz
download CASF-2016_docking.tar.xz
download CASF-2016_screening.tar.xz
download DUD-E.tar.xz
download derivative.tar.xz
