#!/bin/bash

SEED=0
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/processed_features
EXE_DIR=${ROOT_DIR}/src
EXPERIMENT_NAME=only_nda/${SEED}

export CUDA_VISIBLE_DEVICES=$((0+${SEED}))

date
python -u ${EXE_DIR}/exe/train.py \
  experiment_name=${EXPERIMENT_NAME} \
  data=[messi/scoring,messi/docking] \
  data.scoring.processed_data_dir=${DATA_DIR}/ \
  data.scoring.key_dir=${EXE_DIR}/keys/train/PDBbind_v2020/scoring \
  data.docking.root_data_dir=${DATA_DIR}/docking \
  data.docking.key_dir=${EXE_DIR}/keys/train/PDBbind_v2020/docking \
  model=pignet_morse \
  model.short_range_A=2.1 \
  run.dropout_rate=0.1 \
  run.lr=4e-4 \
  run.batch_size=64 \
  run.save_every=1 \
  run.num_epochs=2000 \
  run.num_workers=12 \
  run.pin_memory=false \
  run.seed=${SEED}

date
