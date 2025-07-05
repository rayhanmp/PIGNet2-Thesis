#!/bin/bash

SEED=0
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/dataset/PDBbind-v2020
EXE_DIR=${ROOT_DIR}/src
EXPERIMENT_NAME=profiling/pda_nda/${SEED}

export CUDA_VISIBLE_DEVICES=$((0+${SEED}))

echo "Starting PDA+NDA profiling (multi-dataset complex training)..."
date

python -u ${EXE_DIR}/exe/train.py \
  experiment_name=${EXPERIMENT_NAME} \
  data=[messi/pda,messi/docking,messi/cross,messi/random] \
  data.tpda.root_data_dir=${DATA_DIR}/pda \
  data.tpda.key_dir=${EXE_DIR}/keys/train/PDBbind_v2020/pda \
  data.docking.root_data_dir=${DATA_DIR}/docking \
  data.docking.key_dir=${EXE_DIR}/keys/train/PDBbind_v2020/docking \
  data.cross.root_data_dir=${DATA_DIR}/cross \
  data.cross.key_dir=${EXE_DIR}/keys/train/PDBbind_v2020/cross \
  data.random.root_data_dir=${DATA_DIR}/random \
  data.random.key_dir=${EXE_DIR}/keys/train/PDBbind_v2020/random \
  model=pignet_morse \
  model.short_range_A=2.1 \
  run.dropout_rate=0.1 \
  run.lr=4e-4 \
  run.batch_size=128 \
  run.save_every=1 \
  run.num_epochs=3 \
  run.num_workers=8 \
  run.pin_memory=true \
  run.seed=${SEED} \
  run.enable_profiler=true \
  run.profiler_epochs=[1,2,3] \
  run.profiler_output_dir=profiler_output/pda_nda \
  run.run_name=profiling_pda_nda_${SEED}

echo "PDA+NDA profiling completed."
echo "This profiles complex multi-dataset training bottlenecks."
echo "Check profiler_output/pda_nda/ for results."
echo "Open epoch_X_trace.json files in Chrome at chrome://tracing"
date 