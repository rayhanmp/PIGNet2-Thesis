#!/bin/bash

SEED=0
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/dataset/PDBbind-v2020
EXE_DIR=${ROOT_DIR}/src
EXPERIMENT_NAME=profiling/quick/${SEED}

export CUDA_VISIBLE_DEVICES=$((0+${SEED}))

echo "Starting quick profiling (fast bottleneck identification)..."
date

python -u ${EXE_DIR}/exe/train.py \
  experiment_name=${EXPERIMENT_NAME} \
  data=[messi/scoring] \
  data.scoring.root_data_dir=${DATA_DIR}/scoring \
  data.scoring.key_dir=${EXE_DIR}/keys/train/PDBbind_v2020/scoring \
  model=pignet_morse \
  model.short_range_A=2.1 \
  run.dropout_rate=0.1 \
  run.lr=4e-4 \
  run.batch_size=64 \
  run.save_every=1 \
  run.num_epochs=2 \
  run.num_workers=4 \
  run.pin_memory=true \
  run.seed=${SEED} \
  run.enable_profiler=true \
  run.profiler_epochs=[1,2] \
  run.profiler_output_dir=profiler_output/quick \
  run.run_name=profiling_quick_${SEED}

echo "Quick profiling completed in ~2-5 minutes."
echo "Check profiler_output/quick/ for results."
echo "Open epoch_X_trace.json files in Chrome at chrome://tracing"
date 