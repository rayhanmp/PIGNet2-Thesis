#!/bin/bash

SEED=0
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/dataset/PDBbind-v2020
EXE_DIR=${ROOT_DIR}/src
EXPERIMENT_NAME=profiling/memory/${SEED}

export CUDA_VISIBLE_DEVICES=$((0+${SEED}))

echo "Starting memory-focused profiling (small batch sizes for memory analysis)..."
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
  run.batch_size=16 \
  run.save_every=1 \
  run.num_epochs=3 \
  run.num_workers=4 \
  run.pin_memory=true \
  run.seed=${SEED} \
  run.enable_profiler=true \
  run.profiler_epochs=[1,2,3] \
  run.profiler_output_dir=profiler_output/memory \
  run.run_name=profiling_memory_${SEED}

echo "Memory profiling completed."
echo "This uses small batch sizes to focus on memory allocation patterns."
echo "Check profiler_output/memory/ for results."
echo "Look for memory usage patterns and allocation inefficiencies."
echo "Open epoch_X_trace.json files in Chrome at chrome://tracing"
date 