#!/bin/bash

SEED=0
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/dataset/PDBbind-v2020
EXE_DIR=${ROOT_DIR}/src
EXPERIMENT_NAME=profiling/large_batch/${SEED}

export CUDA_VISIBLE_DEVICES=$((0+${SEED}))

echo "Starting large batch profiling (GPU utilization and scaling analysis)..."
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
  run.batch_size=512 \
  run.save_every=1 \
  run.num_epochs=3 \
  run.num_workers=16 \
  run.pin_memory=true \
  run.seed=${SEED} \
  run.enable_profiler=true \
  run.profiler_epochs=[1,2,3] \
  run.profiler_output_dir=profiler_output/large_batch \
  run.run_name=profiling_large_batch_${SEED}

echo "Large batch profiling completed."
echo "This uses large batch sizes to test GPU utilization and memory scaling."
echo "Check profiler_output/large_batch/ for results."
echo "Look for GPU utilization patterns and memory pressure."
echo "Compare with smaller batch results to see scaling efficiency."
echo "Open epoch_X_trace.json files in Chrome at chrome://tracing"
date 