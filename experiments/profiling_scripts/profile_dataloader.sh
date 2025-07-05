#!/bin/bash

SEED=0
ROOT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=${ROOT_DIR}/dataset/PDBbind-v2020
EXE_DIR=${ROOT_DIR}/src
EXPERIMENT_NAME=profiling/dataloader/${SEED}

export CUDA_VISIBLE_DEVICES=$((0+${SEED}))

echo "Starting dataloader-focused profiling (data loading bottleneck analysis)..."
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
  run.batch_size=128 \
  run.save_every=1 \
  run.num_epochs=3 \
  run.num_workers=1 \
  run.pin_memory=false \
  run.seed=${SEED} \
  run.enable_profiler=true \
  run.profiler_epochs=[1,2,3] \
  run.profiler_output_dir=profiler_output/dataloader \
  run.run_name=profiling_dataloader_${SEED}

echo "Dataloader profiling completed."
echo "This uses single worker and disabled pin_memory to highlight I/O bottlenecks."
echo "Check profiler_output/dataloader/ for results."
echo "Look for data loading gaps and CPU-GPU synchronization issues."
echo "Compare with results from other scripts to see dataloader impact."
echo "Open epoch_X_trace.json files in Chrome at chrome://tracing"
date 