# Profiling Scripts

This directory contains ready-to-use profiling scripts to identify performance bottlenecks in training. Each script is designed to profile specific aspects of the training pipeline.

## Quick Start

```bash
bash experiments/profiling_scripts/profile_baseline.sh
```

## Available Scripts

### 1. `profile_baseline.sh`
**Purpose**: Standard profiling for scoring task
- **Epochs**: 5  
- **Batch size**: 256
- **Workers**: 12
- **Best for**: Comprehensive single-task performance analysis

### 2. `profile_pda_nda.sh`
**Purpose**: Multi-dataset complex training profiling
- **Epochs**: 3
- **Batch size**: 128
- **Datasets**: PDA + Docking + Cross + Random
- **Best for**: Complex training pipeline bottlenecks

### 3. `profile_memory.sh`
**Purpose**: Memory allocation and usage analysis
- **Epochs**: 3
- **Batch size**: 16 (small for memory focus)
- **Workers**: 4
- **Best for**: Memory bottlenecks and allocation patterns

### 4. `profile_dataloader.sh`
**Purpose**: Data loading bottleneck identification
- **Epochs**: 3
- **Batch size**: 128
- **Workers**: 1 (intentionally low)
- **Pin memory**: Disabled
- **Best for**: I/O and data loading performance issues

### 5. `profile_large_batch.sh`
**Purpose**: GPU utilization and scaling analysis
- **Epochs**: 3
- **Batch size**: 512 (large)
- **Workers**: 16
- **Best for**: GPU scaling and utilization efficiency