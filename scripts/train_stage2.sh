#!/bin/bash

# Stage 2 Training Script: Spatial Memory
# Freeze ViT, train Memory + PE

set -e

# Configuration
CONFIG_PATH="configs/stage2_spatial_memory.yaml"
OUTPUT_DIR="outputs/stage2"
CHECKPOINT_DIR="checkpoints/stage2"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINT_DIR}

# Run training with accelerate
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes=8 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=no \
    src/training/train_stage2.py \
    --config ${CONFIG_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR}

echo "Stage 2 training completed!"
