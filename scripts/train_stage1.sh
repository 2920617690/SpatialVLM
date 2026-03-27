#!/bin/bash

# Stage 1 Training Script: 3D Grounding
# Freeze LLM, train 3D-aware ViT

set -e

# Configuration
CONFIG_PATH="configs/stage1_3d_grounding.yaml"
OUTPUT_DIR="outputs/stage1"
CHECKPOINT_DIR="checkpoints/stage1"

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
    src/training/train_stage1.py \
    --config ${CONFIG_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR}

echo "Stage 1 training completed!"
