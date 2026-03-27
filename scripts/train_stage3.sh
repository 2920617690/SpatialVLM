#!/bin/bash

# Stage 3 Training Script: End-to-End Fine-tuning
# Full model training with LoRA on LLM

set -e

# Configuration
CONFIG_PATH="configs/stage3_end2end.yaml"
OUTPUT_DIR="outputs/stage3"
CHECKPOINT_DIR="checkpoints/stage3"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINT_DIR}

# Run training with accelerate (or deepspeed for large models)
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes=8 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=no \
    src/training/train_stage3.py \
    --config ${CONFIG_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR}

# Alternative: Use DeepSpeed for better memory efficiency
# accelerate launch --config_file deepspeed_config.yaml \
#     src/training/train_stage3.py \
#     --config ${CONFIG_PATH} \
#     --output_dir ${OUTPUT_DIR} \
#     --checkpoint_dir ${CHECKPOINT_DIR}

echo "Stage 3 training completed!"
