#!/bin/bash

# Evaluation Script: Theory of Space Benchmark

set -e

# Configuration
MODEL_CHECKPOINT="checkpoints/stage3/best_model.pt"
TEST_DATA_PATH="data/tos_benchmark"
OUTPUT_DIR="outputs/evaluation"
STAGE=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            MODEL_CHECKPOINT="$2"
            shift 2
            ;;
        --data_path)
            TEST_DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run evaluation
python src/evaluation/evaluate.py \
    --checkpoint ${MODEL_CHECKPOINT} \
    --data_path ${TEST_DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --stage ${STAGE}

echo "Evaluation completed! Results saved to ${OUTPUT_DIR}"
