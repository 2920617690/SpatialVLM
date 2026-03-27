#!/bin/bash
#
# One-click data preparation script for SpatialVLM
#
# This script automates the entire data preparation pipeline:
# 1. Download Hypersim dataset
# 2. Preprocess Hypersim data
# 3. Download Theory of Space benchmark
# 4. Generate spatial QA pairs
#

set -e  # Exit on error

# Default paths
DATA_DIR="./data"
HYPERSIM_DIR="${DATA_DIR}/hypersim"
HYPERSIM_PROCESSED_DIR="${DATA_DIR}/hypersim_processed"
TOS_DIR="${DATA_DIR}/tos"
QA_DIR="${DATA_DIR}/spatial_qa"

# Default parameters
NUM_SCENES=5
NUM_WORKERS=4
IMAGE_SIZE=384
NUM_QA_PER_SCENE=25

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --num-scenes)
            NUM_SCENES="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --num-qa)
            NUM_QA_PER_SCENE="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --skip-tos)
            SKIP_TOS=true
            shift
            ;;
        --skip-qa)
            SKIP_QA=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-dir DIR          Base data directory (default: ./data)"
            echo "  --num-scenes N          Number of Hypersim scenes to download (default: 5)"
            echo "  --num-workers N         Number of preprocessing workers (default: 4)"
            echo "  --image-size N          Target image size (default: 384)"
            echo "  --num-qa N              Number of QA pairs per scene (default: 25)"
            echo "  --skip-download         Skip downloading Hypersim"
            echo "  --skip-preprocess       Skip preprocessing Hypersim"
            echo "  --skip-tos              Skip downloading ToS benchmark"
            echo "  --skip-qa               Skip generating QA pairs"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Update paths based on data directory
HYPERSIM_DIR="${DATA_DIR}/hypersim"
HYPERSIM_PROCESSED_DIR="${DATA_DIR}/hypersim_processed"
TOS_DIR="${DATA_DIR}/tos"
QA_DIR="${DATA_DIR}/spatial_qa"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "${SCRIPT_DIR}")"

print_header "SpatialVLM Data Preparation Pipeline"
print_info "Data directory: ${DATA_DIR}"
print_info "Number of Hypersim scenes: ${NUM_SCENES}"
print_info "Preprocessing workers: ${NUM_WORKERS}"
print_info "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
print_info "QA pairs per scene: ${NUM_QA_PER_SCENE}"
echo ""

# Step 1: Download Hypersim
if [ "$SKIP_DOWNLOAD" = true ]; then
    print_info "Skipping Hypersim download"
else
    print_header "Step 1: Downloading Hypersim Dataset"
    
    python3 "${SRC_DIR}/src/data/download_hypersim.py" \
        --output_dir "${HYPERSIM_DIR}" \
        --num_scenes "${NUM_SCENES}" \
        --data_types rgb depth
    
    if [ $? -eq 0 ]; then
        print_success "Hypersim download completed"
    else
        print_error "Hypersim download failed"
        exit 1
    fi
fi

# Step 2: Preprocess Hypersim
if [ "$SKIP_PREPROCESS" = true ]; then
    print_info "Skipping Hypersim preprocessing"
else
    print_header "Step 2: Preprocessing Hypersim Data"
    
    python3 "${SRC_DIR}/src/data/preprocess_hypersim.py" \
        --input_dir "${HYPERSIM_DIR}" \
        --output_dir "${HYPERSIM_PROCESSED_DIR}" \
        --image_size "${IMAGE_SIZE}" \
        --num_workers "${NUM_WORKERS}"
    
    if [ $? -eq 0 ]; then
        print_success "Hypersim preprocessing completed"
        
        # Print statistics
        if [ -d "${HYPERSIM_PROCESSED_DIR}" ]; then
            NUM_PROCESSED_SCENES=$(find "${HYPERSIM_PROCESSED_DIR}" -maxdepth 1 -type d | wc -l)
            NUM_PROCESSED_SCENES=$((NUM_PROCESSED_SCENES - 1))  # Exclude the directory itself
            NUM_FRAMES=$(find "${HYPERSIM_PROCESSED_DIR}" -name "*.png" | wc -l)
            print_info "Processed scenes: ${NUM_PROCESSED_SCENES}"
            print_info "Processed frames: ${NUM_FRAMES}"
        fi
    else
        print_error "Hypersim preprocessing failed"
        exit 1
    fi
fi

# Step 3: Download Theory of Space
if [ "$SKIP_TOS" = true ]; then
    print_info "Skipping ToS download"
else
    print_header "Step 3: Downloading Theory of Space Benchmark"
    
    python3 "${SRC_DIR}/src/data/download_tos.py" \
        --output_dir "${TOS_DIR}" \
        --split all
    
    if [ $? -eq 0 ]; then
        print_success "ToS download completed"
    else
        print_error "ToS download failed"
        exit 1
    fi
fi

# Step 4: Generate Spatial QA
if [ "$SKIP_QA" = true ]; then
    print_info "Skipping QA generation"
else
    print_header "Step 4: Generating Spatial QA Pairs"
    
    python3 "${SRC_DIR}/src/data/generate_spatial_qa.py" \
        --input_dir "${HYPERSIM_PROCESSED_DIR}" \
        --output_dir "${QA_DIR}" \
        --num_qa_per_scene "${NUM_QA_PER_SCENE}" \
        --seed 42
    
    if [ $? -eq 0 ]; then
        print_success "QA generation completed"
        
        # Print statistics
        if [ -f "${QA_DIR}/spatial_qa.json" ]; then
            NUM_QA=$(python3 -c "import json; print(len(json.load(open('${QA_DIR}/spatial_qa.json'))))")
            print_info "Total QA pairs generated: ${NUM_QA}"
        fi
    else
        print_error "QA generation failed"
        exit 1
    fi
fi

# Final summary
print_header "Data Preparation Complete"
print_success "All steps completed successfully!"
echo ""
print_info "Data locations:"
echo "  - Hypersim raw data: ${HYPERSIM_DIR}"
echo "  - Hypersim processed: ${HYPERSIM_PROCESSED_DIR}"
echo "  - Theory of Space: ${TOS_DIR}"
echo "  - Spatial QA pairs: ${QA_DIR}"
echo ""
print_info "You can now start training with the prepared data!"
