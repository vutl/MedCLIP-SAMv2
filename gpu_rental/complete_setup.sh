#!/bin/bash
# Complete setup script - does everything: environment, models, datasets, testing
# Run this on the GPU server after connecting via SSH

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "MedCLIP-SAMv2 Complete Setup"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Step 1: Basic environment setup
echo -e "${BLUE}[Step 1/5] Running basic environment setup...${NC}"
bash "$SCRIPT_DIR/remote_setup.sh"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate medclipsamv2 || {
    echo -e "${RED}Failed to activate conda environment${NC}"
    exit 1
}

# Step 2: Download models
echo ""
echo -e "${BLUE}[Step 2/5] Downloading model checkpoints...${NC}"
bash "$SCRIPT_DIR/download_models.sh"

# Step 3: Download datasets (optional, can be skipped)
echo ""
echo -e "${BLUE}[Step 3/5] Downloading datasets (this may take a while)...${NC}"
read -p "Do you want to download datasets now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "$SCRIPT_DIR/download_datasets.sh"
else
    echo -e "${YELLOW}Skipping dataset download. You can run it later with:${NC}"
    echo "  bash gpu_rental/download_datasets.sh"
fi

# Step 4: Run tests
echo ""
echo -e "${BLUE}[Step 4/5] Running setup verification tests...${NC}"
python "$SCRIPT_DIR/test_setup.py"

# Step 5: Summary
echo ""
echo -e "${GREEN}=========================================="
echo "Complete Setup Finished!"
echo "==========================================${NC}"
echo ""
echo "Setup Summary:"
echo "  ✓ Environment: medclipsamv2"
echo "  ✓ SAM checkpoint: segment-anything/sam_checkpoints/"
echo "  ✓ BiomedCLIP model: saliency_maps/model/"
if [ -d "data" ] && [ "$(ls -A data 2>/dev/null)" ]; then
    echo "  ✓ Datasets: data/"
else
    echo "  ⚠️  Datasets: Not downloaded (run download_datasets.sh)"
fi
echo ""
echo "Quick Start:"
echo "  1. Activate environment: conda activate medclipsamv2"
echo "  2. Test setup: python gpu_rental/test_setup.py"
echo "  3. Run segmentation: bash zeroshot.sh data/your_dataset"
echo ""
echo "Example commands:"
echo "  # Brain tumors"
echo "  bash zeroshot_scripts/zeroshot_brain_tumors.sh data/brain_tumors"
echo ""
echo "  # Breast tumors"
echo "  bash zeroshot_scripts/zeroshot_breast_tumors.sh data/breast_tumors"
echo ""

