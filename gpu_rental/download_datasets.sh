#!/bin/bash
# Download and prepare datasets for MedCLIP-SAMv2
# This script downloads datasets and prepares the directory structure

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Downloading and Preparing Datasets"
echo "=========================================="

# Check if kagglehub is installed
if ! python -c "import kagglehub" 2>/dev/null; then
    echo -e "${YELLOW}Installing kagglehub for dataset downloads...${NC}"
    pip install -q kagglehub
fi

# Check if we're in the repo root
if [ ! -f "scripts/download_and_prepare_datasets.py" ]; then
    echo -e "${RED}Error: Must run from MedCLIP-SAMv2 root directory${NC}"
    exit 1
fi

# Create data directory
mkdir -p data

echo -e "${GREEN}Downloading datasets using the provided script...${NC}"
echo "This will download:"
echo "  - Breast Ultrasound Images (BUSI)"
echo "  - COVID-QU-Ex (Lung Chest X-ray)"
echo "  - Chest CT Segmentation"
echo ""

# Run the dataset preparation script
python scripts/download_and_prepare_datasets.py --root data

echo ""
echo -e "${GREEN}=========================================="
echo "Dataset preparation completed!"
echo "==========================================${NC}"
echo ""
echo "Datasets are available in:"
echo "  - data/breast_tumors/"
echo "  - data/lung_chest_xray/"
echo "  - data/lung_ct/"
echo ""
echo -e "${YELLOW}Note: You may need to manually download:${NC}"
echo "  - Brain Tumors dataset from Kaggle"
echo "  - Pre-processed segmentation datasets from Google Drive"
echo "    https://drive.google.com/file/d/1uYtyg3rClE-XXPNuEz7s6gYq2p48Z08p/view?usp=sharing"

