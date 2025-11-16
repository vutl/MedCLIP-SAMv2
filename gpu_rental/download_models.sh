#!/bin/bash
# Download all required model checkpoints
# This script downloads SAM and BiomedCLIP models

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Downloading Model Checkpoints"
echo "=========================================="

# SAM Checkpoint
SAM_CHECKPOINT_DIR="segment-anything/sam_checkpoints"
SAM_CHECKPOINT_FILE="${SAM_CHECKPOINT_DIR}/sam_vit_h_4b8939.pth"

mkdir -p ${SAM_CHECKPOINT_DIR}

if [ ! -f "$SAM_CHECKPOINT_FILE" ]; then
    echo -e "${GREEN}[1/2] Downloading SAM ViT-H checkpoint...${NC}"
    wget --progress=bar:force https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O "$SAM_CHECKPOINT_FILE"
    echo -e "${GREEN}✓ SAM checkpoint downloaded${NC}"
else
    echo -e "${GREEN}[1/2] SAM checkpoint already exists${NC}"
fi

# BiomedCLIP Model
BIOMEDCLIP_MODEL_DIR="saliency_maps/model"
BIOMEDCLIP_MODEL_FILE="${BIOMEDCLIP_MODEL_DIR}/pytorch_model.bin"

mkdir -p ${BIOMEDCLIP_MODEL_DIR}

if [ ! -f "$BIOMEDCLIP_MODEL_FILE" ]; then
    echo -e "${GREEN}[2/2] Downloading BiomedCLIP model...${NC}"
    
    # Check if gdown is installed
    if ! command -v gdown &> /dev/null; then
        echo "Installing gdown for Google Drive downloads..."
        pip install -q gdown
    fi
    
    # Google Drive file ID from the README
    # https://drive.google.com/file/d/1jjnZabUlc9_gpcP0d2nz_GNS-EGX0lq5/view?usp=sharing
    GDRIVE_ID="1jjnZabUlc9_gpcP0d2nz_GNS-EGX0lq5"
    
    echo "Downloading from Google Drive (this may take a while)..."
    gdown --id "$GDRIVE_ID" -O "$BIOMEDCLIP_MODEL_FILE" || {
        echo -e "${YELLOW}⚠️  gdown failed. Trying alternative method...${NC}"
        # Alternative: direct download if available
        echo "Please download manually from:"
        echo "https://drive.google.com/file/d/${GDRIVE_ID}/view?usp=sharing"
        echo "And place it at: ${BIOMEDCLIP_MODEL_FILE}"
        exit 1
    }
    
    echo -e "${GREEN}✓ BiomedCLIP model downloaded${NC}"
else
    echo -e "${GREEN}[2/2] BiomedCLIP model already exists${NC}"
fi

# Verify file sizes
echo ""
echo "Verifying downloads..."
if [ -f "$SAM_CHECKPOINT_FILE" ]; then
    SAM_SIZE=$(du -h "$SAM_CHECKPOINT_FILE" | cut -f1)
    echo "  SAM checkpoint: $SAM_SIZE"
fi

if [ -f "$BIOMEDCLIP_MODEL_FILE" ]; then
    BIOMED_SIZE=$(du -h "$BIOMEDCLIP_MODEL_FILE" | cut -f1)
    echo "  BiomedCLIP model: $BIOMED_SIZE"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Model downloads completed!"
echo "==========================================${NC}"

