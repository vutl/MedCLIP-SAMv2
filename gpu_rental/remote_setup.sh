#!/bin/bash
# Remote GPU Server Setup Script
# This script should be run on the rented GPU server to set up the environment

set -e  # Exit on error

echo "=========================================="
echo "MedCLIP-SAMv2 GPU Server Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${YELLOW}Warning: Running as root. Some operations may need non-root user.${NC}"
fi

# Update system packages
echo -e "${GREEN}[1/10] Updating system packages...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq git wget curl build-essential tmux
echo -e "${GREEN}✓ tmux installed for persistent sessions${NC}"

# Install Anaconda/Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo -e "${GREEN}[2/10] Installing Miniconda...${NC}"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
else
    echo -e "${GREEN}[2/10] Conda already installed${NC}"
    export PATH="$(conda info --base)/bin:$PATH"
fi

# Initialize conda
eval "$(conda shell.bash hook)"

# Create or activate conda environment
ENV_NAME="medclipsamv2"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${GREEN}[3/10] Conda environment '${ENV_NAME}' already exists, activating...${NC}"
    conda activate ${ENV_NAME}
else
    echo -e "${GREEN}[3/10] Creating conda environment '${ENV_NAME}'...${NC}"
    # Check if environment file exists
    if [ -f "medclipsamv2_env.yml" ]; then
        conda env create -f medclipsamv2_env.yml
    else
        echo -e "${RED}Error: medclipsamv2_env.yml not found!${NC}"
        echo "Please ensure the environment file is in the current directory."
        exit 1
    fi
    conda activate ${ENV_NAME}
fi

# Clone or update repository
REPO_DIR="MedCLIP-SAMv2"
if [ -d "$REPO_DIR" ]; then
    echo -e "${GREEN}[4/10] Repository exists, updating...${NC}"
    cd $REPO_DIR
    git pull || echo "Git pull failed, continuing..."
else
    echo -e "${GREEN}[4/10] Cloning repository...${NC}"
    # Note: Update this URL if you have a private repo or fork
    git clone https://github.com/HealthX-Lab/MedCLIP-SAMv2.git $REPO_DIR || {
        echo -e "${YELLOW}Git clone failed. If you have the code locally, upload it manually.${NC}"
        echo "You can use scp or rsync to transfer files."
        exit 1
    }
    cd $REPO_DIR
fi

# Setup segment-anything
echo -e "${GREEN}[5/10] Setting up segment-anything...${NC}"
cd segment-anything
pip install -e . --quiet
cd ..

# Setup nnUNet
echo -e "${GREEN}[6/8] Setting up nnUNet...${NC}"
cd weak_segmentation
pip install -e . --quiet
cd ..

# Download SAM model checkpoint if not present
echo -e "${GREEN}[7/8] Checking SAM model checkpoint...${NC}"
SAM_CHECKPOINT_DIR="segment-anything/sam_checkpoints"
SAM_CHECKPOINT_FILE="${SAM_CHECKPOINT_DIR}/sam_vit_h_4b8939.pth"

mkdir -p ${SAM_CHECKPOINT_DIR}

if [ ! -f "$SAM_CHECKPOINT_FILE" ]; then
    echo "Downloading SAM ViT-H checkpoint..."
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O "$SAM_CHECKPOINT_FILE"
    echo "SAM checkpoint downloaded."
else
    echo "SAM checkpoint already exists."
fi

# Download BiomedCLIP model if not present
echo -e "${GREEN}[8/10] Checking BiomedCLIP model...${NC}"
BIOMEDCLIP_MODEL_DIR="saliency_maps/model"
BIOMEDCLIP_MODEL_FILE="${BIOMEDCLIP_MODEL_DIR}/pytorch_model.bin"

mkdir -p ${BIOMEDCLIP_MODEL_DIR}

if [ ! -f "$BIOMEDCLIP_MODEL_FILE" ]; then
    echo -e "${YELLOW}BiomedCLIP model not found. Attempting download...${NC}"
    
    # Install gdown if not present
    pip install -q gdown 2>/dev/null || echo "gdown not available, will try manual download"
    
    # Try to download using gdown
    GDRIVE_ID="1jjnZabUlc9_gpcP0d2nz_GNS-EGX0lq5"
    if command -v gdown &> /dev/null; then
        echo "Downloading BiomedCLIP model from Google Drive..."
        gdown --id "$GDRIVE_ID" -O "$BIOMEDCLIP_MODEL_FILE" && {
            echo -e "${GREEN}✓ BiomedCLIP model downloaded${NC}"
        } || {
            echo -e "${YELLOW}⚠️  Automatic download failed.${NC}"
            echo "Please download manually from:"
            echo "https://drive.google.com/file/d/${GDRIVE_ID}/view?usp=sharing"
            echo "Place it at: ${BIOMEDCLIP_MODEL_FILE}"
        }
    else
        echo "gdown not available. Please download manually from:"
        echo "https://drive.google.com/file/d/${GDRIVE_ID}/view?usp=sharing"
        echo "Place it at: ${BIOMEDCLIP_MODEL_FILE}"
    fi
else
    echo "BiomedCLIP model found."
fi

# Download datasets (optional)
echo -e "${GREEN}[9/10] Setting up dataset download script...${NC}"
if [ -f "scripts/download_and_prepare_datasets.py" ]; then
    echo "Dataset download script available."
    echo "Run 'bash gpu_rental/download_datasets.sh' to download datasets."
else
    echo -e "${YELLOW}Dataset download script not found.${NC}"
fi

# Verify GPU availability
echo -e "${GREEN}[10/10] Verifying GPU setup...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" || {
    echo -e "${YELLOW}Warning: Could not verify GPU. Make sure CUDA drivers are installed.${NC}"
}

echo ""
echo -e "${GREEN}=========================================="
echo "Setup completed successfully!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate ${ENV_NAME}"
echo "2. Navigate to the repo: cd ${REPO_DIR}"
echo "3. Download models (if not done): bash gpu_rental/download_models.sh"
echo "4. Download datasets (optional): bash gpu_rental/download_datasets.sh"
echo "5. Test setup: python gpu_rental/test_setup.py"
echo "6. Run segmentation: bash zeroshot.sh data/your_dataset"
echo ""

