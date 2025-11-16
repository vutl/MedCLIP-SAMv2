#!/bin/bash
# Script to transfer the entire MedCLIP-SAMv2 repository to GPU server
# Usage: ./transfer_to_gpu.sh <IP_ADDRESS> [USER]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <IP_ADDRESS> [USER]"
    echo "Example: $0 123.45.67.89 root"
    exit 1
fi

IP_ADDRESS=$1
USER=${2:-root}
REPO_NAME="MedCLIP-SAMv2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Transferring MedCLIP-SAMv2 to GPU Server"
echo "=========================================="
echo ""
echo "Target: ${USER}@${IP_ADDRESS}"
echo "Source: ${REPO_ROOT}"
echo ""

# Check if rsync is available
if command -v rsync &> /dev/null; then
    echo "Using rsync (faster, supports resume)..."
    echo ""
    
    # Exclude unnecessary files
    rsync -avz --progress \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.ipynb_checkpoints' \
        --exclude 'node_modules' \
        --exclude '.env' \
        --exclude '*.log' \
        --exclude 'sam_outputs' \
        --exclude 'saliency_map_outputs' \
        --exclude 'coarse_outputs' \
        "${REPO_ROOT}/" "${USER}@${IP_ADDRESS}:~/${REPO_NAME}/"
    
    echo ""
    echo "✓ Transfer completed!"
    
elif command -v scp &> /dev/null; then
    echo "Using scp (slower, but available)..."
    echo "Note: This may take a while for large repositories"
    echo ""
    
    # Create directory on remote
    ssh "${USER}@${IP_ADDRESS}" "mkdir -p ~/${REPO_NAME}"
    
    # Transfer files
    scp -r "${REPO_ROOT}"/* "${USER}@${IP_ADDRESS}:~/${REPO_NAME}/"
    
    echo ""
    echo "✓ Transfer completed!"
else
    echo "Error: Neither rsync nor scp found. Please install one of them."
    exit 1
fi

echo ""
echo "Next steps on GPU server:"
echo "  1. SSH into server: ssh ${USER}@${IP_ADDRESS}"
echo "  2. Navigate: cd ${REPO_NAME}"
echo "  3. Run setup: bash gpu_rental/complete_setup.sh"
echo ""

