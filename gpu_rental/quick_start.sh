#!/bin/bash
# Quick start script for GPU rental workflow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "MedCLIP-SAMv2 GPU Rental Quick Start"
echo "=========================================="
echo ""

# Check if config exists
CONFIG_FILE="$PROJECT_ROOT/gpu_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating configuration file..."
    cp "$SCRIPT_DIR/gpu_config.json.template" "$CONFIG_FILE"
    echo ""
    echo "⚠️  Please edit gpu_config.json with your API key:"
    echo "   nano $CONFIG_FILE"
    echo ""
    read -p "Press Enter after you've edited the config file..."
fi

# Check Python dependencies
echo "Checking Python dependencies..."
if ! python3 -c "import requests" 2>/dev/null; then
    echo "Installing requests..."
    pip3 install -q requests
fi

# Run the main script
cd "$PROJECT_ROOT"
python3 "$SCRIPT_DIR/run_on_gpu.py" "$@"

