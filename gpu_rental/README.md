# GPU Rental Workflow for MedCLIP-SAMv2

This directory contains scripts and utilities to run MedCLIP-SAMv2 on rented GPUs from ckey.vn.

## Quick Start

**For complete automated setup, see [QUICK_START.md](QUICK_START.md)**

1. **Configure your API key:**
   ```bash
   cp gpu_rental/gpu_config.json.template gpu_config.json
   # Edit gpu_config.json and add your API key
   ```

2. **Rent a GPU:**
   ```bash
   python3 gpu_rental/run_on_gpu.py --action rent
   ```

3. **Transfer code to GPU server** (from your local machine):
   ```bash
   # Get IP address from ckey.vn dashboard, then:
   bash gpu_rental/transfer_to_gpu.sh <IP_ADDRESS> root
   ```

4. **SSH into the GPU server:**
   ```bash
   ssh root@<IP_ADDRESS>
   # Password will be shown in step 2 output
   ```

5. **Run complete setup** (downloads models, datasets, sets up everything):
   ```bash
   cd MedCLIP-SAMv2
   bash gpu_rental/complete_setup.sh
   ```

6. **Run your segmentation:**
   ```bash
   conda activate medclipsamv2
   bash zeroshot.sh data/your_dataset
   ```

### What the Complete Setup Does

The `complete_setup.sh` script automatically:
- ✓ Sets up conda environment with all dependencies
- ✓ Downloads SAM model checkpoint (~2.4GB)
- ✓ Downloads BiomedCLIP model (~1.2GB) using gdown
- ✓ Downloads datasets from Kaggle (optional)
- ✓ Runs verification tests to ensure everything works
- ✓ Prepares directory structure

## Files Overview

### Core Scripts
- **`gpu_api_client.py`**: Python client for interacting with ckey.vn API
- **`run_on_gpu.py`**: Main workflow script for renting GPUs and managing instances
- **`transfer_to_gpu.sh`**: Automatically transfers code to GPU server via rsync/scp

### Setup Scripts (run on GPU server)
- **`complete_setup.sh`**: **Main setup script** - does everything automatically
- **`remote_setup.sh`**: Basic environment setup (called by complete_setup.sh)
- **`download_models.sh`**: Downloads SAM and BiomedCLIP model checkpoints
- **`download_datasets.sh`**: Downloads and prepares datasets from Kaggle
- **`test_setup.py`**: Verification script to test the setup

### Configuration
- **`gpu_config.json.template`**: Configuration template

## Configuration

Edit `gpu_config.json` with your settings:

```json
{
  "api_key": "your_api_key_here",
  "gpu_type": "rtxa4000",
  "count_gpu": 1,
  "count_storage": 200,
  "count_port": "22,80,443",
  "template": 1,
  "dataset_path": "data/your_dataset"
}
```

### Configuration Options

- **`api_key`**: Your ckey.vn API key
- **`gpu_type`**: Type of GPU (e.g., "rtxa4000", "RTX A4000")
- **`count_gpu`**: Number of GPUs to rent (usually 1)
- **`count_storage`**: Storage size in GB (recommended: 200+ GB)
- **`count_port`**: Ports to open (comma-separated)
- **`template`**: OS template (1 = Ubuntu)
- **`dataset_path`**: Path to your dataset on the remote server

## Usage Examples

### Rent a GPU
```bash
python gpu_rental/run_on_gpu.py --action rent
```

### Get GPU Information
```bash
python gpu_rental/run_on_gpu.py --action info --instance-id 1234
```

### Reboot GPU
```bash
python gpu_rental/run_on_gpu.py --action reboot --instance-id 1234
```

### Delete GPU Instance
```bash
python gpu_rental/run_on_gpu.py --action delete --instance-id 1234
```

## Workflow Steps

1. **Rent GPU**: Use the script to find and rent an available GPU
2. **Get Connection Info**: The script will provide SSH credentials
3. **Connect**: SSH into the GPU server
4. **Setup**: Run `remote_setup.sh` to install dependencies
5. **Upload Data**: Transfer your dataset to the server
6. **Run Model**: Execute the segmentation pipeline
7. **Download Results**: Transfer results back to your local machine
8. **Cleanup**: Delete the GPU instance when done

## Manual Setup on GPU Server

If you prefer to set up manually:

```bash
# 1. Install Miniconda (if not installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. Clone repository
git clone https://github.com/HealthX-Lab/MedCLIP-SAMv2.git
cd MedCLIP-SAMv2

# 3. Create environment
conda env create -f medclipsamv2_env.yml
conda activate medclipsamv2

# 4. Setup libraries
cd segment-anything && pip install -e . && cd ..
cd weak_segmentation && pip install -e . && cd ..

# 5. Download SAM checkpoint
mkdir -p segment-anything/sam_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  -O segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth

# 6. Download BiomedCLIP model (manually from Google Drive)
# Place at: saliency_maps/model/pytorch_model.bin

# 7. Prepare your dataset
mkdir -p data/your_dataset
# Upload your images to data/your_dataset/images

# 8. Run segmentation
bash zeroshot.sh data/your_dataset
```

## Transferring Files

### Upload to GPU Server
```bash
# Upload dataset
scp -r data/your_dataset root@<IP>:/root/MedCLIP-SAMv2/data/

# Upload code
rsync -avz --exclude '.git' /path/to/MedCLIP-SAMv2 root@<IP>:/root/
```

### Download Results
```bash
# Download results
scp -r root@<IP>:/root/MedCLIP-SAMv2/sam_outputs data/
```

## Troubleshooting

### GPU Not Available
- Try different GPU types
- Check ckey.vn dashboard for availability
- Wait a few minutes and try again

### Connection Issues
- Verify IP address from ckey.vn dashboard
- Check firewall settings
- Ensure SSH port (22) is open

### Setup Failures
- Check internet connection on GPU server
- Verify conda/miniconda installation
- Check disk space (need at least 50GB free)

### Model Download Issues
- SAM checkpoint downloads automatically
- BiomedCLIP model must be downloaded manually from Google Drive
- Place it at: `saliency_maps/model/pytorch_model.bin`

## Cost Management

- Monitor your GPU usage on ckey.vn dashboard
- Delete GPU instances when not in use
- Set reminders for long-running jobs
- Check pricing before renting (varies by GPU type)

## Notes

- GPU instances are billed hourly
- Always save your password when renting
- Results are stored on the GPU server until you download them
- Make sure to download results before deleting the instance

## Support

For issues with:
- **GPU Rental API**: Check ckey.vn documentation
- **MedCLIP-SAMv2**: See main README.md
- **Setup Scripts**: Check script comments and error messages

