#!/usr/bin/env python3
"""
Main workflow script for renting GPU and running MedCLIP-SAMv2
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

# Add parent directory to path to import gpu_api_client
sys.path.insert(0, str(Path(__file__).parent))

from gpu_api_client import GPUAPIClient, GPUInfo


class GPUWorkflow:
    """Manages the workflow of renting GPU and running MedCLIP-SAMv2"""
    
    def __init__(self, config_path: str = "gpu_config.json"):
        """
        Initialize workflow with configuration
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config = self.load_config(config_path)
        self.client = GPUAPIClient(self.config["api_key"])
        self.instance_id: Optional[int] = None
        self.gpu_info: Optional[GPUInfo] = None
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            print(f"Config file {config_path} not found. Creating template...")
            self.create_config_template(config_path)
            print(f"Please edit {config_path} with your settings and run again.")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def create_config_template(self, config_path: str):
        """Create a template configuration file"""
        template = {
            "api_key": "YOUR_API_KEY_HERE",
            "gpu_type": "rtxa4000",
            "count_gpu": 1,
            "count_storage": 200,
            "count_port": "22,80,443",
            "template": 1,
            "dataset_path": "data/your_dataset",
            "auto_setup": True,
            "auto_delete": False,
            "ssh_user": "root",
            "remote_setup_script": "gpu_rental/remote_setup.sh"
        }
        
        with open(config_path, 'w') as f:
            json.dump(template, f, indent=2)
    
    def find_available_gpu(self) -> Optional[dict]:
        """Find an available GPU"""
        print(f"Searching for available {self.config['gpu_type']} GPUs...")
        gpus = self.client.get_available_gpus(self.config['gpu_type'])
        
        if not gpus:
            print("No available GPUs found!")
            return None
        
        # Filter for available GPUs
        available = [gpu for gpu in gpus if gpu.get('isAvailableForDemand', False)]
        
        if not available:
            print("No GPUs available for demand rental!")
            return None
        
        # Sort by price (lowest first) or other criteria
        available.sort(key=lambda x: x.get('pricePerGpu', float('inf')))
        
        selected = available[0]
        print(f"\nSelected GPU:")
        print(f"  ID: {selected['id']}")
        print(f"  Model: {selected.get('gpuModel', 'N/A')}")
        print(f"  Memory: {selected.get('gpuMemorySize', 'N/A')} MB")
        print(f"  Price: {selected.get('pricePerGpu', 'N/A')} VND/hour")
        print(f"  Region: {selected.get('rig', {}).get('region', 'N/A')}")
        
        return selected
    
    def rent_gpu(self) -> bool:
        """Rent a GPU instance"""
        gpu = self.find_available_gpu()
        if not gpu:
            return False
        
        print(f"\nRenting GPU {gpu['id']}...")
        success, instance_id, message = self.client.rent_gpu(
            gpu_id=gpu['id'],
            count_gpu=self.config['count_gpu'],
            count_storage=self.config['count_storage'],
            count_port=self.config['count_port'],
            template=self.config['template']
        )
        
        if success:
            self.instance_id = instance_id
            print(f"✓ GPU rented successfully! Instance ID: {instance_id}")
            print(f"  Message: {message}")
            return True
        else:
            print(f"✗ Failed to rent GPU: {message}")
            return False
    
    def wait_for_gpu_ready(self) -> bool:
        """Wait for GPU to be ready and get connection info"""
        if not self.instance_id:
            print("No GPU instance ID!")
            return False
        
        print(f"\nWaiting for GPU {self.instance_id} to be ready...")
        self.gpu_info = self.client.wait_for_gpu_ready(self.instance_id)
        
        if self.gpu_info and self.gpu_info.password:
            print(f"\n✓ GPU is ready!")
            print(f"  Instance ID: {self.gpu_info.id}")
            print(f"  GPU Model: {self.gpu_info.gpu_model}")
            print(f"  GPU Memory: {self.gpu_info.gpu_memory_size} MB")
            print(f"  Password: {self.gpu_info.password}")
            print(f"\n⚠️  IMPORTANT: Save the password above!")
            print(f"   You'll need it to SSH into the server.")
            return True
        else:
            print("✗ GPU did not become ready in time")
            return False
    
    def get_connection_info(self) -> Optional[dict]:
        """Get SSH connection information"""
        if not self.gpu_info:
            # Try to get info again
            if self.instance_id:
                self.gpu_info = self.client.get_gpu_info(self.instance_id)
        
        if not self.gpu_info or not self.gpu_info.password:
            print("Cannot get connection info. GPU may not be ready yet.")
            return None
        
        # Note: The API may not return IP directly, you might need to check
        # the ckey.vn dashboard or the API response might have it
        return {
            "instance_id": self.gpu_info.id,
            "password": self.gpu_info.password,
            "user": self.config.get("ssh_user", "root"),
            # IP might need to be retrieved from dashboard or another API call
        }
    
    def print_connection_instructions(self):
        """Print instructions for connecting to the GPU server"""
        info = self.get_connection_info()
        if not info:
            return
        
        print("\n" + "="*60)
        print("CONNECTION INSTRUCTIONS")
        print("="*60)
        print(f"\n1. Get the IP address from ckey.vn dashboard")
        print(f"   Instance ID: {info['instance_id']}")
        print(f"\n2. Transfer code to GPU server (from your local machine):")
        print(f"   bash gpu_rental/transfer_to_gpu.sh <IP_ADDRESS> {info['user']}")
        print(f"\n3. SSH into the server:")
        print(f"   ssh {info['user']}@<IP_ADDRESS>")
        print(f"\n4. Password: {info['password']}")
        print(f"\n5. Once connected, run the complete setup:")
        print(f"   cd MedCLIP-SAMv2")
        print(f"   bash gpu_rental/complete_setup.sh")
        print(f"\n   This will:")
        print(f"   - Set up the environment")
        print(f"   - Download all model checkpoints")
        print(f"   - Download datasets (optional)")
        print(f"   - Run verification tests")
        print(f"\n6. After setup, run segmentation:")
        print(f"   conda activate medclipsamv2")
        print(f"   bash zeroshot.sh {self.config['dataset_path']}")
        print("="*60 + "\n")
    
    def run_setup_remotely(self) -> bool:
        """Run setup script on remote GPU server via SSH"""
        info = self.get_connection_info()
        if not info:
            return False
        
        print("\n⚠️  Automatic remote setup requires:")
        print("   1. IP address (check ckey.vn dashboard)")
        print("   2. SSH key setup or password authentication")
        print("   3. Code uploaded to remote server")
        print("\nFor now, please follow the manual connection instructions above.")
        return False
    
    def delete_gpu(self):
        """Delete the rented GPU instance"""
        if not self.instance_id:
            print("No GPU instance to delete")
            return
        
        print(f"\nDeleting GPU instance {self.instance_id}...")
        if self.client.delete_gpu(self.instance_id):
            print("✓ GPU instance deleted successfully")
            self.instance_id = None
            self.gpu_info = None
        else:
            print("✗ Failed to delete GPU instance")


def main():
    parser = argparse.ArgumentParser(
        description="Rent GPU and run MedCLIP-SAMv2 workflow"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="gpu_config.json",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["rent", "info", "delete", "reboot"],
        default="rent",
        help="Action to perform"
    )
    parser.add_argument(
        "--instance-id",
        type=int,
        help="GPU instance ID (for info/delete/reboot actions)"
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Skip automatic setup (just rent GPU)"
    )
    
    args = parser.parse_args()
    
    workflow = GPUWorkflow(args.config)
    
    if args.action == "rent":
        if workflow.rent_gpu():
            if workflow.wait_for_gpu_ready():
                workflow.print_connection_instructions()
                
                if not args.no_setup and workflow.config.get("auto_setup", False):
                    workflow.run_setup_remotely()
    
    elif args.action == "info":
        if args.instance_id:
            workflow.instance_id = args.instance_id
        elif workflow.config.get("last_instance_id"):
            workflow.instance_id = workflow.config["last_instance_id"]
        else:
            print("Please provide --instance-id")
            return
        
        info = workflow.client.get_gpu_info(workflow.instance_id)
        if info:
            print(f"\nGPU Instance {workflow.instance_id} Info:")
            print(f"  Model: {info.gpu_model}")
            print(f"  Memory: {info.gpu_memory_size} MB")
            print(f"  CPU Cores: {info.cpu_core_count}")
            print(f"  System Memory: {info.system_memory} MB")
            if info.password:
                print(f"  Password: {info.password}")
    
    elif args.action == "delete":
        if args.instance_id:
            workflow.instance_id = args.instance_id
        elif workflow.config.get("last_instance_id"):
            workflow.instance_id = workflow.config["last_instance_id"]
        else:
            print("Please provide --instance-id")
            return
        
        workflow.delete_gpu()
    
    elif args.action == "reboot":
        if args.instance_id:
            workflow.instance_id = args.instance_id
        elif workflow.config.get("last_instance_id"):
            workflow.instance_id = workflow.config["last_instance_id"]
        else:
            print("Please provide --instance-id")
            return
        
        print(f"Rebooting GPU instance {workflow.instance_id}...")
        if workflow.client.reboot_gpu(workflow.instance_id):
            print("✓ GPU rebooted successfully")
        else:
            print("✗ Failed to reboot GPU")


if __name__ == "__main__":
    main()

