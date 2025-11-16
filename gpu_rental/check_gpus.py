#!/usr/bin/env python3
"""
Quick script to check available GPUs
"""
import sys
import json
from gpu_api_client import GPUAPIClient

def main():
    # Load config
    try:
        with open('../gpu_config.json', 'r') as f:
            config = json.load(f)
        api_key = config['api_key']
    except:
        if len(sys.argv) > 1:
            api_key = sys.argv[1]
        else:
            print("Usage: python check_gpus.py [API_KEY]")
            print("Or ensure gpu_config.json exists")
            return
    
    client = GPUAPIClient(api_key)
    
    # Try common GPU types
    gpu_types = [
        'rtxa4000', 'RTX A4000',
        'rtx3060', 'RTX 3060', 
        'rtx3090', 'RTX 3090',
        'rtx3080', 'RTX 3080',
        'rtxa2000', 'RTX A2000'
    ]
    
    print("=" * 60)
    print("Checking Available GPUs")
    print("=" * 60)
    print()
    
    found_any = False
    for gpu_type in gpu_types:
        gpus = client.get_available_gpus(gpu_type)
        if gpus:
            available = [g for g in gpus if g.get('isAvailableForDemand', False)]
            if available:
                found_any = True
                print(f"✓ {gpu_type}: {len(available)} available GPU(s)")
                for g in available[:3]:  # Show first 3
                    print(f"  ID: {g['id']}")
                    print(f"     Model: {g.get('gpuModel', 'N/A')}")
                    print(f"     Memory: {g.get('gpuMemorySize', 'N/A')} MB")
                    print(f"     Price: {g.get('pricePerGpu', 'N/A')} VND/hour")
                    print(f"     Region: {g.get('rig', {}).get('region', 'N/A')}")
                    print()
    
    if not found_any:
        print("⚠️  No GPUs available for demand rental right now.")
        print("   Try again later or check ckey.vn dashboard for availability.")
        print()
        print("To rent when available, run:")
        print("  python3 run_on_gpu.py --action rent")

if __name__ == '__main__':
    main()

