import os
import argparse

# Force CPU-only by clearing CUDA devices before any torch import
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from types import SimpleNamespace
import importlib.util

# Import generate_saliency_maps by path (avoids package import issues)
gen_path = os.path.join(os.getcwd(), 'saliency_maps', 'generate_saliency_maps.py')
spec = importlib.util.spec_from_file_location('generate_saliency_maps', gen_path)
gen_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_mod)
main = gen_mod.main


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--model-name', default='BiomedCLIP')
    parser.add_argument('--finetuned', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--vbeta', type=float, default=0.1)
    parser.add_argument('--vvar', type=float, default=1.0)
    parser.add_argument('--vlayer', type=int, default=7)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json-path', type=str, default='busi.json')
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    # Convert to a simple namespace to match expected args in main
    ns = SimpleNamespace(**vars(args))
    return ns


if __name__ == '__main__':
    args = make_args()
    main(args)
