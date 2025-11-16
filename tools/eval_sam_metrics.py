import os
import cv2
import numpy as np
import argparse
from pathlib import Path


def read_mask(path, size=None):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if size is not None and (img.shape[1], img.shape[0]) != (size[0], size[1]):
        img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
    # binarize
    mask = (img > 127).astype(np.uint8)
    return mask


def compute_metrics(pred, gt):
    tp = np.logical_and(pred == 1, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    eps = 1e-7
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        'acc': float(acc),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }


def find_gt_for_pred(pred_path, gt_dir):
    pred_base = Path(pred_path).stem
    # try exact match with common extensions
    for ext in ['.png', '.jpg', '.jpeg', '.tif']:
        candidate = Path(gt_dir) / (pred_base + ext)
        if candidate.exists():
            return candidate
    # fallback: try any file with same stem
    for p in Path(gt_dir).glob('*'):
        if p.is_file() and p.stem == pred_base:
            return p
    return None


def eval_dataset(pred_root, data_root):
    results = []
    pred_root = Path(pred_root)
    data_root = Path(data_root)
    for ds in sorted([p for p in pred_root.iterdir() if p.is_dir()]):
        gt_dir = data_root / ds.name / 'train_masks'
        pred_dir = ds
        if not gt_dir.exists():
            print(f"  Skipping {ds.name}: no ground-truth directory {gt_dir}")
            continue
        print(f"Evaluating dataset {ds.name}")
        per = []
        for p in sorted(pred_dir.rglob('*')):
            if not p.is_file():
                continue
            # ignore files that are not images
            if p.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif']:
                continue
            gt = find_gt_for_pred(p, gt_dir)
            if gt is None:
                # no GT found for this prediction
                continue
            gt_mask = read_mask(gt)
            pred_mask = read_mask(p, size=(gt_mask.shape[1], gt_mask.shape[0]))
            if pred_mask is None or gt_mask is None:
                continue
            m = compute_metrics(pred_mask, gt_mask > 0)
            m.update({'dataset': ds.name, 'file': p.name})
            per.append(m)
        if not per:
            print(f"  No matched predictions for {ds.name}")
            continue
        # aggregate
        agg = {k: np.mean([x[k] for x in per]) for k in ['acc', 'iou', 'precision', 'recall', 'f1']}
        agg.update({'dataset': ds.name, 'n': len(per)})
        results.append((agg, per))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-root', type=str, default='tmp_full', help='Root folder with dataset subfolders containing SAM outputs')
    parser.add_argument('--data-root', type=str, default='data', help='Data root containing dataset subfolders with train_masks')
    parser.add_argument('--out-csv', type=str, default=None, help='Optional CSV file to write per-image results')
    args = parser.parse_args()

    results = eval_dataset(args.pred_root, args.data_root)
    # print summary
    for agg, per in results:
        print(f"Dataset {agg['dataset']}: n={agg['n']}, IoU={agg['iou']:.4f}, F1={agg['f1']:.4f}, Acc={agg['acc']:.4f}")
    # optional CSV
    if args.out_csv:
        import csv
        rows = []
        for agg, per in results:
            for p in per:
                row = {**p}
                rows.append(row)
        if rows:
            keys = ['dataset', 'file', 'acc', 'iou', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn']
            with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k, '') for k in keys})


if __name__ == '__main__':
    main()
