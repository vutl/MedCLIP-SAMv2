#!/usr/bin/env python3
import csv
from pathlib import Path
import sys

def tsv_to_csv(tsv_path, csv_path):
    tsv_path = Path(tsv_path)
    csv_path = Path(csv_path)
    if not tsv_path.exists():
        print('TSV not found:', tsv_path)
        return 1
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, 'r', encoding='utf-8') as fin, open(csv_path, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin, delimiter='\t')
        writer = csv.writer(fout)
        # header expected by biomedclip open_clip train script
        writer.writerow(['filename','Caption'])
        for row in reader:
            if not row:
                continue
            # row[0] = image path, row[1] = caption
            # ensure there are at least two columns
            img = row[0]
            caption = row[1] if len(row) > 1 else ''
            writer.writerow([img, caption])
    print('Wrote CSV:', csv_path)
    return 0

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: tsv_to_csv.py <in.tsv> <out.csv>')
        sys.exit(2)
    sys.exit(tsv_to_csv(sys.argv[1], sys.argv[2]))
