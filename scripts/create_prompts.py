import os
import json
import sys

def create_prompts_for(dataset, prompt_text):
    imgdir = os.path.join('data', dataset, 'test_images')
    outdir = os.path.join('saliency_maps', 'text_prompts')
    if not os.path.isdir(imgdir):
        print(f"Missing image dir: {imgdir}")
        return 0
    files = sorted([f for f in os.listdir(imgdir) if os.path.isfile(os.path.join(imgdir, f))])
    mapping = {f: prompt_text for f in files}
    os.makedirs(outdir, exist_ok=True)
    tmp = os.path.join(outdir, f"{dataset}_testing.json.tmp")
    out = os.path.join(outdir, f"{dataset}_testing.json")
    with open(tmp, 'w', encoding='utf-8') as fp:
        json.dump(mapping, fp, ensure_ascii=False)
    os.replace(tmp, out)
    print(f"Wrote {out} ({len(files)} entries)")
    return len(files)

def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ['lung_CT', 'lung_Xray']
    prompts = {
        'lung_CT': 'A medical lung CT scan showing a lung region possibly containing an abnormality.',
        'lung_Xray': 'A chest x-ray showing a suspicious area in the lung.'
    }
    for d in datasets:
        p = prompts.get(d, 'A medical image showing an area of concern.')
        create_prompts_for(d, p)

if __name__ == '__main__':
    main()
