from PIL import Image
import numpy as np
import os

inp='tmp_single/input/covid_1.png'
mask='tmp_single/sam_output/covid_1.png'
out='tmp_single/sam_output/overlay_covid_1.png'
if os.path.exists(inp) and os.path.exists(mask):
    im=Image.open(inp).convert('RGB')
    m=Image.open(mask).convert('L').resize(im.size)
    red = Image.new('RGBA', im.size, (255,0,0,120))
    im_rgba = im.convert('RGBA')
    overlay = Image.composite(red, im_rgba, m)
    overlay.save(out)
    print('OVERLAY_SAVED:', out)
    arr = np.array(Image.open(mask))
    print('MASK_STATS min,max,unique_count:', int(arr.min()), int(arr.max()), int(len(np.unique(arr))))
else:
    print('mask or image missing')
