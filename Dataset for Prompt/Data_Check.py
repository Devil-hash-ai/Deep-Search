import os
import base64
from tqdm import tqdm

IMAGE_IDS_FILE = "/home/featurize/WEBQA/webqa_image_ids/image_ids.txt"
IMG_TSV_PATH   = "/home/featurize/WEBQA/imgs.tsv"
IMG_IDX_PATH   = "/home/featurize/WEBQA/imgs.lineidx"
OUTPUT_DIR     = "/home/featurize/WEBQA/webqa_image"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(IMAGE_IDS_FILE, 'r', encoding='utf-8') as f:
    wanted_ids = set(line.strip() for line in f if line.strip())

with open(IMG_IDX_PATH, 'r', encoding='utf-8') as f:
    offsets = [int(line.strip()) for line in f if line.strip()]

count = 0
with open(IMG_TSV_PATH, 'r', encoding='utf-8') as tsv, \
     tqdm(offsets, desc="Extracting images") as bar:
    for offset in bar:
        tsv.seek(offset)
        parts = tsv.readline().strip().split('\t', 1)
        if len(parts) != 2:
            continue
        img_id, b64_data = parts
        if img_id not in wanted_ids:
            continue
        try:
            img_bytes = base64.b64decode(b64_data)
            out_path = os.path.join(OUTPUT_DIR, f"{img_id}.jpg")
            with open(out_path, 'wb') as wf:
                wf.write(img_bytes)
            count += 1
        except Exception as e:
            bar.write(f" Failed to decode {img_id}: {e}")

print(f"\n Completed extraction of {count} images to {OUTPUT_DIR}")
