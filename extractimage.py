import os
import csv
import requests
from tqdm import tqdm

TSV_FILE = "/root/autodl-tmp/rag/WEBQA/webqa_yesno_vqa.tsv"
SAVE_DIR = "/root/autodl-tmp/rag/WEBQA/vqa_images"

os.makedirs(SAVE_DIR, exist_ok=True)

with open(TSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows = list(reader)

for row in tqdm(rows, desc="Downloading images"):
    image_id = row["image_id"]
    image_url = row["image_url"]
    save_path = os.path.join(SAVE_DIR, f"{image_id}.jpg")

    if os.path.exists(save_path):
        continue  # already downloaded

    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"[Warning] Failed to download {image_id}: HTTP {response.status_code}")
    except Exception as e:
        print(f"[Error] {image_id}: {e}")
