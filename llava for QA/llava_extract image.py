import os
import json
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm


qa_json_path = "datasets/WebQA_test_image.json"  
tsv_path = "imgs.tsv"                   
output_dir = "test_images"              

with open(qa_json_path, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

image_ids = set()
for qid, entry in qa_data.items():
    for img in entry.get("img_posFacts", []):
        image_ids.add(str(img["image_id"]))
    for img in entry.get("img_negFacts", []):
        image_ids.add(str(img["image_id"]))

print(f"[INFO] 需要提取 {len(image_ids)} 张图像。")

os.makedirs(output_dir, exist_ok=True)
extracted = 0

with open(tsv_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc=" 正在提取图像"):
        try:
            img_id, img_b64 = line.strip().split('\t')
            if img_id in image_ids:
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                img.save(os.path.join(output_dir, f"{img_id}.jpg"))
                extracted += 1
        except Exception as e:
            print(f"[ERROR] 跳过图像 {img_id}: {e}")

print(f"[DONE] 成功提取 {extracted} 张图像，保存至: {output_dir}")
