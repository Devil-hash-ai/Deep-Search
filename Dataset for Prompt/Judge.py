import json
import os
from tqdm import tqdm


WEBQA_DIR  = "/home/featurize/WEBQA"
IDS_FILE   = os.path.join(WEBQA_DIR, "webqa_image_ids", "image_ids.txt")
JSON_FILE  = os.path.join(WEBQA_DIR, "WebQA_test.json")
CAPS_FILE  = os.path.join(WEBQA_DIR, "webqa_captions", "captions.txt")  


with open(IDS_FILE, 'r', encoding='utf-8') as f:
    ids = [line.strip() for line in f if line.strip()]

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

id2cap = {}
for entry in tqdm(data.values(), desc="Mapping img_Facts"):
    for img in entry.get("img_Facts", []):
        iid = str(img["image_id"])
        if iid in ids:
  
            if iid not in id2cap:
                id2cap[iid] = img["caption"].replace('\n', ' ').strip()


os.makedirs(os.path.dirname(CAPS_FILE), exist_ok=True)
with open(CAPS_FILE, 'w', encoding='utf-8') as f:
    for iid in tqdm(ids, desc="Writing captions"):

        f.write(id2cap.get(iid, "") + "\n")

print(f" 重新生成 {len(ids)} 条 captions 到 {CAPS_FILE}")