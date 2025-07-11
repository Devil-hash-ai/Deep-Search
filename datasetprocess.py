import os
import json
import csv

WEBQA_DIR = "/root/autodl-tmp/rag/WEBQA"
INPUT_JSON = os.path.join(WEBQA_DIR, "WebQA_train_val.json")
OUTPUT_TSV = os.path.join(WEBQA_DIR, "webqa_yesno_vqa.tsv")

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

vqa_data = []

for guid, entry in raw_data.items():
    if entry.get("Qcate", "").strip().lower() != "yesno":
        continue

    question = entry.get("Q", "").strip().strip('"')
    if not question:
        continue

    answer_list = entry.get("A", [])
    if not answer_list:
        continue
    answer = answer_list[0].strip().strip('"').lower()

    if answer.startswith("yes"):
        label = 1
    elif answer.startswith("no"):
        label = 0
    else:
        continue

    img_pos_facts = entry.get("img_posFacts", [])
    if not img_pos_facts:
        continue
    image_id = img_pos_facts[0].get("image_id", "")
    image_url = img_pos_facts[0].get("image_url", "")
    if not image_id or not image_url:
        continue

    vqa_data.append((image_id, image_url, question, label))

with open(OUTPUT_TSV, "w", newline='', encoding="utf-8") as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    writer.writerow(["image_id", "image_url", "question", "answer"])
    for row in vqa_data:
        writer.writerow(row)

print(f"Processed {len(vqa_data)} VQA samples.")
print(f"Saved to: {OUTPUT_TSV}")
