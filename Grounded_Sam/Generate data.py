import os
import json
import subprocess
from tqdm import tqdm

WEBQA_DIR = "/home/featurize/WEBQA"
PROMPT_FILE = os.path.join(WEBQA_DIR, "grounded_sam_prompts", "prompts_fallback.jsonl")
IMAGE_IDS_FILE = os.path.join(WEBQA_DIR, "webqa_image_ids", "image_ids.txt")
IMAGE_DIR = os.path.join(WEBQA_DIR, "webqa_images")
OUTPUT_BASE = os.path.join(WEBQA_DIR, "grounded_sam_outputs")

CONFIG_PATH = "/home/featurize/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CKPT = "/home/featurize/Grounded-Segment-Anything/checkpoints/groundingdino_swint_ogc.pth"
SAM_CKPT = "/home/featurize/Grounded-Segment-Anything/checkpoints/sam_vit_h_4b8939.pth"
SAM_VERSION = "vit_h"

DEVICE = "cuda"
BOX_THRESH = 0.3
TEXT_THRESH = 0.25

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_lines = [json.loads(line.strip()) for line in f]

with open(IMAGE_IDS_FILE, "r", encoding="utf-8") as f:
    image_ids = [line.strip() for line in f]

assert len(prompt_lines) == len(image_ids), 

MAX_ENTRIES = None


total = len(prompt_lines) if MAX_ENTRIES is None else min(len(prompt_lines), MAX_ENTRIES)

for i, (image_id, prompt_list) in enumerate(tqdm(zip(image_ids, prompt_lines), total=total, desc="▶️ Running Grounded-SAM")):
    if MAX_ENTRIES and i >= MAX_ENTRIES:
        break

    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        print(f" 图像不存在: {image_path}")
        continue

    prompt = ", ".join(prompt_list) if isinstance(prompt_list, list) else str(prompt_list)
    out_dir = os.path.join(OUTPUT_BASE, str(i))  
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "grounded_sam_demo.py",
        "--config", CONFIG_PATH,
        "--grounded_checkpoint", DINO_CKPT,
        "--sam_version", SAM_VERSION,
        "--sam_checkpoint", SAM_CKPT,
        "--input_image", image_path,
        "--text_prompt", prompt,
        "--output_dir", out_dir,
        "--box_threshold", str(BOX_THRESH),
        "--text_threshold", str(TEXT_THRESH),
        "--device", DEVICE
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 0:
        print(f" [{i}] 成功: {image_id}.jpg")
    else:
        print(f" [{i}] 失败: {image_id}.jpg")
