import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm

from groundingdino.util.inference import Model as GroundingModel
from segment_anything import sam_model_registry, SamPredictor

WEBQA_DIR = "/home/featurize/WEBQA"
IMG_DIR = os.path.join(WEBQA_DIR, "webqa_images")
ID_FILE = os.path.join(WEBQA_DIR, "webqa_image_ids", "image_ids.txt")
PROMPT_FILE = os.path.join(WEBQA_DIR, "grounded_sam_prompts", "prompts_fallback.jsonl")
OUTPUT_DIR = os.path.join(WEBQA_DIR, "grounded_sam_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CKPT = "groundingdino_swint_ogc.pth"
SAM_CKPT = "sam_vit_h_4b8939.pth"
SAM_VERSION = "vit_h"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

dino_model = GroundingModel(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CKPT)
sam = sam_model_registry[SAM_VERSION](checkpoint=SAM_CKPT).to(DEVICE)
sam_predictor = SamPredictor(sam)

with open(ID_FILE, "r") as f:
    ids = [line.strip() for line in f]
with open(PROMPT_FILE, "r") as f:
    prompts = [json.loads(line.strip()) for line in f]
assert len(ids) == len(prompts)

def segment_boxes(image_bgr, xyxy):
    sam_predictor.set_image(image_bgr)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        best_mask = masks[np.argmax(scores)]
        result_masks.append(best_mask)
    return np.array(result_masks)

for image_id, prompt in tqdm(zip(ids, prompts), total=len(ids), desc="🔍 Running Grounded-SAM"):
    image_path = os.path.join(IMG_DIR, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        print(f" Missing image: {image_id}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f" Failed to load: {image_path}")
        continue

    prompt_str = ", ".join(prompt)

    detections = dino_model.predict_with_classes(
        image=image,
        classes=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    if len(detections.xyxy) == 0:
        print(f" No boxes detected for {image_id}")
        continue

    masks = segment_boxes(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections.xyxy)

    for i, (box, mask) in enumerate(zip(detections.xyxy, masks)):
        save_prefix = os.path.join(OUTPUT_DIR, f"{image_id}_{i}")
        box_img = image.copy()
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(box_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imwrite(f"{save_prefix}_box.jpg", box_img)
        cv2.imwrite(f"{save_prefix}_mask.png", (mask * 255).astype(np.uint8))

    print(f" Saved {len(masks)} masks for {image_id}")
