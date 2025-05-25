import argparse
import os
import json
import base64
import torch
import faiss
import random
import numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import alpha_clip
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="WebQA + CLIP + Alpha-CLIP + SAM Pipeline")

    parser.add_argument('--webqa_dir', type=str, default="/home/featurize/WEBQA", help="Root directory of WebQA")
    parser.add_argument('--image_ids_file', type=str, default="webqa_image_ids/image_ids.txt", help="Relative path to image IDs")
    parser.add_argument('--captions_file', type=str, default="webqa_captions/captions.txt", help="Relative path to captions")
    parser.add_argument('--image_subdir', type=str, default="webqa_images", help="Subdirectory for images")

    parser.add_argument('--faiss_index_name', type=str, default="webqa_clip.index", help="FAISS index file name")
    parser.add_argument('--alpha_ckpt', type=str, default="/home/featurize/new work/Clip/clip_l14_336_grit_20m_4xe.pth", help="Alpha-CLIP checkpoint")
    parser.add_argument('--sam_ckpt', type=str, default="/home/featurize/new work/GroundingDINO/checkpoints/sam_vit_h_4b8939.pth", help="SAM checkpoint")
    parser.add_argument('--precomputed_mask_dir', type=str, default="/home/featurize/WEBQA/grounded_sam_outputs", help="Directory containing precomputed mask.json files")

    parser.add_argument('--sample_size', type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument('--top_k', type=int, default=5, help="Top-K images to rerank")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use")

    return parser.parse_args()

def build_or_load_faiss(image_ids, image_dir, faiss_index_path, faiss_ids_path,
                        clip_model, clip_processor, device):

    d = 512
    error_log_path = os.path.join(WEBQA_DIR, "invalid_images.txt")

    if os.path.exists(FAISS_INDEX_PATH):
        index      = faiss.read_index(FAISS_INDEX_PATH)
        imgid_list = np.load(FAISS_IDS_PATH, allow_pickle=True).tolist()
        print(f" Loaded FAISS index with {len(imgid_list)} images.")
    else:
        index      = faiss.IndexFlatIP(d)
        imgid_list = []

        for img_id in tqdm(image_ids, desc=" Building FAISS"):
            path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")

            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")  
                    np_img = np.array(img)


                    if np_img.ndim != 3 or np_img.shape[2] != 3:
                        raise ValueError(f"Invalid image shape: {np_img.shape}")
                    
                    image = Image.fromarray(np_img)
            except Exception as e:
                with open(error_log_path, "a") as logf:
                    logf.write(f"{img_id}\t{str(e)}\n")
                print(f" Skipping invalid image {img_id}: {e}")
                continue

            try:
                inputs = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    feat = clip_model.get_image_features(**inputs)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                vec = feat.cpu().numpy().astype(np.float32)
                index.add(vec)
                imgid_list.append(img_id)
            except Exception as e:
                print(f" Failed CLIP encoding for {img_id}: {e}")
                continue

        faiss.write_index(index, FAISS_INDEX_PATH)
        np.save(FAISS_IDS_PATH, np.array(imgid_list))
        print(f" Built FAISS index with {len(imgid_list)} images.")
    
    return index, imgid_list


def clip_faiss_topk(query_text, clip_model, clip_processor, faiss_index, imgid_list, device, k):

    inputs = clip_processor(text=query_text, return_tensors="pt").to(device)
    text_feat = clip_model.get_text_features(**inputs)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    vec = text_feat.cpu().numpy().astype(np.float32)
    D, I = faiss_index.search(vec, k)

    return [imgid_list[i] for i in I[0] if i != -1]





def load_precomputed_sam_masks(image_id, sam_output_dir, max_masks=30, min_logit=0.3):
    """
    Load precomputed SAM masks from JSON output and filter based on logit confidence and label.

    Args:
        image_id (str or int): Identifier of the image.
        sam_output_dir (str): Directory where SAM masks are stored.
        max_masks (int): Maximum number of high-confidence masks to return.
        min_logit (float): Minimum logit threshold for valid masks.

    Returns:
        List[dict]: A list of selected mask dictionaries with high confidence and valid labels.
    """
    mask_path = os.path.join(sam_output_dir, str(image_id), "mask.json")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    with open(mask_path, 'r', encoding='utf-8') as f:
        try:
            mask_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {mask_path}: {e}")

    valid_masks = []
    for m in mask_data:

        label = m.get("label", "").strip().lower()
        logit = m.get("logit", 0.0)
        if label == "background":
            continue
        if logit >= min_logit and "box" in m and isinstance(m["box"], list) and len(m["box"]) == 4:
            valid_masks.append(m)

    if not valid_masks:
        print(f"[Warning] No valid masks found for image ID {image_id} with logit ≥ {min_logit}")

    valid_masks.sort(key=lambda m: m["logit"], reverse=True)
    return valid_masks[:max_masks]





def crop_box_from_raw_image(image_path, box):

    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)
    return image.crop((x1, y1, x2, y2))


def alpha_clip_rerank(query_text, global_image_pil, mask_region_list, alpha_model, alpha_preprocess, device):

   
    image_tensor = alpha_preprocess(global_image_pil).unsqueeze(0).to(device).half()
    text_tokens = alpha_clip.tokenize([query_text]).to(device)
    text_feat = alpha_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    
    full_mask = torch.ones(1, 1, 336, 336).to(device).half()
    full_feat = alpha_model.visual(image_tensor, full_mask)
    full_feat = full_feat / full_feat.norm(dim=-1, keepdim=True)
    global_sim = (full_feat @ text_feat.T).item()

    best_sim = -1
    for region in mask_region_list:
        try:
            region_tensor = alpha_preprocess(region).unsqueeze(0).to(device).half()
            region_mask = torch.ones(1, 1, 336, 336).to(device).half()
            feat = alpha_model.visual(region_tensor, region_mask)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sim = (feat @ text_feat.T).item()
            best_sim = max(best_sim, sim)
        except Exception as e:
            print(f"[Rerank Mask Error] {e}")
            continue

    return 0.7 * best_sim + 0.3 * global_sim if best_sim >= 0 else global_sim



def sam_segment_full_image(image_pil, max_masks=30):
    image = np.array(image_pil)
    masks_data = mask_generator.generate(image)
    masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)[:max_masks]
    masks = [torch.tensor(m['segmentation']).unsqueeze(0) for m in masks_data]
    masks.append(torch.ones(1, image.shape[0], image.shape[1]))
    return masks

def main():
    args = parse_args()

    image_ids_path = os.path.join(args.webqa_dir, args.image_ids_file)
    captions_path = os.path.join(args.webqa_dir, args.captions_file)
    image_dir = os.path.join(args.webqa_dir, args.image_subdir)
    faiss_index_path = os.path.join(args.webqa_dir, args.faiss_index_name)
    faiss_ids_path = faiss_index_path + ".ids.npy"

    with open(image_ids_path, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    with open(captions_path, 'r') as f:
        all_lines = f.readlines()
    queries = [line.strip().split('\t') for line in random.sample(all_lines, args.sample_size) if '\t' in line]

    print("[INFO] Loading models...")
    faiss_index = faiss.read_index(faiss_index_path)

    id_mapping = np.load(faiss_ids_path, allow_pickle=True).tolist()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    alpha_model, alpha_preprocess = alpha_clip.load(
        "ViT-L/14@336px", alpha_vision_ckpt_pth=args.alpha_ckpt, device=args.device
    )

    correct = 0
    for gt_image_id, query_text in tqdm(queries, desc="Running Evaluation"):
        inputs = clip_processor(text=query_text, return_tensors="pt").to(args.device)
        text_feat = clip_model.get_text_features(**inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        vec = text_feat.cpu().numpy().astype(np.float32)
        D, I = faiss_index.search(vec, args.top_k)
        topk_ids = [id_mapping[i] for i in I[0] if i != -1]

        best_score = -1
        best_id = None

        for image_id in topk_ids:
            try:
                raw_image_path = os.path.join(image_dir, f"{image_id}.jpg")
                raw_image = Image.open(raw_image_path).convert("RGB")
                masks = load_precomputed_sam_masks(image_id, args.precomputed_mask_dir)
                regions = [crop_box_from_raw_image(raw_image_path, m["box"]) for m in masks]
                score = alpha_clip_rerank(query_text, raw_image, regions, alpha_model, alpha_preprocess, args.device)
                if score > best_score:
                    best_score = score
                    best_id = image_id
            except Exception as e:
                print(f"[Error processing {image_id}] {e}")

        if best_id == gt_image_id:
            correct += 1

    print(f"\n Recall@{args.top_k}: {correct / len(queries):.4f} ({correct}/{len(queries)})")


if __name__ == "__main__":
    main()