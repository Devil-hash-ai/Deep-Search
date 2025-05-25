import sys
import os

import json
import base64
import random
import numpy as np
from io import BytesIO
from tqdm import tqdm
from PIL import Image
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
sys.path.append(os.path.abspath("new_work"))
sys.path.append(os.path.abspath("new work/AlphaCLIP"))
import alpha_clip
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


WEBQA_DIR = "WEBQA"

IMG_TSV_PATH = os.path.join(WEBQA_DIR, "imgs.tsv")
IMG_IDX_PATH = os.path.join(WEBQA_DIR, "imgs.lineidx")

ALPHA_CLIP_CKPT = "/home/featurize/new work/Clip/clip_l14_336_grit_20m_4xe.pth"
SAM_CKPT = "/home/featurize/new work/GroundingDINO/checkpoints/sam_vit_h_4b8939.pth"

FAISS_INDEX_PATH = "./webqa_faiss_clip.index"


IMAGE_INDEX_JSON = os.path.join(WEBQA_DIR, "WebQA_train_val_image_index_to_id.json")
QUESTION_JSON = os.path.join(WEBQA_DIR, "WebQA_train_val.json")



SAMPLE_SIZE = 10
TOP_K_CLIP = 20
TOP_K_FINAL = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


print(" Loading models...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

alpha_model, alpha_preprocess = alpha_clip.load(
    "ViT-L/14@336px",
    alpha_vision_ckpt_pth=ALPHA_CLIP_CKPT,
    device=DEVICE
)


sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
sam.to(DEVICE)
sam_predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

IMG_IDX_PATH = "../WEBQA/imgs.lineidx"
IMG_TSV_PATH = "../WEBQA/imgs.tsv"



print("📦 Loading image index...")
with open(IMG_IDX_PATH, "r") as f:
    id2offset = {str(i): int(offset.strip()) for i, offset in enumerate(f)}

img_tsv = open(IMG_TSV_PATH, "r")

def load_image_by_id(image_id):
    pos = id2offset[str(image_id)]
    img_tsv.seek(pos)
    line = img_tsv.readline()
    parts = line.strip().split('\t')  
    image_data = base64.b64decode(parts[1])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image


@torch.no_grad()
def build_faiss_index(image_ids):
    index = faiss.IndexFlatIP(768)
    imgid_list = []

    for img_id in tqdm(image_ids, desc="Building FAISS"):
        try:
            image = load_image_by_id(img_id)
            inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
            img_feat = clip_model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            index.add(img_feat.cpu().numpy().astype(np.float32))
            imgid_list.append(img_id)
        except Exception as e:
            print(f"[WARN] Failed to process {img_id}: {e}")

    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(FAISS_INDEX_PATH + ".ids.npy", np.array(imgid_list))
    return index, imgid_list


image_ids = list(id2offset.keys())  


if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_INDEX_PATH + ".ids.npy"):
    print("✅ FAISS index already exists. Loading...")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    imgid_list = np.load(FAISS_INDEX_PATH + ".ids.npy")
else:
    print("⚠️ FAISS index not found. Building...")
    image_ids = list(id2offset.keys())
    faiss_index, imgid_list = build_faiss_index(image_ids)




@torch.no_grad()
def clip_faiss_topk(query_text, faiss_index, imgid_list, k=TOP_K_CLIP):
    inputs = clip_processor(text=query_text, return_tensors="pt").to(DEVICE)
    text_feat = clip_model.get_text_features(**inputs)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_vec = text_feat.cpu().numpy().astype(np.float32)
    D, I = faiss_index.search(text_vec, k)
    return [str(imgid_list[i]) for i in I[0] if i != -1]



def sam_segment_full_image(image_pil, max_masks=30):
    image = np.array(image_pil)
    masks_data = mask_generator.generate(image)


    masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)[:max_masks]

    masks = []
    for m in masks_data:
        mask = torch.tensor(m['segmentation']).unsqueeze(0) 
        masks.append(mask)

    full_mask = torch.ones(1, image.shape[0], image.shape[1])
    masks.append(full_mask)

    return masks



def alpha_clip_rerank(query_text, image_pil, masks):
    image_tensor = alpha_preprocess(image_pil).unsqueeze(0).to(DEVICE).half()
    text_tokens = alpha_clip.tokenize([query_text]).to(DEVICE)
    text_feat = alpha_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)


    full_mask = torch.ones(1, 1, 336, 336).to(DEVICE).half()
    full_img_feat = alpha_model.visual(image_tensor, full_mask)
    full_img_feat = full_img_feat / full_img_feat.norm(dim=-1, keepdim=True)
    global_sim = (full_img_feat @ text_feat.T).item()

    best_mask_sim = -1
    for mask in masks:
        binary = mask[0].cpu().numpy() > 0.5
        if binary.sum() < 10:
            continue
        alpha = torch.from_numpy(binary.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        alpha = torch.nn.functional.interpolate(alpha, size=(336, 336), mode='bilinear', align_corners=False)
        alpha = (alpha - 0.5) / 0.26
        alpha = alpha.to(DEVICE).half()
        img_feat = alpha_model.visual(image_tensor, alpha)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ text_feat.T).item()
        best_mask_sim = max(best_mask_sim, sim)

    final_score = 0.7 * best_mask_sim + 0.3 * global_sim
    return final_score

if __name__ == "__main__":
    import os
    import json
    import random
    from tqdm import tqdm


    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    WEBQA_DIR = os.path.join(BASE_DIR, "WEBQA")

    QUESTION_JSON = os.path.join(WEBQA_DIR, "WebQA_train_val.json")  
    IMAGE_INDEX_JSON = os.path.join(WEBQA_DIR, "WebQA_train_val_image_index_to_id.json")  

    print(f" Loading WebQA data from: {QUESTION_JSON}")
    if not os.path.exists(QUESTION_JSON):
        raise FileNotFoundError(f"❌ Dataset file not found: {QUESTION_JSON}")

    with open(QUESTION_JSON, "r") as f:
        questions = json.load(f)

    question_ids = list(questions.keys())
    print(f" Total questions loaded: {len(question_ids)}")


    valid_ids = [qid for qid in question_ids if questions[qid].get("img_posFacts")]
    print(f" Questions with image annotations: {len(valid_ids)}")

    if len(valid_ids) == 0:
        raise ValueError(" No valid annotated samples found.")

    sampled_ids = random.sample(valid_ids, min(SAMPLE_SIZE, len(valid_ids)))
    print(f" Sampling {len(sampled_ids)} question(s) for evaluation.")

 
    with open(IMAGE_INDEX_JSON, "r") as f:
        index_to_image_id = json.load(f)
    id2offset = {v: int(k) for k, v in index_to_image_id.items()}

  
    print(" Building or loading FAISS...")
    if not os.path.exists(FAISS_INDEX_PATH):
        all_image_ids = list(id2offset.keys())
        faiss_index, imgid_list = build_faiss_index(all_image_ids)
    else:
        print(" FAISS index already exists. Loading...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        imgid_list = np.load(FAISS_INDEX_PATH + ".ids.npy")

    hit_count = 0

    for qid in tqdm(sampled_ids, desc="Full Pipeline"):
        try:
            query = questions[qid]["Q"]
            print(f"\n QID: {qid}")
            print(f" Query: {query}")

            pos_imgs = [str(x["image_id"]) for x in questions[qid].get("img_posFacts", [])]
            print(f"📷 Positive images: {pos_imgs}")

            topk_clip_results = clip_faiss_topk(query, faiss_index, imgid_list)
            print(f" Top-{TOP_K_CLIP} retrieved: {topk_clip_results[:5]}...")

            alpha_scores = []
            for img_id in topk_clip_results:
                try:
                    image = load_image_by_id(img_id)  
                    masks = sam_segment_full_image(image)
                    score = alpha_clip_rerank(query, image, masks)
                    alpha_scores.append((img_id, score))
                except Exception as e:
                    print(f"[ WARN] Failed on image {img_id}: {e}")
                    continue

            if not alpha_scores:
                print(f"[ WARN] No valid scores for QID: {qid}")
                continue

            topk_final = [i for i, _ in sorted(alpha_scores, key=lambda x: x[1], reverse=True)[:TOP_K_FINAL]]
            print(f" Final top-{TOP_K_FINAL}: {topk_final}")

            if any(img in pos_imgs for img in topk_final):
                hit_count += 1

        except Exception as e:
            print(f"[ ERROR] Failed on QID {qid}: {e}")
            continue

    recall = hit_count / len(sampled_ids)
    print(f"\n Final Recall@{TOP_K_FINAL}: {recall:.4f} over {len(sampled_ids)} samples.")