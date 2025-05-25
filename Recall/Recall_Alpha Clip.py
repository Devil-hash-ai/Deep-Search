import os
import json
import torch
import faiss
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import alpha_clip


WEBQA_DIR        = "/home/featurize/WEBQA"
IMAGE_IDS_FILE   = os.path.join(WEBQA_DIR, "webqa_image_ids", "image_ids.txt")
CAPTIONS_FILE    = os.path.join(WEBQA_DIR, "webqa_captions",  "captions.txt")
IMAGE_DIR        = os.path.join(WEBQA_DIR, "webqa_images")
FAISS_INDEX_PATH = os.path.join(WEBQA_DIR, "webqa_alpha_clip.index")
FAISS_IDS_PATH   = FAISS_INDEX_PATH + ".ids.npy"
ALPHA_CLIP_CKPT  = "/home/featurize/new work/Clip/clip_l14_336_grit_20m_4xe.pth"

SAMPLE_SIZE = 100
TOP_K       = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


print("\u2705 Loading Alpha-CLIP model...")
alpha_model, alpha_preprocess = alpha_clip.load(
    "ViT-L/14@336px",
    alpha_vision_ckpt_pth=ALPHA_CLIP_CKPT,
    device=DEVICE
)

mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336)),
    transforms.Normalize(0.5, 0.26)
])


def build_or_load_faiss(image_ids):
    d = 768
    error_log_path = os.path.join(WEBQA_DIR, "invalid_images.txt")

    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        imgid_list = np.load(FAISS_IDS_PATH, allow_pickle=True).tolist()
        print(f"\u2705 Loaded FAISS index with {len(imgid_list)} images.")
    else:
        index = faiss.IndexFlatIP(d)
        imgid_list = []

        for img_id in tqdm(image_ids, desc="\ud83d\udd27 Building Alpha-CLIP FAISS"):
            path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
            try:
                image = Image.open(path).convert("RGB")
                image_tensor = alpha_preprocess(image).unsqueeze(0).to(DEVICE).half()


                H, W = image.size
                mask = np.ones((H, W), dtype=bool)
                binary_mask = mask_transform((mask.astype(np.uint8) * 255)).unsqueeze(0).to(DEVICE).half()

                feat = alpha_model.visual(image_tensor, binary_mask)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                vec = feat.detach().cpu().numpy().astype(np.float32)
                index.add(vec)
                imgid_list.append(img_id)
            except Exception as e:
                with open(error_log_path, "a") as f:
                    f.write(f"{img_id}\t{str(e)}\n")
                print(f"[WARN] Skipping image {img_id}: {e}")

        faiss.write_index(index, FAISS_INDEX_PATH)
        np.save(FAISS_IDS_PATH, np.array(imgid_list))
        print(f"\u2705 Built FAISS index with {len(imgid_list)} images.")

    return index, imgid_list



def recall_at_k_alpha(sampled_idxs, top_k=TOP_K):
    hit_count = 0

    for idx in tqdm(sampled_idxs, desc=f"\ud83d\udcca Evaluating Recall@{top_k} (Alpha-CLIP)"):
        try:
            query_text = captions[idx]
            gt_image_id = image_ids[idx]

            tokens = alpha_clip.tokenize([query_text]).to(DEVICE)
            text_feat = alpha_model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            vec = text_feat.detach().cpu().numpy().astype(np.float32)

            D, I = faiss_index.search(vec, top_k)
            pred_ids = [imgid_list[i] for i in I[0] if i != -1]

            if gt_image_id in pred_ids:
                hit_count += 1
        except Exception as e:
            print(f"[ERROR] idx {idx} failed: {e}")
            continue

    return hit_count / len(sampled_idxs) if sampled_idxs else 0.0


def main():
    global image_ids, captions, faiss_index, imgid_list

    with open(IMAGE_IDS_FILE) as f:
        image_ids = [line.strip() for line in f if line.strip()]
    with open(CAPTIONS_FILE) as f:
        captions = [line.strip() for line in f if line.strip()]

    assert len(image_ids) == len(captions), " image_ids and captions must match."

    faiss_index, imgid_list = build_or_load_faiss(image_ids)
    sampled_idxs = random.sample(range(len(image_ids)), min(SAMPLE_SIZE, len(image_ids)))
    recall = recall_at_k_alpha(sampled_idxs)

    print(f"\n Final Recall@{TOP_K} (Alpha-CLIP): {recall:.4f} over {len(sampled_idxs)} samples.")


if __name__ == "__main__":
    main()
