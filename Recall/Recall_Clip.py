import os
import random
import numpy as np
import torch
import faiss
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split




WEBQA_DIR = "/root/autodl-tmp/rag/WEBQA"

IMAGE_IDS_FILE = os.path.join(WEBQA_DIR, "webqa_image_ids", "image_ids.txt")
CAPTIONS_FILE  = os.path.join(WEBQA_DIR, "webqa_captions",  "captions.txt")
IMAGE_DIR      = os.path.join(WEBQA_DIR, "webqa_image")

FAISS_INDEX_PATH = os.path.join(WEBQA_DIR, "webqa_clip.index")
FAISS_IDS_PATH   = FAISS_INDEX_PATH + ".ids.npy"


CLIP_MODEL_PATH = os.path.join(WEBQA_DIR, "clip-finetuned")
#CLIP_MODEL_PATH = "/root/autodl-tmp/rag/clip"   

TOP_K  = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)



def build_or_load_faiss(image_ids):
    d = 512
    error_log_path = os.path.join(WEBQA_DIR, "invalid_images_test.txt")

    if os.path.exists(FAISS_INDEX_PATH):
        index      = faiss.read_index(FAISS_INDEX_PATH)
        imgid_list = np.load(FAISS_IDS_PATH, allow_pickle=True).tolist()
        print(f"[FAISS] Loaded test index with {len(imgid_list)} images.")
    else:
        index = faiss.IndexFlatIP(d)
        imgid_list = []

        for img_id in tqdm(image_ids, desc="Building FAISS from test images"):
            img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    np_img = np.array(img)
                    if np_img.ndim != 3 or np_img.shape[2] != 3:
                        raise ValueError(f"Invalid shape: {np_img.shape}")
                    image = Image.fromarray(np_img)
            except Exception as e:
                with open(error_log_path, "a") as f:
                    f.write(f"{img_id}\t{str(e)}\n")
                print(f" [!] Skipped {img_id}: {e}")
                continue

            try:
                inputs = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                with torch.no_grad():
                    feat = clip_model.get_image_features(**inputs)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                index.add(feat.cpu().numpy().astype(np.float32))
                imgid_list.append(img_id)
            except Exception as e:
                print(f" [!] Encoding failed for {img_id}: {e}")
                continue

        faiss.write_index(index, FAISS_INDEX_PATH)
        np.save(FAISS_IDS_PATH, np.array(imgid_list))
        print(f"[FAISS] Built new index with {len(imgid_list)} images.")

    return index, imgid_list


def recall_at_k(index, imgid_list, captions, top_k=TOP_K):
    hits = 0
    total = len(captions)
    for idx in tqdm(range(total), desc=f" Recall@{top_k}"):
        text = captions[idx]
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            txt_feat = clip_model.get_text_features(**inputs)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        vec = txt_feat.cpu().numpy().astype(np.float32)
        _, I = index.search(vec, top_k)
        preds = [imgid_list[i] for i in I[0]]
        if imgid_list[idx] in preds:
            hits += 1
    return hits / total



def main():
    with open(IMAGE_IDS_FILE) as f:
        all_ids = [line.strip() for line in f if line.strip()]
    with open(CAPTIONS_FILE) as f:
        all_caps = [line.strip() for line in f if line.strip()]
    assert len(all_ids) == len(all_caps), "image_ids.txt and captions.txt must match!"
    train_ids, temp_ids, train_caps, temp_caps = train_test_split(all_ids, all_caps, test_size=0.2, random_state=42)
    val_ids, test_ids, val_caps, test_caps = train_test_split(temp_ids, temp_caps, test_size=0.5, random_state=42)

    print(f"[Split] Using full test set: {len(test_ids)} samples")


    index, imgid_list = build_or_load_faiss(test_ids)

    recall = recall_at_k(index, imgid_list, test_caps, top_k=TOP_K)

    print(f"\n[Final] Test Recall@{TOP_K}: {recall:.4f} over {len(test_caps)} samples")

if __name__ == "__main__":
    main()
