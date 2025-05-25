import os
import random
import numpy as np
import torch
import faiss
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

WEBQA_DIR        = "/home/featurize/WEBQA"
IMAGE_IDS_FILE   = os.path.join(WEBQA_DIR, "webqa_image_ids", "image_ids.txt")
CAPTIONS_FILE    = os.path.join(WEBQA_DIR, "webqa_captions",  "captions.txt")  
IMAGE_DIR        = os.path.join(WEBQA_DIR, "webqa_images")                   
FAISS_INDEX_PATH = os.path.join(WEBQA_DIR, "webqa_clip.index")
FAISS_IDS_PATH   = FAISS_INDEX_PATH + ".ids.npy"

SAMPLE_SIZE = 100     
TOP_K       = 5       
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def build_or_load_faiss(image_ids):
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
                inputs = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
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



def recall_at_k(index, imgid_list, captions, sampled_idxs, top_k=TOP_K):
    hits = 0
    N    = len(sampled_idxs)
    for idx in tqdm(sampled_idxs, desc=f" Recall@{top_k}"):
        text = captions[idx]
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        txt_feat = clip_model.get_text_features(**inputs)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        vec      = txt_feat.cpu().numpy().astype(np.float32)
        _, I     = index.search(vec, top_k)
        preds    = [imgid_list[i] for i in I[0]]
        if imgid_list[idx] in preds:
            hits += 1
    return hits / N if N > 0 else 0


def main():
  
    with open(IMAGE_IDS_FILE) as f: 
        ids = [l.strip() for l in f if l.strip()]
    with open(CAPTIONS_FILE) as f:
        caps = [l.strip() for l in f if l.strip()]
    assert len(ids) == len(caps), "image_ids.txt and captions.txt must have the same number of lines!"

    faiss_index, imgid_list = build_or_load_faiss(ids)

    sample_n = min(SAMPLE_SIZE, len(ids))
    sampled_idxs = random.sample(range(len(ids)), sample_n)


    rec = recall_at_k(faiss_index, imgid_list, caps, sampled_idxs, TOP_K)
    print(f"\n Self-recall@{TOP_K}: {rec:.4f} over {sample_n} samples.")

if __name__ == "__main__":
    main()
