import os
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from pycocotools.coco import COCO
import random

DATASET_PATH = '/home/featurize/COCO2017/val2017'
ANNOTATION_FILE = '/home/featurize/COCO2017/annotations_trainval2017/captions_val2017.json'
FAISS_INDEX_PATH = './mscoco_clip.index'
SAMPLE_SIZE = 50  # Number of test samples for Recall@5
TOP_K = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

coco = COCO(ANNOTATION_FILE)
image_ids = list(coco.imgs.keys())
print(f"Found {len(image_ids)} images in COCO.")

image_id_to_path = {
    img_id: os.path.join(DATASET_PATH, coco.loadImgs(img_id)[0]['file_name'])
    for img_id in image_ids
}
valid_img_ids = [img_id for img_id, path in image_id_to_path.items() if os.path.exists(path)]
print(f"Valid images: {len(valid_img_ids)}")

@torch.no_grad()
def build_faiss_index(img_ids, index_path=FAISS_INDEX_PATH):
    d = 512  
    index = faiss.IndexFlatIP(d)  
    imgid_list = []

    for img_id in tqdm(img_ids, desc="Extracting & indexing"):
        path = image_id_to_path[img_id]
        try:
            image = Image.open(path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
            img_feat = clip_model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            vec = img_feat.cpu().numpy().astype(np.float32)
            index.add(vec)
            imgid_list.append(img_id)
        except Exception as e:
            print(f"[WARN] Failed: {path} - {e}")

    print(f"✅ Built Faiss index with {index.ntotal} vectors.")

    faiss.write_index(index, index_path)
    np.save(index_path + ".ids.npy", np.array(imgid_list))
    print(f"✅ Saved index to: {index_path} (+ .ids.npy)")

if not os.path.exists(FAISS_INDEX_PATH):
    build_faiss_index(valid_img_ids)
else:
    print("Index already exists, skip building.")

index = faiss.read_index(FAISS_INDEX_PATH)
imgid_list = np.load(FAISS_INDEX_PATH + ".ids.npy")
print(f"Loaded Faiss index with {index.ntotal} vectors.")

def recall_at_k(sampled_img_ids, top_k=TOP_K):
    hit_count = 0

    for img_id in tqdm(sampled_img_ids, desc="Evaluating Recall@5"):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        captions = [coco.loadAnns(aid)[0]['caption'] for aid in ann_ids]
        if not captions:
            continue
        query_text = random.choice(captions)

        with torch.no_grad():
            inputs = clip_processor(text=[query_text], return_tensors="pt").to(DEVICE)
            text_feat = clip_model.get_text_features(**inputs)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            vec = text_feat.cpu().numpy().astype(np.float32)

            D, I = index.search(vec, top_k)  # I: indices
            topk_img_ids = [imgid_list[i] for i in I[0]]

        if img_id in topk_img_ids:
            hit_count += 1

    recall = hit_count / len(sampled_img_ids)
    return recall

if __name__ == "__main__":
    sampled_img_ids = random.sample(valid_img_ids, min(SAMPLE_SIZE, len(valid_img_ids)))
    recall = recall_at_k(sampled_img_ids, top_k=TOP_K)
    print(f"\n Final Recall@{TOP_K}: {recall:.4f} on {len(sampled_img_ids)} samples.")
