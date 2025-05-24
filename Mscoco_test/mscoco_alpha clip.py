import os
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor  # Optional: if you want to compare
from pycocotools.coco import COCO
import random
import alpha_clip
from torchvision import transforms

DATASET_PATH = '/home/featurize/COCO2017/val2017'
ANNOTATION_FILE = '/home/featurize/COCO2017/annotations_trainval2017/captions_val2017.json'
FAISS_INDEX_PATH = './mscoco_alpha_clip.index'
ALPHA_CLIP_CKPT = '/home/featurize/Clip/clip_l14_336_grit_20m_4xe.pth'

SAMPLE_SIZE = 50  # Number of test samples for Recall@5
TOP_K = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(" Loading Alpha-CLIP model...")
alpha_model, alpha_preprocess = alpha_clip.load(
    "ViT-L/14@336px",
    alpha_vision_ckpt_pth=ALPHA_CLIP_CKPT,
    device=DEVICE
)

# Mask transform (normalize mask input)
mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336)),
    transforms.Normalize(0.5, 0.26)
])


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
def build_faiss_index_alpha(img_ids, index_path=FAISS_INDEX_PATH):
    d = 768  
    index = faiss.IndexFlatIP(d)  
    imgid_list = []

    for img_id in tqdm(img_ids, desc="Extracting & indexing (Alpha-CLIP)"):
        path = image_id_to_path[img_id]
        try:
            image = Image.open(path).convert("RGB")
            image_tensor = alpha_preprocess(image).unsqueeze(0).to(DEVICE).half()

 
            H, W = image.size
            mask = np.ones((H, W), dtype=bool)
            binary_mask = mask_transform((mask.astype(np.uint8) * 255)).unsqueeze(0).to(DEVICE).half()

            img_feat = alpha_model.visual(image_tensor, binary_mask)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            vec = img_feat.cpu().numpy().astype(np.float32)
            index.add(vec)
            imgid_list.append(img_id)
        except Exception as e:
            print(f"[WARN] Failed: {path} - {e}")

    print(f" Built Alpha-CLIP Faiss index with {index.ntotal} vectors.")


    faiss.write_index(index, index_path)
    np.save(index_path + ".ids.npy", np.array(imgid_list))
    print(f" Saved index to: {index_path} (+ .ids.npy)")

if not os.path.exists(FAISS_INDEX_PATH):
    build_faiss_index_alpha(valid_img_ids)
else:
    print("Index already exists, skip building.")

index = faiss.read_index(FAISS_INDEX_PATH)
imgid_list = np.load(FAISS_INDEX_PATH + ".ids.npy")
print(f"Loaded Alpha-CLIP Faiss index with {index.ntotal} vectors.")

@torch.no_grad()
def recall_at_k_alpha(sampled_img_ids, top_k=TOP_K):
    hit_count = 0

    for img_id in tqdm(sampled_img_ids, desc="Evaluating Recall@5 (Alpha-CLIP)"):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        captions = [coco.loadAnns(aid)[0]['caption'] for aid in ann_ids]
        if not captions:
            continue
        query_text = random.choice(captions)

        text_tokens = alpha_clip.tokenize([query_text]).to(DEVICE)
        text_feat = alpha_model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        vec = text_feat.cpu().numpy().astype(np.float32)

        D, I = index.search(vec, top_k)
        topk_img_ids = [imgid_list[i] for i in I[0]]

        if img_id in topk_img_ids:
            hit_count += 1

    recall = hit_count / len(sampled_img_ids)
    return recall

if __name__ == "__main__":
    sampled_img_ids = random.sample(valid_img_ids, min(SAMPLE_SIZE, len(valid_img_ids)))
    recall = recall_at_k_alpha(sampled_img_ids, top_k=TOP_K)
    print(f"\n Final Recall@{TOP_K} (Alpha-CLIP): {recall:.4f} on {len(sampled_img_ids)} samples.")
