import os
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from pycocotools.coco import COCO
import random
import alpha_clip
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


DATASET_PATH = '/home/featurize/COCO2017/val2017'
ANNOTATION_FILE = '/home/featurize/COCO2017/annotations_trainval2017/captions_val2017.json'
ALPHA_CLIP_CKPT = '/home/featurize/Clip/clip_l14_336_grit_20m_4xe.pth'
SAM_CKPT = '/home/featurize/GroundingDINO/checkpoints/sam_vit_h_4b8939.pth'
FAISS_INDEX_PATH = './faiss_clip_vitl14.index'  

SAMPLE_SIZE = 10
TOP_K_CLIP = 100
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


coco = COCO(ANNOTATION_FILE)
image_ids = list(coco.imgs.keys())
print(f"Found {len(image_ids)} images in COCO.")

image_id_to_path = {
    img_id: os.path.join(DATASET_PATH, coco.loadImgs(img_id)[0]['file_name'])
    for img_id in image_ids
}
valid_img_ids = [img_id for img_id, path in image_id_to_path.items() if os.path.exists(path)]
print(f"Valid images: {len(valid_img_ids)}")



def build_faiss_index(img_ids, index_path=FAISS_INDEX_PATH):
    d = 768  
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

    print(f" Built Faiss index with {index.ntotal} vectors.")
    faiss.write_index(index, index_path)
    np.save(index_path + ".ids.npy", np.array(imgid_list))
    print(f" Saved index to: {index_path} (+ .ids.npy)")



print("⚠️ Rebuilding FAISS index to ensure consistency...")
build_faiss_index(valid_img_ids)


faiss_index = faiss.read_index(FAISS_INDEX_PATH)
imgid_list = np.load(FAISS_INDEX_PATH + ".ids.npy")




def clip_faiss_topk(query_text, k=TOP_K_CLIP):
    inputs = clip_processor(text=query_text, return_tensors="pt").to(DEVICE)
    text_feat = clip_model.get_text_features(**inputs)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_vec = text_feat.cpu().numpy().astype(np.float32)
    D, I = faiss_index.search(text_vec, k)
    print(f"🔍 FAISS search got indices: {I[0]}")
    return [int(imgid_list[i]) for i in I[0] if i != -1]



def sam_segment_full_image(image_pil, max_masks=30):
    image = np.array(image_pil)
    masks_data = mask_generator.generate(image)


    masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)[:max_masks]

    masks = []
    for m in masks_data:
        mask = torch.tensor(m['segmentation']).unsqueeze(0)  # [1, H, W]
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
    sampled_img_ids = random.sample(valid_img_ids, min(SAMPLE_SIZE, len(valid_img_ids)))
    hit_count = 0

    for query_img_id in tqdm(sampled_img_ids, desc="Full Pipeline"):
        ann_ids = coco.getAnnIds(imgIds=query_img_id)
        captions = [coco.loadAnns(aid)[0]['caption'] for aid in ann_ids]
        if not captions:
            continue
        query_text = random.choice(captions)
        print(f"\n Query: {query_text}")


        topk_clip_results = clip_faiss_topk(query_text, k=TOP_K_CLIP)
        print(f" Retrieved {len(topk_clip_results)} candidates from FAISS")

        if not topk_clip_results:
            print(" No candidates found, skipping this query.")
            continue
        alpha_scores = []
        for img_id in topk_clip_results:
            img_path = image_id_to_path[img_id]
            image_pil = Image.open(img_path).convert("RGB")

            masks = sam_segment_full_image(image_pil)
            print(f" Rerank on image_id: {img_id}, got {len(masks)} masks")

            if masks:
                score = alpha_clip_rerank(query_text, image_pil, masks)
                alpha_scores.append((img_id, score))

        if not alpha_scores:
            print(" No valid masks for rerank.")
            continue

        alpha_scores = sorted(alpha_scores, key=lambda x: x[1], reverse=True)
        topk_final = [i for i, _ in alpha_scores[:TOP_K_FINAL]]

        print(f" Final top-{TOP_K_FINAL}: {topk_final}")
        if query_img_id in topk_final:
            hit_count += 1

    recall = hit_count / len(sampled_img_ids)
    print(f"\n Final Recall@{TOP_K_FINAL}: {recall:.4f} on {len(sampled_img_ids)} samples.")