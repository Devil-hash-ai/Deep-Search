import os
import json
import argparse
import torch
import faiss
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM

import alpha_clip
from groundingdino.util.inference import load_model as load_dino_model, predict as dino_predict
from segment_anything import sam_model_registry, SamPredictor


def parse_args():
    parser = argparse.ArgumentParser("Full RAG-VQA pipeline with GroundingDINO + SAM + Alpha-CLIP")
    parser.add_argument('--webqa_dir', type=str, required=True)
    parser.add_argument('--tsv_path', type=str, default="webqa_yesno_vqa.tsv")
    parser.add_argument('--image_subdir', type=str, default="vqa_images")
    parser.add_argument('--faiss_index_name', type=str, default="rag_vqa.index")
    parser.add_argument('--faiss_ids_name', type=str, default="rag_vqa.index.ids.npy")
    parser.add_argument('--clip_model_path', type=str, required=True)
    parser.add_argument('--alpha_ckpt', type=str, required=True)
    parser.add_argument('--alpha_model_type', type=str, default="ViT-L/14@336px")
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--vqa', type=str, default="qwen", choices=["qwen", "llava"])
    parser.add_argument('--dino_config', type=str, default="xxxx/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--dino_ckpt', type=str, default="xxxx/groundingdino_swint_ogc.pth")
    parser.add_argument('--sam_type', type=str, default="vit_h")
    parser.add_argument('--sam_ckpt', type=str, default="xxxx/sam_vit_h.pth")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--phrase_model', type=str, default="gpt2", help="Model for phrase extraction")
    return parser.parse_args()


def extract_entity_from_question(question, model, tokenizer, device):
    prompt = f"Extract the main object from this question:\nQ: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=10)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.strip().lower()


def load_vqa_model(name, device):
    if name == "qwen":
        from modelscope import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

        def answer(img, question):
            prompt = tokenizer.from_list_format([{"image": img}, {"text": question}])
            out = model.generate(**prompt)
            return tokenizer.decode(out[0], skip_special_tokens=True).lower()

    elif name == "llava":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        processor = LlavaNextProcessor.from_pretrained("liuhaotian/llava-v1.5-7b")
        model = LlavaNextForConditionalGeneration.from_pretrained("liuhaotian/llava-v1.5-7b", device_map="auto").eval()

        def answer(img, question):
            inputs = processor(images=img, text=question, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=20)
            return processor.batch_decode(out, skip_special_tokens=True)[0].lower()

    else:
        raise ValueError("Unknown VQA model")
    return answer


def run_groundingdino(model, image_path, phrase, box_threshold=0.3, text_threshold=0.25):
    image_pil = Image.open(image_path).convert("RGB")
    boxes, logits, phrases = dino_predict(
        model=model,
        image=image_pil,
        caption=phrase,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=model.device
    )
    return boxes, phrases, image_pil


def run_sam(image_pil, boxes, sam_model):
    predictor = SamPredictor(sam_model)
    image = np.array(image_pil)
    predictor.set_image(image)
    H, W = image.shape[:2]
    input_boxes = boxes * torch.Tensor([W, H, W, H]).to(boxes.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, (H, W))
    masks, _, _ = predictor.predict_torch(boxes=transformed_boxes, multimask_output=False)
    return masks


def rerank_alpha_clip(query, full_img, masks, alpha_model, alpha_preproc, device):
    text_tokens = alpha_clip.tokenize([query]).to(device)
    text_feat = alpha_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    full_tensor = alpha_preproc(full_img).unsqueeze(0).to(device).half()
    full_mask = torch.ones(1, 1, 336, 336).to(device).half()
    global_feat = alpha_model.visual(full_tensor, full_mask)
    global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
    global_sim = (global_feat @ text_feat.T).item()

    best_sim = -1
    for m in masks:
        try:
            seg = m.squeeze().cpu().numpy().astype(np.uint8) * 255
            seg_img = Image.fromarray(seg).convert("RGB").resize(full_img.size)
            crop_tensor = alpha_preproc(seg_img).unsqueeze(0).to(device).half()
            region_mask = torch.ones(1, 1, 336, 336).to(device).half()
            region_feat = alpha_model.visual(crop_tensor, region_mask)
            region_feat = region_feat / region_feat.norm(dim=-1, keepdim=True)
            sim = (region_feat @ text_feat.T).item()
            best_sim = max(best_sim, sim)
        except:
            continue
    return 0.7 * best_sim + 0.3 * global_sim if best_sim > 0 else global_sim


def main():
    args = parse_args()
    tsv_path = os.path.join(args.webqa_dir, args.tsv_path)
    img_dir = os.path.join(args.webqa_dir, args.image_subdir)
    faiss_path = os.path.join(args.webqa_dir, args.faiss_index_name)
    faiss_ids_path = os.path.join(args.webqa_dir, args.faiss_ids_name)

    with open(tsv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))

    clip_model = CLIPModel.from_pretrained(args.clip_model_path).to(args.device)
    clip_proc = CLIPProcessor.from_pretrained(args.clip_model_path)

    faiss_index = faiss.read_index(faiss_path)
    id_map = list(np.load(faiss_ids_path, allow_pickle=True))

    alpha_model, alpha_preproc = alpha_clip.load(args.alpha_model_type, alpha_vision_ckpt_pth=args.alpha_ckpt, device=args.device)
    vqa_answer = load_vqa_model(args.vqa, args.device)
    dino_model = load_dino_model(args.dino_config, args.dino_ckpt, args.device)
    sam_model = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt).to(args.device)


    phrase_tokenizer = AutoTokenizer.from_pretrained(args.phrase_model)
    phrase_model = AutoModelForCausalLM.from_pretrained(args.phrase_model).to(args.device).eval()

    correct, total = 0, 0

    for row in tqdm(rows, desc="[VQA Eval]"):
        q = row['question']
        label = int(row['answer'])

        txt_inputs = clip_proc(text=[q], return_tensors="pt").to(args.device)
        with torch.no_grad():
            txt_feat = clip_model.get_text_features(**txt_inputs)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        D, I = faiss_index.search(txt_feat.cpu().numpy().astype("float32"), args.top_k)
        cand_ids = [id_map[i] for i in I[0] if i != -1]

        best_score, best_id = -1, None
        object_phrase = extract_entity_from_question(q, phrase_model, phrase_tokenizer, args.device)

        for img_id in cand_ids:
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                continue
            try:
                boxes, _, img_pil = run_groundingdino(dino_model, img_path, object_phrase)
                if boxes is None or len(boxes) == 0:
                    continue
                masks = run_sam(img_pil, boxes, sam_model)
                score = rerank_alpha_clip(q, img_pil, masks, alpha_model, alpha_preproc, args.device)
                if score > best_score:
                    best_score = score
                    best_id = img_id
            except Exception as e:
                print(f"[Error] {img_id}: {e}")
                continue

        if best_id:
            try:
                img = Image.open(os.path.join(img_dir, f"{best_id}.jpg")).convert("RGB")
                pred = vqa_answer(img, q)
                pred_label = 1 if "yes" in pred else 0 if "no" in pred else -1
                if pred_label != -1:
                    correct += int(pred_label == label)
                    total += 1
            except Exception as e:
                print(f"[VQA Error] {best_id}: {e}")

    acc = correct / total if total else 0.0
    print(f"\n[RESULT] Accuracy@Top-{args.top_k}: {acc:.4f} ({correct}/{total})")


if __name__ == '__main__':
    main()