import numpy as np
import os, csv, json, requests, faiss, torch, argparse
from tqdm import tqdm
from collections import Counter
from PIL import Image

from transformers import CLIPModel, CLIPProcessor
from modelscope import AutoModelForCausalLM, AutoTokenizer


WEBQA_DIR    = "/root/autodl-tmp/rag/WEBQA"
TSV_PATH     = os.path.join(WEBQA_DIR, "webqa_yesno_vqa.tsv")
IMG_DIR      = os.path.join(WEBQA_DIR, "vqa_images")         
CLIP_PATH    = os.path.join(WEBQA_DIR, "clip-finetuned")     
FAISS_PATH   = os.path.join(WEBQA_DIR, "rag_vqa.index")
FAISS_IDS    = FAISS_PATH + ".ids.npy"
TOP_K        = 3                                            
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

def download_missing_images():
    os.makedirs(IMG_DIR, exist_ok=True)
    with open(TSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in tqdm(rows, desc="Downloading images"):
        img_id, url = row["image_id"], row["image_url"]
        path = os.path.join(IMG_DIR, f"{img_id}.jpg")
        if os.path.exists(path) or not url:
            continue
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(path, "wb") as o:
                    o.write(r.content)
        except Exception:
            pass


def build_or_load_faiss(image_ids, clip_model, clip_proc):
    if os.path.exists(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
        id_map = list(np.load(FAISS_IDS, allow_pickle=True))
        print(f"[FAISS] Loaded index ({len(id_map)} vectors)")
        return index, id_map

    d = clip_model.config.projection_dim
    index = faiss.IndexFlatIP(d)
    id_map = []

    for img_id in tqdm(image_ids, desc="Building FAISS"):
        img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = clip_proc(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            index.add(feat.cpu().numpy().astype("float32"))
            id_map.append(img_id)
        except Exception:
            continue

    faiss.write_index(index, FAISS_PATH)
    np.save(FAISS_IDS, np.array(id_map))
    print(f"[FAISS] New index built ({len(id_map)} vectors)")
    return index, id_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa", type=str, default="qwen", choices=["qwen", "llava"],
                        help="Choose VQA model: qwen or llava")
    args = parser.parse_args()

    # (B) read TSV
    with open(TSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    all_img_ids = {r["image_id"] for r in rows}
    print(f"[DATA] {len(rows)} QA pairs, {len(all_img_ids)} unique images")

    # (C) load CLIP
    clip_model = CLIPModel.from_pretrained(CLIP_PATH).to(DEVICE)
    clip_proc  = CLIPProcessor.from_pretrained(CLIP_PATH)

    # (D) faiss
    index, id_map = build_or_load_faiss(all_img_ids, clip_model, clip_proc)

    # (E) load VQA model
    if args.vqa == "qwen":
        from modelscope import AutoModelForCausalLM, AutoTokenizer
        print("[Qwen] loading ...")
        vqa_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat",
                                                        device_map="auto", trust_remote_code=True).eval()
        vqa_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        def get_answer(img, q):
            prompt = vqa_tokenizer.from_list_format([{"image": img}, {"text": q}])
            out = vqa_model.generate(**prompt)
            return vqa_tokenizer.decode(out[0], skip_special_tokens=True).lower()
    elif args.vqa == "llava":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        print("[LLaVA] loading ...")
        processor = LlavaNextProcessor.from_pretrained("liuhaotian/llava-v1.5-7b")
        vqa_model = LlavaNextForConditionalGeneration.from_pretrained(
            "liuhaotian/llava-v1.5-7b", device_map="auto").eval()
        def get_answer(img, q):
            inputs = processor(images=img, text=q, return_tensors="pt").to(vqa_model.device)
            output = vqa_model.generate(**inputs, max_new_tokens=20)
            return processor.batch_decode(output, skip_special_tokens=True)[0].lower()

    # (F) evaluation loop
    correct, total = 0, 0
    for row in tqdm(rows, desc=f"RAG-VQA Top-{TOP_K}"):
        question = row["question"]
        label    = int(row["answer"])

        txt_inputs = clip_proc(text=[question], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            txt_feat = clip_model.get_text_features(**txt_inputs)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        D, I = index.search(txt_feat.cpu().numpy().astype("float32"), TOP_K)
        cand_ids = [id_map[i] for i in I[0] if i != -1]

        votes = []
        for img_id in cand_ids:
            img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                ans = get_answer(img, question)
                if "yes" in ans:
                    votes.append(1)
                elif "no" in ans:
                    votes.append(0)
            except Exception:
                continue

        if votes:
            pred = Counter(votes).most_common(1)[0][0]
            correct += int(pred == label)
            total += 1

    acc = correct / total if total else 0
    print(f"\n[RESULT] Accuracy@Top-{TOP_K}: {acc:.4f} ({correct}/{total})")
