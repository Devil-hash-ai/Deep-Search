import os
import json
import re
from tqdm import tqdm

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from peft import get_peft_model, LoraConfig, TaskType

WEBQA_DIR = "/home/featurize/WEBQA"
IMAGE_IDS_FILE = os.path.join(WEBQA_DIR, "webqa_image_ids", "image_ids.txt")
CAPTIONS_FILE = os.path.join(WEBQA_DIR, "webqa_captions", "captions.txt")
PROMPT_SAVE_FILE = os.path.join(WEBQA_DIR, "grounded_sam_prompts", "prompts.txt")
os.makedirs(os.path.dirname(PROMPT_SAVE_FILE), exist_ok=True)

base_model = GPT2LMHeadModel.from_pretrained("gpt2")
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['c_attn'],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_cfg)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def make_prompt(caption):
    return f"""从下面这个图像描述中提取适合图像定位任务的关键词或短语（object-centric，适用于 grounded-SAM 的文本输入）:
Caption: "{caption}"
输出:"""

def clean_prompt_text(text, fallback_caption=""):
    text = text.strip().replace("\n", " ")

    if "输出:" in text:
        text = text.split("输出:")[-1].strip()

    text = text.strip().strip('"')
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    phrases = [p.strip() for p in text.strip("[]").split(",") if p.strip()]
    cleaned = ", ".join(phrases)

    if len(cleaned) < 3:
        fallback = " ".join(fallback_caption.split()[:5])
        return fallback if fallback else "unknown"

    return cleaned


with open(IMAGE_IDS_FILE) as f:
    ids = [line.strip() for line in f]
with open(CAPTIONS_FILE) as f:
    captions = [line.strip() for line in f]
assert len(ids) == len(captions), "image_ids 和 captions 不匹配"

prompts = []

for i, (image_id, caption) in enumerate(tqdm(zip(ids, captions), total=len(ids), desc="🧠 Generating prompts (fallback guaranteed)")):
    if i > 50: break

    prompt = make_prompt(caption)
    try:
        response = pipe(prompt, max_new_tokens=30, do_sample=False)[0]['generated_text']
        raw_text = response[len(prompt):]
        cleaned = clean_prompt_text(raw_text, fallback_caption=caption)
        prompts.append(cleaned)
    except Exception as e:
        print(f" Error for {image_id}: {e}")
        fallback = " ".join(caption.split()[:5]) or "unknown"
        prompts.append(fallback)

with open(PROMPT_SAVE_FILE, "w", encoding="utf-8") as f:
    for prompt in prompts:
        f.write(prompt + "\n")

print(f" 已保存 {len(prompts)} 条 prompts（无空行）到 {PROMPT_SAVE_FILE}")