import os
import json
from PIL import Image
from tqdm import tqdm

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def main(jsonl_path, image_root, output_path, model_path="liuhaotian/llava-v1.5-7b", load_4bit=True):

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_path,
        load_4bit=load_4bit
    )

    if DEFAULT_IMAGE_TOKEN not in tokenizer.get_vocab():
        print(f"[DEBUG] Adding missing token: {DEFAULT_IMAGE_TOKEN}")
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    else:
        print(f"[DEBUG] Token {DEFAULT_IMAGE_TOKEN} already exists.")


    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    results = []

    for sample in tqdm(data, desc="LLaVA Inference"):
        qid = sample["qid"]
        question = sample["question"]
        pos_images = sample.get("pos_images", [])

        if not pos_images:
            print(f"[WARNING] No image for qid {qid}")
            continue

        image_id = pos_images[0]["image_id"]
        image_path = os.path.join(image_root, f"{image_id}.jpg")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Failed to open image {image_path}: {e}")
            continue


        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        image_tensor = process_images([image], image_processor, model.config).to(model.device)

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        )

        if input_ids is None:
            print(f"[ERROR] tokenizer_image_token failed for qid: {qid}")
            continue

        input_ids = input_ids.unsqueeze(0).to(model.device)


        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=128
            )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        results.append({
            "qid": qid,
            "image_id": image_id,
            "question": question,
            "answer": response,
            "gt_answers": sample.get("answers", [])
        })


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n Inference complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main(
        jsonl_path="results/llava_webqa_outputs/processed_webqa.jsonl",
        image_root="test_images",
        output_path="results/llava_webqa_outputs/predictions.jsonl",
        model_path="liuhaotian/llava-v1.5-7b",
        load_4bit=True
    )
