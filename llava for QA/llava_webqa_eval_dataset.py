import json
import os
from tqdm import tqdm

def convert_webqa_to_llava_format(
    webqa_json_path: str,
    output_jsonl_path: str,
    image_root_dir: str
):
    """
    Convert a processed WebQA dataset to LLaVA-style instruction tuning format (tpdm-style).
    """
    with open(webqa_json_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    converted = []

    for item in tqdm(dataset, desc="Converting to LLaVA format"):
        qid = item["qid"]
        question = item["question"]
        answers = item["answers"]
        pos_images = item.get("pos_images", [])

        if not pos_images:
            continue  

        image_id = pos_images[0]["image_id"]
        caption = pos_images[0].get("caption", "")

        image_path = os.path.join(image_root_dir, f"{image_id}.jpg")


        sample = {
            "id": qid,
            "image": os.path.relpath(image_path, start=image_root_dir),  
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + question
                },
                {
                    "from": "gpt",
                    "value": answers[0] if answers else "I don't know."
                }
            ]
        }

        converted.append(sample)


    with open(output_jsonl_path, 'w', encoding='utf-8') as fout:
        for entry in converted:
            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n Saved {len(converted)} samples to {output_jsonl_path}")



if __name__ == "__main__":
    convert_webqa_to_llava_format(
        webqa_json_path="results/llava_webqa_outputs/processed_webqa.jsonl",
        output_jsonl_path="results/llava_webqa_outputs/llava_train_format.jsonl",
        image_root_dir="test_images"
    )

