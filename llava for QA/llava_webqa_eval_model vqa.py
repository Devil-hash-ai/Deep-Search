import json
import os

def convert_processed_to_vqa_input(processed_jsonl, output_jsonl, image_root_dir):
    with open(processed_jsonl, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for item in dataset:
            if not item.get("pos_images"):
                continue

            qid = item["qid"]
            question = item["question"]
            image_id = item["pos_images"][0]["image_id"]

            fout.write(json.dumps({
                "question_id": qid,
                "image": f"{image_id}.jpg",
                "text": question
            }, ensure_ascii=False) + "\n")

    print(f" Converted to model_vqa input format: {output_jsonl}")


if __name__ == "__main__":
    convert_processed_to_vqa_input(
        processed_jsonl="results/llava_webqa_outputs/processed_webqa.jsonl",
        output_jsonl="results/llava_webqa_outputs/model_vqa_input.jsonl",
        image_root_dir="test_images"
    )
