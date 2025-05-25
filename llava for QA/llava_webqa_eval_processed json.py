import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class WebQADataset(Dataset):
    def __init__(self, 
                 qa_json_path, 
                 caption_json_path,
                 image_index_json_path,
                 image_root_dir,
                 transform=None):
        with open(qa_json_path, 'r', encoding='utf-8') as f:
            self.qa_data = json.load(f)

        with open(caption_json_path, 'r', encoding='utf-8') as f:
            self.caption_map = json.load(f)

        with open(image_index_json_path, 'r', encoding='utf-8') as f:
            self.index_to_imgid = json.load(f)

        self.image_root = image_root_dir
        self.transform = transform
        self.keys = list(self.qa_data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        qid = self.keys[idx]
        entry = self.qa_data[qid]

        question = entry['Q'].strip('"')
        answers = [a.strip('"') for a in entry['A']]
        category = entry.get('Qcate', 'Others')
        split = entry.get('split', 'train')


        pos_images = []
        for item in entry['img_posFacts']:
            image_id = str(item['image_id'])
            img_path = os.path.join(self.image_root, f"{image_id}.jpg")
            caption = self.caption_map.get(image_id, "")
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                pos_images.append({
                    "image": image,
                    "caption": caption,
                    "image_id": image_id
                })
            except Exception as e:
                print(f"[WARNING] Failed to load image: {img_path} ({e})")


        neg_images = []
        for item in entry.get("img_negFacts", []):
            image_id = str(item['image_id'])
            img_path = os.path.join(self.image_root, f"{image_id}.jpg")
            caption = self.caption_map.get(image_id, "")
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                neg_images.append({
                    "image": image,
                    "caption": caption,
                    "image_id": image_id
                })
            except Exception as e:
                print(f"[WARNING] Failed to load image: {img_path} ({e})")

  
        txt_pos_facts = entry.get("txt_posFacts", [])
        txt_neg_facts = entry.get("txt_negFacts", [])

        return {
            "qid": qid,
            "question": question,
            "answers": answers,
            "category": category,
            "split": split,
            "pos_images": pos_images,
            "neg_images": neg_images,
            "txt_posFacts": txt_pos_facts,
            "txt_negFacts": txt_neg_facts
        }
def build_webqa_dataloader(qa_json_path, 
                            caption_json_path,
                            image_index_json_path,
                            image_root_dir,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4,
                            transform=None):

    dataset = WebQADataset(
        qa_json_path=qa_json_path,
        caption_json_path=caption_json_path,   
        image_index_json_path=image_index_json_path, 
        image_root_dir=image_root_dir,  
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x  
    )
    return dataloader

def save_webqa_as_jsonl(dataset, save_path):
    with open(save_path, 'w', encoding='utf-8') as fout:
        for sample in dataset:
            sample_to_save = {
                "qid": sample["qid"],
                "question": sample["question"],
                "answers": sample["answers"],
                "category": sample["category"],
                "split": sample["split"],
                "pos_images": [
                    {
                        "image_id": img["image_id"],
                        "caption": img["caption"]
                    } for img in sample["pos_images"]
                ],
                "neg_images": [
                    {
                        "image_id": img["image_id"],
                        "caption": img["caption"]
                    } for img in sample["neg_images"]
                ],
                "txt_posFacts": sample["txt_posFacts"],
                "txt_negFacts": sample["txt_negFacts"]
            }
            fout.write(json.dumps(sample_to_save, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    qa_path = "datasets/WebQA_test_image.json"
    caption_path = "datasets/WebQA_caption_test.json"
    index_path = "datasets/WebQA_test_image_index_to_id.json"
    image_dir = "test_images"


    save_path = "results/llava_webqa_outputs/processed_webqa.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = WebQADataset(
        qa_json_path=qa_path,
        caption_json_path=caption_path,
        image_index_json_path=index_path,
        image_root_dir=image_dir
    )

    dataloader = build_webqa_dataloader(
        qa_json_path=qa_path,
        caption_json_path=caption_path,
        image_index_json_path=index_path,
        image_root_dir=image_dir
    )

    save_webqa_as_jsonl(dataset, save_path)
    print(f" Processed dataset saved to: {save_path}")
