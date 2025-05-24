import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List

MODEL_CHECKPOINT = "dslim/bert-base-NER"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8


tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForTokenClassification.from_pretrained(MODEL_CHECKPOINT).to(DEVICE)
model.eval()


label2id = model.config.label2id
id2label = model.config.id2label


class WebQANERDataset(Dataset):
    def __init__(self, json_path: str):
        self.data = self._load_json(json_path)

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        queries = []
        if isinstance(raw_data, dict):
            for entry_id, entry in raw_data.items():
                question = entry.get("Q", "").replace("\n", "").strip().strip('"')
                if question:
                    queries.append(question)
        return queries

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"sentence": self.data[idx]}


def webqa_collate_fn(batch):
    texts = [item['sentence'] for item in batch]
    encoded = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128,
        return_offsets_mapping=True
    )
    return encoded, texts


def extract_named_entities(batch_logits, batch_inputs, texts):
    preds = batch_logits.argmax(dim=-1).cpu().tolist()
    results = []

    for i, (pred_seq, text) in enumerate(zip(preds, texts)):
        input_ids = batch_inputs['input_ids'][i].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        tags = [id2label.get(p, "O") for p in pred_seq]

        entities = []
        current = ""
        current_label = ""
        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                if current:
                    entities.append(current)
                current = token
                current_label = tag[2:]
            elif tag.startswith("I-") and current:
                current += token
            else:
                if current:
                    entities.append(current)
                    current = ""
                    current_label = ""
        if current:
            entities.append(current)

        results.append({"query": text, "entities": entities})
    return results

if __name__ == "__main__":
    dataset = WebQANERDataset("/home/featurize/WEBQA/WebQA_train_val.json")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=webqa_collate_fn)

    all_results = []
    with torch.no_grad():
        for batch_inputs, raw_texts in dataloader:
            batch_inputs = {k: v.to(DEVICE) for k, v in batch_inputs.items() if k != 'offset_mapping'}
            logits = model(**batch_inputs).logits
            batch_result = extract_named_entities(logits, batch_inputs, raw_texts)
            all_results.extend(batch_result)

    with open("webqa_entities_output.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(" Extracted entity spans saved to webqa_entities_output.json")
