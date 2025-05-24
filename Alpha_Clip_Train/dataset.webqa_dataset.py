import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class WebQAMaskDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(224, 224)):
        self.root_dir = root_dir
        self.output_dir = os.path.join(root_dir, "grounded_sam_outputs")
        self.image_ids_file = os.path.join(root_dir, "webqa_image_ids", "image_ids.txt")
        self.captions_file = os.path.join(root_dir, "webqa_captions", "captions.txt")
        self.prompts_file = os.path.join(root_dir, "grounded_sam_prompts", "prompts_fallback.jsonl")

        with open(self.image_ids_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        with open(self.captions_file, "r") as f:
            self.captions = [line.strip() for line in f.readlines()]

        with open(self.prompts_file, "r", encoding="utf-8") as f:
            self.prompts = [json.loads(line.strip()) for line in f.readlines()]

        assert len(self.image_ids) == len(self.captions) == len(self.prompts), "Mismatch between ids, captions, and prompts"

        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        output_path = os.path.join(self.output_dir, str(idx))
        image_path = os.path.join(output_path, "raw_image.jpg")
        mask_path = os.path.join(output_path, "mask.jpg")
        mask_json_path = os.path.join(output_path, "mask.json")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.mask_transform(mask)

        image_id = self.image_ids[idx]
        caption = self.captions[idx]
        prompt = self.prompts[idx]
        with open(mask_json_path, "r") as f:
            mask_info = json.load(f)
        return {
            'image': image,             
            'caption': caption,         
            'prompt': prompt,           
            'mask': mask,               
            'image_id': image_id,       
            'mask_info': mask_info      
        }
