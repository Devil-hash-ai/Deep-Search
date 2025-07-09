import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging
from tqdm import tqdm
class WebQAMaskDataset(Dataset):
    def __init__(self, root_dir, image_size=(336, 336)):
        self.data_dir = root_dir  # 你已经传进来的是 grounded_sam_outputs 目录
        self.samples = []

        for folder in os.listdir(self.data_dir):
            sample_dir = os.path.join(self.data_dir, folder)
            if not os.path.isdir(sample_dir):
                continue
            if all(os.path.exists(os.path.join(sample_dir, f)) for f in ['mask.json', 'mask.jpg', 'grounded_sam_output.jpg']):
                self.samples.append({
                    'image_id': folder,
                    'dir': sample_dir
                })

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_dir = sample['dir']
        image_id = sample['image_id']
    
        image = Image.open(os.path.join(sample_dir, 'grounded_sam_output.jpg')).convert('RGB')
        mask = Image.open(os.path.join(sample_dir, 'mask.jpg')).convert('L')
        image = self.transform(image)
        mask = self.mask_transform(mask)
    
        with open(os.path.join(sample_dir, 'mask.json'), 'r') as f:
            meta = json.load(f)
            if isinstance(meta, list):
                meta = meta[0] if len(meta) > 0 else {}
    
        caption = meta.get('caption', f"A photo of object in image {image_id}")
        prompt = meta.get('prompt', f"Describe the mask region in image {image_id}")
    
        return {
            'image': image,
            'mask': mask,
            'caption': caption,
            'prompt': prompt,
            'image_id': image_id,
            'mask_info': meta
        }

