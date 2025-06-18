import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging
from tqdm import tqdm

class WebQAMaskDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(336, 336), skip_errors=True):
        """
        Args:
            root_dir: 包含 grounded_sam_outputs 的目录的上层路径（即 /home/featurize/WEBQA）
            image_size: 输入图像大小，需匹配 CLIP 模型（ViT-L/14@336px → 336x336）
            skip_errors: 是否跳过错误样本（强烈建议保持True）
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.data_dir = os.path.join(root_dir.rstrip('/'), 'grounded_sam_outputs')
        self.logger.info(f"数据目录设置为: {self.data_dir}")

        meta_dir = "/home/featurize/WEBQA"
        self.image_ids = self._load_file(os.path.join(meta_dir, "webqa_image_ids", "image_ids.txt"))
        self.captions = self._load_file(os.path.join(meta_dir, "webqa_captions", "captions.txt"))
        self.prompts = self._load_jsonl(os.path.join(meta_dir, "grounded_sam_prompts", "prompts_fallback.jsonl"))
        self._validate_metadata()
        self.valid_indices = self._prefilter_samples(skip_errors)
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    def _load_file(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _load_jsonl(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def _validate_metadata(self):
        assert len(self.image_ids) == len(self.captions) == len(self.prompts), \
            f"元数据长度不匹配: 图片ID({len(self.image_ids)}) 描述({len(self.captions)}) 提示({len(self.prompts)})"

    def _prefilter_samples(self, skip_errors):
        valid_indices = []
        for idx in tqdm(range(len(self.image_ids)), desc="验证样本"):
            sample_dir = os.path.join(self.data_dir, self.image_ids[idx])
            required_files = ['raw_image.jpg', 'mask.jpg', 'mask.json']
            if all(os.path.exists(os.path.join(sample_dir, f)) for f in required_files):
                valid_indices.append(idx)
            elif not skip_errors:
                raise FileNotFoundError(f"样本 {self.image_ids[idx]} 缺少必要文件于 {sample_dir}")
        
        self.logger.info(f"有效样本: {len(valid_indices)}/{len(self.image_ids)}")
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):

        idx = self.valid_indices[index]
        sample_dir = os.path.join(self.data_dir, self.image_ids[idx])
        
        try:

            img_path = os.path.join(sample_dir, 'raw_image.jpg')
            with Image.open(img_path) as img:
                image = self.transform(img.convert('RGB'))
            
            mask_path = os.path.join(sample_dir, 'mask.jpg')
            with Image.open(mask_path) as mask:
                mask = self.mask_transform(mask.convert('L'))
            
            with open(os.path.join(sample_dir, 'mask.json'), 'r') as f:
                mask_info = json.load(f)
            
            return {
                'image': image,
                'mask': mask,
                'caption': self.captions[idx],
                'prompt': self.prompts[idx],
                'image_id': self.image_ids[idx],
                'mask_info': mask_info
            }
        except Exception as e:
            raise RuntimeError(f"已验证样本 {self.image_ids[idx]} 加载失败: {str(e)}")

    def get_missing_samples(self):
        return sorted(set(range(len(self.image_ids))) - set(self.valid_indices))

