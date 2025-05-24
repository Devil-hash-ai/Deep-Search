import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoTokenizer


def cosine_lr(optimizer, base_lr, warmup_length, steps, para_gamma=1.0):
    def lr_lambda(current_step):
        if current_step < warmup_length:
            return float(current_step) / float(max(1, warmup_length))
        return max(
            0.0,
            0.5 * (1.0 + torch.cos(
                torch.tensor((current_step - warmup_length) / (steps - warmup_length) * 3.141592653589793))
        )) * para_gamma

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class CLIP_Clean_Train:
    def __init__(
        self,
        model,
        local_rank=0,
        lr=4e-5,
        weigth_decay=0.02,
        log_scale=4.6052,
        para_gamma=0.01,
        exp_name="auto",
        warmup_length=200,
        epoch_num=4,
        subnum=10000
    ):
        self.local_rank = local_rank
        torch.cuda.set_device(local_rank)

        self.model = model.float().cuda()
        self.batch_size = 64 // max(1, dist.get_world_size())
        self.lr = lr
        self.epoch_num = epoch_num
        self.subnum = subnum

        if exp_name == "auto":
            self.logdir = f"log/webqa/lr={lr}_wd={weigth_decay}_wl={warmup_length}_logs={log_scale}_e{self.epoch_num}"
        else:
            self.logdir = exp_name
        self.ckptdir = os.path.join(self.logdir, "ckpt")
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        self.model.module.logit_scale = torch.nn.Parameter(torch.ones([]) * log_scale)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=weigth_decay)
        self.scheduler = None
        self.scaler = GradScaler()
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def train_webqa(self, dataloader, resume=False, amp=False, warmup_length=200):
        self.scheduler = cosine_lr(
            self.optimizer, base_lr=self.lr,
            warmup_length=warmup_length,
            steps=5000, para_gamma=1.0
        )
        step = 0

        for epoch in range(self.epoch_num):
            self.model.train()
            if hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
            loop = tqdm(dataloader, disable=(dist.get_rank() != 0))
            epoch_loss = 0.0

            for batch in loop:
                images = batch['image'].cuda(non_blocking=True)
                masks = batch['mask'].cuda(non_blocking=True)
                captions = batch['caption']
                texts = self.tokenize(captions).cuda()

                self.optimizer.zero_grad()
                self.scheduler.step(step)

                if amp:
                    with autocast():
                        loss = self.forward(images, masks, texts)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.forward(images, masks, texts)
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()
                step += 1
                loop.set_postfix(loss=loss.item())

                if step % 50 == 0 and dist.get_rank() == 0:
                    avg_loss = epoch_loss / max(1, step)
                    self.writer.add_scalar("Loss/train", avg_loss, step)
                    self.writer.add_scalar("logit_scale", self.model.module.logit_scale.item(), step)

                if step % 1000 == 0 and dist.get_rank() == 0:
                    torch.save(self.model.module.state_dict(), os.path.join(self.ckptdir, f"model_step{step}.pth"))

    def forward(self, images, masks, texts):
        image_features = self.model.module.encode_image(images, masks)
        text_features = self.model.module.encode_text(texts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.model.module.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        labels = torch.arange(images.size(0), device=images.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2

    def tokenize(self, texts):
        encoding = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        return encoding['input_ids'].to(torch.device(f"cuda:{self.local_rank}"))
