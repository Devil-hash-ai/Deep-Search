import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist


if not dist.is_available():
    raise RuntimeError("torch.distributed is not available")

if not dist.is_initialized():
    dist.init_process_group = lambda *args, **kwargs: None
    dist.get_rank = lambda: 0
    dist.get_world_size = lambda: 1

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

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
        subnum=10000,
        amp=True
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

        #self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * log_scale)
        self.model.logit_scale = torch.nn.Parameter(
            torch.ones([], device=f"cuda:{local_rank}", dtype=torch.float32) * log_scale
        )


        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=weigth_decay)
        self.scheduler = None

        self.use_amp = amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        self.best_val_loss = float("inf")

        print(f"[INFO] AMP enabled: {self.use_amp}, device: {torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}")

    def train_webqa(self, train_loader, val_loader, early_stopper, resume=False, warmup_length=200):
        self.scheduler = cosine_lr(
            self.optimizer, base_lr=self.lr,
            warmup_length=warmup_length,
            steps=5000, para_gamma=1.0
        )
        step = 0

        for epoch in range(self.epoch_num):
            self.model.train()
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            loop = tqdm(train_loader, disable=(dist.get_rank() != 0))
            epoch_loss = 0.0

            for batch in loop:
                images = batch['image'].cuda(non_blocking=True)
                masks = batch['mask'].cuda(non_blocking=True)
                texts = batch['text_input'].cuda(non_blocking=True)

                self.optimizer.zero_grad()
                if not self.use_amp:
                    self.scheduler.step(step)

                if self.use_amp:
                    with autocast():
                        loss = self.forward(images, masks, texts)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step(step)
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
                    self.writer.add_scalar("logit_scale", self.model.logit_scale.item(), step)

                if step % 1000 == 0 and dist.get_rank() == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.ckptdir, f"model_step{step}.pth"))

            # ===== 验证阶段 =====
            val_loss = self.evaluate(val_loader)
            if dist.get_rank() == 0:
                print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.ckptdir, "best_model.pth"))

            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"[EarlyStopping] Triggered at epoch {epoch+1}")
                break

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].cuda(non_blocking=True)
                masks = batch['mask'].cuda(non_blocking=True)
                texts = batch['text_input'].cuda(non_blocking=True)

                loss = self.forward(images, masks, texts)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        return total_loss / max(1, total_samples)

    def forward(self, images, masks, texts):
        image_features = self.model.encode_image(images, masks)
        text_features = self.model.encode_text(texts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        labels = torch.arange(images.size(0), device=images.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2
