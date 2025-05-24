import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import numpy as np
import faiss
from dataset.webqa_dataset import WebQAMaskDataset
import alpha_clip  # Load pre-trained alpha_clip model
from trainer.a_trainer import CLIP_Clean_Train

def setup_distributed(backend="nccl", port="29501"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return local_rank

def train_main(args, local_rank):
    dataset = WebQAMaskDataset(root_dir=args.data_root)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size // dist.get_world_size(), sampler=sampler, num_workers=4, pin_memory=True)

    model, _ = alpha_clip.load("ViT-L/14@336px", device='cpu', lora_adapt=False, rank=-1)

    trainer = CLIP_Clean_Train(
        model=model,
        local_rank=local_rank,
        lr=args.lr,
        weigth_decay=args.weight_decay,
        log_scale=args.log_scale,
        para_gamma=args.para_gamma,
        exp_name=args.exp_name,
        warmup_length=args.warmup_length,
        epoch_num=args.epoch_num
    )

    trainer.train_webqa(
        dataloader=dataloader,
        resume=args.resume,
        amp=args.amp,
        warmup_length=args.warmup_length
    )

    if dist.get_rank() == 0:
        final_ckpt_path = os.path.join(args.data_root, "webqa_alpha.pth")
        torch.save(trainer.model.module.state_dict(), final_ckpt_path)
        print(f" Final model saved to {final_ckpt_path}")

def test_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = alpha_clip.load("ViT-L/14@336px", device='cpu', lora_adapt=False, rank=-1)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        text_input = alpha_clip.tokenize([args.query_text]).to(device)
        text_feat = model.encode_text(text_input)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    index = faiss.read_index(args.faiss_index)
    image_ids = np.load(args.faiss_ids)

    D, I = index.search(text_feat.cpu().numpy(), args.topk)
    matched_ids = [image_ids[i] for i in I[0]]

    print("Top-{} results for query: {}".format(args.topk, args.query_text))
    for i, img_id in enumerate(matched_ids):
        print(f"{i+1}. Image ID: {img_id}  |  Distance: {D[0][i]:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/featurize/WEBQA")
    parser.add_argument("--query_text", type=str, default="a photo of a street")
    parser.add_argument("--faiss_index", type=str, default="/home/featurize/WEBQA/webqa_alpha_clip.index")
    parser.add_argument("--faiss_ids", type=str, default="/home/featurize/WEBQA/webqa_alpha_clip.index.ids.npy")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--log_scale", type=float, default=4.6052)
    parser.add_argument("--para_gamma", type=float, default=0.01)
    parser.add_argument("--exp_name", type=str, default="auto")
    parser.add_argument("--warmup_length", type=int, default=200)
    parser.add_argument("--epoch_num", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test")
    args = parser.parse_args()

    if args.mode == "train":
        local_rank = setup_distributed()
        train_main(args, local_rank)
    else:
        test_main(args)

if __name__ == "__main__":
    main()
