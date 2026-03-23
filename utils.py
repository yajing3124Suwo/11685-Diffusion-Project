# -*- coding: utf-8 -*-
import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def randn_tensor(shape, generator=None, device=None, dtype=torch.float32):
    """Sample standard normal noise with optional generator (shape can be tuple or int tuple)."""
    if isinstance(shape, int):
        shape = (shape,)
    noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return noise


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def init_distributed_device(args):
    """Single-GPU / CPU by default; multi-GPU if launched with torch.distributed."""
    args.local_rank = int(getattr(args, "local_rank", -1))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        if args.local_rank < 0:
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        if args.local_rank < 0:
            args.local_rank = 0

    if torch.cuda.is_available():
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend="nccl")
            device = torch.device("cuda", args.local_rank)
        else:
            device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    args.device = device
    return device


def is_primary(args):
    return not args.distributed or args.rank == 0


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def save_checkpoint(unet, noise_scheduler, vae, class_embedder, optimizer, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "checkpoint_epoch_{:04d}.pt".format(epoch))
    state = {
        "epoch": epoch,
        "unet": unet.state_dict(),
        "noise_scheduler": noise_scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if vae is not None:
        state["vae"] = vae.state_dict()
    if class_embedder is not None:
        state["class_embedder"] = class_embedder.state_dict()
    torch.save(state, path)


def load_checkpoint(unet, scheduler, vae=None, class_embedder=None, checkpoint_path=None):
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        raise ValueError("checkpoint_path must be a valid file")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    unet.load_state_dict(ckpt["unet"])
    if "noise_scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["noise_scheduler"])
    if vae is not None and ckpt.get("vae") is not None:
        vae.load_state_dict(ckpt["vae"])
    if class_embedder is not None and ckpt.get("class_embedder") is not None:
        class_embedder.load_state_dict(ckpt["class_embedder"])
    return ckpt
