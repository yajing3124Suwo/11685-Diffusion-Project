# -*- coding: utf-8 -*-
import logging
import os
from logging import getLogger as get_logger

import torch
from tqdm import tqdm
from PIL import Image

from torchvision import datasets, transforms

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def build_val_dataset(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    if args.dataset == "cifar10":
        root = os.path.join(args.data_dir, "cifar10")
        return datasets.CIFAR10(root=root, train=False, download=args.cifar_download, transform=transform)
    return datasets.ImageFolder(root=os.path.join(args.data_dir, "val"), transform=transform)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("Creating model")
    attn_levels = list(args.unet_attn) if (args.unet_attn is not None and len(args.unet_attn) > 0) else []
    unet = UNet(
        input_size=args.unet_in_size,
        input_ch=args.unet_in_ch,
        T=args.num_train_timesteps,
        ch=args.unet_ch,
        ch_mult=tuple(args.unet_ch_mult),
        attn=attn_levels,
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.unet_dropout,
        conditional=args.use_cfg,
        c_dim=args.unet_ch,
    )
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info("Number of parameters: %.2fM", num_params / 1.0e6)

    vae = None
    if args.latent_ddpm:
        vae = VAE()
        vae.init_from_ckpt("pretrained/model.ckpt")
        vae.eval()

    class_embedder = None
    if args.use_cfg:
        class_embedder = ClassEmbedder(args.unet_ch, n_classes=args.num_classes)

    if args.use_ddim:
        scheduler_class = DDIMScheduler
    else:
        scheduler_class = DDPMScheduler

    scheduler = scheduler_class(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )

    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae is not None:
        vae = vae.to(device)
    if class_embedder is not None:
        class_embedder = class_embedder.to(device)

    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)

    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler, vae=vae, class_embedder=class_embedder)

    logger.info("***** Running inference *****")

    all_images = []
    if args.use_cfg:
        for i in tqdm(range(args.num_classes)):
            logger.info("Generating 50 images for class %s", i)
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device,
            )
            all_images.extend(gen_images)
    else:
        batch_size = 50
        for _ in tqdm(range(0, 5000, batch_size)):
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )
            all_images.extend(gen_images)

    val_dataset = build_val_dataset(args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False)

    ref_images = []
    for batch_idx, (imgs, _) in enumerate(val_loader):
        ref_images.append(imgs)
        if batch_idx >= 99:
            break
    ref_images = torch.cat(ref_images, dim=0)[:5000]
    ref_images = (ref_images + 1) / 2

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        logger.warning("torchmetrics not installed; skip FID/IS. pip install torchmetrics")
        return

    try:
        from torchmetrics.image.inception import InceptionScore
    except ImportError:
        logger.warning("torchmetrics.image.inception.InceptionScore not available; skip IS, FID only")
        InceptionScore = None

    gen_tensors = []
    for img in all_images:
        gen_tensors.append(transforms.ToTensor()(img))
    gen_tensors = torch.stack(gen_tensors)

    fid = FrechetInceptionDistance(feature=64, normalize=False).to(device)

    fid.update(ref_images, real=True)
    fid.update(gen_tensors, real=False)

    fid_score = fid.compute()
    logger.info("FID Score: %s", fid_score)

    if InceptionScore is not None:
        inception = InceptionScore(normalize=False).to(device)
        inception.update(gen_tensors)
        is_mean, is_std = inception.compute()
        logger.info("Inception Score: %s +/- %s", is_mean, is_std)


if __name__ == "__main__":
    main()
