# -*- coding: utf-8 -*-
import os
import argparse
import logging
from logging import getLogger as get_logger

import torch
import torch.nn.functional as F
import ruamel.yaml as yaml
from tqdm import tqdm
from PIL import Image

from torchvision import datasets, transforms

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint
from ddpm_runtime import apply_runtime_to_args

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM.")

    parser.add_argument("--config", type=str, default="configs/ddpm.yaml", help="YAML config")

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "imagefolder"],
        help="cifar10: torchvision CIFAR-10 ([-1,1] normalized). imagefolder: ImageFolder layout.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root folder: CIFAR-10 is downloaded under data/cifar10; ImageFolder uses data_dir/train.",
    )
    parser.add_argument("--image_size", type=int, default=32, help="Spatial size (CIFAR-10 is 32).")
    parser.add_argument("--batch_size", type=int, default=64, help="Per-device batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of dataset classes (CIFAR-10: 10)")

    parser.add_argument("--run_name", type=str, default=None, help="Optional run name suffix")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Experiment root")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="none", choices=["fp16", "bf16", "fp32", "none"])

    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=200)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--variance_type", type=str, default="fixed_small")
    parser.add_argument("--prediction_type", type=str, default="epsilon")
    parser.add_argument("--clip_sample", type=str2bool, default=True)
    parser.add_argument("--clip_sample_range", type=float, default=1.0)

    parser.add_argument("--unet_in_size", type=int, default=32)
    parser.add_argument("--unet_in_ch", type=int, default=3)
    parser.add_argument("--unet_ch", type=int, default=128)
    parser.add_argument("--unet_ch_mult", type=int, nargs="+", default=[1, 2, 2, 2])
    parser.add_argument("--unet_attn", type=int, nargs="*", default=[], help="Attention at these level indices; empty disables.")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2)
    parser.add_argument("--unet_dropout", type=float, default=0.1)

    parser.add_argument("--latent_ddpm", type=str2bool, default=False)
    parser.add_argument("--use_cfg", type=str2bool, default=False)
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.0)
    parser.add_argument("--use_ddim", type=str2bool, default=False)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="Set automatically for distributed")
    parser.add_argument(
        "--cifar_download",
        type=str2bool,
        default=True,
        help="If false, use CIFAR files already under data/cifar10 (no HTTPS download).",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default="auto",
        choices=["auto", "local", "colab", "psc"],
        help="Training profile: auto (detect Colab), local, colab, or psc. "
        "Override with env DDPM_RUNTIME=colab|psc|local.",
    )

    args = parser.parse_args()

    if args.config is not None and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            if config_args:
                parser.set_defaults(**config_args)
        args = parser.parse_args()

    return args


def build_dataset(args):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    if args.dataset == "cifar10":
        root = os.path.join(args.data_dir, "cifar10")
        train_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=args.cifar_download,
            transform=transform,
        )
    else:
        train_dir = os.path.join(args.data_dir, "train")
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    return train_dataset


def main():
    args = parse_args()
    apply_runtime_to_args(args)

    seed_everything(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("DDPM runtime profile: %s", getattr(args, "_ddpm_runtime", "local"))

    device = init_distributed_device(args)
    if args.distributed:
        logger.info(
            "Distributed: rank %s / %s on %s",
            args.rank,
            args.world_size,
            args.device,
        )
    else:
        logger.info("Single process on %s", args.device)
    assert args.rank >= 0

    logger.info("Creating dataset (%s)", args.dataset)
    train_dataset = build_dataset(args)

    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    shuffle = sampler is None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    total_batch_size = args.batch_size * args.world_size
    args.total_batch_size = total_batch_size

    os.makedirs(args.output_dir, exist_ok=True)
    if args.run_name is None:
        args.run_name = "exp-{}".format(len(os.listdir(args.output_dir)))
    else:
        args.run_name = "exp-{}-{}".format(len(os.listdir(args.output_dir)), args.run_name)
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, "checkpoints")
    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    num_update_steps_per_epoch = len(train_loader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

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
    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %.2fM", n_params / 1.0e6)

    noise_scheduler = DDPMScheduler(
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

    vae = None
    if args.latent_ddpm:
        vae = VAE()
        vae.init_from_ckpt("pretrained/model.ckpt")
        vae.eval()

    class_embedder = None
    if args.use_cfg:
        class_embedder = ClassEmbedder(args.unet_ch, n_classes=args.num_classes)

    unet = unet.to(device)
    noise_scheduler = noise_scheduler.to(device)
    if vae is not None:
        vae = vae.to(device)
    if class_embedder is not None:
        class_embedder = class_embedder.to(device)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.max_train_steps, 1), eta_min=1e-6)

    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )
        unet_wo_ddp = unet.module
        if class_embedder is not None:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False,
            )
            class_embedder_wo_ddp = class_embedder.module
        else:
            class_embedder_wo_ddp = None
    else:
        unet_wo_ddp = unet
        class_embedder_wo_ddp = class_embedder

    vae_wo_ddp = vae

    if args.use_ddim:
        infer_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            num_inference_steps=args.num_inference_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            prediction_type=args.prediction_type,
            clip_sample=args.clip_sample,
            clip_sample_range=args.clip_sample_range,
        ).to(device)
    else:
        infer_scheduler = noise_scheduler

    pipeline = DDPMPipeline(
        unet=unet_wo_ddp,
        scheduler=infer_scheduler,
        vae=vae_wo_ddp,
        class_embedder=class_embedder_wo_ddp,
    )

    if is_primary(args):
        file_yaml = yaml.YAML()
        with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            file_yaml.dump(vars(args), f)

    wandb_logger = None
    if is_primary(args) and args.use_wandb:
        import wandb

        wandb_logger = wandb.init(project="ddpm", name=args.run_name, config=vars(args))

    def wb_log(d):
        if wandb_logger is not None:
            wandb_logger.log(d)

    if is_primary(args):
        logger.info("***** Training *****")
        logger.info("  Examples = %s", len(train_dataset))
        logger.info("  Epochs = %s", args.num_epochs)
        logger.info("  Batch/device = %s", args.batch_size)
        logger.info("  Global batch = %s", total_batch_size)
        logger.info("  Steps/epoch = %s", num_update_steps_per_epoch)
        logger.info("  Total steps = %s", args.max_train_steps)

    progress_bar = tqdm(range(args.max_train_steps), disable=not is_primary(args))

    for epoch in range(args.num_epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        args.epoch = epoch
        if is_primary(args):
            logger.info("Epoch %s / %s", epoch + 1, args.num_epochs)

        loss_m = AverageMeter()
        unet.train()
        noise_scheduler.train()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if vae is not None:
                with torch.no_grad():
                    images = vae.encode(images)
                images = images * 0.1845

            optimizer.zero_grad()

            if class_embedder is not None:
                class_emb = class_embedder(labels)
            else:
                class_emb = None

            noise = torch.randn_like(images)
            timesteps = torch.randint(0, args.num_train_timesteps, (images.shape[0],), device=device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            model_pred = unet(noisy_images, timesteps, class_emb)

            if args.prediction_type == "epsilon":
                target = noise
            else:
                raise NotImplementedError(args.prediction_type)

            loss = F.mse_loss(model_pred, target)
            loss_m.update(loss.item())

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)

            if step % 100 == 0 and is_primary(args):
                logger.info(
                    "Epoch %s step %s / %s  loss=%.5f avg=%.5f",
                    epoch + 1,
                    step,
                    num_update_steps_per_epoch,
                    loss.item(),
                    loss_m.avg,
                )
                wb_log({"loss": loss_m.avg})

        unet.eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(epoch + args.seed)

        with torch.no_grad():
            if args.use_cfg:
                classes = torch.randint(0, args.num_classes, (4,), device=device)
                gen_images = pipeline(
                    batch_size=4,
                    num_inference_steps=args.num_inference_steps,
                    classes=classes,
                    guidance_scale=args.cfg_guidance_scale,
                    generator=generator,
                    device=device,
                )
            else:
                gen_images = pipeline(
                    batch_size=4,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    device=device,
                )

        grid_image = Image.new("RGB", (4 * args.image_size, args.image_size))
        for i, image in enumerate(gen_images):
            x = (i % 4) * args.image_size
            grid_image.paste(image, (x, 0))
        if is_primary(args):
            if wandb_logger is not None:
                import wandb

                wandb_logger.log({"gen_images": wandb.Image(grid_image)})
            else:
                grid_path = os.path.join(output_dir, "sample_epoch_{:04d}.png".format(epoch))
                grid_image.save(grid_path)

        if is_primary(args):
            save_checkpoint(
                unet_wo_ddp,
                noise_scheduler,
                vae_wo_ddp,
                class_embedder_wo_ddp,
                optimizer,
                epoch,
                save_dir=save_dir,
            )


if __name__ == "__main__":
    main()
