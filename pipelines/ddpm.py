# -*- coding: utf-8 -*-
from typing import List, Optional, Union

from PIL import Image
from tqdm import tqdm
import torch

from utils import randn_tensor


class DDPMPipeline:
    def __init__(self, unet, scheduler, vae=None, class_embedder=None):
        self.unet = unet
        self.scheduler = scheduler

        self.vae = None
        if vae is not None:
            self.vae = vae

        self.class_embedder = None
        if class_embedder is not None:
            self.class_embedder = class_embedder

    def numpy_to_pil(self, images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                "`self._progress_bar_config` should be of type `dict`, but is {}.".format(
                    type(self._progress_bar_config)
                )
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        if total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        classes: Optional[Union[int, List[int]]] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device=None,
    ):
        image_shape = (batch_size, self.unet.input_ch, self.unet.input_size, self.unet.input_size)
        if device is None:
            device = next(self.unet.parameters()).device

        class_embeds = None
        uncond_embeds = None

        if classes is not None or guidance_scale is not None:
            assert self.class_embedder is not None, "class_embedder is not defined"

        if classes is not None:
            if isinstance(classes, int):
                classes = [classes] * batch_size
            elif isinstance(classes, list):
                assert len(classes) == batch_size, "Length of classes must be equal to batch_size"
            classes = torch.tensor(classes, device=device, dtype=torch.long)
            uncond_classes = torch.full_like(classes, self.class_embedder.num_classes)
            class_embeds = self.class_embedder(classes)
            uncond_embeds = self.class_embedder(uncond_classes)

        image = randn_tensor(image_shape, generator=generator, device=device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for t in self.progress_bar(self.scheduler.timesteps):
            t_i = int(t) if not isinstance(t, int) else t

            use_cfg = (
                class_embeds is not None
                and guidance_scale is not None
                and abs(float(guidance_scale) - 1.0) > 1e-6
            )

            if use_cfg:
                model_input = torch.cat([image, image], dim=0)
                c = torch.cat([uncond_embeds, class_embeds], dim=0)
            else:
                model_input = image
                c = class_embeds

            b = model_input.shape[0]
            ts = torch.full((b,), t_i, device=device, dtype=torch.long)
            model_output = self.unet(model_input, ts, c)

            if guidance_scale is not None and abs(float(guidance_scale) - 1.0) > 1e-6 and class_embeds is not None:
                uncond_model_output, cond_model_output = model_output.chunk(2, dim=0)
                model_output = uncond_model_output + guidance_scale * (
                    cond_model_output - uncond_model_output
                )

            image = self.scheduler.step(model_output, t_i, image, generator=generator)

        if self.vae is not None:
            image = image / 0.1845
            image = self.vae.decode(image)
            image = image.clamp(-1, 1)

        image = (image / 2 + 0.5).clamp(0, 1)

        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)

        return image
