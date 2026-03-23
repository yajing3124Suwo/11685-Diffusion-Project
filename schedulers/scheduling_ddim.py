# -*- coding: utf-8 -*-
import torch

from utils import randn_tensor

from .scheduling_ddpm import DDPMScheduler


class DDIMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super(DDIMScheduler, self).__init__(*args, **kwargs)
        assert self.num_inference_steps is not None, (
            "Please set `num_inference_steps` before running inference using DDIM."
        )
        self.set_timesteps(self.num_inference_steps)

    def _get_variance(self, t):
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        if prev_t >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        else:
            alpha_prod_t_prev = self.alphas_cumprod.new_tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def step(self, model_output, timestep, sample, generator=None, eta=0.0):
        t = int(timestep)
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        if prev_t >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        else:
            alpha_prod_t_prev = self.alphas_cumprod.new_tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t

        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
            pred_epsilon = model_output
        else:
            raise NotImplementedError(self.prediction_type)

        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)

        variance = self._get_variance(t)
        std_dev_t = eta * variance.sqrt()

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2).sqrt() * pred_epsilon

        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

        if eta > 0:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample + std_dev_t * variance_noise

        return prev_sample
