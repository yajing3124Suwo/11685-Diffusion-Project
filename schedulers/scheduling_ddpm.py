# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from utils import randn_tensor


class DDPMScheduler(nn.Module):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: Optional[int] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        variance_type: str = "fixed_small",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        super(DDPMScheduler, self).__init__()

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        if self.beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        else:
            raise NotImplementedError(self.beta_schedule)
        self.register_buffer("betas", betas)

        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self.register_buffer("timesteps", torch.from_numpy(timesteps.copy()).long())

    def set_timesteps(self, num_inference_steps: int = 250, device: Union[str, torch.device] = None):
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                "num_inference_steps cannot be larger than num_train_timesteps ({})".format(
                    self.num_train_timesteps
                )
            )
        stride = max(1, self.num_train_timesteps // num_inference_steps)
        ts = np.arange(self.num_train_timesteps - 1, -1, -stride, dtype=np.int64)
        if ts[-1] != 0:
            ts = np.append(ts, 0)
        new_ts = torch.from_numpy(ts).long()
        if device is not None:
            new_ts = new_ts.to(device)
        self.register_buffer("timesteps", new_ts)
        self.num_inference_steps = num_inference_steps

    def __len__(self):
        return self.num_train_timesteps

    def previous_timestep(self, timestep):
        num_inference_steps = self.num_inference_steps if self.num_inference_steps else self.num_train_timesteps
        stride = max(1, self.num_train_timesteps // num_inference_steps)
        prev_t = int(timestep) - stride
        return prev_t

    def _get_variance(self, t):
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        if prev_t >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        else:
            alpha_prod_t_prev = self.alphas_cumprod.new_tensor(1.0)
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        if self.variance_type == "fixed_small":
            pass
        elif self.variance_type == "fixed_large":
            variance = current_beta_t
        else:
            raise NotImplementedError(self.variance_type)

        return variance

    def add_noise(self, original_samples, noise, timesteps):
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype, device=original_samples.device)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def step(self, model_output, timestep, sample, generator=None):
        t = int(timestep)
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        if prev_t >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        else:
            alpha_prod_t_prev = torch.tensor(1.0, device=sample.device, dtype=sample.dtype)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        else:
            raise NotImplementedError(self.prediction_type)

        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)

        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t ** 0.5 * beta_prod_t_prev) / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        variance = 0
        if t > 0:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            variance = (self._get_variance(t) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
