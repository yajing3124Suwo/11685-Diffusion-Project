# -*- coding: utf-8 -*-
"""Stub VAE for latent DDPM; load real weights when latent_ddpm is True."""
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

    def init_from_ckpt(self, path):
        raise NotImplementedError(
            "Download pretrained/model.ckpt and implement VAE.encode/decode for latent DDPM."
        )

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError
