# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super(AttnBlock, self).__init__()
        ng = min(32, ch)
        self.norm = nn.GroupNorm(ng, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        h_ = self.norm(x)
        q = self.q(h_).view(n, c, h * w).permute(0, 2, 1)
        k = self.k(h_).view(n, c, h * w)
        v = self.v(h_).view(n, c, h * w).permute(0, 2, 1)
        scale = 1.0 / math.sqrt(float(c))
        att = torch.bmm(q, k) * scale
        att = F.softmax(att, dim=-1)
        out = torch.bmm(att, v).permute(0, 2, 1).view(n, c, h, w)
        return x + self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout, cond_dim=None):
        super(ResBlock, self).__init__()
        ng1 = min(32, in_ch)
        ng2 = min(32, out_ch)
        self.conv1 = nn.Sequential(
            nn.GroupNorm(ng1, in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(ng2, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.cond_mlp = None
        if cond_dim is not None:
            self.cond_mlp = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb=None):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        if self.cond_mlp is not None and c_emb is not None:
            h = h + self.cond_mlp(c_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """DDPM U-Net; skip layout matches OpenAI-style DDPM (Ho et al.)."""

    def __init__(
        self,
        input_size,
        input_ch,
        T,
        ch=128,
        ch_mult=(1, 2, 2, 2),
        attn=(1, 2, 3),
        num_res_blocks=2,
        dropout=0.0,
        conditional=False,
        c_dim=128,
    ):
        super(UNet, self).__init__()
        self.input_size = input_size
        self.input_ch = input_ch
        if isinstance(ch_mult, list):
            ch_mult = tuple(ch_mult)
        attn_levels = set(attn) if not isinstance(attn, set) else attn

        time_dim = ch * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(ch),
            nn.Linear(ch, time_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )

        block_ch = [ch * m for m in ch_mult]
        in0 = block_ch[0]
        self.input_conv = nn.Conv2d(input_ch, in0, 3, padding=1)

        skip_channels = [in0]
        h_ch = in0
        self.down = nn.ModuleList()
        for level, mult in enumerate(ch_mult):
            out_c = block_ch[level]
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(h_ch, out_c, time_dim, dropout, c_dim if conditional else None))
                h_ch = out_c
                skip_channels.append(h_ch)
            if level in attn_levels:
                self.down.append(AttnBlock(h_ch))
                skip_channels.append(h_ch)
            if level != len(ch_mult) - 1:
                self.down.append(Downsample(h_ch))
                skip_channels.append(h_ch)

        self.mid = nn.ModuleList(
            [
                ResBlock(h_ch, h_ch, time_dim, dropout, c_dim if conditional else None),
                AttnBlock(h_ch),
                ResBlock(h_ch, h_ch, time_dim, dropout, c_dim if conditional else None),
            ]
        )

        pop_channels = list(reversed(skip_channels))
        self.up = nn.ModuleList()
        h_ch = block_ch[-1]
        for level in reversed(range(len(ch_mult))):
            out_c = block_ch[level]
            for _ in range(num_res_blocks + 1):
                sc = pop_channels.pop(0)
                self.up.append(ResBlock(h_ch + sc, out_c, time_dim, dropout, c_dim if conditional else None))
                h_ch = out_c
            if level in attn_levels:
                self.up.append(AttnBlock(h_ch))
            if level > 0:
                self.up.append(Upsample(h_ch))

        assert len(pop_channels) == 0, "skip / up mismatch: {} left".format(len(pop_channels))

        self.out_norm = nn.GroupNorm(min(32, h_ch), h_ch)
        self.out_conv = nn.Conv2d(h_ch, input_ch, 3, padding=1)

    def forward(self, x, timesteps, c=None):
        t_emb = self.time_embed(timesteps)
        h = self.input_conv(x)
        skips = [h]

        for layer in self.down:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb, c)
            elif isinstance(layer, AttnBlock):
                h = layer(h)
            else:
                h = layer(h)
            skips.append(h)

        for layer in self.mid:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb, c)
            else:
                h = layer(h)

        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = layer(h, t_emb, c)
            elif isinstance(layer, AttnBlock):
                h = layer(h)
            else:
                h = layer(h)

        h = self.out_norm(h)
        h = F.relu(h, inplace=True)
        return self.out_conv(h)
