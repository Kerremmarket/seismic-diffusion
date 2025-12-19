#!/usr/bin/env python3
"""Conditional 1D UNet baseline for Stage 1 seismic diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Construct sinusoidal timestep embeddings."""
    half_dim = dim // 2
    exponent = torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
    exponent = exponent / max(half_dim - 1, 1)
    emb = timesteps.float().unsqueeze(1) * torch.exp(-torch.log(torch.tensor(10_000.0)) * exponent)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    """Feature-wise linear modulation."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_dim * 2),
        )

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale_shift = self.net(cond)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        return scale, shift


def make_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class ResBlock1D(nn.Module):
    """Residual block with timestep and conditional modulation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        cond_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        self.norm1 = make_group_norm(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation)

        self.time_proj = nn.Linear(time_dim, out_ch)
        self.cond_film = FiLM(cond_dim, out_ch)

        self.norm2 = make_group_norm(out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation)

        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))

        time = self.time_proj(t_emb).unsqueeze(-1)
        h = h + time

        scale, shift = self.cond_film(cond)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention block for 1D feature maps."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = make_group_norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        h = self.norm(x).permute(0, 2, 1)  # B, T, C
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        attn_out = self.proj(self.act(attn_out))
        attn_out = attn_out.permute(0, 2, 1)
        return x + attn_out


@dataclass
class UNet1DConfig:
    in_channels: int = 1
    base_channels: int = 128
    channel_multipliers: Sequence[int] = (1, 2, 4, 8)
    num_res_blocks: int = 3
    kernel_size: int = 5
    time_embedding_dim: int = 512
    cond_dim: int = 7
    station_embedding_dim: int = 0
    num_stations: int = 0
    use_attention: bool = True
    attention_heads: int = 4
    mid_dilations: Sequence[int] = (1, 2, 4)


class ConditionalUNet1D(nn.Module):
    """Conditional 1D UNet backbone for diffusion noise prediction."""

    def __init__(self, config: UNet1DConfig) -> None:
        super().__init__()
        self.cfg = config
        self.time_embed_dim = config.time_embedding_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(config.cond_dim, config.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim),
        )

        if config.num_stations > 0 and config.station_embedding_dim > 0:
            self.station_embed = nn.Embedding(
                config.num_stations,
                config.station_embedding_dim,
                padding_idx=0,
            )
            self.station_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.station_embedding_dim, config.time_embedding_dim),
            )
        else:
            self.station_embed = None
            self.station_proj = None

        base_ch = config.base_channels
        padding = (config.kernel_size - 1) // 2
        self.input_conv = nn.Conv1d(config.in_channels, base_ch, kernel_size=config.kernel_size, padding=padding)

        dims = [base_ch * m for m in config.channel_multipliers]
        self.downs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        prev_ch = base_ch

        for ch in dims:
            blocks = nn.ModuleList(
                [
                    ResBlock1D(
                        prev_ch if i == 0 else ch,
                        ch,
                        config.time_embedding_dim,
                        config.time_embedding_dim,
                        kernel_size=config.kernel_size,
                    )
                    for i in range(config.num_res_blocks)
                ]
            )
            self.downs.append(blocks)
            prev_ch = ch
            self.down_samples.append(nn.Conv1d(prev_ch, prev_ch, kernel_size=4, stride=2, padding=1))

        mid_blocks = []
        for dilation in config.mid_dilations:
            mid_blocks.append(
                ResBlock1D(
                    prev_ch,
                    prev_ch,
                    config.time_embedding_dim,
                    config.time_embedding_dim,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                )
            )
            if config.use_attention:
                mid_blocks.append(SelfAttention1D(prev_ch, num_heads=config.attention_heads))
        self.mid_blocks = nn.ModuleList(mid_blocks)

        self.ups = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for ch in reversed(dims):
            first_in_ch = ch * 2
            blocks = nn.ModuleList()
            blocks.append(
                ResBlock1D(
                    first_in_ch,
                    ch,
                    config.time_embedding_dim,
                    config.time_embedding_dim,
                    kernel_size=config.kernel_size,
                )
            )
            for _ in range(1, config.num_res_blocks):
                blocks.append(
                    ResBlock1D(
                        ch,
                        ch,
                        config.time_embedding_dim,
                        config.time_embedding_dim,
                        kernel_size=config.kernel_size,
                    )
                )
            self.ups.append(blocks)
            self.up_samples.append(
                nn.ConvTranspose1d(prev_ch, ch, kernel_size=4, stride=2, padding=1)
            )
            prev_ch = ch

        self.output_norm = make_group_norm(base_ch)
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv1d(base_ch, config.in_channels, kernel_size=config.kernel_size, padding=padding)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
        station_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise given noisy input, timestep, and condition."""
        t_emb = self.time_mlp(sinusoidal_time_embedding(timesteps, self.cfg.time_embedding_dim))
        c_emb = self.cond_mlp(cond)

        if self.station_embed is not None:
            if station_ids is None:
                station_ids = torch.zeros(cond.shape[0], dtype=torch.long, device=cond.device)
            station_emb = self.station_embed(station_ids)
            station_emb = self.station_proj(station_emb)
            c_emb = c_emb + station_emb

        joint_emb = t_emb + c_emb

        h = self.input_conv(x)
        skips: list[torch.Tensor] = []

        for blocks, down in zip(self.downs, self.down_samples):
            for block in blocks:
                h = block(h, joint_emb, joint_emb)
            skips.append(h)
            h = down(h)

        for block in self.mid_blocks:
            if isinstance(block, SelfAttention1D):
                h = block(h)
            else:
                h = block(h, joint_emb, joint_emb)

        for blocks, up in zip(self.ups, self.up_samples):
            h = up(h)
            skip = skips.pop()
            if skip.shape[-1] != h.shape[-1]:
                skip = F.interpolate(skip, size=h.shape[-1], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, joint_emb, joint_emb)

        h = self.output_conv(self.output_act(self.output_norm(h)))
        return h


