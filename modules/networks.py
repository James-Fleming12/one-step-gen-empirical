import math

import torch
import torch.nn as nn

def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Standard sinusoidal embedding for a batch of scalars in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * 1000.0 * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class MLPVectorField(nn.Module):
    def __init__(self, dim: int = 2, hidden: int = 256, depth: int = 4, time_dim: int = 64, use_r: bool = False):
        super().__init__()
        self.use_r = use_r
        self.time_dim = time_dim
        in_dim = dim + time_dim + (time_dim if use_r else 0)
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, r: torch.Tensor = None) -> torch.Tensor:
        temb = timestep_embedding(t, self.time_dim)
        feats = [x, temb]
        if self.use_r:
            remb = timestep_embedding(r if r is not None else t, self.time_dim)
            feats.append(remb)
        h = torch.cat(feats, dim=-1)
        return self.net(h)

class FiLMConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.conv(x)))
        scale, shift = self.emb_proj(emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return h

class SmallUNet(nn.Module):
    def __init__(self, channels: int = 1, base: int = 32, time_dim: int = 128, use_r: bool = False):
        super().__init__()
        self.use_r = use_r
        self.time_dim = time_dim
        emb_in = time_dim * (2 if use_r else 1)
        emb_dim = base * 4
        self.emb_mlp = nn.Sequential(nn.Linear(emb_in, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.in_conv = nn.Conv2d(channels, base, 3, padding=1)
        self.down1 = FiLMConvBlock(base, base, emb_dim)
        self.pool1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.down2 = FiLMConvBlock(base * 2, base * 2, emb_dim)
        self.pool2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.mid = FiLMConvBlock(base * 4, base * 4, emb_dim)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = FiLMConvBlock(base * 4, base * 2, emb_dim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = FiLMConvBlock(base * 2, base, emb_dim)
        self.out_conv = nn.Conv2d(base, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, r: torch.Tensor = None) -> torch.Tensor:
        temb = timestep_embedding(t, self.time_dim)
        if self.use_r:
            remb = timestep_embedding(r if r is not None else t, self.time_dim)
            emb = torch.cat([temb, remb], dim=-1)
        else:
            emb = temb
        emb = self.emb_mlp(emb)

        h0 = self.in_conv(x)
        h1 = self.down1(h0, emb)
        h1p = self.pool1(h1)
        h2 = self.down2(h1p, emb)
        h2p = self.pool2(h2)
        hm = self.mid(h2p, emb)
        u2 = self.up2(hm)
        d2 = self.dec2(torch.cat([u2, h2], dim=1), emb)
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, h1], dim=1), emb)
        return self.out_conv(d1)