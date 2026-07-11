import time

import numpy as np
import torch
import torch.nn as nn

def rbf_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = None) -> float:
    x = x.reshape(x.shape[0], -1).float()
    y = y.reshape(y.shape[0], -1).float()
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2
    if sigma is None:
        sigma = torch.median(xy).item() + 1e-8
    kxx = torch.exp(-xx / (2 * sigma))
    kyy = torch.exp(-yy / (2 * sigma))
    kxy = torch.exp(-xy / (2 * sigma))
    return float((kxx.mean() + kyy.mean() - 2 * kxy.mean()).item())

def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_proj: int = 128, seed: int = 0) -> float:
    x = x.reshape(x.shape[0], -1).cpu().numpy()
    y = y.reshape(y.shape[0], -1).cpu().numpy()
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    proj = rng.normal(size=(n_proj, dim))
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    xp = np.sort(x @ proj.T, axis=0)
    yp = np.sort(y @ proj.T, axis=0)
    m = min(len(xp), len(yp))
    return float(np.mean(np.abs(xp[:m] - yp[:m])))

class RandomFeatureEncoder(nn.Module):
    """Fixed-weight small CNN used to compute an FID-like Frechet distance
    without needing a pretrained Inception network."""

    def __init__(self, channels: int = 1, out_dim: int = 128, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 16, out_dim),
        )
        with torch.no_grad():
            for p in self.net.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=0.05, generator=g)
                else:
                    nn.init.zeros_(p)
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        return self.net(x)

def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.T) / 2)
    vals = np.clip(vals, 0, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

def frechet_distance(feat_x: torch.Tensor, feat_y: torch.Tensor) -> float:
    feat_x = feat_x.cpu().numpy()
    feat_y = feat_y.cpu().numpy()
    mu_x, mu_y = feat_x.mean(0), feat_y.mean(0)
    cov_x = np.cov(feat_x, rowvar=False)
    cov_y = np.cov(feat_y, rowvar=False)
    diff = mu_x - mu_y
    covmean = _sqrtm_psd(cov_x @ cov_y)
    fd = diff @ diff + np.trace(cov_x + cov_y - 2 * covmean)
    return float(fd.real if np.iscomplexobj(fd) else fd)

def time_sample_fn(sample_fn, n: int, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.time()
        sample_fn(n)
        times.append(time.time() - t0)
    return float(np.mean(times))

def save_scatter(gen: torch.Tensor, ref: torch.Tensor, path: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = gen.detach().cpu().numpy()
    ref = ref.detach().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.scatter(ref[:, 0], ref[:, 1], s=4, alpha=0.4, label="real")
    plt.scatter(gen[:, 0], gen[:, 1], s=4, alpha=0.4, label="generated")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8):
    import torchvision.utils as vutils

    images = (images.detach().clamp(-1, 1) + 1) / 2
    vutils.save_image(images, path, nrow=nrow)