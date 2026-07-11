""""Improved" MeanFlow based on the Improved Mean Flows paper.
"""
import copy

import torch

from .mean_flow import MeanFlow

class ImprovedMeanFlow(MeanFlow):
    name = "improved_mean_flow"

    def __init__(self, data_info, device="cpu", hidden=256, depth=4, ratio_same_rt=0.5,
                 logit_normal_mu=-0.4, logit_normal_sigma=1.0, ema_decay=0.9999, **kwargs):
        super().__init__(data_info, device=device, hidden=hidden, depth=depth, ratio_same_rt=ratio_same_rt, **kwargs)
        self.logit_normal_mu = logit_normal_mu
        self.logit_normal_sigma = logit_normal_sigma
        self.ema_decay = ema_decay
        self.ema_net = copy.deepcopy(self.net)
        for p in self.ema_net.parameters():
            p.requires_grad_(False)

    def _sample_rt(self, batch_size):
        normal = torch.randn(batch_size, device=self.device) * self.logit_normal_sigma + self.logit_normal_mu
        t = torch.sigmoid(normal)
        r = torch.rand(batch_size, device=self.device) * t
        same = torch.rand(batch_size, device=self.device) < self.ratio_same_rt
        r = torch.where(same, t, r)
        return r, t

    @torch.no_grad()
    def _update_ema(self):
        for p_ema, p in zip(self.ema_net.parameters(), self.net.parameters()):
            p_ema.mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)

    def training_step(self, x1):
        x1 = x1.to(self.device)
        bsz = x1.shape[0]
        x0 = torch.randn_like(x1)
        r, t = self._sample_rt(bsz)
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))
        z = (1 - t_) * x0 + t_ * x1
        
        # Ground-truth marginal velocity
        v = x1 - x0

        # Boundary condition of u_theta: v_theta = u_theta(z_t, t, t)
        with torch.no_grad():
            v_theta = self.net(z, t, t)

        def u_fn(z_, r_, t_scalar):
            return self.net(z_, t_scalar, r_)

        zeros_r = torch.zeros_like(r)
        ones_t = torch.ones_like(t)
        
        # Predict u and dudt using the live network and v_theta as tangent
        u_pred, dudt = torch.func.jvp(u_fn, (z, r, t), (v_theta, zeros_r, ones_t))

        delta = (t - r).view(-1, *([1] * (x1.dim() - 1)))
        
        # Compound function V = u + (t - r) * stopgrad(dudt)
        V = u_pred + delta * dudt.detach()
        
        # Regression error against ground truth v
        diff = V - v
        
        # Adaptive weighting L2 loss as in original MF
        sq_err = diff.pow(2).flatten(1).mean(1)
        weight = 1.0 / (sq_err.detach() + 1e-3).pow(self.adaptive_p)
        loss = (weight * sq_err).mean()

        self._update_ema()
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def sample(self, n, steps=1):
        """steps=1 is the canonical single-NFE MeanFlow sampler. steps>1 is
        supported for comparison by chaining several average-velocity jumps
        over sub-intervals of [0, 1]."""
        x = torch.randn(n, *self.shape, device=self.device)
        bounds = torch.linspace(0, 1, steps + 1, device=self.device)
        for i in range(steps):
            r = torch.full((n,), bounds[i].item(), device=self.device)
            t = torch.full((n,), bounds[i + 1].item(), device=self.device)
            # Use EMA network for sampling
            u = self.ema_net(x, t, r)
            x = x + (bounds[i + 1] - bounds[i]) * u
        return x
