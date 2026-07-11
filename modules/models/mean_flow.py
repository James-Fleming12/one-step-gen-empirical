import torch

from ..networks import MLPVectorField, SmallUNet

class MeanFlow:
    name = "mean_flow"

    def __init__(self, data_info, device="cpu", hidden=256, depth=4, ratio_same_rt=0.25,
                 adaptive_p=1.0, **kwargs):
        self.device = device
        self.data_info = data_info
        self.ratio_same_rt = ratio_same_rt
        self.adaptive_p = adaptive_p
        if data_info["type"] == "synthetic":
            self.net = MLPVectorField(dim=data_info["dim"], hidden=hidden, depth=depth, use_r=True)
            self.shape = (data_info["dim"],)
        else:
            self.net = SmallUNet(channels=data_info["channels"], base=32, use_r=True)
            s = data_info["image_size"]
            self.shape = (data_info["channels"], s, s)
        self.net.to(device)

    def parameters(self):
        return self.net.parameters()

    def _sample_rt(self, batch_size):
        t = torch.rand(batch_size, device=self.device)
        r = torch.rand(batch_size, device=self.device) * t

        same = torch.rand(batch_size, device=self.device) < self.ratio_same_rt
        r = torch.where(same, t, r)
        return r, t

    def training_step(self, x1):
        x1 = x1.to(self.device)
        bsz = x1.shape[0]
        x0 = torch.randn_like(x1)
        r, t = self._sample_rt(bsz)
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))
        z = (1 - t_) * x0 + t_ * x1
        v = x1 - x0

        def u_fn(z_, r_, t_scalar):
            return self.net(z_, t_scalar, r_)

        zeros_r = torch.zeros_like(r)
        ones_t = torch.ones_like(t)
        u_pred, dudt = torch.func.jvp(u_fn, (z, r, t), (v, zeros_r, ones_t))

        delta = (t - r).view(-1, *([1] * (x1.dim() - 1)))
        u_target = (v - delta * dudt).detach()

        diff = u_pred - u_target
        sq_err = diff.pow(2).flatten(1).mean(1)

        weight = 1.0 / (sq_err.detach() + 1e-3).pow(self.adaptive_p)
        loss = (weight * sq_err).mean()
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
            u = self.net(x, t, r)
            x = x + (bounds[i + 1] - bounds[i]) * u
        return x

    def nfe_for_sampling(self, steps=1):
        return steps

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)
