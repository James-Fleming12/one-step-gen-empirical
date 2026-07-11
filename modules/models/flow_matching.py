import torch

from ..networks import MLPVectorField, SmallUNet

class FlowMatching:
    name = "flow_matching"

    def __init__(self, data_info, device="cpu", hidden=256, depth=4, sample_steps=50, **kwargs):
        self.device = device
        self.data_info = data_info
        self.sample_steps = sample_steps
        if data_info["type"] == "synthetic":
            self.net = MLPVectorField(dim=data_info["dim"], hidden=hidden, depth=depth, use_r=False)
            self.shape = (data_info["dim"],)
        else:
            self.net = SmallUNet(channels=data_info["channels"], base=32, use_r=False)
            s = data_info["image_size"]
            self.shape = (data_info["channels"], s, s)
        self.net.to(device)

    def parameters(self):
        return self.net.parameters()

    def training_step(self, x1):
        x1 = x1.to(self.device)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=self.device)
        t_ = t.view(-1, *([1] * (x1.dim() - 1)))
        xt = (1 - t_) * x0 + t_ * x1
        target = x1 - x0
        pred = self.net(xt, t)
        loss = ((pred - target) ** 2).mean()
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def sample(self, n, steps=None):
        steps = steps or self.sample_steps
        x = torch.randn(n, *self.shape, device=self.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=self.device)
            v = self.net(x, t)
            x = x + v * dt
        return x

    def nfe_for_sampling(self, steps=None):
        return steps or self.sample_steps

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)