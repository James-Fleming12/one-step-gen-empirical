import copy
import torch
from ..networks import MLPVectorField, SmallUNet

class DriftModel:
    name = "drifting_model"
    def __init__(self, data_info, device="cpu", hidden=256, depth=4, ema_decay=0.999, temperatures=(0.02, 0.05, 0.2), **kwargs):
        self.device = device
        self.data_info = data_info
        self.ema_decay = ema_decay
        self.temperatures = temperatures
        if data_info["type"] == "synthetic":
            self.net = MLPVectorField(dim=data_info["dim"], hidden=hidden, depth=depth, use_r=False)
            self.shape = (data_info["dim"],)
        else:
            self.net = SmallUNet(channels=data_info["channels"], base=32, use_r=False)
            s = data_info["image_size"]
            self.shape = (data_info["channels"], s, s)
        self.net.to(device)
        self.ema_net = copy.deepcopy(self.net)
        for p in self.ema_net.parameters():
            p.requires_grad_(False)

    def parameters(self):
        return self.net.parameters()

    @torch.no_grad()
    def _update_ema(self):
        for p_ema, p in zip(self.ema_net.parameters(), self.net.parameters()):
            p_ema.mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)

    def _compute_V(self, x, y_pos, y_neg, T):
        # x: [N, D]
        # y_pos: [N_pos, D]
        # y_neg: [N_neg, D]
        N = x.shape[0]
        N_pos = y_pos.shape[0]
        N_neg = y_neg.shape[0]
        
        # compute pairwise distance
        dist_pos = torch.cdist(x, y_pos) # [N, N_pos]
        dist_neg = torch.cdist(x, y_neg) # [N, N_neg]
        
        # ignore self (if y_neg is x)
        if x is y_neg:
            dist_neg += torch.eye(N, device=x.device) * 1e6
            
        # compute logits
        logit_pos = -dist_pos / T
        logit_neg = -dist_neg / T
        
        # concat for normalization
        logit = torch.cat([logit_pos, logit_neg], dim=1) # [N, N_pos + N_neg]
        
        # normalize along both dimensions
        A_row = torch.nn.functional.softmax(logit, dim=-1)
        A_col = torch.nn.functional.softmax(logit, dim=-2)
        A = torch.sqrt(A_row * A_col)
        
        # back to [N, N_pos] and [N, N_neg]
        A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=1)
        
        # compute the weights
        sum_neg = A_neg.sum(dim=1, keepdim=True)
        sum_pos = A_pos.sum(dim=1, keepdim=True)
        
        W_pos = A_pos * sum_neg
        W_neg = A_neg * sum_pos
        
        drift_pos = W_pos @ y_pos
        drift_neg = W_neg @ y_neg
        
        V = drift_pos - drift_neg
        return V

    def training_step(self, x1):
        x1 = x1.to(self.device)
        y_pos = x1.flatten(1) # [N_pos, D]
        
        bsz = x1.shape[0]
        e = torch.randn(bsz, *self.shape, device=self.device)
        t0 = torch.zeros(bsz, device=self.device)
        
        x = self.net(e, t0)
        x_flat = x.flatten(1)
        y_neg = x_flat # reuse x as negatives
        
        with torch.no_grad():
            V_total = 0
            for T in self.temperatures:
                V_total = V_total + self._compute_V(x_flat, y_pos, y_neg, T)
        
        x_drifted = (x_flat + V_total).detach()
        
        loss = torch.nn.functional.mse_loss(x_flat, x_drifted)
        
        self._update_ema()
        return loss, {
            "loss": loss.item()
        }

    @torch.no_grad()
    def sample(self, n, steps=1):
        x = torch.randn(n, *self.shape, device=self.device)
        t0 = torch.zeros(n, device=self.device)
        
        # Drifting Models natively perform one-step generation mapping noise directly to data
        x = self.ema_net(x, t0)
        
        return x

    def nfe_for_sampling(self, steps=1):
        return 1

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)