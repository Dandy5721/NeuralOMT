import torch
import torch.nn as nn

class AD_MAE(nn.Module):

    def __init__(self,
                p = 0.5,
                max_= 1.0,
                 bins = 10):
        super(AD_MAE, self).__init__()
        self.p = p
        self.max_ = max_
        self.bins = bins
        self.edges = torch.arange(bins + 1).float() / bins
        self.edges[-1] = 1e3

    # TODO: support reduction parameter
    def forward(self,
                pred,
                target):

        edges = self.edges
        loss = torch.abs(pred - target)

        # gradient length
        g = (torch.abs(pred - target) ** self.p/ self.max_).detach()
        weights = torch.zeros_like(g)

        tot = pred.shape[0]
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n
        res = loss * (weights * torch.abs(pred - target) ** 0.5).detach()
        return res.mean()