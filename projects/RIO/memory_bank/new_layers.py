# This code is modified from https://github.com/wyharveychen/CloserLookFewShot/

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, "weight", dim=0)

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = (
                torch.norm(self.L.weight.data, p=2, dim=1)
                .unsqueeze(1)
                .expand_as(self.L.weight.data)
            )
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)

        return cos_dist
