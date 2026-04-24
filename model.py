import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):

    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(out_f))
        # one gate score per weight, sigmoid will squash it to (0,1)
        self.gate_scores = nn.Parameter(torch.zeros(out_f, in_f))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        w = self.weight * gates
        return F.linear(x, w, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, thresh=0.01):
        g = self.get_gates()
        return (g < thresh).float().mean().item()


class SelfPruningNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = PrunableLinear(3072, 512)
        self.l2 = PrunableLinear(512,  256)
        self.l3 = PrunableLinear(256,  10)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        return self.l3(x)

    def sparsity_loss(self):
        total = 0
        for layer in [self.l1, self.l2, self.l3]:
            total += torch.sigmoid(layer.gate_scores).sum()
        return total

    def overall_sparsity(self):
        vals = []
        for layer in [self.l1, self.l2, self.l3]:
            vals.append(layer.sparsity())
        return sum(vals) / len(vals)