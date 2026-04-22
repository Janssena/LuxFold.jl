py"""
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm
from functools import partial

LinearNoBias = partial(Linear, bias=False)

class Boltz2AdaLN(nn.Module):
    def __init__(self, dim, dim_single_cond):
        super().__init__()
        self.a_norm = LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = LayerNorm(dim_single_cond, bias=False)
        self.s_scale = Linear(dim_single_cond, dim)
        self.s_bias = LinearNoBias(dim_single_cond, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        a = self.sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a
"""
