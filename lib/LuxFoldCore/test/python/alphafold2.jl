py"""
import math
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm
from typing import List, Optional, Tuple
from functools import partialmethod

class AF2OuterProductMean(nn.Module):
    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c_hidden)
        self.linear_2 = nn.Linear(c_m, c_hidden)
        self.linear_out = nn.Linear(c_hidden ** 2, c_z)

    def forward(self, m, mask=None):
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        ln = self.layer_norm(m)
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        return outer / norm
"""
