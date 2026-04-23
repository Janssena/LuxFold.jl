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

class Boltz2OuterProductMean(nn.Module):
    def __init__(self, c_in, c_hidden, c_out):
        super().__init__()
        self.c_hidden = c_hidden
        self.norm = nn.LayerNorm(c_in)
        self.proj_a = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out)

    def forward(self, m, mask, chunk_size=None):
        mask = mask.unsqueeze(-1).to(m)
        m = self.norm(m)
        a = self.proj_a(m) * mask
        b = self.proj_b(m) * mask

        mask_expanded = mask[:, :, None, :] * mask[:, :, :, None]
        num_mask = mask_expanded.sum(1).clamp(min=1)
        z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())
        z = z.reshape(*z.shape[:3], -1)
        z = z / num_mask

        z = self.proj_o(z.to(m))
        return z
"""
