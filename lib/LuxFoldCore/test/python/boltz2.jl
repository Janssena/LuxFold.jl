py"""
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm
from typing import Optional
from torch import Tensor
from einops.layers.torch import Rearrange
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


class Boltz2PairWeightedAveraging(nn.Module):
    def __init__(self, c_m, c_z, c_h, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.c_h = c_h

        self.m_norm = LayerNorm(c_m)
        self.z_norm = LayerNorm(c_z)
        self.z_proj = LinearNoBias(c_z, num_heads)
        self.v_proj = LinearNoBias(c_m, c_h * num_heads)
        self.g_proj = LinearNoBias(c_m, c_h * num_heads)
        self.o_proj = LinearNoBias(c_h * num_heads, c_m)

        self.sigmoid = nn.Sigmoid()

    def forward(self, m, z, mask=None, chunk_heads=None):
        m = self.m_norm(m)
        z = self.z_norm(z)

        v = self.v_proj(m)
        v = v.view(*v.shape[:-1], self.num_heads, self.c_h)
        v = v.permute(0, 3, 1, 2, 4) # [B, H, S, N, C_h] (b h s j d)

        w = self.z_proj(z)
        w = w.permute(0, 3, 1, 2)

        if mask is not None:
            mask = mask.to(m)
            w = w + (1e9 * (mask[:, None, :, :] - 1.0))

        w = torch.softmax(w.float(), dim=-1).to(m)

        o = torch.einsum("bhij,bhsjd->bhsid", w, v) # [B, H, S, N, C_h]

        g = self.sigmoid(self.g_proj(m))
        g = g.view(*g.shape[:-1], self.num_heads, self.c_h) # [B, S, N, H, C_h]

        o = o.permute(0, 2, 3, 1, 4) # [B, S, N, H, C_h]
        o = o * g
        o = o.reshape(*o.shape[:-2], -1)
        o = self.o_proj(o)

        return o


class Boltz2AttentionPairBias(nn.Module):
    # Attention pair bias layer.

    def __init__(
        self,
        c_s: int,
        c_z: Optional[int] = None,
        num_heads: Optional[int] = None,
        inf: float = 1e6,
        compute_pair_bias: bool = True,
    ) -> None:
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.proj_q = nn.Linear(c_s, c_s, bias=False)

        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.compute_pair_bias = compute_pair_bias
        if compute_pair_bias:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, num_heads, bias=False),
                Rearrange("b ... h -> b h ..."),
            )
        else:
            self.proj_z = Rearrange("b ... h -> b h ...")

        self.proj_o = nn.Linear(c_s, c_s, bias=False)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        k_in: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        B = s.shape[0]

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        bias = self.proj_z(z)
        bias = bias.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        with torch.autocast("cuda", enabled=False):
            # Compute attention weights
            attn = torch.einsum("...ihd,...jhd->...hij", q.float(), k.float())
            attn = attn / (self.head_dim**0.5) + bias.float()
            
            # Match Julia's key-only masking if mask is (B, N)
            if len(mask.shape) == 2:
                mask_val = mask.view(mask.shape[0], 1, 1, mask.shape[1])
            else:
                mask_val = mask.unsqueeze(-3)
                
            attn = attn + (1 - mask_val.float()) * -self.inf
            attn = attn.softmax(dim=-1)


            # Compute output
            o = torch.einsum("...hij,...jhd->...ihd", attn, v.float()).to(v.dtype)

        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)

        return o
"""
