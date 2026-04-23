py"""
import math
import torch
import torch.nn as nn
from torch.nn import Linear

def af3_permute_final_dims(tensor: torch.Tensor, inds):
    num_first_dims = len(tensor.shape)-len(inds)
    first_inds = list(range(num_first_dims))
    return tensor.permute(first_inds + [num_first_dims + i for i in inds])

def af3_flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))

class LayerNorm(nn.Module):
    # Basic LayerNorm layer with learnable scale and offset.

    def __init__(
        self, c_in: int, create_scale: bool = True, create_offset: bool = True, eps=1e-5
    ):
        # Args:
        #     c_in: Number of input channels
        #     create_scale: Whether to create a learnable scale parameter
        #     create_offset: Whether to create a learnable offset parameter
        #     eps: Epsilon value for numerical stability
        super().__init__()

        self.c_in = (c_in,)
        self.eps = eps
        self.weight = None
        self.bias = None

        if create_scale:
            self.weight = nn.Parameter(torch.ones(c_in))

        if create_offset:
            self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x) -> torch.Tensor:
        d = x.dtype
        deepspeed_is_initialized = False

        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.amp.autocast("cuda", enabled=False):
                weight = self.weight.to(dtype=d) if self.weight is not None else None
                bias = self.bias.to(dtype=d) if self.bias is not None else None

                out = nn.functional.layer_norm(
                    input=x,
                    normalized_shape=self.c_in,
                    weight=weight,
                    bias=bias,
                    eps=self.eps,
                )
        else:
            out = nn.functional.layer_norm(
                input=x,
                normalized_shape=self.c_in,
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
            )

        return out


class AF3AdaLN(nn.Module):
    # Adaptive LayerNorm.

    # Implements AF3 Algorithm 26.

    def __init__(
        self,
        c_a: int,
        c_s: int,
        eps: float = 1e-5,
    ):
        # Args:
        #     c_a: Number of input channels for input tensor
        #     c_s: Number of input channels for shift/scale tensor
        #     eps: Epsilon value for numerical stability
        #     linear_init_params: Linear layer initialization parameters
        super().__init__()

        self.c_a = c_a
        self.c_s = c_s
        self.eps = eps

        self.layer_norm_a = LayerNorm(
            self.c_a, create_scale=False, create_offset=False, eps=self.eps
        )
        self.layer_norm_s = LayerNorm(
            self.c_s, create_scale=True, create_offset=False, eps=self.eps
        )

        self.sigmoid = nn.Sigmoid()
        self.linear_g = Linear(self.c_s, self.c_a)
        self.linear_s = Linear(self.c_s, self.c_a)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # Args:
        #     a: Input tensor to be normalized
        #     s: Input tensor to compute shift/scale

        # Returns:
        #     Normalized tensor

        a = self.layer_norm_a(a)
        s = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s))
        a = g * a + self.linear_s(s)

        return a


class AF3Attention(nn.Module):
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        super().__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, bias=False
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, bias=False
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ):
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = af3_flatten_final_dims(o, 2)
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: [torch.Tensor] = None,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=True)

        attn_dtype = torch.float32 if use_high_precision else q.dtype
        with torch.amp.autocast("cuda", dtype=attn_dtype, enabled=False):
            scores = torch.einsum("...qc, ...kc->...qk", q, k)
            for b in biases:
                scores += b
            scores = nn.functional.softmax(scores, dim=-1)

        attention = torch.einsum("...qk, ...kc->...qc", scores.to(dtype=v.dtype), v)

        o = attention.transpose(-2, -3)
        o = self._wrap_up(o, q_x)

        return o


class AF3AttentionPairBias(nn.Module):
    # Attention layer with pair bias.

    # Implements AF3 Algorithm 24 for the trunk, where no sequence local
    # or adaptive layernorm are needed by default.
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf=1e9,
    ):
        # Args:
        #     c_q:
        #         Input dimension of query data
        #     c_k:
        #         Input dimension of key data
        #     c_v:
        #         Input dimension of value data
        #     c_s:
        #         Single activation channel dimension
        #     c_z:
        #         Pair activation channel dimension
        #     c_hidden:
        #         Per-head hidden dimension
        #     no_heads:
        #         Number of attention heads
        #     use_ada_layer_norm:
        #         Whether to apply AdaLN-Zero conditioning
        #     gating:
        #         Whether the output should be gated using query data
        #     inf:
        #         Large constant used to create mask for attention logits
        #     linear_init_params:
        #         Linear layer initialization parameters
        super().__init__()

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf

        self.use_ada_layer_norm = use_ada_layer_norm

        if self.use_ada_layer_norm:
            self.layer_norm_a = AF3AdaLN(
                c_a=self.c_q, c_s=self.c_s
            )

            self.linear_ada_out = Linear(
                self.c_s, self.c_q
            )
        else:
            self.layer_norm_a = LayerNorm(c_in=self.c_q)

        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(self.c_z, no_heads)

        self.mha = AF3Attention(
            c_q=c_q,
            c_k=c_k,
            c_v=c_v,
            c_hidden=c_hidden,
            no_heads=no_heads,
            gating=gating,
        )

        self.sigmoid = nn.Sigmoid()

    def _prep_bias(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        # Args:
        #     a:
        #         [*, N, C_token] Token or atom-level embedding
        #     z:
        #         [*, N, N, C_z] Pair embedding
        #     mask:
        #         [*, N] Mask for token or atom-level embedding

        # Returns:
        #     List of bias terms. Includes the pair bias and attention mask.
        if mask is None:
            # [*, N]
            mask = a.new_ones(
                a.shape[:-1],
            )

        # DS kernel has strict shape asserts and expects the mask to be
        # tiled to the correct shape for the batch dims
        batch_dims = a.shape[:-2]
        mask = mask.expand((*batch_dims, -1))

        # [*, 1, 1, N]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :]
        biases = [mask_bias]

        # [*, N, N, C_z]
        z = self.layer_norm_z(z)

        # [*, N, N, no_heads]
        z = self.linear_z(z)

        # [*, no_heads, N, N]
        z = af3_permute_final_dims(z, [2, 0, 1])

        biases.append(z)

        return biases

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
    ) -> torch.Tensor:
        # Args:
        #     a:
        #         [*, N, C_q] Token or atom-level embedding
        #     z:
        #         [*, N, N, C_z] Pair embedding
        #     s:
        #         [*, N, C_s] Single embedding. Used in AdaLN if use_ada_layer_norm is
        #         True
        #     mask:
        #         [*, N] Mask for token or atom-level embedding
        #     use_deepspeed_evo_attention:
        #         Whether to use DeepSpeed Evo Attention kernel
        #     use_lma:
        #         Whether to use LMA
        #     use_high_precision_attention:
        #         Whether to run attention in high precision
        # Returns
        #     [*, N, C_q] attention updated token or atom-level embedding
        a = self.layer_norm_a(a, s) if self.use_ada_layer_norm else self.layer_norm_a(a)

        biases = self._prep_bias(a=a, z=z, mask=mask)

        # TODO: Make this less awkward, DS kernel has strict shape asserts
        #  and expects batch and seq dims to exist
        #  Current reshape function only expects missing batch dim
        batch_dims = a.shape[:-2]
        reshape_for_ds_kernel = (
            use_deepspeed_evo_attention or use_cueq_triangle_kernels
        ) and len(batch_dims) == 1

        if reshape_for_ds_kernel:
            a = a.unsqueeze(1)
            biases = [b.unsqueeze(1) for b in biases]

        a = self.mha(
            q_x=a,
            kv_x=a,
            biases=biases,
            # use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            # use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            # use_lma=use_lma,
            # use_high_precision=use_high_precision_attention,
        )

        if reshape_for_ds_kernel:
            a = a.squeeze(1)

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a


class AF3CrossAttentionPairBias(nn.Module):
    # Attention layer with pair bias and neighborhood mask.
    # Unlike AttentionPairBias, inputs are blocked for sequence-local attention
    # and AdaLN is applied by default.

    # Implements AF3 Algorithm 24.

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        use_ada_layer_norm: bool = False,
        n_query: int | None = None,
        n_key: int | None = None,
        gating: bool = True,
        inf=1e9,
        linear_init_params = None,
    ):
        # Args:
        #     c_q:
        #         Input dimension of query data
        #     c_k:
        #         Input dimension of key data
        #     c_v:
        #         Input dimension of value data
        #     c_s:
        #         Single activation channel dimension
        #     c_z:
        #         Pair activation channel dimension
        #     c_hidden:
        #         Per-head hidden dimension
        #     no_heads:
        #         Number of attention heads
        #     use_ada_layer_norm:
        #         Whether to apply AdaLN-Zero conditioning
        #     n_query:
        #         Number of queries (block height). If provided, inputs are split into
        #         q/k blocks of n_query and n_key prior to attention.
        #     n_key:
        #         Number of keys (block width). If provided, inputs are split into
        #         q/k blocks of n_query and n_key prior to attention.
        #     gating:
        #         Whether the output should be gated using query data
        #     inf:
        #         Large constant used to create mask for attention logits
        #     linear_init_params:
        #         Linear layer initialization parameters
        super().__init__()

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf

        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if linear_init_params is None:
            linear_init_params = (
                lin_init.diffusion_att_pair_bias_init
                if self.use_ada_layer_norm
                else lin_init.att_pair_bias_init
            )

        if self.use_ada_layer_norm:
            self.layer_norm_a_q = AF3AdaLN(
                c_a=self.c_q, c_s=self.c_s, linear_init_params=linear_init_params.ada_ln
            )
            self.layer_norm_a_k = AF3AdaLN(
                c_a=self.c_q, c_s=self.c_s, linear_init_params=linear_init_params.ada_ln
            )

            self.linear_ada_out = Linear(
                self.c_s, self.c_q, **linear_init_params.linear_ada_out
            )
        else:
            self.layer_norm_a_q = LayerNorm(c_in=self.c_q)
            self.layer_norm_a_k = LayerNorm(c_in=self.c_q)

        self.linear_z = Linear(self.c_z, no_heads, **linear_init_params.linear_z)

        self.mha = Attention(
            c_q=c_q,
            c_k=c_k,
            c_v=c_v,
            c_hidden=c_hidden,
            no_heads=no_heads,
            gating=gating,
            linear_init_params=linear_init_params.mha,
        )

        self.sigmoid = nn.Sigmoid()

    def _prep_block_inputs(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ):
        # Args:
        #     a:
        #         [*, N, C_token] Token or atom-level embedding
        #     z:
        #         [*, N, N, C_z] Pair embedding
        #     mask:
        #         [*, N] Mask for token or atom-level embedding

        # Returns:
        #     List of bias terms. Includes the pair bias and attention mask.
        a_query, a_key, mask = convert_single_rep_to_blocks(
            ql=a, n_query=self.n_query, n_key=self.n_key, atom_mask=mask
        )

        # [*, 1, 1, N]
        mask_bias = (self.inf * (mask - 1))[..., None, :, :]
        biases = [mask_bias]

        # [*, N, N, no_heads]
        z = self.linear_z(z)

        # [*, no_heads, N, N]
        z = af3_permute_final_dims(z, [2, 0, 1])

        biases.append(z)

        return a_query, a_key, biases

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_high_precision_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
    ) -> torch.Tensor:
        # Args:
        #     a:
        #         [*, N, C_q] Token or atom-level embedding
        #     z:
        #         [*, N, N, C_z] Pair embedding
        #     s:
        #         [*, N, C_s] Single embedding. Used in AdaLN if use_ada_layer_norm is
        #         True
        #     mask:
        #         [*, N] Mask for token or atom-level embedding
        #     use_high_precision_attention:
        #         Whether to run attention in high precision
        # Returns
        #     [*, N, C_q] attention updated token or atom-level embedding
        batch_dims = a.shape[:-2]
        n_atom, n_dim = a.shape[-2:]

        if mask is None:
            # [*, N]
            mask = a.new_ones(
                a.shape[:-1],
            )

        a_q, a_k, biases = self._prep_block_inputs(a=a, z=z, mask=mask)

        if self.use_ada_layer_norm:
            s_q, s_k, _ = convert_single_rep_to_blocks(
                ql=s, n_query=self.n_query, n_key=self.n_key, atom_mask=mask
            )
            a_q = self.layer_norm_a_q(a_q, s_q)
            a_k = self.layer_norm_a_k(a_k, s_k)
        else:
            a_q = self.layer_norm_a_q(a_q)
            a_k = self.layer_norm_a_k(a_k)

        a = self.mha(
            q_x=a_q,
            kv_x=a_k,
            biases=biases,
            use_high_precision=use_high_precision_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
        )

        # Convert back to unpadded and flattened atom representation
        # [*, N_blocks, N_query, c_atom] -> [*, N_atom, c_atom]
        a = a.reshape((*batch_dims, -1, n_dim))[..., :n_atom, :]

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a

class AF3OuterProductMean(nn.Module):
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


class AF3MSAPairWeightedAveraging(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_z,
        no_heads,
        inf=1e9,
    ):
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.c_z = c_z
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)
        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(self.c_z, self.no_heads)

        self.linear_v = Linear(self.c_in, self.c_hidden * self.no_heads)
        self.linear_o = Linear(c_hidden * no_heads, c_in)
        self.linear_g = Linear(self.c_in, self.c_hidden * self.no_heads)

        self.sigmoid = nn.Sigmoid()

    def _prep_inputs(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        z = af3_permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)
        z = z + mask_bias

        return z

    def _get_pair_weighted_avg(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        v = self.linear_v(m)
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        v = v.transpose(-2, -3)

        o = torch.nn.functional.softmax(z, -1)
        o = torch.einsum("...hqk,...hkc->...qhc", o, v)

        return o

    def forward(
        self, m: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        z = self._prep_inputs(z=z, mask=mask)
        m = self.layer_norm_m(m)
        
        o = self._get_pair_weighted_avg(m=m, z=z)
        g = self.sigmoid(self.linear_g(m))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))
        o = o * g
        
        o = o.reshape(o.shape[:-2] + (-1,))
        o = self.linear_o(o)

        return o
"""
