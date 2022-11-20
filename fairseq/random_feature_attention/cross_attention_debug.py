"""Cross RFA Debug."""

from typing import Optional

import torch
from fairseq.modules.quant_noise import quant_noise
from fairseq.random_feature_attention.utils import upgrade_state_dict_named
from fairseq.random_feature_attention.utils import random_project
from fairseq.random_feature_attention.utils import build_random_matrices
from fairseq.random_feature_attention.utils import normalize_attn_weights
from torch import Tensor, nn
from torch.nn import Parameter


class CrossAttentionDebug(nn.Module):
    """Random feature cross attention."""

    def __init__(
        self,
        *,
        args,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        tau: float = 1.0,
        reparam_proj: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bias = True
        self.tau = tau
        self.reparam_proj = reparam_proj

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        bias = True
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.k_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=True), q_noise, qn_block_size
        )
        if reparam_proj:
            self.sigma = Parameter(Tensor(num_heads, 1, head_dim))
        self.reset_parameters(args)
        self.upgrade_state_dict_named = upgrade_state_dict_named

    def reset_parameters(self, args):
        gain = args.init_scale ** -0.5
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)

        # std = 0.02 * args.init_scale ** -0.5
        # nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        # nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        # nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        # nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.reparam_proj:
            nn.init.constant_(self.sigma, 1.)
        # if self.q_proj.bias is not None:
        #     nn.init.constant_(self.q_proj.bias, 0.0)
        #     nn.init.constant_(self.k_proj.bias, 0.0)
        #     nn.init.constant_(self.v_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        random_matrices: Tensor,
        key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            state: s, z, random_matrices
                s [bsz, num_heads, 2 * proj_dim, head_dim]
                z [bsz, num_heads, 2 * proj_dim]
                random_matrices: [num_heads, proj_dim, head_dim]
        Return:
            attn: [tgt_len, bsz, embed_dim]
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert embed_dim == self.embed_dim

        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(key)
        q = q.contiguous().view(
            tgt_len, bsz, self.num_heads, self.head_dim)
        k = k.contiguous().view(
            src_len, bsz, self.num_heads, self.head_dim)
        v = v.contiguous().view(
            src_len, bsz, self.num_heads, self.head_dim)

        random_matrices = build_random_matrices(
            random_matrices=random_matrices,
            tau=self.tau,
            sigma=self.sigma if self.reparam_proj else None,
            reparam_proj=self.reparam_proj)

        # [src_len, bsz, num_heads, 1]
        # k_norm = torch.norm(k, p=2, dim=-1, keepdim=True)

        # [tgt_len, bsz, num_heads, 2 * proj_dim]
        phi_q = random_project(
            x=q,
            random_matrices=random_matrices
        )
        phi_k = random_project(
            x=k,
            random_matrices=random_matrices
        )
        # scale = torch.exp(k_norm / 2)
        # k = k * scale

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        # [bsz, num_heads, tgt_len, src_len]
        attn_weights = torch.einsum("tbhk,sbhk->bhts", phi_q, phi_k)
        assert list(attn_weights.size()) == [bsz, self.num_heads, tgt_len, src_len]
        if key_padding_mask is not None:
            # [bsz, 1, 1, src_len]: bool
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            attn_weights = attn_weights.masked_fill(mask, 0.0)
            # attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = normalize_attn_weights(attn_weights, dtype=attn_weights.dtype)
        # attn_weights = torch.softmax(attn_weights * 2, dim=-1)
        attn = torch.einsum("bhts,sbhd->tbhd", attn_weights, v)

        assert list(attn.size()) == [tgt_len, bsz, self.num_heads, self.head_dim]
        attn = attn.contiguous().view(tgt_len, bsz, self.num_heads * self.head_dim)
        # [tgt_len, bsz, embed_dim]
        attn = self.out_proj(attn)
        return attn
