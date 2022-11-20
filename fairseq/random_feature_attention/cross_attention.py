""""Cross RFA."""

from typing import List, Optional, Tuple, Dict

import torch
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.quant_noise import quant_noise
from fairseq.models.fairseq_encoder import CrossAttentionState
from fairseq.random_feature_attention.utils import upgrade_state_dict_named
from fairseq.random_feature_attention.utils import random_project
from fairseq.random_feature_attention.utils import load_random_matrices
from fairseq.random_feature_attention.utils import build_random_matrices
from fairseq.random_feature_attention.utils import sample_random_matrices
from fairseq.random_feature_attention.utils import attention_activation
from fairseq.random_feature_attention.utils import EPS
from fairseq.random_feature_attention.utils import tau
from fairseq.random_feature_attention.causal_attention import CausalAttention, masked_rfa
from torch import Tensor, nn
from torch.nn import Parameter

def cross_rfa(*,
               phi_q: Tensor,
               s: Tensor,
               z: Tensor,
               training = False,
               dropout_p = 0.0) -> Tensor:
    """Masked causal RFA implementation.

    Args:
        phi_q: [tgt_len, bsz, num_heads, 2 * proj_dim]
        phi_k: [src_len, bsz, num_heads, 2 * proj_dim]
        v: [src_len, bsz, num_heads, head_dim]
        key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        attn_mask (ByteTensor, optional): typically used to
            implement causal attention, where the mask prevents the
            attention from looking forward in time (default: None).
            [tgt_len, src_len]
    Return:
        attn: [tgt_len, bsz, num_heads * head_dim]
    """

    tgt_len, bsz, num_heads, proj_dim = phi_q.size()
    head_dim = s.size(-1)
    # This is part of a workaround to get around fork/join parallelism
    # not supporting Optional types.

    qs = torch.einsum("tbhk,bhkd->tbhd", phi_q, s)
    qz = torch.einsum("tbhk,bhk->tbh", phi_q, z).abs().clamp_min(EPS)
    # [tgt_len, bsz, num_heads, head_dim]
    attn = qs / qz.unsqueeze(-1)
    assert list(attn.size()) == [tgt_len, bsz, num_heads, head_dim]
    attn = attn.contiguous().view(tgt_len, bsz, num_heads * head_dim)

    return attn

def compute_sz(*,
               phi_k: Tensor,
               v: Tensor,
               key_padding_mask: Optional[Tensor] = None,
               training = False,
               dropout_p = 0.0) -> Tensor:

    if key_padding_mask is not None:
        mask = key_padding_mask.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).to(torch.bool)
        phi_k = phi_k.masked_fill(mask, 0.0)

    
    if training:
        src_len, bsz, num_heads, _ = phi_k.size()
        dropout_mask = phi_k.new_ones([src_len, bsz, num_heads, 1])
        dropout_mask = nn.functional.dropout(dropout_mask, p=dropout_p, training=training)
        phi_k = phi_k*dropout_mask

    s = torch.einsum("sbhk,sbhd->bhkd", phi_k, v)
    z = torch.sum(phi_k, dim=0)  # [bsz, num_heads, head_dim]
    return s, z


class CrossAttention(CausalAttention):
    """Random feature cross attention."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        random_matrices: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            x: [tgt_len, bsz, embed_dim]
            random_matrices: [num_heads, proj_dim, head_dim]
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
                [tgt_len, src_len]
        Return:
            attn: [tgt_len, bsz, embed_dim]
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert embed_dim == self.embed_dim

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)

        else:
            saved_state = None

        q = self.q_proj(query)

        q = q.contiguous().view(
            tgt_len, bsz, self.num_heads, self.head_dim)
        if self.random_feature == 'mlp' or self.random_feature == 'mlp-norm':
            random_matrices, bias = random_matrices.weight, random_matrices.bias
            random_matrices = random_matrices.view(self.num_heads, -1, self.head_dim)
            bias = bias.view(self.num_heads, -1)
        else:
            bias = None
        random_matrices = build_random_matrices(
            random_matrices=random_matrices,
            tau=tau(self.tau),
            sigma=self.sigma if self.reparam_proj else None,
            reparam_proj=self.reparam_proj)
        # q, k = self.attn_act(q), self.attn_act(k)
        scaling = self.head_dim ** -0.5
        q = q*scaling
        phi_q = random_project(
            x=q,
            random_matrices=random_matrices,
            tau=tau(self.tau),
            norm_rescale=False,
            random_feature=self.random_feature,
            bias = bias,
            activation = self.attn_act,
        )
        if self.use_input_gate:
            # [tgt_len, bsz, num_heads]
            g = self.g_proj(x)
            g = torch.sigmoid(g).unsqueeze(-1)
            phi_k = g * phi_k
        if saved_state is not None:
            if "prev_s" in saved_state:
                assert "prev_z" in saved_state
                del key
                if key_padding_mask is not None:
                    del key_padding_mask
                s, z = saved_state["prev_s"], saved_state["prev_z"]
            else:
                k, v = self.k_proj(key), self.v_proj(key)

                # [s, b, h, d]
                k = (
                    k.contiguous()
                    .view(-1, bsz, self.num_heads, self.head_dim)
                )
                v = (
                    v.contiguous()
                    .view(-1, bsz, self.num_heads, self.head_dim)
                )
                # [s, b, h, k]
                phi_k = random_project(
                    x=k,
                    random_matrices=random_matrices,
                    tau=tau(self.tau),
                    norm_rescale=False,
                    random_feature=self.random_feature,
                    bias = bias,
                    activation = self.attn_act,
                )
                s, z = compute_sz(phi_k=phi_k, v=v, key_padding_mask=key_padding_mask)
                saved_state["prev_s"], saved_state["prev_z"] = s, z
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        else:
            k, v = self.k_proj(key), self.v_proj(key)

            # [s, b, h, d]
            k = (
                k.contiguous()
                .view(-1, bsz, self.num_heads, self.head_dim)
            )
            v = (
                v.contiguous()
                .view(-1, bsz, self.num_heads, self.head_dim)
            )
            # [s, b, h, k]
            phi_k = random_project(
                x=k,
                random_matrices=random_matrices,
                tau=tau(self.tau),
                norm_rescale=False,
                random_feature=self.random_feature,
                bias = bias,
                activation = self.attn_act,
            )
            s, z = compute_sz(phi_k=phi_k, v=v, key_padding_mask=key_padding_mask)
        attn = cross_rfa(phi_q=phi_q, s=s, z=z,
                          training=self.training,
                          dropout_p=self.dropout_p
                          )
        #attn = masked_rfa(phi_q=phi_q, phi_k=phi_k, v=v,
        #                  attn_mask=attn_mask,
        #                  key_padding_mask=key_padding_mask,
        #                  training=self.training,
        #                  dropout_p=self.dropout_p
        #                  )

        # [tgt_len, bsz, embed_dim]
        attn = self.out_proj(attn)
        return attn
