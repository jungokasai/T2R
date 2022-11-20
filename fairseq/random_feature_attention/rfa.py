"""RFA cuda.

Einsum notations:
    b: bsz
    s: seq_len
    n: num_layers
    h: num_heads
    k: proj_dim
    d: head_dim
"""


import torch
import rfa_cuda


EPS = 1.

def reverse_cumsum(x, dim):
    return torch.flip(torch.cumsum(torch.flip(x, [dim]), dim), [dim])


def equal(a, b, threshold=1e-4):
    return (a - b) ** 2 < threshold


def rfa_debug(q, k, v):
    """
    Args:
        q: [tgt_len, bsz * num_heads, proj_dim]
        k: [tgt_len, bsz * num_heads, proj_dim]
        v: [tgt_len, bsz * num_heads, head_dim]

    Return:
        attn: [tgt_len, bsz * num_heads, head_dim]
    """
    s = torch.einsum("tbk,tbd->tbkd", k, v)
    s = torch.cumsum(s, dim=0)
    qs = torch.einsum("tbkd,tbk->tbd", s, q)

    z = torch.cumsum(k, dim=0)
    qz = torch.einsum("tbk,tbk->tb", q, z).clamp_min(EPS)
    attn = qs / qz.unsqueeze(-1)
    return attn


class RFA(torch.autograd.Function):

    @staticmethod
    def forward_torch(q, k, v):
        """
        Args:
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]

        Return:
            attn: [tgt_len, bsz * num_heads, head_dim]
        """
        s = torch.einsum("tbk,tbd->tbkd", k, v)
        s = torch.cumsum(s, dim=0)
        qs = torch.einsum("tbkd,tbk->tbd", s, q)

        z = torch.cumsum(k, dim=0)
        qz = torch.einsum("tbk,tbk->tb", q, z).clamp_min(EPS)
        attn = qs / qz.unsqueeze(-1)
        return attn

    @staticmethod
    def backward_torch(q, k, v, grad_attn):
        """
        Args:
            grad_attn: [tgt_len, bsz * num_heads, head_dim]
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]
        Return:
            grad_q: [tgt_len, bsz * num_heads, proj_dim]
            grad_k: [tgt_len, bsz * num_heads, proj_dim]
            grad_v: [tgt_len, bsz * num_heads, head_dim]
        """
        s = torch.einsum("tbk,tbd->tbkd", k, v)
        s = torch.cumsum(s, dim=0)
        qs = torch.einsum("tbkd,tbk->tbd", s, q)

        z = torch.cumsum(k, dim=0)
        qz = torch.einsum("tbk,tbk->tb", q, z).clamp_min(EPS)

        # [bsz, tgt_len, head_dim]
        grad_qs = grad_attn / qz.unsqueeze(-1)

        grad_qz = torch.einsum("tbd,tbd->tb", grad_attn, qs)
        grad_qz = -grad_qz / (qz ** 2)

        grad_q = torch.einsum("tbd,tbkd->tbk", grad_qs, s) \
            + grad_qz.unsqueeze(-1) * z

        grad_s = torch.einsum("tbk,tbd->tbkd", q, grad_qs)
        grad_s = reverse_cumsum(grad_s, dim=0)
        grad_k = torch.einsum("tbkd,tbd->tbk", grad_s, v)
        grad_v = torch.einsum("tbkd,tbk->tbd", grad_s, k)

        grad_k = grad_k + reverse_cumsum(q * grad_qz.unsqueeze(-1), dim=0)

        return grad_q, grad_k, grad_v

    @staticmethod
    def eval_torch(q, k, v, w, b):
        q = torch.einsum("hkd,tbhd->tbhk", w, q)
        k = torch.einsum("hkd,tbhd->tbhk", w, k)
        # q = torch.relu(q + b.unsqueeze(0).unsqueeze(0))
        # k = torch.relu(k + b.unsqueeze(0).unsqueeze(0))
        q = q + b.unsqueeze(0).unsqueeze(0)
        k = k + b.unsqueeze(0).unsqueeze(0)
        s = torch.einsum("tbhk,tbhd->tbhkd", k, v)
        s = torch.cumsum(s, dim=0)
        z = torch.cumsum(k, dim=0)  # [t, b, h, k]
        qs = torch.einsum("tbhk,tbhkd->tbhd", q, s)
        qz = torch.einsum("tbhk,tbhk->tbh", q, z).clamp_min(EPS)
        attn = qs / qz.unsqueeze(-1)
        return attn

    @staticmethod
    def forward_cuda(q, k, v):
        return rfa_cuda.forward(q, k, v)

    @staticmethod
    def backward_cuda(q, k, v, grad_attn):
        return rfa_cuda.backward(q, k, v, grad_attn)

    @staticmethod
    def eval_cuda(q, k, v, w, b):
        return rfa_cuda.eval(q, k, v, w, b)

    @staticmethod
    def forward(ctx, q, k, v):
        """
        Args:
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]

        Return:
            attn: [tgt_len, bsz * num_heads, head_dim]
        """
        ctx.save_for_backward(q, k, v)
        attn = RFA.forward_cuda(q, k, v)
        #attn = RFA.forward_torch(q, k, v)
        return attn

    @staticmethod
    def backward(ctx, grad_attn):
        """
        Args:
            q: [tgt_len, bsz * num_heads, proj_dim]
            k: [tgt_len, bsz * num_heads, proj_dim]
            v: [tgt_len, bsz * num_heads, head_dim]
            grad_attn: [tgt_len, bsz * num_heads, head_dim]
        Return:
            grad_q: [tgt_len, bsz * num_heads, proj_dim]
            grad_k: [tgt_len, bsz * num_heads, proj_dim]
            grad_v: [tgt_len, bsz * num_heads, head_dim]
        """
        q, k, v = ctx.saved_tensors
        grad_q, grad_k, grad_v = RFA.backward_cuda(q, k, v, grad_attn)
        # grad_q, grad_k, grad_v = RFA.backward_torch(q, k, v, grad_attn)
        return grad_q, grad_k, grad_v
