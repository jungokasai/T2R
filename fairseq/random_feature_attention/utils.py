"""RFA Utils."""
from typing import Optional, Text, Tuple
import numpy as np
import torch, os
from torch import Tensor
import torch.nn.functional as F


EPS = 1e-2
SCALE = 1e-2
RANDOM_MATRICES_PATH = os.path.join(os.path.dirname(__file__), '../../random_matrices')



def get_rfa_attr(args):
    args.use_rfa = getattr(args, "use_rfa", False)
    args.cross_proj_dim = getattr(args, "cross_proj_dim", 64)
    args.causal_proj_dim = getattr(args, "causal_proj_dim", 64)
    args.cross_tau = getattr(args, "cross_tau", 1.0)
    args.causal_tau = getattr(args, "causal_tau", 1.0)
    args.reparam_proj = getattr(args, "reparam_proj", False)
    args.learned_tau = getattr(args, "learned_tau", False)
    args.norm_rescale = getattr(args, "norm_rescale", False)
    args.cuda_causal_rfa = getattr(args, "cuda_causal_rfa", False)
    args.init_scale = getattr(args, "init_scale", 1.0)
    args.random_feature = getattr(args, "random_feature", "rrf")
    args.attn_act = getattr(args, "attn_act", "none")
    args.use_input_gate = getattr(args, "use_input_gate", False)
    return args


def add_rfa_args(parser):
    parser.add_argument('--use-rfa', action='store_true',
                        help='whether or not to use rfa')
    parser.add_argument('--cross-proj-dim', type=int, metavar='N',
                        help='projection size for cross rfa')
    parser.add_argument('--causal-proj-dim', type=int, metavar='N',
                        help='projection size for causal rfa')
    parser.add_argument('--cross-tau', type=float, metavar='D',
                        help='tau for rfa')
    parser.add_argument('--causal-tau', type=float, metavar='D',
                        help='tau for rfa')
    parser.add_argument('--reparam-proj', action='store_true',
                        help='whether or not to reparameterze random matrices in rfa')
    parser.add_argument('--learned-tau', action='store_true',
                        help='whether or not to learn tau in rfa')
    parser.add_argument('--norm-rescale', action='store_true',
                        help='whether or not to rescale keys by their norms')
    parser.add_argument('--cuda-causal-rfa', action='store_true',
                        help='whether or not to use custom cuda kernel for causal rfa')
    parser.add_argument('--init-scale', type=float, metavar='D',
                        help='init scale')
    parser.add_argument('--random-feature', help='random feature')
    parser.add_argument('--attn-act', type=str, metavar='STR',
                        default='none', help='path to pre-trained decoder embedding')
    parser.add_argument('--use-input-gate', action='store_true',
                        help='whether or not to use input gate')
    return parser


def load_random_matrices(
        *,
        head_dim: int,
        proj_dim: int,
        dtype: torch.dtype = torch.half) -> Tensor:

    # [num_random_matrices, proj_dim, head_dim]
    random_matrices = np.load(
        f"{RANDOM_MATRICES_PATH}/{head_dim}_{proj_dim}.npy")[:400]
    return torch.nn.Parameter(
        torch.tensor(random_matrices, dtype=dtype), requires_grad=False)


def sample_random_matrices(
        *,
        num_layers: int,
        num_heads: int,
        random_matrices: Tensor,
        is_training: bool = True):
    # random_matrices
    # [num_random_matrices, proj_dim, head_dim]

    if is_training:
        num_random_matrices = random_matrices.size(0)
        indices = np.random.choice(
            num_random_matrices,
            size=num_layers * num_heads,
            replace=False)
        # [num_layers * num_heads, proj_dim, head_dim]
        random_matrices = random_matrices[indices]
        sampled_random_matrices = []
        for i in range(num_layers):
            sampled_random_matrices.append(
                random_matrices[i * num_heads: (i + 1) * num_heads])
        return sampled_random_matrices
    else:
        indices = list(range(num_heads))
        # [num_layers * num_heads, proj_dim, head_dim]
        return random_matrices[indices]


def build_random_matrices(
        random_matrices: Tensor,
        tau: Tensor,
        sigma: Optional[Tensor] = None,
        reparam_proj: bool = False) -> Tensor:
    if reparam_proj:
        random_matrices = sigma * random_matrices
    return torch.div(random_matrices, tau)


def _normalize(x: Tensor) -> Tuple[Tensor, Tensor]:
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    return x / (norm + 1e-3), norm


def tau(x: Tensor) -> Tensor:
    return x
    # tau \in (0.5, 1.5) for better numerical stability
    # return torch.sigmoid(x) + 0.5

def tanh(x):
    return F.tanh(x)*0.01
def cos(x):
    return torch.cos(x)*0.01


def attention_activation(act):
    if act == "relu":
        return F.relu
    elif act == "elu":
        return lambda x: F.elu(x.float()).type_as(x) + 1.
    elif act == "gelu":
        return lambda x: F.gelu(x.float()).type_as(x) + 1.
    elif act == "sigmoid":
        return F.sigmoid
    elif act == "tanh":
        return tanh
    elif act == "cos":
        return cos
    elif act == "none":
        return lambda x: x
    else:
        assert False


def random_project(
        *,
        x: Tensor,
        random_matrices: Tensor,
        tau: Optional[Tensor] = None,
        norm_rescale: bool = False,
        scale: float = 1.0,
        random_feature: Text = "rrf",
        bias: Tensor = None,
        activation = None,
        ) -> Tensor:
    # x: [seq_len, bsz, num_heads, head_dim]
    # random_matrices: [num_heads, proj_dim, head_dim]
    # tau: [num_heads, 1, 1]

    # [1, 1, num_heads, 1]
    tau = tau.contiguous().view(1, 1, tau.size(0), 1)
    if random_feature == "rrf":
        x, x_norm = _normalize(x)
        x = x / tau
        input_x = x
        # [seq_len, bsz, num_heads, proj_dim]
        x = torch.einsum("sbhd,hkd->sbhk", x, random_matrices)
        x_sin, x_cos = torch.sin(x), torch.cos(x)

        # [seq_len, bsz, num_heads, 2 * proj_dim]
        phi_x = torch.cat([x_sin, x_cos], dim=-1) * SCALE
        if norm_rescale:
            x_norm = 0.5 * (x_norm * scale / tau) ** 2.
            maxes = torch.max(x_norm, dim=0, keepdim=True).values.detach()
            phi_x = phi_x * torch.exp(x_norm - maxes)
        return phi_x
    elif random_feature == "prf":
        scale = x.size(-1) ** -0.25
        x = x * scale  # / tau
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True) ** 2

        # [seq_len, bsz, num_heads, proj_dim]
        x = torch.einsum("sbhd,hkd->sbhk", x, random_matrices)
        maxes = torch.max(x, dim=-1, keepdim=True).values
        x = x - 0.5 * x_norm - maxes
        phi_x = torch.exp(x) * SCALE
        return phi_x
    elif random_feature == "mlp" or random_feature == "mlp-norm":
        if random_feature == "mlp-norm":
            x, x_norm = _normalize(x)
        #T, bsz, num_heads, C = x.size()
        #out_dim = weight.size(0) // num_heads
        #weight = weight.view(num_heads, out_dim, C)
        #bias = bias.view(num_heads, out_dim)
        phi_x = torch.einsum('sbhd,hkd->sbhk', x, random_matrices)
        phi_x = phi_x + bias.unsqueeze(0).unsqueeze(0)
        phi_x = activation(phi_x)
        return phi_x
    else:
        assert False, "random project setting can either be `rrf` or `ppf` or `mlp(-norm)`"


def normalize_attn_weights(
        x: Tensor,
        dim: int = -1,
        dtype: torch.dtype = torch.float32) -> Tensor:
    x = x.type(torch.float32)
    # [..., 1]
    s = x.sum(dim=dim, keepdim=True).clamp(EPS)
    return torch.div(x, s).type(dtype)


def append_prev_key_padding_mask(
    key_padding_mask: Optional[Tensor],
    prev_key_padding_mask: Optional[Tensor],
    batch_size: int,
    src_len: int,
    static_kv: bool,
) -> Optional[Tensor]:
    # saved key padding masks have shape (bsz, seq_len)
    if prev_key_padding_mask is not None and static_kv:
        new_key_padding_mask = prev_key_padding_mask
    elif prev_key_padding_mask is not None and key_padding_mask is not None:
        new_key_padding_mask = torch.cat(
            [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
        )
    # During incremental decoding, as the padding token enters and
    # leaves the frame, there will be a time when prev or current
    # is None
    elif prev_key_padding_mask is not None:
        filler = torch.zeros(
            (batch_size, src_len - prev_key_padding_mask.size(1)),
            device=prev_key_padding_mask.device,
        )
        new_key_padding_mask = torch.cat(
            [prev_key_padding_mask.float(), filler.float()], dim=1
        )
    elif key_padding_mask is not None:
        filler = torch.zeros(
            (batch_size, src_len - key_padding_mask.size(1)),
            device=key_padding_mask.device,
        )
        new_key_padding_mask = torch.cat(
            [filler.float(), key_padding_mask.float()], dim=1
        )
    else:
        new_key_padding_mask = prev_key_padding_mask
    return new_key_padding_mask


def upgrade_state_dict_named(state_dict, name):
    prefix = name + "." if name != "" else ""
    items_to_add = {}
    keys_to_remove = []
    for k in state_dict.keys():
        if k.endswith(prefix + "in_proj_weight"):
            # in_proj_weight used to be q + k + v with same dimensions
            dim = int(state_dict[k].shape[0] / 3)
            items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
            items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
            items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

            keys_to_remove.append(k)

            k_bias = prefix + "in_proj_bias"
            if k_bias in state_dict.keys():
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                    dim: 2 * dim
                ]
                items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                keys_to_remove.append(prefix + "in_proj_bias")

    for k in keys_to_remove:
        del state_dict[k]

    for key, value in items_to_add.items():
        state_dict[key] = value
