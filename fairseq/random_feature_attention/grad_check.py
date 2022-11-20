import torch
import time
from rfa import RFA, equal


device = torch.device("cuda:0")
dtype = torch.half


def normalize(x):
    return x / x.norm(p=2, dim=-1, keepdim=True)


def test_forward():
    bsz, head_dim, proj_dim, tgt_len = 1, 16, 16, 50
    torch.manual_seed(3)
    q = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype) - .5
    k = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype) - .5
    v = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype) - .5

    rfa_func = RFA()
    attn_torch = rfa_func.forward_torch(q, k, v)
    attn_cuda = rfa_func.forward_cuda(q, k, v)
    e = (equal(attn_torch, attn_cuda, threshold=1e-5))
    print(torch.all(e))
    print(attn_cuda)
    print(attn_torch)
    print(e)


def test_forward_speed():
    bsz, head_dim, proj_dim = 2 * 8, 64, 128
    tgt_len = 1024
    num_ite = 10000
    rfa_func = RFA()

    q = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype)
    k = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype)
    v = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype)
    start_time = time.time()
    for _ in range(num_ite):
        _ = rfa_func.forward_torch(q, k, v)
    print(f"torch forward: {time.time() - start_time}s")

    q = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype)
    k = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype)
    v = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype)
    start_time = time.time()
    for _ in range(num_ite):
        _ = rfa_func.forward_cuda(q, k, v)
    print(f"cuda forward: {time.time() - start_time}s")


def test_backward():
    torch.manual_seed(3)
    bsz, head_dim, proj_dim = 5, 64, 128
    tgt_len = 256
    torch.manual_seed(3)
    q = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype)
    k = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype)
    v = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype)
    grad_attn = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype)

    q, k, v, grad_attn = normalize(q), normalize(k), normalize(v), normalize(grad_attn)
    rfa_func = RFA()

    gq_torch, gk_torch, gv_torch = rfa_func.backward_torch(q, k, v, grad_attn)
    gq_cuda, gk_cuda, gv_cuda = rfa_func.backward_cuda(q, k, v, grad_attn)
    e = (equal(gq_cuda, gq_torch))
    print(torch.all(e))
    e = (equal(gk_cuda, gk_torch))
    print(torch.all(e))
    e = (equal(gv_cuda, gv_torch))
    print(torch.all(e))


def test_backward_speed():
    torch.manual_seed(3)
    bsz, head_dim, proj_dim = 5, 64, 128
    tgt_len = 3072
    num_ite = 1000
    rfa_func = RFA()
    torch.manual_seed(3)
    q = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype)
    k = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype)
    v = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype)
    grad_attn = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype)
    q, k, v, grad_attn = normalize(q), normalize(k), normalize(v), normalize(grad_attn)

    start_time = time.time()
    for _ in range(num_ite):
        _ = rfa_func.backward_torch(q, k, v, grad_attn)
    print(f"torch backward: {time.time() - start_time}s")

    q = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype)
    k = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype)
    v = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype)
    grad_attn = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype)
    q, k, v, grad_attn = normalize(q), normalize(k), normalize(v), normalize(grad_attn)
    start_time = time.time()
    for _ in range(num_ite):
        _ = rfa_func.backward_cuda(q, k, v, grad_attn)
    print(f"cuda backward: {time.time() - start_time}s")


def test_eval():
    bsz, num_heads, head_dim, proj_dim, tgt_len = 20, 8, 128, 128, 50
    torch.manual_seed(3)
    q = torch.rand(tgt_len, bsz, num_heads, head_dim,
                   device=device, dtype=dtype) - 0.5
    k = torch.rand(tgt_len, bsz, num_heads, head_dim,
                   device=device, dtype=dtype) - 0.5
    v = torch.rand(tgt_len, bsz, num_heads, head_dim, device=device, dtype=dtype) - 0.5

    w = torch.rand(num_heads, proj_dim, head_dim, device=device, dtype=dtype) - 0.5
    b = torch.rand(num_heads, proj_dim, device=device, dtype=dtype) - 0.5

    rfa_func = RFA()

    attn_torch = rfa_func.eval_torch(q, k, v, w, b)
    attn_cuda = rfa_func.eval_cuda(q, k, v, w, b)
    e = (equal(attn_torch, attn_cuda, threshold=1e-4))
    print(torch.all(e))
    print(attn_cuda)
    print(attn_torch)
    print(e)


test_eval()
# test_forward()
# test_forward_speed()
# test_backward()
# test_backward_speed()
