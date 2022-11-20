#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <stdio.h>
typedef torch::Tensor Tensor;

Tensor RFAForward(
        Tensor const& q, 
        Tensor const& k, 
        Tensor const& v
);

std::vector<Tensor> RFABackward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v, 
        Tensor const& grad_attn);

Tensor RFAEval(
        Tensor const& q, 
        Tensor const& k, 
        Tensor const& v, 
        Tensor const& w, 
        Tensor const& b
);

Tensor forward(
        Tensor q,
        Tensor k,
        Tensor v) {
    TORCH_CHECK(q.type().is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.type().is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.type().is_cuda(), "v must be a CUDA tensor");
    return RFAForward(q, k, v);
}

std::vector<Tensor> backward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v, 
        Tensor const& grad_attn) {
    return RFABackward(q, k, v, grad_attn);
}

Tensor eval(
        Tensor q,
        Tensor k,
        Tensor v,
        Tensor w,
        Tensor b) {
    TORCH_CHECK(q.type().is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.type().is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.type().is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(w.type().is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(b.type().is_cuda(), "b must be a CUDA tensor");
    return RFAEval(q, k, v, w, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RFA forward");
    m.def("backward", &backward, "RFA backward");
    m.def("eval", &eval, "RFA eval");
}
