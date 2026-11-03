#include "register_graph_kernels.h"
#include <ATen/ATen.h>

// Explicit out contracts: outputs and scratch are owned by the graph planner.
void tensor_unary(at::Tensor x, at::Tensor out, int64_t kind) {
    switch (kind) {
    case 0: at::square_out(out, x); break;
    case 1: at::sqrt_out(out, x); break;
    case 2: at::reciprocal_out(out, x); break;
    case 3: at::exp_out(out, x); break;
    case 4: at::sigmoid_out(out, x); break;
    case 5: at::silu_out(out, x); break;
    default: TORCH_CHECK(false, "Unknown tensor_unary operation");
    }
}
void tensor_binary(at::Tensor x, at::Tensor y, at::Tensor out, int64_t kind) {
    if (kind == 0) at::add_out(out, x, y);
    else if (kind == 1) at::mul_out(out, x, y);
    else TORCH_CHECK(false, "Unknown tensor_binary operation");
}
void tensor_scalar(at::Tensor x, at::Tensor out, double value, int64_t kind) {
    if (kind == 0) at::add_out(out, x, at::Scalar(value));
    else if (kind == 1) at::mul_out(out, x, at::Scalar(value));
    else if (kind == 2) at::gt_out(out, x, at::Scalar(value));
    else TORCH_CHECK(false, "Unknown tensor_scalar operation");
}
void tensor_mean_last(at::Tensor x, at::Tensor out) {
    at::mean_out(out, x, {-1}, true, x.scalar_type());
}
void tensor_softplus(at::Tensor x, at::Tensor out, double beta, double threshold) {
    at::softplus_out(out, x, beta, threshold);
}
void tensor_select(at::Tensor x, at::Tensor indices, at::Tensor out) {
    at::index_select_out(out, x, 0, indices);
}
void tensor_scatter(at::Tensor x, at::Tensor indices, at::Tensor source) {
    x.index_copy_(0, indices, source);
}
void tensor_where(at::Tensor condition, at::Tensor x, at::Tensor y, at::Tensor out) {
    at::where_out(out, condition, x, y);
}
#define GRAPH_REGISTER(name) static KernelRegistrar r_##name(#name, make_dispatch_v<decltype(&name), &name>);
GRAPH_REGISTER(tensor_unary)
GRAPH_REGISTER(tensor_binary)
GRAPH_REGISTER(tensor_scalar)
GRAPH_REGISTER(tensor_mean_last)
GRAPH_REGISTER(tensor_softplus)
GRAPH_REGISTER(tensor_select)
GRAPH_REGISTER(tensor_scatter)
GRAPH_REGISTER(tensor_where)

void tensor_mm(at::Tensor x, at::Tensor weight, at::Tensor out) {
    at::mm_out(out, x, weight.t());
}
GRAPH_REGISTER(tensor_mm)
