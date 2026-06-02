#include <torch/extension.h>
#include <kupl.h>
#include "sgl_kernel_ops.h"
#include <fstream>
#include <cstdint>
#include "../utils/prf_memcpy.h"

at::Tensor hbw_allocator_kunpeng(int64_t size) {
    void* ptr = kupl_hbw_malloc(size);
    if (!ptr) {
        throw std::runtime_error("kupl_hbw_malloc failed");
    }

    int64_t ptr_value = reinterpret_cast<int64_t>(ptr);
    at::Tensor ptr_tensor = at::full({1}, ptr_value, at::dtype(at::kLong).device(at::kCPU));
    return ptr_tensor;
}

void hbw_destroy_kunpeng(at::Tensor ptr_tensor) {
    int64_t ptr_value = ptr_tensor.item<int64_t>();
    void* ptr = reinterpret_cast<void*>(ptr_value);

    if (ptr) {
        kupl_hbw_free(ptr);
    }
}

void sync_swap_kunpeng(at::Tensor dst, at::Tensor src, int64_t byte_size) {
    void* dst_ptr = dst.data_ptr();
    void* src_ptr = src.data_ptr();

    utils::prf_memcpy<true, true, 3 * 1024, SV_PLDL2STRM>(dst_ptr, src_ptr, static_cast<size_t>(byte_size));
}