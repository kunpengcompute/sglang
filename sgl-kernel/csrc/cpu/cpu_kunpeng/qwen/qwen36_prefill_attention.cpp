// Qwen chunked prefill: pack each KV head once, then share the packed
// buffers between query tiles through KuTACC's existing kv_is_packed API.
#include "../utils/utils.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <torch/library.h>

static void shared_from_packed(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
                               at::Tensor packed_k, at::Tensor packed_v, at::Tensor workspace,
                               int64_t count, int64_t length, double scale) {
    const int64_t dim = q.size(2), threads = kutacc::get_thread_num();
    auto [br, bc] = kutacc::get_flash_attention_block();
    int64_t offset = 0;
    auto alloc = [&](at::ScalarType dtype, std::vector<int64_t> shape) {
        int64_t bytes = at::elementSize(dtype);
        for (auto n : shape)
            bytes *= n;
        TORCH_CHECK(offset + bytes <= workspace.numel(), "Qwen prefill scratch too small");
        auto view = workspace.slice(0, offset, offset + bytes).view(dtype).reshape(shape);
        offset += (bytes + 63) / 64 * 64;
        return view;
    };
    auto pq = alloc(at::kBFloat16, {threads, br * dim});
    auto s = alloc(at::kFloat, {threads, bc * br});
    auto old_o = alloc(at::kFloat, {threads, br, v.size(2)});
    auto new_o = alloc(at::kFloat, {threads, br, v.size(2)});
    auto old_m = alloc(at::kFloat, {threads, br});
    auto new_m = alloc(at::kFloat, {threads, br});
    auto old_l = alloc(at::kFloat, {threads, br});
    auto new_l = alloc(at::kFloat, {threads, br});
    auto qsl = at::tensor({0, static_cast<int>(count)}, at::kInt);
    auto ksl = at::tensor({0, static_cast<int>(length)}, at::kInt);
    std::vector<int64_t> lengths{length}, counts{count};
    kutacc::flash_attention(
        to_kutacc<bfloat16_t, 3>(q), to_kutacc<bfloat16_t, 3>(k), to_kutacc<bfloat16_t, 3>(v),
        to_kutacc<bfloat16_t, 3>(out), to_kutacc<bfloat16_t, 2>(pq),
        to_kutacc<bfloat16_t, 3>(packed_k), to_kutacc<bfloat16_t, 3>(packed_v),
        to_kutacc<float, 2>(s), to_kutacc<float, 3>(old_o), to_kutacc<float, 3>(new_o),
        to_kutacc<float, 2>(old_m), to_kutacc<float, 2>(new_m), to_kutacc<float, 2>(old_l),
        to_kutacc<float, 2>(new_l), true, scale, to_kutacc<int, 1>(qsl), to_kutacc<int, 1>(ksl),
        count, lengths, counts, true);
}
void qwen_prefill_flash_shared(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
                               at::Tensor packed_k, at::Tensor packed_v, at::Tensor workspace,
                               int64_t count, double scale) {
    const int64_t heads = q.size(1), dim = q.size(2), length = k.size(0);
    const int64_t capacity = packed_k.size(0), threads = kutacc::get_thread_num();
    auto [br, bc] = kutacc::get_flash_attention_block();
    TORCH_CHECK(count > 0 && count <= length && count <= q.size(0) && length <= capacity &&
                    capacity % 1024 == 0,
                "Qwen prefill requires positive live lengths within the packed capacity");
    TORCH_CHECK(k.size(1) == 1 || k.size(1) == heads,
                "Qwen prefill KV heads must be one or match query heads");
    const int64_t head_stride = k.size(1) == 1 ? 0 : k.stride(1);
    kutacc::flash_attention_k_block_pack(length, heads, dim, capacity, k.stride(0), head_stride,
                                         reinterpret_cast<bfloat16_t *>(k.data_ptr()),
                                         reinterpret_cast<bfloat16_t *>(packed_k.data_ptr()));
    kutacc::flash_attention_v_block_pack(length, heads, v.size(2), capacity, v.stride(0),
                                         v.size(1) == 1 ? 0 : v.stride(1),
                                         reinterpret_cast<bfloat16_t *>(v.data_ptr()),
                                         reinterpret_cast<bfloat16_t *>(packed_v.data_ptr()));
    shared_from_packed(q, k, v, out, packed_k, packed_v, workspace, count, length, scale);
}

// Indexed gather writes the exact existing Pack layouts without an L*D staging
// tensor. Only integer/BF16 bit copies occur here; attention arithmetic is reused.
void qwen_prefill_flash_shared_gather(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor indices,
                                      at::Tensor out, at::Tensor packed_k, at::Tensor packed_v,
                                      at::Tensor workspace, int64_t count, double scale) {
    for (const auto &x : {q, k, v, out, packed_k, packed_v})
        TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
                    "Indexed shared prefill requires dense CPU BF16 tensors");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.sizes() == k.sizes() && packed_k.dim() == 3 &&
                    packed_v.sizes() == packed_k.sizes() && out.sizes() == q.sizes(),
                "Indexed shared prefill tensor ranks/shapes");
    TORCH_CHECK(indices.device().is_cpu() && indices.is_contiguous() && indices.dim() == 1 &&
                    (indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong),
                "Indexed shared prefill indices must be dense CPU int32/int64");
    TORCH_CHECK(workspace.device().is_cpu() && workspace.is_contiguous() &&
                    workspace.scalar_type() == at::kByte && workspace.dim() == 1,
                "Indexed shared prefill scratch");
    const auto heads = q.size(1), dim = q.size(2), length = indices.numel(),
               capacity = packed_k.size(0);
    const auto threads = kutacc::get_thread_num();
    auto [br, bc] = kutacc::get_flash_attention_block();
    TORCH_CHECK(heads > 0 && dim > 0 && dim % 64 == 0 && k.size(0) > 0 && k.size(2) == dim &&
                    (k.size(1) == 1 || k.size(1) == heads) && packed_k.size(1) == heads &&
                    packed_k.size(2) == dim && count > 0 && count <= length && count <= q.size(0) &&
                    length <= INT32_MAX && capacity > 0 && capacity % 1024 == 0 &&
                    length <= capacity && br > 0 && bc > 0 && bc % 16 == 0 && capacity % bc == 0 &&
                    threads > 0 && std::isfinite(scale) && scale > 0,
                "Indexed shared prefill lengths/configuration");
    const auto head_span = (capacity / 1024) * 1024 / heads * heads;
    const auto blocks = (length + bc - 1) / bc;
    TORCH_CHECK(head_span >= blocks * bc, "Indexed shared prefill packed head capacity");
    int64_t scratch_bytes = 0;
    for (auto bytes : {threads * br * dim * 2, threads * bc * br * 4, threads * br * dim * 4,
                       threads * br * dim * 4, threads * br * 4, threads * br * 4, threads * br * 4,
                       threads * br * 4})
        scratch_bytes += (bytes + 63) / 64 * 64;
    TORCH_CHECK(workspace.numel() >= scratch_bytes, "Indexed shared prefill scratch too small");
    auto index = [&](int64_t row) -> int64_t {
        return indices.scalar_type() == at::kInt ? indices.data_ptr<int32_t>()[row]
                                                 : indices.data_ptr<int64_t>()[row];
    };
    for (int64_t row = 0; row < length; ++row)
        TORCH_CHECK(index(row) >= 0 && index(row) < k.size(0), "Indexed shared prefill KV index");
    std::vector<at::Tensor> tensors{q, k, v, indices, out, packed_k, packed_v, workspace};
    for (size_t write : {4, 5, 6, 7})
        for (size_t other = 0; other < tensors.size(); ++other)
            if (write != other) {
                const auto a = reinterpret_cast<uintptr_t>(tensors[write].data_ptr());
                const auto b = reinterpret_cast<uintptr_t>(tensors[other].data_ptr());
                TORCH_CHECK(a + tensors[write].nbytes() <= b || b + tensors[other].nbytes() <= a,
                            "Indexed shared prefill mutated buffers must not overlap");
            }
    const auto *keys = static_cast<const uint8_t *>(k.data_ptr());
    const auto *values = static_cast<const uint8_t *>(v.data_ptr());
    auto *pk = static_cast<uint8_t *>(packed_k.data_ptr());
    auto *pv = static_cast<uint8_t *>(packed_v.data_ptr());
    kutacc::parallel_for(0, heads * blocks, 1, [&](int64_t begin, int64_t end) {
        for (auto task = begin; task < end; ++task) {
            const auto head = task / blocks, block = task % blocks, first = block * bc;
            const auto live = std::min(bc, length - first), kv_head = k.size(1) == 1 ? 0 : head;
            auto *dst_k = pk + (head * head_span + first) * dim * 2;
            auto *dst_v = pv + (head * head_span + first) * dim * 2;
            // K: [16-row panel, D/2, row-in-panel, BF16 pair]. The
            // original ZA Pack zeros inactive rows in its final 16-row panel.
            for (int64_t panel = 0; panel < live; panel += 16) {
                const auto rows = std::min<int64_t>(16, live - panel);
                if (rows < 16)
                    std::memset(dst_k + panel * dim * 2, 0, 16 * dim * 2);
                const uint8_t *source[16];
                for (int64_t row = 0; row < rows; ++row)
                    source[row] =
                        keys +
                        (index(first + panel + row) * k.stride(0) + kv_head * k.stride(1)) * 2;
                for (int64_t pair = 0; pair < dim / 2; ++pair)
                    for (int64_t row = 0; row < rows; ++row)
                        std::memcpy(dst_k + (panel * dim + pair * 32 + row * 2) * 2,
                                    source[row] + pair * 4, 4);
            }
            // V: [bc-row block, 16-column group, row-pair, column, BF16 pair].
            // The original Pack zeros every V row through the last bc boundary.
            if (live < bc)
                std::memset(dst_v, 0, bc * dim * 2);
            for (int64_t group = 0; group < dim / 16; ++group) {
                const auto column = (group % (dim / 32)) * 32 + (group / (dim / 32)) * 16;
                for (int64_t row = 0; row < live; row += 2) {
                    const auto *a =
                        values +
                        (index(first + row) * v.stride(0) + kv_head * v.stride(1) + column) * 2;
                    const uint8_t *b = row + 1 < live
                                           ? values + (index(first + row + 1) * v.stride(0) +
                                                       kv_head * v.stride(1) + column) *
                                                          2
                                           : nullptr;
                    auto *target = dst_v + (group * 16 * bc + row * 16) * 2;
                    for (int64_t col = 0; col < 16; ++col) {
                        std::memcpy(target + col * 4, a + col * 2, 2);
                        if (b)
                            std::memcpy(target + col * 4 + 2, b + col * 2, 2);
                    }
                }
            }
        }
    });
    shared_from_packed(q, k, v, out, packed_k, packed_v, workspace, count, length, scale);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("qwen_prefill_flash_shared(Tensor q, Tensor k, Tensor v, Tensor(a!) out, Tensor(b!) "
          "packed_k, Tensor(c!) packed_v, Tensor(d!) workspace, int count, float scale) -> ()",
          &qwen_prefill_flash_shared);
    m.def("qwen_prefill_flash_shared_gather(Tensor q, Tensor k, Tensor v, Tensor indices, "
          "Tensor(a!) out, Tensor(b!) packed_k, Tensor(c!) packed_v, Tensor(d!) workspace, int "
          "count, float scale) -> ()",
          &qwen_prefill_flash_shared_gather);
}
