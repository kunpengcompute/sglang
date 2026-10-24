// Large-row copy traversal only: identical feature-partition payload and bytes.
// Root must confirm the tested node's 768 KiB private L2 premise before execution.
#include <ATen/ATen.h>
#include <algorithm>
#include <arm_sve.h>
#include <cstdint>
#include <kutacc.h>
#include <string>
#include <torch/library.h>
#include <variant>
#include <vector>

using ScalarArg = std::variant<bool, int64_t, double, std::string>;
namespace {
void layout(std::vector<at::Tensor> &t, int64_t partitions, bool unpack) {
    TORCH_CHECK(t.size() == 2, "AR blocked layout ABI");
    for (const auto &x : t)
        TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
                    "AR blocked layout requires dense CPU BF16 tensors");
    TORCH_CHECK(t[0].dim() == 2 && t[1].sizes() == t[0].sizes(), "AR blocked layout shapes");
    const auto rows = t[0].size(0), cols = t[0].size(1);
    TORCH_CHECK(rows > 128 && rows <= 4096 && cols > 0 && partitions > 0 && cols % partitions == 0,
                "AR blocked layout geometry");
    const auto a = reinterpret_cast<uintptr_t>(t[0].data_ptr());
    const auto b = reinterpret_cast<uintptr_t>(t[1].data_ptr());
    TORCH_CHECK(a <= b ? t[0].nbytes() <= b - a : t[1].nbytes() <= a - b,
                "AR blocked layout buffers must be disjoint");
    const auto width = cols / partitions;
    const auto part_tile = std::min<int64_t>(partitions, 32);
    // At width20: two 128-row x32-partition BF16 tiles =327680 B (41.67% L2).
    // No prefetch or additional payload buffers. Dynamic row and partition tails.
    const int64_t target_bytes = 768 * 1024 * 2 / 5;
    const auto row_bytes = 2 * part_tile * width * static_cast<int64_t>(sizeof(uint16_t));
    const auto aligned_rows = std::max<int64_t>(16, ((target_bytes / row_bytes + 8) / 16) * 16);
    const auto row_tile = std::min<int64_t>(128, aligned_rows);
    const auto row_blocks = (rows + row_tile - 1) / row_tile;
    const auto part_blocks = (partitions + part_tile - 1) / part_tile;
    const auto *src = reinterpret_cast<const uint16_t *>(t[unpack ? 1 : 0].data_ptr());
    auto *dst = reinterpret_cast<uint16_t *>(t[unpack ? 0 : 1].data_ptr());
    kutacc::parallel_for(0, row_blocks * part_blocks, 1, [&](int64_t begin, int64_t end) {
        for (auto tile = begin; tile < end; ++tile) {
            const auto row0 = (tile / part_blocks) * row_tile;
            const auto part0 = (tile % part_blocks) * part_tile;
            const auto row_end = std::min(rows, row0 + row_tile);
            const auto part_end = std::min(partitions, part0 + part_tile);
            for (auto part = part0; part < part_end; ++part)
                for (auto row = row0; row < row_end; ++row) {
                    const auto logical = row * cols + part * width;
                    const auto packed = (part * rows + row) * width;
                    const auto from = unpack ? packed : logical, to = unpack ? logical : packed;
                    for (int64_t col = 0; col < width; col += svcnth()) {
                        const auto mask = svwhilelt_b16(col, width);
                        svst1_u16(mask, dst + to + col, svld1_u16(mask, src + from + col));
                    }
                }
        }
    });
}
at::Tensor pack(at::Tensor x, int64_t partitions) {
    auto out = at::empty_like(x);
    std::vector<at::Tensor> t{x, out};
    layout(t, partitions, false);
    return out;
}
void pack_into(at::Tensor packed, at::Tensor x, int64_t partitions) {
    std::vector<at::Tensor> t{x, packed};
    layout(t, partitions, false);
}
void unpack(at::Tensor x, at::Tensor packed, int64_t partitions) {
    std::vector<at::Tensor> t{x, packed};
    layout(t, partitions, true);
}
} // namespace
extern "C" void qwen_graph_ar_pack_block_dispatch(std::vector<at::Tensor> &t,
                                                  const std::vector<ScalarArg> &s) {
    TORCH_CHECK(s.size() == 1, "AR blocked pack scalar ABI");
    layout(t, std::get<int64_t>(s[0]), false);
}
extern "C" void qwen_graph_ar_unpack_block_dispatch(std::vector<at::Tensor> &t,
                                                    const std::vector<ScalarArg> &s) {
    TORCH_CHECK(s.size() == 1, "AR blocked unpack scalar ABI");
    layout(t, std::get<int64_t>(s[0]), true);
}
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("qwen_registered_ar_pack_block(Tensor x, int partitions) -> Tensor", &pack);
    m.def("qwen_registered_ar_pack_block_into(Tensor(a!) packed, Tensor x, int partitions) -> ()",
          &pack_into);
    m.def("qwen_registered_ar_unpack_block(Tensor(a!) x, Tensor packed, int partitions) -> ()",
          &unpack);
}
