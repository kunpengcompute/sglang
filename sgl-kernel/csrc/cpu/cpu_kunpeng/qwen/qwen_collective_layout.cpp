// Shared AR copy layout. Large and small routes retain their original admission,
// tile traversal, schemas, output ownership and small-route counters.
#include <ATen/ATen.h>
#include <algorithm>
#include <arm_sve.h>
#include <atomic>
#include <cstdint>
#include <kutacc.h>
#include <string>
#include <torch/library.h>
#include <variant>
#include <vector>

using ScalarArg = std::variant<bool, int64_t, double, std::string>;
namespace {
std::atomic<uint64_t> pack_calls{0}, unpack_calls{0};

template <bool Small>
void layout(std::vector<at::Tensor> &t, int64_t partitions, bool unpack) {
    TORCH_CHECK(t.size() == 2, "AR blocked layout ABI");
    for (const auto &x : t) {
        if constexpr (Small) {
            TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kBFloat16 &&
                            x.dim() == 2 && x.is_contiguous() && x.stride(1) == 1 &&
                            x.stride(0) == x.size(1),
                        "AR blocked layout requires dense CPU BF16 tensors");
        } else {
            TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kBFloat16 &&
                            x.is_contiguous(),
                        "AR blocked layout requires dense CPU BF16 tensors");
        }
    }
    TORCH_CHECK(t[0].dim() == 2 && t[1].sizes() == t[0].sizes(), "AR blocked layout shapes");
    const auto rows = t[0].size(0), cols = t[0].size(1);
    TORCH_CHECK((Small ? rows >= 1 && rows <= 32 : rows > 128 && rows <= 4096) &&
                    cols > 0 && partitions > 0 && cols % partitions == 0,
                "AR blocked layout geometry");
    const auto a = reinterpret_cast<uintptr_t>(t[0].data_ptr());
    const auto b = reinterpret_cast<uintptr_t>(t[1].data_ptr());
    TORCH_CHECK(a <= b ? t[0].nbytes() <= b - a : t[1].nbytes() <= a - b,
                "AR blocked layout buffers must be disjoint");
    const auto width = cols / partitions;
    const auto part_tile = std::min<int64_t>(partitions, 32);
    const auto part_blocks = (partitions + part_tile - 1) / part_tile;
    const auto *src = reinterpret_cast<const uint16_t *>(t[unpack ? 1 : 0].data_ptr());
    auto *dst = reinterpret_cast<uint16_t *>(t[unpack ? 0 : 1].data_ptr());
    const auto copy = [&](int64_t row, int64_t part) {
        const auto logical = row * cols + part * width;
        const auto packed = (part * rows + row) * width;
        const auto from = unpack ? packed : logical, to = unpack ? logical : packed;
        for (int64_t col = 0; col < width; col += svcnth()) {
            const auto mask = svwhilelt_b16(col, width);
            svst1_u16(mask, dst + to + col, svld1_u16(mask, src + from + col));
        }
    };
    if constexpr (Small) {
        kutacc::parallel_for(0, part_blocks, 1, [&](int64_t begin, int64_t end) {
            for (auto tile = begin; tile < end; ++tile) {
                const auto part0 = tile * part_tile;
                const auto part_end = std::min(partitions, part0 + part_tile);
                for (auto part = part0; part < part_end; ++part)
                    for (int64_t row = 0; row < rows; ++row)
                        copy(row, part);
            }
        });
        (unpack ? unpack_calls : pack_calls).fetch_add(1, std::memory_order_relaxed);
    } else {
        // Preserve the original 768 KiB L2 premise and row-tile formula.
        const int64_t target_bytes = 768 * 1024 * 2 / 5;
        const auto row_bytes = 2 * part_tile * width * static_cast<int64_t>(sizeof(uint16_t));
        const auto aligned_rows = std::max<int64_t>(16, ((target_bytes / row_bytes + 8) / 16) * 16);
        const auto row_tile = std::min<int64_t>(128, aligned_rows);
        const auto row_blocks = (rows + row_tile - 1) / row_tile;
        kutacc::parallel_for(0, row_blocks * part_blocks, 1, [&](int64_t begin, int64_t end) {
            for (auto tile = begin; tile < end; ++tile) {
                const auto row0 = (tile / part_blocks) * row_tile;
                const auto part0 = (tile % part_blocks) * part_tile;
                const auto row_end = std::min(rows, row0 + row_tile);
                const auto part_end = std::min(partitions, part0 + part_tile);
                for (auto part = part0; part < part_end; ++part)
                    for (auto row = row0; row < row_end; ++row)
                        copy(row, part);
            }
        });
    }
}
template <bool Small>
at::Tensor pack(at::Tensor x, int64_t partitions) {
    auto out = at::empty_like(x);
    std::vector<at::Tensor> t{x, out};
    layout<Small>(t, partitions, false);
    return out;
}
template <bool Small>
void pack_into(at::Tensor packed, at::Tensor x, int64_t partitions) {
    std::vector<at::Tensor> t{x, packed};
    layout<Small>(t, partitions, false);
}
template <bool Small>
void unpack(at::Tensor x, at::Tensor packed, int64_t partitions) {
    std::vector<at::Tensor> t{x, packed};
    layout<Small>(t, partitions, true);
}
} // namespace
extern "C" void qwen_graph_ar_pack_block_dispatch(std::vector<at::Tensor> &t,
                                                const std::vector<ScalarArg> &s) {
    TORCH_CHECK(s.size() == 1, "AR blocked pack scalar ABI");
    layout<false>(t, std::get<int64_t>(s[0]), false);
}
extern "C" void qwen_graph_ar_unpack_block_dispatch(std::vector<at::Tensor> &t,
                                                  const std::vector<ScalarArg> &s) {
    TORCH_CHECK(s.size() == 1, "AR blocked unpack scalar ABI");
    layout<false>(t, std::get<int64_t>(s[0]), true);
}
extern "C" void qwen_graph_p2_ar_pack_smallrow_dispatch(std::vector<at::Tensor> &t,
                                                     const std::vector<ScalarArg> &s) {
    TORCH_CHECK(s.size() == 1, "AR blocked pack scalar ABI");
    layout<true>(t, std::get<int64_t>(s[0]), false);
}
extern "C" void qwen_graph_p2_ar_unpack_smallrow_dispatch(std::vector<at::Tensor> &t,
                                                       const std::vector<ScalarArg> &s) {
    TORCH_CHECK(s.size() == 1, "AR blocked unpack scalar ABI");
    layout<true>(t, std::get<int64_t>(s[0]), true);
}
extern "C" uint64_t qwen_graph_p2_ar_pack_smallrow_calls() { return pack_calls.load(std::memory_order_relaxed); }
extern "C" uint64_t qwen_graph_p2_ar_unpack_smallrow_calls() { return unpack_calls.load(std::memory_order_relaxed); }
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("qwen_registered_ar_pack_block(Tensor x, int partitions) -> Tensor", &pack<false>);
    m.def("qwen_registered_ar_pack_block_into(Tensor(a!) packed, Tensor x, int partitions) -> ()", &pack_into<false>);
    m.def("qwen_registered_ar_unpack_block(Tensor(a!) x, Tensor packed, int partitions) -> ()", &unpack<false>);
    m.def("qwen_registered_p2_ar_pack_smallrow(Tensor x, int partitions) -> Tensor", &pack<true>);
    m.def("qwen_registered_p2_ar_pack_smallrow_into(Tensor(a!) packed, Tensor x, int partitions) -> ()", &pack_into<true>);
    m.def("qwen_registered_p2_ar_unpack_smallrow(Tensor(a!) x, Tensor packed, int partitions) -> ()", &unpack<true>);
}
