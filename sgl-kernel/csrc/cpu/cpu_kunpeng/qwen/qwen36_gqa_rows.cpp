// Isolated batched GQA scheduling candidate. Every row calls an existing
// validated C ABI; no dot-product, softmax or accumulation formula is changed.
#include <cmath>
#include <cstdint>
#include <kutacc.h>
#include <vector>

extern "C" int qwen36_gqa_decode_bf16_serial_with_workspace(const uint16_t *, const uint16_t *,
                                                            const uint16_t *, const int32_t *,
                                                            uint16_t *, float *, int64_t, int64_t,
                                                            int64_t, int64_t, float, float *,
                                                            int64_t);
extern "C" int qwen36_gqa_decode_bf16_parallel_with_workspace(const uint16_t *, const uint16_t *,
                                                              const uint16_t *, const int32_t *,
                                                              uint16_t *, float *, int64_t, int64_t,
                                                              int64_t, int64_t, float, float *,
                                                              int64_t);

extern "C" int qwen36_gqa_decode_batch_with_workspace(
    const uint16_t *q, const uint16_t *k, const uint16_t *v, const int32_t *table,
    const int32_t *requests, const int32_t *lengths, uint16_t *output, float *scratch,
    int64_t batch, int64_t heads, int64_t capacity, int64_t cache_tokens, int64_t request_slots,
    float scale, int64_t parallel_rows, float *pv_workspace, int64_t pv_numel) {
    if (!pv_workspace || pv_numel < batch * heads * 256)
        return 4;
    if (!q || !k || !v || !table || !requests || !lengths || !output || !scratch || batch < 1 ||
        batch > 32 || (heads != 2 && heads != 6) || capacity < 1 || cache_tokens < 1 ||
        request_slots < 1 || !std::isfinite(scale) || (parallel_rows != 0 && parallel_rows != 1))
        return 1;
    // Validate on the caller before entering worker callbacks.
    for (int64_t row = 0; row < batch; ++row) {
        if (requests[row] < 0 || requests[row] >= request_slots || lengths[row] < 1 ||
            lengths[row] > capacity)
            return 2;
        const auto *indices = table + int64_t(requests[row]) * capacity;
        for (int64_t j = 0; j < lengths[row]; ++j)
            if (indices[j] < 0 || indices[j] >= cache_tokens)
                return 3;
    }
    int status[32] = {};
    auto row_call = [&](int64_t row, bool serial) {
        auto fn = serial || lengths[row] < 384 ? qwen36_gqa_decode_bf16_serial_with_workspace
                                               : qwen36_gqa_decode_bf16_parallel_with_workspace;
        status[row] =
            fn(q + row * heads * 256, k, v, table + int64_t(requests[row]) * capacity,
               output + row * heads * 256, scratch + (parallel_rows ? row * heads * capacity : 0),
               lengths[row], cache_tokens, heads, 256, scale,
               pv_workspace + (parallel_rows ? row * heads * 256 : 0), heads * 256);
    };
    if (parallel_rows) {
        // One launch for the batch; private scratch/output per request.
        // The called serial kernel never recursively enters the worker pool.
        kutacc::parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row)
                row_call(row, true);
        });
    } else {
        for (int64_t row = 0; row < batch; ++row)
            row_call(row, false);
    }
    for (int64_t row = 0; row < batch; ++row)
        if (status[row])
            return 10 + status[row];
    return 0;
}

extern "C" int qwen36_gqa_decode_batch(const uint16_t *q, const uint16_t *k, const uint16_t *v,
                                       const int32_t *table, const int32_t *requests,
                                       const int32_t *lengths, uint16_t *output, float *scratch,
                                       int64_t batch, int64_t heads, int64_t capacity,
                                       int64_t cache_tokens, int64_t request_slots, float scale,
                                       int64_t parallel_rows) {
    if (batch < 1 || batch > 32 || (heads != 2 && heads != 6))
        return 1;
    std::vector<float> pv(batch * heads * 256);
    return qwen36_gqa_decode_batch_with_workspace(
        q, k, v, table, requests, lengths, output, scratch, batch, heads, capacity, cache_tokens,
        request_slots, scale, parallel_rows, pv.data(), pv.size());
}
