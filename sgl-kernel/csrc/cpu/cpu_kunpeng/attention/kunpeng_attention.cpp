/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#include <torch/extension.h>
#include <kutacc.h>
#include <kupl.h>
#include "sgl_kernel_ops.h"
#include <fstream>
#include <cstdint>

template <typename T, int Dims>
kutacc::Tensor<T, Dims> to_kutacc(at::Tensor t) {
    TORCH_CHECK(t.dim() == Dims);
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        TORCH_CHECK(t.scalar_type() == at::kBFloat16);
    } else {
        TORCH_CHECK(t.scalar_type() == c10::CppTypeToScalarType<T>::value);
    }
    return kutacc::Tensor<T, Dims>(
        reinterpret_cast<T*>(t.data_ptr()),
        t.sizes().data(),
        t.strides().data()
    );
}

at::Tensor flash_mla_meta_create_kunpeng() {
    kutacc::FlashMLAMetaHandle meta;
    kutacc::flash_mla_meta_create(meta);
    int64_t ptr_val = reinterpret_cast<int64_t>(meta);
    return at::tensor(ptr_val, at::dtype(at::kLong));
}

at::Tensor flash_mla_meta_destroy_kunpeng(at::Tensor meta_tensor) {
    TORCH_CHECK(meta_tensor.defined(), "meta_tensor is not defined");
    TORCH_CHECK(meta_tensor.scalar_type() == at::kLong,
                "meta_tensor must be int64 type to store pointer");
    TORCH_CHECK(meta_tensor.numel() == 1,
                "meta_tensor must be a scalar tensor");

    int64_t ptr_val = meta_tensor.item<int64_t>();
    if (ptr_val != 0) {
        auto meta = reinterpret_cast<kutacc::FlashMLAMetaHandle>(ptr_val);
        kutacc::flash_mla_meta_destory(meta);
        meta_tensor.fill_(0);
    }
    return meta_tensor;
}

// Wrapper 函数
void flash_mla_dense_decode_kunpeng(
    at::Tensor q,
    at::Tensor kcache,
    c10::optional<at::Tensor> vcache,
    at::Tensor block_table,
    at::Tensor seqlens_kv,
    at::Tensor o,
    at::Tensor softmax_lse,
    double softmax_scale,
    bool is_causal,
    at::Tensor extra_buffer,
    c10::optional<at::Tensor> meta
) {

    auto print_tensor_shape = [](const std::string& name, const at::Tensor& t) {
        if (t.defined()) {
            std::cout << name << " shape: " << t.sizes() << std::endl;
        } else {
            std::cout << name << " is undefined/None" << std::endl;
        }
    };

    auto kt_q      = to_kutacc<bfloat16_t, 4>(q);
    auto kt_kcache = to_kutacc<bfloat16_t, 3>(kcache);

    std::optional<kutacc::Tensor<bfloat16_t, 3>> kt_vcache = std::nullopt;
    if (vcache.has_value()) {
        kt_vcache = kutacc::Tensor<bfloat16_t, 3>(
            reinterpret_cast<bfloat16_t*>(vcache->data_ptr<at::BFloat16>()),
            vcache.value().sizes().data(),
            vcache.value().strides().data()
        );
    }

    auto kt_block_table = to_kutacc<int, 2>(block_table);
    auto kt_seqlens_kv = to_kutacc<int, 1>(seqlens_kv);
    auto kt_o = to_kutacc<bfloat16_t, 4>(o);
    auto kt_softmax_lse = to_kutacc<float, 3>(softmax_lse);

    void* extra_ptr = extra_buffer.data_ptr();
    kutacc::FlashMLAMetaHandle meta_handle = reinterpret_cast<kutacc::FlashMLAMetaHandle>(meta.value().item<int64_t>());

    kutacc::flash_mla_dense_decode(
        kt_q,
        kt_kcache,
        kt_vcache,
        kt_block_table,
        kt_seqlens_kv,
        kt_o,
        kt_softmax_lse,
        static_cast<float>(softmax_scale),
        is_causal,
        extra_ptr,
        meta_handle
    );
}

int64_t flash_mla_dense_decode_sched_kunpeng(
    const at::Tensor& seqlens_kv,
    int64_t seqlen_q,
    int64_t num_heads_q,
    int64_t head_dim,
    int64_t head_dim_v,
    int64_t page_block_size,
    bool is_kv_packed,
    c10::optional<at::Tensor> meta
) {
    kutacc::Tensor<int, 1> kt_seqlens = to_kutacc<int, 1>(seqlens_kv);
    kutacc::FlashMLAMetaHandle meta_handle= reinterpret_cast<kutacc::FlashMLAMetaHandle>(meta.value().item<int64_t>());
    int64_t extra_bytes_sizes = 0;

    kutacc::flash_mla_dense_decode_sched(
        kt_seqlens,
        seqlen_q,
        num_heads_q,
        head_dim,
        head_dim_v,
        page_block_size,
        is_kv_packed,
        extra_bytes_sizes,
        meta_handle
    );
    return extra_bytes_sizes;
}

std::tuple<int64_t, int64_t> get_flash_attention_block_kunpeng() {
    return kutacc::get_flash_attention_block();
}

int64_t get_flash_attention_thread_num() {
    return kutacc::get_thread_num();
}

void flash_attention_k_block_pack_kunpeng(
    int64_t kv_len,
    int64_t num_heads,
    int64_t qk_head_dim,
    int64_t output_len,
    int64_t input_stride0,
    int64_t input_stride1,
    at::Tensor input,
    at::Tensor output
) {
    bfloat16_t* input_ptr  = reinterpret_cast<bfloat16_t*>(input.data_ptr());
    bfloat16_t* output_ptr = reinterpret_cast<bfloat16_t*>(output.data_ptr());

    kutacc::flash_attention_k_block_pack(
        kv_len, num_heads, qk_head_dim,
        output_len, input_stride0, input_stride1,
        input_ptr, output_ptr
    );
}

void flash_attention_v_block_pack_kunpeng(
    int64_t kv_len,
    int64_t num_heads,
    int64_t vo_head_dim,
    int64_t output_len,
    int64_t input_stride0,
    int64_t input_stride1,
    at::Tensor input,
    at::Tensor output
) {
    bfloat16_t* input_ptr  = reinterpret_cast<bfloat16_t*>(input.data_ptr());
    bfloat16_t* output_ptr = reinterpret_cast<bfloat16_t*>(output.data_ptr());

    kutacc::flash_attention_v_block_pack(
        kv_len, num_heads, vo_head_dim,
        output_len, input_stride0, input_stride1,
        input_ptr, output_ptr
    );
}

void flash_attention_kunpeng(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor pack_attn_q,
    at::Tensor pack_attn_k,
    at::Tensor pack_attn_v,
    at::Tensor attn_s,
    at::Tensor attn_out_block_old,
    at::Tensor attn_out_block_new,
    at::Tensor attn_max_block_old,
    at::Tensor attn_max_block_new,
    at::Tensor attn_base_block_old,
    at::Tensor attn_base_block_new,
    bool causal,
    double softmax_scale,
    at::Tensor query_start_loc,
    at::Tensor key_start_loc,
    int64_t chunked_prefill_size,
    std::vector<int64_t> seq_lens,
    std::vector<int64_t> cur_lens
) {
    auto kt_q                = to_kutacc<bfloat16_t, 3>(q);
    auto kt_k                = to_kutacc<bfloat16_t, 3>(k);
    auto kt_v                = to_kutacc<bfloat16_t, 3>(v);
    auto kt_out              = to_kutacc<bfloat16_t, 3>(out);

    auto kt_pack_attn_q      = to_kutacc<bfloat16_t, 2>(pack_attn_q);
    auto kt_pack_attn_k      = to_kutacc<bfloat16_t, 3>(pack_attn_k);
    auto kt_pack_attn_v      = to_kutacc<bfloat16_t, 3>(pack_attn_v);

    auto kt_attn_s           = to_kutacc<float, 2>(attn_s);
    auto kt_attn_out_old     = to_kutacc<float, 3>(attn_out_block_old);
    auto kt_attn_out_new     = to_kutacc<float, 3>(attn_out_block_new);
    auto kt_attn_max_old     = to_kutacc<float, 2>(attn_max_block_old);
    auto kt_attn_max_new     = to_kutacc<float, 2>(attn_max_block_new);
    auto kt_attn_base_old    = to_kutacc<float, 2>(attn_base_block_old);
    auto kt_attn_base_new    = to_kutacc<float, 2>(attn_base_block_new);

    auto kt_query_start_loc  = to_kutacc<int, 1>(query_start_loc);
    auto kt_key_start_loc    = to_kutacc<int, 1>(key_start_loc);

    kutacc::flash_attention(
        kt_q,
        kt_k,
        kt_v,
        kt_out,
        kt_pack_attn_q,
        kt_pack_attn_k,
        kt_pack_attn_v,
        kt_attn_s,
        kt_attn_out_old,
        kt_attn_out_new,
        kt_attn_max_old,
        kt_attn_max_new,
        kt_attn_base_old,
        kt_attn_base_new,
        causal,
        softmax_scale,
        kt_query_start_loc,
        kt_key_start_loc,
        chunked_prefill_size,
        seq_lens,
        cur_lens
    );
}

void varlen_attention_kunpeng(
    at::Tensor q,                    // [total_q_tokens, num_heads, qk_head_dim]
    at::Tensor k,                    // [total_kv_tokens, num_heads, qk_head_dim]
    at::Tensor v,                    // [total_kv_tokens, num_heads, vo_head_dim]
    at::Tensor out,                  // [total_q_tokens, num_heads, vo_head_dim]
    bool causal,
    double softmax_scale,
    at::Tensor query_start_loc,
    at::Tensor key_start_loc
) {
    auto kt_q = to_kutacc<bfloat16_t, 3>(q);
    auto kt_k = to_kutacc<bfloat16_t, 3>(k);
    auto kt_v = to_kutacc<bfloat16_t, 3>(v);
    auto kt_out = to_kutacc<bfloat16_t, 3>(out);

    auto kt_query_start_loc = to_kutacc<int, 1>(query_start_loc);
    auto kt_key_start_loc = to_kutacc<int, 1>(key_start_loc);

    kutacc::varlen_attention(
        kt_q,
        kt_k,
        kt_v,
        kt_out,
        causal,
        softmax_scale,
        kt_query_start_loc,
        kt_key_start_loc
    );
}