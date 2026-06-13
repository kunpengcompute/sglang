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

template <typename T, int Dims>
kutacc::Tensor<T, Dims> to_kutacc(at::Tensor t)
{
    TORCH_CHECK(t.dim() == Dims);
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        TORCH_CHECK(t.scalar_type() == at::kBFloat16);
    } else {
        TORCH_CHECK(t.scalar_type() == c10::CppTypeToScalarType<T>::value);
    }
    return kutacc::Tensor<T, Dims>(reinterpret_cast<T *>(t.data_ptr()), t.sizes().data(), t.strides().data());
}
