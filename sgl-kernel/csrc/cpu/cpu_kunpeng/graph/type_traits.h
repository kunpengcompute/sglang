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

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using ScalarArg = std::variant<bool, int64_t, double, std::string>;
using DispatchFn = void (*)(std::vector<at::Tensor>&, const std::vector<ScalarArg>&);

namespace graph_detail {

template <typename T>
constexpr bool _is_tensor_v = std::is_same_v<std::decay_t<T>, at::Tensor>;

template <typename T>
struct _function_traits;

template <typename R, typename... Args>
struct _function_traits<R(*)(Args...)> {
    using return_type = R;
    using arg_tuple = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
};

template <typename R, typename... Args>
struct _function_traits<R(Args...)> {
    using return_type = R;
    using arg_tuple = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
};

// count_tensors_before<I, Tuple>: number of at::Tensor args before position I
template <size_t I, typename Tuple, size_t... Is>
constexpr size_t _count_tensors_before(std::index_sequence<Is...>) {
    return ((0 + ... + (Is < I && _is_tensor_v<std::tuple_element_t<Is, Tuple>> ? 1 : 0)));
}

template <size_t I, typename Tuple, size_t... Is>
constexpr size_t _count_scalars_before(std::index_sequence<Is...>) {
    return ((0 + ... + (Is < I && !_is_tensor_v<std::tuple_element_t<Is, Tuple>> ? 1 : 0)));
}

template <size_t I, typename Tuple>
constexpr size_t _tensor_offset_v =
    _count_tensors_before<I, Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});

template <size_t I, typename Tuple>
constexpr size_t _scalar_offset_v =
    _count_scalars_before<I, Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});

template <size_t I, typename Tuple>
auto _extract_arg(std::vector<at::Tensor>& tensors, const std::vector<ScalarArg>& scalars) {
    using ArgType = std::decay_t<std::tuple_element_t<I, Tuple>>;

    if constexpr (std::is_same_v<ArgType, at::Tensor>) {
        return tensors[_tensor_offset_v<I, Tuple>];
    } else if constexpr (std::is_same_v<ArgType, bool>) {
        return std::get<bool>(scalars[_scalar_offset_v<I, Tuple>]);
    } else if constexpr (std::is_same_v<ArgType, int64_t>) {
        return std::get<int64_t>(scalars[_scalar_offset_v<I, Tuple>]);
    } else if constexpr (std::is_same_v<ArgType, double>) {
        return std::get<double>(scalars[_scalar_offset_v<I, Tuple>]);
    } else if constexpr (std::is_same_v<ArgType, int>) {
        return static_cast<int>(std::get<int64_t>(scalars[_scalar_offset_v<I, Tuple>]));
    } else if constexpr (std::is_same_v<ArgType, float>) {
        return static_cast<float>(std::get<double>(scalars[_scalar_offset_v<I, Tuple>]));
    } else if constexpr (std::is_same_v<ArgType, std::string>) {
        // Optional string scalar (e.g. print_hash op name): fall back to empty
        // when the call site did not pass it during capture.
        return _scalar_offset_v<I, Tuple> < scalars.size() ? std::get<std::string>(scalars[_scalar_offset_v<I, Tuple>])
                                                           : std::string();
    } else {
        static_assert(sizeof(ArgType) == 0, "Unsupported argument type in graph dispatch");
    }
}

template <typename FnPtr, FnPtr fn, typename ArgTuple, size_t... Is>
void _dispatch_impl(std::vector<at::Tensor>& tensors, const std::vector<ScalarArg>& scalars,
                    std::index_sequence<Is...>) {
    if constexpr (std::is_void_v<typename _function_traits<FnPtr>::return_type>) {
        fn(_extract_arg<Is, ArgTuple>(tensors, scalars)...);
    } else {
        (void)fn(_extract_arg<Is, ArgTuple>(tensors, scalars)...);
    }
}

}  // namespace graph_detail

template <typename FnPtr, FnPtr fn>
struct DispatchAdapter {
    static void call(std::vector<at::Tensor>& tensors, const std::vector<ScalarArg>& scalars) {
        using Traits = graph_detail::_function_traits<FnPtr>;
        graph_detail::_dispatch_impl<FnPtr, fn, typename Traits::arg_tuple>(
            tensors, scalars, std::make_index_sequence<Traits::arity>{});
    }
};

template <typename FnPtr, FnPtr fn>
constexpr DispatchFn make_dispatch_v = &DispatchAdapter<FnPtr, fn>::call;

// Convenience: auto-registration helper that deduces FnPtr from fn
template <auto fn>
using DispatchAdapterFor = DispatchAdapter<decltype(fn), fn>;
