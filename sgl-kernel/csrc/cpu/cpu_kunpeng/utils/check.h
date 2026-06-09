#pragma once

#include <fmt/format.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace utils {
struct parameter_error : std::exception {
    std::string message;

    parameter_error(std::string message) : message(std::move(message)) {}

    const char* what() const noexcept { return message.c_str(); }
};
}  // namespace utils

#define TO_STR_(x) #x
#define TO_STR(x) TO_STR_(x)
#define SOURCE_LOCATION __FILE__ ":" TO_STR(__LINE__)

namespace internal {
template <typename... Args>
inline void parameter_check(bool condition, std::string source_location, Args&&... args)
{
    if (__builtin_expect(!(condition), 0)) {
        std::stringstream stream;
        stream << "PARAMETER_CHECK fail at " << source_location;
        if constexpr (sizeof...(args)) {
            stream << ", ";
            (stream << ... << std::forward<Args>(args));
        }
        throw utils::parameter_error{stream.str()};
    }
}

}  // namespace internal

#ifdef ENABLE_CHECK
#define PARAMETER_CHECK(condition, ...) (::internal::parameter_check((condition), SOURCE_LOCATION, ##__VA_ARGS__))
#else
#define PARAMETER_CHECK(condition, ...) (void(condition))
#endif
