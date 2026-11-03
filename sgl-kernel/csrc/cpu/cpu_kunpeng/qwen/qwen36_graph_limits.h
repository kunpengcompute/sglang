#pragma once

#include <cstdint>

// Compiled wrapper capacities. Python capabilities may select lower startup
// limits; changing these capacities requires native validation, not a new tile.
namespace qwen_graph_limits {
constexpr int64_t decode_tokens = 128;
constexpr int64_t gated_norm_rows = 3 * decode_tokens;
}  // namespace qwen_graph_limits
