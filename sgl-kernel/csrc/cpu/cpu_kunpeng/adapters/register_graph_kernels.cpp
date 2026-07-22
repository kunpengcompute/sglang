#include "register_graph_kernels.h"
#include "cpu_kunpeng/graph/capture.h"

std::unordered_map<std::string, DispatchFn>& dispatch_map() {
    static std::unordered_map<std::string, DispatchFn> map;
    return map;
}

void register_graph_kernels() {
    auto& reg = GraphOpRegistry::instance();
    for (const auto& [name, fn] : dispatch_map()) {
        reg.register_op(name, {fn});
    }
}
