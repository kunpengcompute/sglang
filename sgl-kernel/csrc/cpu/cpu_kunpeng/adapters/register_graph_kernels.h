#pragma once

#include "cpu_kunpeng/graph/type_traits.h"
#include <string>
#include <unordered_map>

std::unordered_map<std::string, DispatchFn>& dispatch_map();

struct KernelRegistrar {
    KernelRegistrar(const char* name, DispatchFn fn) {
        dispatch_map().emplace(name, fn);
    }
};

void register_graph_kernels();
