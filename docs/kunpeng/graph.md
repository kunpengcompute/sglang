# Kunpeng 计算图（Graph Capture / Replay）

## 概述

Kunpeng 计算图面向 920F 平台：对模型一次 forward 进行捕获（capture），把算子序列、张量生命周期与内存布局在编译期固定，之后相同形状的请求直接回放（replay），消除 Python 分发与内存分配开销。

特性：零堆分配回放、确定性内存规划、异构内存（普通内存 + SHM）、decode/extend/idle 三种模式分别捕获独立图。

## 工作原理

流程分三段：

1. **捕获**：`capture(inputs, fixed)` 上下文内执行一次真实 `model.forward`。已注册算子只记录参数与张量元数据，不执行 kernel。未注册算子的行为分两类：
   - 纯 view / slice / transpose 等**共享 storage** 的算子：照常执行，没有问题
   - 产生**新内存**的算子（`torch.empty`、`torch.cat`、`torch.matmul` 等）：其输出张量一旦被后续已注册算子消费，捕获会因 `lookup_or_register` 断言失败（`non-return-value parameter tensor not registered`）而崩溃。因此模型 forward 热路径中所有产生新内存的算子**必须**走已注册图算子（见"换模型适配检查"）
2. **编译**：`finalize(outputs, external_pool, external_shm_pool)` 做生命周期分析（`compute_death_ops`）、内存规划（`plan_memory`）、预计算回放数据（`precompute_replay`）
3. **回放**：`graph.run(inputs)` 逐算子调用预解析的 `DispatchFn`，张量全部来自预分配池，零堆分配。**只执行捕获时记录过的算子**——未注册的通信/集合算子不会在回放时执行，会造成结果错误

核心数据结构：`StorageBuf`（内存块，含 `memory_type` 与生命周期）、`TensorView`（指向 storage 的视图）、`OpRecord`（算子记录）、`MemoryPool`（预分配池，Graph 持有 REGULAR 与 SHM 两个）、`GraphOpRegistry`（op 名 → DispatchFn 注册表）。

图内的张量分三类，由 `capture(inputs=..., fixed=...)` 的传参与算子输出决定：

- **输入张量**：`capture(inputs=[...])` 传入（如 `input_ids`、`positions`）。**内容、形状、`data_ptr` 每次 `graph.run([...])` 时都可变**；约束是必须**直接传给算子**，不能先经过 `slice` 等转换视图操作。不参与内存规划，回放时换入
- **固定张量**：`capture(fixed=[...])` 传入（如权重、`cos_sin_cache`、KV cache buffer）。**内容可变**（可原地更新，如 KV cache 写入），但**形状和 `data_ptr` 不可变**；允许在其上做 `slice` / `view` 等视图转换。内存由外部持有，不参与内存规划，回放时直接读取
- **中间张量**：图内算子产出的激活值。**形状不可变**（捕获时由 `shape_infer` 决定），**`data_ptr` 由图决定**（`plan_memory` 打包进 REGULAR 或 SHM 池，回放时从池内复用、零堆分配；生命周期不重叠的中间张量可复用同一块池内存）；允许视图转换

内存规划：按 `memory_type` 把入池 storage 分成 REGULAR/SHM 两组，各自独立做 interval packing（按 size 降序、同 size 按 id 升序，全序确定）。REGULAR 池可用外部池（如 HBW）或内部 `torch::empty`；SHM 池必须外部提供。确定性是正确性前提——no-copy 通信依赖所有 rank 对同一 storage 的池内 offset 完全一致。

图按 `(forward_mode, total_tokens, batch_size, is_pp_graph)` 为 key 做 LRU 缓存（默认容量 4），批大小（序列数）变化会触发新捕获。batch_size 必须参与 key：`last_tokens`、MTP pad/unpad 等按序列数固定输出形状的算子要求同一张图的 batch 一致；is_pp_graph 区分 PP 代理张量是否作为图输入。decode/idle 模式的 REGULAR 张量可用 HBW 池，extend 不用。

## 开启与配置

前提：920F 平台（`SGLANG_USE_CPU_920F`）且 `SGLANG_ENABLE_GRAPH_CAPTURE=1`。

其他环境变量：

- `SGLANG_ENABLE_HBW_POOL`（默认 0）：HBW 池开关，decode/idle 中间张量放入 HBW
- `SGLANG_ENABLE_GRAPH_PROFILE`（默认 0）：逐 op 计时输出 JSONL
- `SGLANG_KUNPENG_GRAPH_CACHE_SIZE`（默认 4）：图 LRU 容量
- `SGLANG_KUNPENG_ENABLE_SHM_FENCE`（默认 0）：是否在 allreduce/allgather 前插入显式 fence 图算子

启动后日志出现 `[graph] captured mode=DECODE, total_tokens=N` 表示捕获成功，相同 token 数的请求直接回放。若捕获崩溃，检查是否命中"未注册算子产生新内存"约束（见下文）。

### 最小可运行示例

将下列代码保存为 `graph_capture_demo.py` 后可直接运行（`python graph_capture_demo.py`，需已编译安装 sgl-kernel）。演示捕获 → 编译 → 回放全流程，并校验回放结果与 eager 参照逐位一致：

```python
#!/usr/bin/env python3
"""Kunpeng 计算图最小可运行示例。"""
import torch

# 引入图模块：自动注册内置算子并绑定 C++ DispatchFn
from sglang.srt.graph import capture, finalize, is_capturing
from sglang.srt.graph import ops as kunpeng


def eager_reference(x, weight):
    """与图内完全相同的计算（eager 路径），作为正确性参照。"""
    y = torch.empty_like(x)
    torch.ops.sgl_kernel.rmsnorm_kunpeng(x, weight, 1e-6, y)
    z = torch.cat([y, y], dim=-1)
    torch.ops.sgl_kernel.mul_scalar_add_kunpeng(z, z, 2.0)  # z = z + 2*z
    return z


def main():
    torch.manual_seed(0)
    dtype = torch.bfloat16

    # 图输入（内容/形状/地址每次 run 可换，必须直接传给算子，不能先 slice 等视图操作）
    x = torch.randn(4, 8, dtype=dtype)
    # 固定张量（内容可原地更新，但形状和地址不可变；可对其 slice/view）
    weight = torch.randn(8, dtype=dtype)

    # ── 1. 捕获：只记录算子与张量元数据，不执行 kernel ──
    with capture(inputs=[x], fixed=[weight]):
        assert is_capturing()
        y = kunpeng.rmsnorm_kunpeng(x, weight, 1e-6)
        z = kunpeng.cat_kunpeng(y, y, -1)
        kunpeng.mul_scalar_add_kunpeng(z, z, 2.0)  # inplace，无返回值

    # ── 2. 编译：生命周期分析 + 确定性内存规划 + 预计算回放数据 ──
    # 可传 external_pool（如 HBW/SHM 大块）让中间张量落在指定内存
    graph = finalize([z])

    # ── 3. 回放：多次执行验证可复用，结果与 eager 参照逐位一致 ──
    ref = eager_reference(x, weight)
    for i in range(3):
        out, = graph.run([x])
        assert torch.equal(out, ref), f"第 {i} 次回放结果不一致"
        print(f"run {i}: 与 eager 参照一致, shape={tuple(out.shape)}")
    print("OK: 捕获/编译/回放全流程通过")


if __name__ == "__main__":
    main()
```

## 注册自定义算子（开发者）

所有可捕获算子集中在 `python/sglang/srt/graph/adapters.py` 注册。

```python
def register_op(name, shape_infer, eager_fn=None, shm_fn=None):
```

`shape_infer` 返回输出形状列表 `[(shape, dtype), ...]`，空列表表示 inplace；`eager_fn` 是图外回退执行；`shm_fn` 可选，返回需要放 SHM 的张量列表，收到的是 kernel 展开参数（输入张量 + 输出张量 + 标量，顺序与 C++ 签名一致）。

普通算子示例（RMSNorm）：

```python
def _setup_rmsnorm_kunpeng():
    def shape_infer(acts, weights, eps):
        return [(acts.shape, acts.dtype)]

    def eager_fn(acts, weights, eps):
        out = torch.empty(acts.shape, dtype=acts.dtype)
        torch.ops.sgl_kernel.rmsnorm_kunpeng(acts, weights, eps, out)
        return out

    register_op('rmsnorm_kunpeng', shape_infer, eager_fn)
```

inplace 通信算子（allreduce，输入需要 SHM）：

```python
register_op('shm_allreduce_kunpeng', shape_infer, eager_fn,
            shm_fn=lambda input: [input])
```

带输出通信算子（batched_allgather，输入输出都要 SHM）：

```python
register_op('shm_batched_allgather_kunpeng', shape_infer, eager_fn,
            shm_fn=lambda input, output, comm_size: [input, output])
```

C++ 侧在 `sgl-kernel/csrc/cpu/cpu_kunpeng/adapters/` 下新建适配器文件，用 `KernelRegistrar` + `make_dispatch_v` 注册即可（CMake `GLOB_RECURSE` 自动编译）：

```cpp
#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void xxx_kunpeng(at::Tensor input, at::Tensor out, int64_t scalar);

static KernelRegistrar _r("xxx_kunpeng",
    make_dispatch_v<decltype(&xxx_kunpeng), &xxx_kunpeng>);
```

### 常用 torch 算子的替换对照

forward 热路径中产生新内存的 torch 算子需替换为对应图算子：

- `x = torch.empty([numel], dtype)` → `x = kunpeng.alloc_buffer(numel, dtype=dtype)` 生成形状 `(numel,)` 的一维张量；`dtype` 关键字参数默认 `torch.uint8`（此时 numel 即字节数）。buffer 从图池分配，回放时零分配
- `x = torch.empty(shape, dtype)` → `x = kunpeng.alloc_buffer(numel, dtype=dtype).view(shape)`
- `x = torch.cat([a, b], dim)` → `x = kunpeng.cat_kunpeng(a, b, dim)`
- `y = x.contiguous()` → `y = kunpeng.contiguous_kunpeng(x)`
- `y = x.clone()` → `y = kunpeng.alloc_buffer(numel, dtype=dtype).view(shape)` 然后 `kunpeng.copy_kunpeng(y, x)`
- `x = torch.zeros(shape, dtype)` → `x = kunpeng.alloc_buffer(numel, dtype=dtype).view(shape)` 然后 `kunpeng.zero_(x)`
- `y = x.to(dtype)` → `y = kunpeng.alloc_buffer(x.numel(), dtype=dtype)` 然后 `kunpeng.copy_kunpeng(y, x)` 拷贝

以下算子**共享 storage**，不需要替换：`view`、`slice`、`tensor[n:m]`、`transpose`、`permute`、`unsqueeze`、`squeeze`。

部分算子不一定共享 storage，视情况替换：`reshape`、`to`、`contiguous`。

性能问题：目前 cat_kunpeng、contiguous_kunpeng、copy_kunpeng、zero_ 回放直接调用 torch 算子，有较大开销。

## 异构内存（SHM）

`shm_allreduce_kunpeng` / `shm_batched_allgather_kunpeng` 采用双路径：输入（和输出）已在 SHM 时走 no-copy 路径——直接以输入为共享 buffer 原地通信，省两次 memcpy 且不再惰性创建内部 SHM buffer；否则走老 copy 路径（eager 兼容）。

在 `adapters.py` 用 `shm_fn` 标记：新输出张量的 storage 以 SHM 注册；已有输入张量的 storage 被升级为 SHM（`upgrade_storage_memory_type`，SHM 优先级高于 REGULAR）。

SHM 池：`model_runner` 首次捕获后把 SHM bump 分配器剩余全部字节申请为图池（与 HBW"取剩余"一致）。SHM 只分配不释放，图池大小在首次捕获固定，后续更大 token 数的图若超池会报错。no-copy 要求输入输出均在 SHM；单端 SHM 无法走 no-copy。

## 调试

`SGLANG_ENABLE_GRAPH_PROFILE=1` 会把每次回放的逐算子耗时追加写入 `sglang_graph_rank{RANK}.jsonl`（目录由 `SGLANG_TORCH_PROFILER_DIR` 指定）。`scripts/cpu_kunpeng/` 下提供三个分析脚本：

**转 Chrome tracing（时间线视图）**：

```bash
python scripts/cpu_kunpeng/stats_to_trace.py sglang_graph_rank0.jsonl sglang_graph_rank1.jsonl ... trace.json
```

生成的 `trace.json` 用 chrome://tracing 或 Perfetto 打开，每个 rank 一条线程，查看各算子的起止耗时。建议最多分析 16 个文件避免文件过大。

**生成算子统计 CSV（按 forward_mode 分段，含 count/min/max/avg/total/percent）**：

```bash
python scripts/cpu_kunpeng/stats_to_csv.py sglang_graph_rank0.jsonl graph_stats.csv
```

**对比两份 CSV 的 extend/decode 段差异**：

```bash
python scripts/cpu_kunpeng/stats_csv_compare.py run1.csv run2.csv -o compare
# 生成 compare_extend_summary.csv / compare_decode_summary.csv
```

把 `sgl-kernel/csrc/cpu/cpu_kunpeng/graph/capture.h` 的 `kGraphDebugPrint` 改为 `true` 重新编译，可打印捕获期算子视图、plan_memory 放置（offset/born/death）、回放期张量信息。

## 关键代码位置

- 图引擎（C++）：`sgl-kernel/csrc/cpu/cpu_kunpeng/graph/`
- 图算子适配器（C++）：`sgl-kernel/csrc/cpu/cpu_kunpeng/adapters/`
- Python 图模块：`python/sglang/srt/graph/`
- 图集成入口：`python/sglang/srt/model_executor/model_runner.py`（`_kunpeng_graph_forward`）
- SHM 内存管理：`sgl-kernel/csrc/cpu/cpu_kunpeng/memory/kunpeng_shm.cpp`
- SHM 通信 kernel：`sgl-kernel/csrc/cpu/cpu_kunpeng/comm/`
