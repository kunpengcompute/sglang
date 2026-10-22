# Qwen 920F PR 整理与验证状态

## 版本与范围

| 项目 | 起点 |
|---|---|
| SGLang 上游 | `v0.5.11`, 初始`f82f42a07aa2c5fc0152bf90d35755f2153ca3af`；同步到`16cc02ca2c1f5593eb1267487161eeee828d404e` |
| KuTACC 上游 | `main`, `7b042b250a79851d19f4fefc80aa53d889c3ab71` |
| 历史性能/实现参考 | 冻结 job1935371；MTP OFF |
| 当前工作 | PR草稿准备；以远端分支和PR状态为准，本文历史测试不替代当前候选验收 |

遵循最新要求：冲突处优先使用上游新实现，先测试，再依据实际失败增加最小兼容处理。以历史上游 `152f6d6` 作为暂定三方合并祖先，不把它当作已证明的原始 fork commit。

## 集成结构

```text
SGLang 正常调度 / model.forward
  ├─ ModelSlim scheme → 动态激活量化 → KuTACC INT8 GEMM → BF16 输出
  ├─ Qwen 特有 GDN / GQA / BA / RoPE / gate 等结构适配
  └─ 通用 KunpengGraphRunner
       ├─ backend provider 提供动态输入、持久状态和形状契约
       ├─ AutomaticTorchCapture / GraphOpRegistry
       └─ 通用 C++ Graph 重放 → common_ops → KuTACC / Qwen native primitive
```

没有另建一套 Qwen 模型执行器。provider 管绑定与准入，正常 forward 决定运算顺序。图覆盖不等同于每个操作都被融合；通用 boxed ATen 重放仍可能产生临时结果再写入图输出。

## 代码分组

| 部分 | 位置 | 用途 |
|---|---|---|
| Qwen 后端适配 | `python/sglang/srt/hardware_backend/cpu_kunpeng/qwen/` | GQA/GDN、布局、prefill、provider、轻量权重映射 |
| 公共图能力 | `python/sglang/srt/graph/` 与 `sgl-kernel/csrc/cpu/cpu_kunpeng/graph/` | 输入/固定存储绑定、捕获、boxed ATen、统一重放 |
| W8A8 scheme | `layers/quantization/modelslim/schemes/cpu920f_w8a8_dynamic.py` | INT8 权重 pack、激活量化、scale 物理尾部、GEMM 调用 |
| 完整 checkpoint loader | `model_loader/unified_checkpoint.py` | 保留全部组件，使用原参数 loader；不依赖私有 hash receipt |
| Qwen native primitives | `sgl-kernel/csrc/cpu/cpu_kunpeng/qwen/` | GDN/GQA/布局原语，统一进入 common_ops |
| 通用小输出 GEMM 接口 | `sgl-kernel/.../matmul/bf16_small_output.cpp`、KuTACC `src/core/matmul/` | BA 所用 BF16 小 N 点积与 workspace 接口 |
| HBM | 公共 allocator/graph runner、`cpu_hbw/scoped_hbw_allocator.cpp` | 共享前缀、私有池、Storage 生命周期和 forward arena |
| 部署辅助（不进入PR） | 独立工具包 | 构建、NUMA副本、启动、选核、工具调用与基准测试均已移出仓库 |

## 保留上游的内容

- 最新 DeepSeek/GLM 模型路径与其公共分布式实现；没有用旧部署的整份目录覆盖。
- 上游 AllReduce profile 标记及专家权重压缩/分类计数。
- 上游 MTP position 的 clone 与 `-1` 修复；不还原旧版 alias 行为。
- CPU/CUDA 构建入口维持独立；KuTACC C++ 标准库默认值维持上游，只加可配置项。

## 对部署代码做的清理

- 删除默认关闭、依赖旧私有 DSO 的固定 M attention gate 实验支路。
- 重建 native 源码依赖，不提交历史 `.o`/`.so` 或复制 object 的编译方案。
- 用轻量名称/分片规则保留真实加载行为，移除默认启动对 67 MB expected plan、全模型重复 hash、私有 receipt 路径的要求。
- 将原 HBW 构造时的外置 monkey patch 收进可选的原生分配策略；默认上游行为保持不变。
- 部署工具不依赖特定用户目录、指定节点或旧实验窗口。

## 实编发现的上游接口不匹配

未经兼容处理的最新上游组合在作业1936218编译失败：SGLang使用了公开KuTACC main不存在的近期可选接口。增加显式CMake能力探测开关；公共默认严格构建保持原样。独立Qwen构建工具显式允许缺失可选API，保留有能力时的原函数体，缺失时明确报错。没有回退整份旧SGLang或改写DeepSeek算法。

覆盖DSA indexer、full allgather、broadcast、MLA CP read/reduce、批量BF16 GEMM、独立RDMA控制域及MoE combine注册容量参数。批量BF16与控制域两项还通过源码与公开头文件的符号对照发现；最终native与wheel已完成实编。

## 最终基本验收（2026-10-21）

编译节点cn22523；最终对照在cn22526同一租约、同一512核表顺序执行，作业1936599成功。两版本之间未重新调频，无并行编译。输入/输出均为1024 token，C1/C8各两轮，每版本18请求全部完成。

| 并发 | 轮次 | 旧版TTFT ms | 新版TTFT ms | 旧版TPOT ms/token | 新版TPOT ms/token |
|---:|---|---:|---:|---:|---:|
| 1 | 1 | 3340.10 | 3258.14 | 28.49 | 28.00 |
| 1 | 2 | 570.98 | 572.44 | 28.48 | 28.02 |
| 1 | 平均 | 1955.54 | 1915.29 | 28.48 | 28.01 |
| 8 | 1 | 3477.34 | 3506.25 | 53.23 | 53.26 |
| 8 | 2 | 3631.92 | 3481.74 | 52.43 | 52.27 |
| 8 | 平均 | 3554.63 | 3493.99 | 52.83 | 52.76 |

首轮C1含该形状首次图捕获成本，不能把两轮均值与历史暖态TTFT直接比较。未额外清空缓存。两轮短测未观察到明显性能回退；0.13%的C8差异不能当作已证明的收益。

新版109568次、旧版113152次频率读数均覆盖512计算核，范围1.999996～2.000059GHz，越界/缺失均为0。这是周期采样证据，不是全周期硬件认证。最终两版进程退出、HBM hugepages恢复均通过。

| 检查 | 结果及范围 |
|---|---|
| 冻结Python源码 | 2576文件SHA256一致 |
| KuTACC完整构建 | 作业1936145通过，版本26.2.0 |
| SGLang CPU native / wheel | 作业1936359成功编译产出；其后续短测亲和性失败，已在独立1936374修正并通过10项检查 |
| 原生算子 | BA M1/7/33/128、INT8投影M16/129、零/极小量化、融合SiLU/Norm、Storage别名生命周期通过；Storage短测为DDR参考backing |
| checkpoint路由 | 97规则覆盖冻结2079源路由；不代表完整精度等价 |
| TP16与普通对话 | 新旧版均通过 |
| thinking | 默认关闭；请求显式开启后返回reasoning；开启测试在128token处截断，只验证开关 |
| 工具调用 | 强制multiply调用、客户端计算、回传结果、最终回答56088，完整往返通过 |
| 实际输出长度 | 每版18请求均输出1024token；最长连续零token旧4/新3 |
| NUMA私有ELF | 完整依赖闭包与绑定后复制；另一次V2运行关键DSO驻留采样16rank均近端，不等于所有激活近端 |

## 合入前仍需关注

1. **完整精度尚未验收。** 最终18组相同输入的长输出，新旧版0/18逐token完全一致，首次分叉在第67～203个token（从0计）。原因尚未隔离，不能直接认定都是正常量化舍入。代表算子和基本功能通过不替代模型质量/logits误差验收。
2. 公共graph/allocator改动应独立审查，特别是存储别名、重放缓存键和共享窗口退出顺序。当前验证不能覆盖其他模型/设备的全部分支。
3. 已使用匹配920F Torch/KuPL环境实跑；部署仍须保持相同ABI和厂商补丁，详见工具版本清单。
4. 视觉/MTP保留结构与参数，但执行不在此次验收范围。多请求prefill仍存在继承的eager回落；不能声称完整图覆盖。
5. 本轮仅C1/C8短测，未重新验收长上下文/C16/C32。所有激活的实际物理放置尚无完整账本；embedding保留DDR。
6. 两仓库分别提交草稿PR供评审；SGLang依赖配套KuTACC分支。构建和额外部署材料说明见`kunpeng_qwen_integration.md`第9节。

## 2026-10-22 PR范围整理（区别于上述已实跑版本）

- 整包15个部署/选频/压测/工具调用文件已移出PR；原文件及整理前补丁保存在仓库外审计目录。PR不携带站点操作、性能测试或工具调用测试脚本。
- 清除已有函数的纯格式变化；weight_mapping保留相同97规则，以每行一条规则呈现，JSON值完全一致。
- KuTACC M96改为默认关闭的显式通用调优；独立Qwen环境开启。capture输入别名及TorchDispatch屏蔽改为provider显式能力；forward arena增加Qwen开关；移除运行期加载外部replay检查脚本的试验入口。
- 无provider时不执行Qwen专用decode/prefill覆盖统计。C++ Graph执行/plan_memory算法没有重写；新增plan描述属于只读诊断。
- 已通过主机C++实际分流函数检查（SME计算callee替身）、Python实际捕获/runner入口的接口替身检查、语法及diff检查。它们验证开关和默认路径，不等于DeepSeek模型回归或920F重新实编。
- 上表性能对应整理前已实跑快照。线上冻结部署不随本地文件移动或门控修改而更新；本次整理后的源码尚需独立构建与有限模型回归后才能称为新的已验收版本。

## 2026-10-22 同步上游16cc02ca2

- `graph/adapters.py`两处相邻新增注册产生文本冲突；同时保留上游`bf16_packed_gemm_f32_kunpeng`及Qwen的small-output/workspace注册，不改写上游FP32 router或MoE实现。
- 上游新增FP32 split-K入口调用的`kutacc::bf16_packed_gemm_f32`尚不在本次公开KuTACC起点中。沿用已有显式兼容开关添加能力探测：有API时保留原函数体；缺API且开启兼容构建时，该可选入口调用即报错。默认严格构建不变，不提供伪FP32替代计算。
- 其余自动合并保留上游PP per-key传输、MTP/Indexer与FP32 MoE gate更新；同步后的完整原生构建和DeepSeek/Qwen实机回归仍待补齐。
