# Kunpeng Qwen 适配的合入边界

> **2026-11-06，PR #301 attached MTP 与 draft 图捕获**：在 #299 的显式图路径之上，
> MTP 层改为 target-owned 的 attached 组件：EAGLE draft 借用 target 的 runner、
> `embed_tokens` 与 `lm_head`，不再加载第二份 checkpoint；draft decode 与
> draft-extend 首次纳入图捕获；verify 的 GDN 递推与 causal conv 融合为单次调用。
> #299 中“不启用 Provider／boxed 桥接”的边界只对 target 文本路径继续成立，attached
> draft 单独使用 `QwenAttachedMTPGraphProvider`、`AutomaticTorchCapture` 与
> `aten_boxed` 重放；下文相关段落属于旧实现记录，不能作为本版启动要求。

本次适配内容（相对 #299）：

- **模型挂载**：`models/qwen3_5.py` 的显式类在 modelslim + TP16 + PP1 时创建
  `self.mtp`（`Qwen3_5ForCausalLMMTP`，共享 target 的 `embed_tokens` / `lm_head`），
  经 `ModelCapabilities` 与 `mtp_bridge.attach_mtp` 登记为 target-owned 组件；
  `UnifiedCheckpointLoader` 用一份整模型 loader 加载含 `mtp.*` 的 checkpoint。
  MTP 是否启用由 `--speculative-*` 参数决定，不再依赖 config.json 的能力清单。
- **draft 借用 target**：`mtp_bridge.attached_target_for_draft` /
  `configure_borrowed_draft` / `bind_borrowed_model` / `validate_borrowed_pools`
  校验并复用 target 的 req pool、allocator 与 HBW 池；draft 只有 1 层独立 KV。
- **draft 图捕获**：`qwen/graph_provider.py` 新增 `QwenAttachedMTPGraphProvider`。
  draft decode 按每步 attention backend 各捕获一张图（bs 1..32）；draft-extend 覆盖
  ≤128 行，`SGLANG_KUNPENG_QWEN_DRAFT_EXTEND_PAD=1`（默认）把 ragged 行数规整为
  `bs × speculative_num_draft_tokens`，行映射 `requests/lengths` 作为普通动态输入，
  KV 写入按存活 token 前缀裁剪。总开关 `SGLANG_KUNPENG_QWEN_DRAFT_GRAPH_CAPTURE`。
- **verify 融合**：`qwen/ops.py` 新增 `qwen_gdn_verify` / `qwen_conv_verify`，
  `SGLANG_KUNPENG_GDN_VERIFY_FUSED=1`（默认）把全部 MTP verify 步压成一次 GDN 递推与
  一次 causal conv；verify 行元数据在 `prepare_for_verify` 里只构建一次。
- **MTP 权重路径**：`qwen/mtp_pre_fc.py::prepare_mtp_fc` 在权重落位后预打包 pre-fc，
  `fc_linear` 走 native BF16 dense GEMM；`bf16_lm_head` 增加 `prepare_weight` /
  `try_linear`，在 `KUTACC_BF16_SMALL_M_FULL_K=1` 且 tiling 一致时把 M1..128 的
  lm_head 留在 packed GEMM（draft 每步是 M1），否则回落到原 M1 ATen mm / M≥2 packed。
- **公共图能力**：`install_boxed_bridge` 注册 `aten_boxed`（仅 draft 的自动捕获需要）；
  `capture_options` 可从 attention backend 回落；borrowed runner 只接受 draft provider
  支持的形状，其余仍 eager 回落并记录原因。
- **加载边界**：checkpoint 不含 `visual.*` 时不加载视觉张量，也不计入覆盖度校验。

本 PR 新增的环境变量：

| 变量 | 默认 | 作用 |
|---|---|---|
| `SGLANG_KUNPENG_QWEN_DRAFT_GRAPH_CAPTURE` | `1` | `0` = draft 保持 eager（默认捕获 attached MTP 的 draft decode 链与 draft-extend） |
| `SGLANG_KUNPENG_QWEN_DRAFT_EXTEND_PAD` | `1` | `1` = 把 ragged draft-extend 规整为 `bs × speculative_num_draft_tokens` 行 |
| `SGLANG_KUNPENG_GDN_VERIFY_FUSED` | `1` | `1` = 全部 MTP verify 步合并为一次 GDN 递推；`0` = 逐步 decode，此时 verify 不捕获图、整段 eager |

`SGLANG_KUNPENG_QWEN_DRAFT_EXTEND_PAD` 是 `SGLANG_KUNPENG_QWEN_DRAFT_GRAPH_CAPTURE`
的子开关，只在 draft 图捕获开启时被 provider 读取：捕获关闭时 draft decode 与
draft-extend 全程 eager，ragged 行数由 row-wise attention 与 `loc[:tokens]` 裁剪直接
处理，不做 padding；捕获开启但 padding 关闭时，draft-extend 只在
`tokens ≤ speculative_num_draft_tokens` 的形状上捕获，其余回落 eager
（reason `draft_extend_tokens_outside_1_N`），且单请求的 tokens 随接受数变化会产生
多张图。

另有 `KUTACC_BF16_SMALL_M_FULL_K`（已有 KuTACC 开关，默认 `0`）：本版在 lm_head 的
`try_linear` 上追加读取，用于 M1..128 的 full-K packed GEMM。

验证范围为 CPU 920F / TP16 / PP1、modelslim W8A8 权重、EAGLE topk1 且 steps 2~3；
视觉路径与其他模型未回归。具体验证结论以 PR #301 描述为准。

> **2026-11-03，PR #299 显式图整合版**：当前 Kunpeng Qwen 文本路径使用
> `qwen/backend.py` 与公共显式 GraphOp，经 backend 的 `get_graph_inputs` /
> `get_graph_fixed_tensors` 接入原 runner / finalize，不解析 Qwen Provider，
> 不启用 AutomaticTorchCapture / boxed 桥接或 forward arena。
> 下文涉及 Provider、boxed、arena 的段落及配方属于旧实现记录，不能作为本版启动要求。
> 本版仅验证 Qwen3.6-27B-W8A8 文本服务；视觉、MTP 及其他模型回归未验收。
> 本次合并及验证结果以 PR #299 描述为准，旧状态文档中的历史结果不代表本版。
> 若启动继承 DDR 的 `--membind`，需启用上游现有的
> `SGLANG_KUNPENG_HBW_BIND_BEFORE_MMAP=1`，使大页申请先绑定 HBM 并恢复调用者策略。
> Qwen 在 HBW 与 POSIX SHM 启用时，默认复用现有 `SGLANG_KUNPENG_SHARED_HBW_BACKING=split`
> 初始化：SHM payload 在本地 HBM 首次触页，其容量从总 HBW 预算扣除。
> 现有 split 模式要求每 rank 总预算 4000MiB；SHM 为 384MiB 时私有池为 3616MiB。
> 其他自定义总预算不自动切换 split，保持原放置方式。
> 不新增启动参数，显式的共享模式/容量配置仍优先；其他模型和非 POSIX 路径不变。

本说明对应本目录待审代码；实际验证状态见 `kunpeng_qwen_pr_status.md`。构建与部署边界见第9节；选频、启动、压测和工具调用工具包独立维护，不进入本PR。

## 1. 一次前向如何进入原生算子

```text
HTTP / OpenAI API
 → TokenizerManager → Scheduler → TpModelWorker → ModelRunner
 → Qwen3_5ForConditionalGeneration / 原模型 forward
 → LinearBase.quant_method.apply
 → CPU920FModelSlimW8A8Dynamic.apply_weights
 → 通用 graph ops / sgl_kernel Torch dispatcher
 → common_ops → KuTACC → KuPL worker
```

- `modelslim.py` 仅在 CPU920F 和显式 Qwen 后端同时启用时选择新增 scheme。其他平台与未启用 Qwen 的 ModelSlim 分流保留上游实现。
- `cpu920f_w8a8_dynamic.py` 创建 INT8 参数、加载 F32 scale/offset、初始化时 pack 权重。前向输入为 BF16 时进行行量化；输入已经是 `(INT8, F32 scale)` 时复用，避免 Norm/SiLU 融合后再次量化。
- GEMM 用 INT8×INT8、INT32 累加，经两侧 scale 反量化写出 BF16。BF16 BA/LM head 仍使用各自的 BF16 实现，不能把所有模块都称为 INT8。
- `qwen/backend.py` 处理 Qwen 的 GDN、compact GQA、QK Norm/RoPE、gate 和布局契约。它不重复实现 Scheduler 或模型的层遍历。
- 新增原生文件统一编入 `sgl_kernel/common_ops`。只有安装 C10 CPU allocator 的小 DSO 独立构建；没有运行期拼装旧 `.o` 或加载历史实验 DSO 的步骤。

## 2. 图模式复用哪些能力

`ModelRunner` 仍通过公共 `KunpengGraphRunner.can_run/replay` 进入图。Qwen provider 负责：

1. 判断当前 forward、请求和形状是否受支持。
2. 绑定动态输入、KV/GDN 可变状态、固定权重与通信缓冲，形成缓存键。
3. 在普通 `model.forward` 执行期间用 `AutomaticTorchCapture` 捕获 ATen 操作；显式 native primitive 通过 `GraphOpRegistry` 注册。
4. 交给公共 C++ Graph 执行器重放。

DeepSeek 已有的图执行器、allocator、分布式接口及公共量化/GEMM primitive 继续复用。新增 provider 是模型结构适配；GDN 状态、GQA KV 布局不能直接按 DeepSeek MLA/MoE 的结构解释。

公共图代码确有新增能力：输入别名/固定存储绑定、boxed ATen 捕获与重放、图缓存和 profile 支持。输入别名及GraphOp占位分配的TorchDispatch屏蔽均默认关闭，只由provider的`capture_options`显式开启；Qwen provider开启这两项，没有provider时保持原契约。它们应作为公共改动审查，不能全部归为 Qwen 私有算子。boxed ATen 重放有时仍产生临时结果后写入固定输出，并不等于每个 ATen 都已融合。

本次基线 decode 为最多32个请求；native 算子更大的行数支持不自动等于整模型/MTP 图覆盖。多请求 prefill 仍存在 eager 回落，视觉/MTP执行未验收。

## 3. 例一：GDN QKVZ/BA 布局

原始 CPU 通用路径可概括为：

```python
q, k, v, z, b, a = split_and_reshape(qkvz, ba)
b, a = b.contiguous(), a.contiguous()
mixed_qkv = torch.cat((flatten(q), flatten(k), flatten(v)), dim=-1)
# 下游卷积、GDN、gate/norm 读取这些输出
```

QKVZ 投影输出是按 token 行交错排列的。在多行情况下，某个分区跨行的 stride 仍是整个 QKVZ 行宽；它不是下游 native 内核要求的紧密行布局。只删 `.contiguous()` 会改变内核读地址，不能视为等价优化。

`qwen3_5.py` 现有实现按连续范围 `1 <= rows <= 4096` 处理：

- **一行**：QKV、Z、B、A 都是各自输入的只读、不相交 view。没有跨行空隙，可直接使用 `as_strided` 别名，保持生产者 Storage 到最后一个消费者。
- **多行**：用输入的真实行 stride 建立四个源 view；一次分配一块图管理的连续存储，再切成四个紧密输出区域；各分区直接 `copy_` 到最终区域。

多行仍有必要的布局搬运。省掉的是多次临时分配、cat/contiguous/reshape 组合及 boxed 临时输出再次转存，不是声称所有数据都不复制。M1 单独处理来自行布局的数学性质，不是仅支持某个压测并发的特例。

## 4. 例二：AllReduce 布局还原

为保留已验收的256个特征分区与累加顺序，`qwen/backend.py::allreduce` 仍有 pack/collective/unpack：

```python
# 通用概念写法，最后的 reshape 可能触发临时连续副本
packed = x.view(M, 256, 20).transpose(0, 1).contiguous()
collective(packed)
x.copy_(packed.view(256, M, 20).transpose(0, 1).reshape(M, 5120))

# 已采用的还原方法，直接写进原输出的三维 view
x.view(M, 256, 20).copy_(packed.view(256, M, 20).transpose(0, 1))
```

大M还使用原生 block pack/unpack；适合的 eager 输入可复用已有共享 scratch。图不能借用随其他 eager 调用复写的 scratch；代码保留不同生命周期处理和同步互斥。没有合入“把全部 pack/unpack 都移到通信库内部”的未验收设想。

## 5. KuTACC 与部署文件

| 边界 | 改动/依赖 |
|---|---|
| KuTACC 通用源码 | BF16小输出接口、可选小M full-K路径、大M INT8分块、量化零行处理、attention V偏移修复 |
| BA接口限制 | 当前新点积接口限定 N=6、K=5120、SVE512；workspace重载支持M1～4096，不代表任意N/K均已支持 |
| 编译兼容 | 标准库可选项保留上游默认；本厂商Torch使用libstdc++；显式补齐上游依赖的 `<memory>` |
| 独立部署工具包（不进入PR） | build、NUMA镜像、launch/rank入口、environment、tiling、chat template、选频、HTTP压测和工具调用检查；另行分发 |
| 外部必须提供 | 厂商Torch环境、KuPL、Bisheng、libnuma/numactl、模型checkpoint、集群调度与频率/hugepage权限 |

部署脚本的NUMA私有镜像不改变模型层的调度与TP语义；它把关键ELF/库分成独立文件，在绑核后首次加载，减少跨NUMA共享代码/数据页的访问。HBM采用公共池，KV仍在DDR；实际内存放置与频率要在目标机器记录。

## 6. 推荐审查顺序

1. KuTACC独立改动、接口契约及上游默认路径。
2. SGLang公共graph/Storage/共享HBM生命周期能力。
3. ModelSlim scheme、完整checkpoint loader与Qwen模型结构接入。
4. 原生GQA/GDN/BA/布局原语与动态行数边界。
5. 使用文档；部署与测试工具另行维护。

这只是拆分审查建议；当前两个本地工作分支尚未推送或创建远端PR。最终是否继续拆分提交，以真实构建和模型验证结果为依据。

## 7. 与最新上游冲突时的处理方式

本次以冻结job1935371为功能/性能参考，以2026-10-21拉取的公共上游为代码起点。历史提交152f6d6仅作暂定三方比较祖先，真实初始fork点尚未完全确认。因此逐文件核对语义；冲突解决不以保留旧文件为目标。

### 7.1 量化接入保留ModelSlim的公共入口

`ModelSlimConfig`用`is_cpu_920f() and qwen_enabled()`确定新增CPU scheme是否适用。随后仍通过普通Linear的quant_method进入`apply_weights`，其他平台的scheme选择保留。

```python
# 概念示意，具体实现见modelslim.py
if is_cpu_920f() and qwen_enabled():
    return CPU920FModelSlimW8A8Dynamic(...)
return existing_scheme_selection(...)
```

这个CPU scheme要求checkpoint声明W8A8_DYNAMIC。权重在加载后pack一次；BF16激活按行量化，已融合得到的(INT8, F32 scale)直接复用。输出仍为BF16。当前INT8权重pack要求`(N/tile_n)*(K/tile_k)==32`，与32个owned worker匹配；不能把BA支持动态行数等同于所有INT8 pack任务限制均已移除。

scale逻辑形状仍为[N,1]，物理backing为[N+16,1]。native的N尾部可能向量读取额外scale，因此HBM搬迁必须保留完整Storage和padding；只复制逻辑view会丢掉可读尾部。这是接口契约，不是额外的模型权重通道。

### 7.2 公共图执行器通过provider取得模型结构信息

`resolve_graph_provider`询问attention backend是否提供`get_kunpeng_graph_provider`。公共runner再调用provider的`supports/bind`；无provider的原模型继续沿用原有输入构建路径。

```python
provider = resolve_graph_provider(model_runner)
if provider is not None:
    bindings = provider.bind(model_runner, forward_batch, kwargs).validate()
    inputs = list(bindings.inputs.values())
else:
    inputs = existing_input_builder(...)
```

Qwen provider负责KV/GDN状态、输入别名、权重和workspace绑定；层顺序仍由模型forward决定，执行仍使用公共C++ Graph。代码里仍有Qwen条件分支，特别是模型入口、量化入口和TP16 communicator准入；不能宣传成“完全没有模型判断”。目标是把结构差异集中在有边界的接入点。

图缓存同时考虑shape/stride/offset及固定Storage签名。动态输入可更换地址；固定状态的Storage变化必须让旧图失效。Hybrid后端没有DSA的forward_metadata时，缓存键使用空值；保留上游键的其他部分。

### 7.3 保留上游batched AllGather，修正本配置容量

公共上游的batched AllGather和其预分配策略保留。Qwen配置最大请求数32、分块4096；直接将两者相乘会使接收区预分配达到1792MiB，超出768MiB共享前缀。

明确设置`SGLANG_KUNPENG_COLLECTIVE_MAX_TOKENS=4096`，让显式容量对batched AllGather生效，当前接收区为56MiB。没有显式覆盖时保留上游原默认值。没有替换成旧的非batched通信实现；此覆盖值仍需与实际调度上限一致。

### 7.4 SGLang/KuTACC公开接口不同步

两仓库分别最新不保证配套。实际编译发现SGLang引用的部分DSA/MLA/RDMA/MoE/批量BF16及FP32输出packed GEMM接口在公共KuTACC起点不存在。

新增CMake开关默认OFF。只有显式开启后，逐接口做编译探测：存在则编译原实现，缺失则生成调用时明确报错的适配器。独立Qwen构建工具显式开启；当前验收路径不使用这些缺失接口。没有返回零张量或静默吞掉计算。

这是一项需要维护者重点审查的兼容策略。当前Qwen实跑不证明DeepSeek所有可选路径可用；最好由上游确认配套版本/能力矩阵，减少对缺失接口适配的长期依赖。

## 8. 哪些改动会影响公共能力

| 改动 | 开关/边界 | 仍需关注 |
|---|---|---|
| BF16小输出 | 新公共API；N=6/K=5120/SVE512 | 名称通用不代表任意N/K；显式workspace同步有效且64B对齐 |
| BF16 full-K | `KUTACC_BF16_SMALL_M_FULL_K=1`；M1..128、N>=32、K>=4096且为偶数、无bias等条件 | 保留已验收行为，未重新开展完整K5120性能试验；其他形状需另测 |
| INT8内部macro-M96 | m>128且tile_k=2560时启用 | 公共形状分派，会影响满足条件的其他模型，当前没有完整跨模型性能证明 |
| attention第二半V偏移 | 由固定64改为vo_head_dim/2 | head_dim128时仍为64，256时为128；不改变GQA头分片规则 |
| 量化/Norm/SiLU零行处理 | 公共算术原语边界语义 | 明确scale、舍入、饱和与非有限值处理；需要针对性数值用例 |
| C++标准库 | 默认libc++不变；厂商Torch构建显式选libstdc++ | 所有依赖须匹配C++ ABI，不能只依据.so存在就判断可用 |
| scoped HBM allocator | 显式forward scope外走原allocator | 核查多线程scope、Storage别名和释放顺序；不能按Python对象回收就提前复用内存 |

BA workspace每行32个BF16，即64B；最终只有6个BF16输出，即12B。中间行slot按64B隔开以避免多worker写相邻输出造成伪共享；它是最终BF16输出暂存，非FP32分片累加区。工作完成后再紧凑写回[M,6]。

动态M须按层说明：布局适配连续支持1..4096，不是每个M都做过独立计时；当前整体decode provider的batch/token准入仍为1..32。LM head/native支持128行不等于32并发×MTP已整模型通过。

## 9. 从编译到服务启动的顺序

部署工具包已移出仓库；引用下列`build.sh`、`launch.py`等脚本时，它们均来自独立工具包，不是本PR中的文件。工具包构建需显式设置`SGLANG_SOURCE=/path/to/sglang`，启动需传`--source-root /path/to/sglang`，不能根据工具包目录反推源码位置。以下说明各阶段的职责。

1. 在920F计算节点准备独立厂商Python/Torch环境、Bisheng、KuPL、libnuma及RDMA运行库。普通PyPI Torch不能代替带KuPL集成的厂商构建。记录Python/Torch/C++ ABI和头文件、动态库来源。
2. 设置PYTHON、CXX、KUPL_PREFIX、KUTACC_SOURCE、KUTACC_INSTALL，执行build.sh：先KuTACC，再CPU wheel，再SGLang Python。CPU包装使用独立受管理构建目录，不覆盖仓库CUDA入口；不从旧实验复用.o。
3. 默认INSTALL_ENV=1只适合独立环境。共享服务环境用INSTALL_ENV=0构建，另行隔离安装/解包新wheel。编译子进程的环境变量不会自动传回外层终端；后续运行需重新配置LD_LIBRARY_PATH。
4. 在已分配的目标节点核查HBM和其他服务；保存hugepages/SHM初始状态。总HBM4000MiB/NUMA包含768MiB共享前缀与3232MiB私有池；864MiB arena属于私有池，KV在DDR。
5. 请求约2GHz并重新生成核表。root_proxy是集群外部工具，PR不携带；其返回码不代替512计算核实际采样。调频后不能沿用此前核表。
6. prepare_numa_images生成16份关键ELF私有副本，解析完整ldd依赖闭包，保留SONAME别名；复制和加载都先绑NUMA。纯Python软链接仍依赖原环境，必须保留它。
7. 使用launch.py启动TP16，对外访问需`--host 0.0.0.0`（默认127.0.0.1）。模型加载后依据实际PID/TID执行compute_frequency.py apply；正式短负载期间verify。
8. 验证health/models、普通聊天、默认thinking关闭、显式开启及工具往返；再做少量实际1024输入/输出短测，关闭逐token日志。不能只看scheduler RUNNING或HTTP监听就认定模型可用。
9. 长期运行由集群owner/dsub包装负责独占租约、大页生命周期、运行时限和退出清理。独立launch.py本身不申请节点、不设置频率、不自动准备系统大页。仓库原生CLI加NUMA私有镜像的替代启动方式尚未单独验收，不把单独checkout称为具备全部站点部署能力。

常见部署坑：taskset需在numactl之前展开初始单核亲和性；Unix IPC使用/tmp短路径；高HTTP端口可能使上游派生的HTTP+10000 RPC端口溢出；libnuma常在系统目录而非conda/lib；/usr/lib64/libibverbs是独立provider目录；部分可选CUDA导入warning在CPU服务中不是致命错误，应看首个真正异常。

## 10. 提交PR前的建议

- 当前改动尚未commit/push。提交前重新fetch目标分支，记录实际base SHA；2026-10-21的冻结起点不保证之后上游没有更新。
- 先提交KuTACC公共接口/算术变更，再提交依赖它的SGLang集成，说明匹配commit、头文件、SO版本及ABI。跨仓库接口改动应让reviewer能构建同一组合。
- 优先拆分审查提交：通用算子/正确性；公共graph与allocator；ModelSlim/完整loader；Qwen结构原语；使用文档。部署工具独立维护。把单纯格式化与语义改动分开。
- MTP关闭不代表diff中没有MTP代码：目前保留bridge/speculative等结构。建议将未验收的执行优化单列审查，同时保留完整checkpoint加载与能力声明，避免范围失控。
- 给公共INT8分块、allocator、图输入别名补最小针对性回归；特别检查未启用Qwen时的DeepSeek原路径及其他平台导入/构建。当前尚不能承诺这些分支全无回退。
- 基本功能/性能通过与完整精度分开报告。冻结旧版与新版18组长输出在67..203token处分叉，原因未隔离；合入前补固定集质量/logits误差验收，不直接归因为正常INT8舍入。
- 不提交旧object/DSO、模型权重、运行日志、私有receipt、密码/token或特定用户/节点的生产路径。独立工具参数化；集群root_proxy/dsub包装可单独作为站点部署材料。
- 推荐先以Draft PR做结构审查。给出已通过/未覆盖列表、实际短测口径、已知eager回落及视觉/MTP限制；不要把两轮小幅差异或历史性能包装成本次新增收益。

## 11. 默认行为与DeepSeek隔离（2026-10-22收紧）

- KuTACC大M调优新增`KUTACC_INT8_LARGE_M_MACRO_M96=1`。默认仍用上游macro-M128；显式开启且M>128、tile_k=2560时使用96。Qwen的独立启动配置开启该变量。KuTACC不识别模型名称，避免在数学库内部新增模型依赖；此开关是进程级调优，不能给DeepSeek进程继承Qwen的配置。
- Qwen模型入口、CPU920F ModelSlim scheme、compact GQA与AllReduce布局分流由Qwen后端开关或模块标记启用。原DeepSeek communicator调用保留在else分支。
- 公共capture默认不允许新增输入别名、不屏蔽TorchDispatch；Qwen provider通过`capture_options`选择能力。普通DeepSeek捕获不使用AutomaticTorchCapture或boxed ATen适配链。
- forward HBM arena只在Qwen后端启用且arena容量非零时安装。关闭Qwen后，即便误留arena容量变量也不替换C10 allocator。共享HBM backing、bind-before-mmap、通信容量覆盖都是默认关闭/不覆盖的显式能力。
- Storage/DataPtr lease让最后一个别名释放后才归还arena区间；共享HBM模式延迟window销毁到进程退出。普通非共享HBM模式仍走原销毁路径。该进程级生命周期策略不承诺同一进程内无损热重建多个服务。
- 通用bug修复仍会影响其他调用者的边界行为：KV最后一页分配、零尺寸HBM分配、未初始化SHM与peer范围检查、量化全零/极小值、attention V偏移。它们应按独立正确性变更审查，不能把“默认禁用Qwen优化”写成“任何输入下逐指令/逐bit都与上游相同”。
- 2026-10-22的隔离整理已完成主机侧分流检查；尚未重新完成920F native实编、DeepSeek完整前向及性能回归。10月21日短测及已运行的独立冻结服务不能替代这些新增改动的运行验收。
