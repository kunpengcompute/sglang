# SGLang鲲鹏优化版本参考实现

## Release Notes

- \[2026/05] 本版本基于SGLang官方v0.5.11版本进行优化适配。

## 1. 简介

SGLang鲲鹏优化版本是基于开源高性能推理框架SGLang与鲲鹏平台推出的DeepSeek V3模型服务化部署参考实现。该实现充分发挥鲲鹏处理器的众核并行与大带宽互联特性，结合SGLang的KV缓存管理机制、PD分离架构与高效并发调度能力，为鲲鹏生态用户提供低时延、高吞吐的请求处理能力。

### 1.1 版本配套

- 运行平台
  - 鲲鹏 920 专业版
- 系统规格
  - openEuler 22.03（LTS-SP4）AArch64

## 2. SGLang安装部署过程

### 2.1 SGLang依赖安装

在正式安装SGLang相关组件之前，需先完成Python虚拟环境的创建与激活，参考命令如下所示，更多信息可参考SGLang官方社区提供的[安装指南](https://docs.sglang.com.cn/platforms/cpu_server.html#install-from-source)。

```shell
source ~/anaconda3/start_conda.sh
conda create -n sgl-cpu python=3.12 -y
conda activate sgl-cpu
```

此外，编译SGLang内核模块需要毕昇编译器，请从<https://www.hikunpeng.com/developer/hpc/hpckit-download>获取HPCKit安装包，参考官方安装步骤完成安装。

在PD（Prefill-Decode）分离部署模式下，需额外安装SGLang Router与Mooncake两个组件，以实现Prefill实例与Decode实例之间的请求分配以及KVCache数据传输：

- **SGLang Router**：安装方式请参考SGLang[官方文档](https://docs.sglang.com.cn/advanced_features/sgl_model_gateway.html)。
- **Mooncake**：安装方式请参考Mooncake官方[GitHub仓库](https://github.com/kvcache-ai/Mooncake#-quick-start)。

### 2.2 SGLang应用安装

在安装SGLang之前，先激活已配置好的Python虚拟环境。随后，根据SGLang安装路径获取SGLang源代码，并依次执行以下命令完成编译安装：

```shell
# 安装SGLang主模块（包含CPU相关依赖）
cd sglang/python
pip install --upgrade pip setuptools
pip install -e .
pip install torchvision==0.24.0 triton==3.5.0 --force-reinstall

# 加载毕昇编译器环境变量
HPCKIT_PATH=/path-to-HPCKit
source ${HPCKIT_PATH}/latest/compiler/bisheng/env/setvars.sh
export CC=$(which clang)
export CXX=$(which clang++)

# 加载kutacc环境变量
KUTACC_PATH=/path-to-KUTACC
export KUTACC_LIB=${KUTACC_PATH}/install/lib
export KUTACC_INCLUDE=${KUTACC_PATH}/install/include

# 进入sgl-kernel子目录并安装内核模块
cd ../sgl-kernel
pip install -v . --no-build-isolation
```

### 2.3 PyTorch v2.9.0安装

SGLang鲲鹏优化版本需依赖通过毕昇编译器构建的PyTorch v2.9.0，并启用kupl多线程后端以获得最佳性能。安装前请先从 <https://gitcode.com/kunpengcompute/kunpeng-extension-for-pytorch/tree/main/thirdparty> 获取适配补丁，然后按后续命令完成补丁应用。

```shell
git clone -b v2.9.0 --depth=1 --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
git apply pytorch-v2.9.0-kupl.patch
```

## 3. SGLang部署和启动

### 3.1 DeepSeek模型使用

以DeepSeek V3 int8量化版本为例，推荐配置为`DP=16`、`TP=16`（总共256个TP Worker），需要16个节点。在`scripts/cpu_kunpeng`目录下的`env.sh`脚本中配置`NATIVE_IP_SPEC`、`NATIVE_MASTER_ADDR`以及模型路径、conda环境名等信息。

由于V3量化版本权重加载较慢，可以在启动参数中添加`--load-format sharded_state`，从而支持权重分片加载以加速启动。首次使用前，需要用`scripts/cpu_kunpeng/split_weights.py`脚本对原始权重进行预处理。

启动和终止命令如下：

```shell
# native模式（非PD分离）
sh launch.sh native
sh stop.sh native

# PD分离模式（分别在prefill、decode、router节点执行）
sh launch.sh prefill
sh launch.sh decode
sh launch.sh router
```

### 3.2 非DeepSeek模型使用

使用非DeepSeek模型时，需要注释掉`srt/models/registry.py`中第98\~99行，这两行会跳过非DeepSeek模型依赖的加载，其目的是提升启动加载速度。也可以将条件判断的值改成具体所用模型对应的值，例如`sglang.srt.models.qwen3`。

### 3.3 部署模式

四种部署模式通过 `source env.sh [模式]` 选择：

- **native**（默认）：非PD分离，单集群全量服务。节点从 `NATIVE_IP_SPEC` 读取，负载均衡 `round_robin`。
- **prefill**：PD分离的prefill角色。节点从 `PREFILL_IP_SPEC` 读取。
- **decode**：PD分离的decode角色。节点从 `DECODE_IP_SPEC` 读取。
- **router**：PD分离的router节点，单节点运行 `sglang_router.launch_router`。

### 3.4 环境变量说明

#### 3.4.1 功能类

**1. SGLANG\_ENABLE\_BINARY\_LAUNCH / SGLANG\_ENABLE\_NUMA\_DUPLICATION — 二进制启动模式**

```
export SGLANG_ENABLE_BINARY_LAUNCH=1      # 默认开
export SGLANG_ENABLE_NUMA_DUPLICATION=1   # 默认开
```

- **作用**：每个scheduler进程独立启动，使用 `--tp-rank-in-node` 区分rank号。`server.sh` 遍历 `ATTN_TP_RANK` 逐个启动进程，绑CPU核（`taskset -c`）并配置大页。关闭时仅有一个 `python -m sglang.launch_server` 进程。
- **SGLANG\_ENABLE\_NUMA\_DUPLICATION**：在上述基础上，每个NUMA节点使用 `PYINSTALL_PATH/dist/sglang_server_tp{rank}/` 下的独立预编译二进制（`SGLANG_ENABLE_BINARY_LAUNCH` 开启前提下），实现各NUMA节点内存本地化，减少跨NUMA访存。关闭时所有rank共享一个二进制，同时影响 `LD_LIBRARY_PATH` 的设置逻辑。

**2. SGLANG\_ENABLE\_TOKENIZER\_SEPERATE — 分离 Tokenizer**

```
export SGLANG_ENABLE_TOKENIZER_SEPERATE=0  # 默认关
```

- **作用**：将tokenizer从推理进程中拆出。=1时触发三个子函数：

| 子模式 | 判断条件 |
|---|---|
| `is_tokenizer_separate()` | `=1` |
| `is_http_only()` | 且 `SGLANG_LAUNCH_HTTP_ONLY=1` |
| `is_skip_http()` | 且 `SGLANG_SKIP_HTTP=1` |

- **相关设置**：`SGLANG_LAUNCH_HTTP_ONLY=1`（router节点仅起HTTP）、`SGLANG_SKIP_HTTP=1`（推理节点跳过HTTP）、`ROUTER_IP`（HTTP server绑到router IP）。
- **效果**：PD分离场景下router节点统一做tokenize/detokenize，推理节点专心forward。非PD场景（native）无需开启。

**3. SGLANG\_KUNPENG\_DISABLE\_MLA\_ALL2ALL — 禁用 MLA all2all**

```
export SGLANG_KUNPENG_DISABLE_MLA_ALL2ALL=1  # 默认开
```

- **作用**：控制MLA attention的通信策略。
  - `=0`（不禁用）：TP=8或16时，attention内的Q和attention_output走per-socket all2all（group_size=8），两个socket之间的合并靠o_proj的allreduce，减少跨socket SHM通信。
  - `=1`（禁用）：不走per-socket all2all，attn_mqa在完整TP group上计算，通信由RowParallelLinear的allreduce承担。

**5. SGLANG\_ENABLE\_HBW\_POOL / SGLANG\_ENABLE\_HBW\_SWAP — HBW 内存管理**

```
export SGLANG_ENABLE_HBW_POOL=1   # 启用 HBW KV cache 分配
export SGLANG_ENABLE_HBW_SWAP=0   # 关闭 HBW↔DRAM 交换
```

- **SGLANG\_ENABLE\_HBW\_POOL=1**：KV cache优先从高带宽内存分配，减少DRAM访存延迟。
- **SGLANG\_ENABLE\_HBW\_SWAP=1**：允许HBW页与DRAM之间通过SDMA异步迁移，内存压力大时换出冷数据。
- **相关设置**：`SGLANG_KUNPENG_WEIGTHS_HBW_POOL_SIZE_MB`、`SGLANG_KUNPENG_SDMA_MAX_EVENTS`、`SGLANG_KUNPENG_SDMA_THRESHOLD`。

**6. SGLANG\_ENABLE\_MTP — 多 Token 预测**

```
export SGLANG_ENABLE_MTP=0  # 默认关
```

- **作用**：启用DeepSeek模型的Multi-Token Prediction投机解码特性，=1时开启多Token预测以加速推理。

#### 3.4.2 其他开关类

- **SGLANG\_USE\_CPU\_920F**：标识鲲鹏920F平台，触发fork启动、仅加载deepseek_v2模型、CPU绑核等平台优化。
- **SGLANG\_USE\_CPU\_ENGINE**：强制走CPU推理引擎。
- **SGLANG\_SET\_CPU\_AFFINITY**：启用CPU绑核。
- **SGLANG\_LAUNCH\_HTTP\_ONLY**：仅启动HTTP server不做推理，router节点使用。需配合 `SGLANG_ENABLE_TOKENIZER_SEPERATE`。
- **SGLANG\_SKIP\_HTTP**：跳过HTTP server启动，prefill/decode推理节点使用。需配合 `SGLANG_ENABLE_TOKENIZER_SEPERATE`。
- **SGLANG\_DISAGGREGATION\_FORCE\_QUERY\_PREFILL\_DP\_RANK**：PD分离模式下强制指定查询prefill的DP rank，绕过 `follow_bootstrap_room` 的自动路由。
- **SGLANG\_ENABLE\_TORCH\_COMPILE** / **TORCH\_COMPILE\_DISABLE**：启用torch.compile（此版本默认关闭）。
- **SGLANG\_LOG\_MS**：日志添加毫秒级时间戳。
- **SGLANG\_KUNPENG\_PROFILE**：函数调用耗时打印，帮助定位性能瓶颈。
- **SGLANG\_ENABLE\_TP\_MEMORY\_INBALANCE\_CHECK**：TP间显存容量不均检查（此处设为0关闭）。

#### 3.4.3 大小/阈值类

**共享内存池**
- **SGLANG\_KUNPENG\_PREFILL\_SHM\_SIZE\_MB**：prefill阶段共享内存池大小，默认476MB。
- **SGLANG\_KUNPENG\_DECODE\_SHM\_SIZE\_MB**：decode阶段共享内存池大小，默认100MB。

**HBW内存池**
- **SGLANG\_KUNPENG\_WEIGTHS\_HBW\_POOL\_SIZE\_MB**：权重缓存在高带宽内存中的预分配大小，默认3800MB。

**SHM批处理容量**
- **SGLANG\_KUNPENG\_MAX\_SEQ\_NUM**：SHM预分配最大序列数。prefill时默认4，decode时默认128。
- **SGLANG\_KUNPENG\_MAX\_CUR\_LEN**：SHM预分配最大序列长度。prefill时默认1024，decode时默认1。

**SDMA传输**
- **SGLANG\_KUNPENG\_SDMA\_MAX\_EVENTS**：SDMA最大并发事件数，默认10。
- **SGLANG\_KUNPENG\_SDMA\_THRESHOLD**：SDMA传输阈值，超过此大小使用硬件DMA而非CPU memcpy，默认5。

**超时控制**
- **SGLANG\_WARMUP\_TIMEOUT**：warmup超时秒数，默认1600。

**模型加载**
- **LOAD\_FORMAT**：权重加载格式，设为`sharded_state`启用分片加载加速启动（DeepSeek V3 INT8推荐）。

**多线程后端**
- **KUPL\_EXECUTOR\_BACKEND** / **KUPL\_EXECUTOR\_COUNT**：仅在`TORCH_USE_KUPL`设为1时生效，指定kupl多线程后端。常规PyTorch仅支持omp。`KUPL_EXECUTOR_COUNT`默认32。

## 4. 正确性验证

验证脚本 `scripts/cpu_kunpeng/curl.sh` 支持多种功能，以下为常用用法。

### 4.1 基本用法

```shell
sh curl.sh                    # 基础推理
sh curl.sh -s -m 50           # 流式，50 tokens
sh curl.sh -n 10 -m 20        # 10个请求
sh curl.sh -s -f prompts/1k.txt  # 长prompt
```

### 4.2 路由测试

```shell
# 指定DP rank
sh curl.sh -d 3 -n 10 -m 20
```

### 4.3 性能分析

```shell
# 开启profile模式（自动调用 start_profile / stop_profile）
sh curl.sh -p -s -f prompts/ragged.txt
```

### 4.4 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `-p` | 启用 profile（start/stop） | 关闭 |
| `-s` | 启用流式输出 | 关闭 |
| `-d RANK` | 路由到指定 DP rank | - |
| `-n NUM` | 请求数量 | 文件所有行 |
| `-m TOKENS` | 最大 token 数 | 10 |
| `-f FILE` | prompt 文件 | prompts/5.txt |

根据请求的返回结果对正确性进行验证。

## 5. Prefill 多实例部署

### 5.1 配置说明

在 `scripts/cpu_kunpeng/env.sh` 中配置两个 prefill 实例：

```bash
# 默认实例（短请求）
PREFILL_IP_SPEC="xxx.xxx.xxx. | 1-16"
PREFILL_MASTER_ADDR="xxx.xxx.xxx.1"
PREFILL_MASTER_PORT="5000"
PREFILL_PP_SIZE=1

# 长请求实例
PREFILL_LONG_PROMPT_IP_SPEC="xxx.xxx.xxx. | 17-32"
PREFILL_LONG_PROMPT_MASTER_ADDR="xxx.xxx.xxx.17"
PREFILL_LONG_PROMPT_MASTER_PORT="5020"
PREFILL_LONG_PROMPT_PP_SIZE=2
```

### 5.2 启动方式

#### 1. 单实例模式（默认）

```bash
# prefill（在节点1上执行）
sh launch.sh prefill

# decode（在 decode master 节点上执行）
sh launch.sh decode

# router（在 router 节点上执行）
sh launch.sh router
```

#### 2. 双实例模式（Bucket 负载均衡策略）

```bash
# 实例1（短请求 prefill，节点1~16，在节点1上执行）
sh launch.sh prefill

# 实例2（长请求 prefill，节点17~32，在节点17上执行）
sh launch.sh prefill long_prompt

# decode（在 decode master 节点上执行）
sh launch.sh decode

# router（在 router 节点上执行，启用 bucket 策略）
sh launch.sh router prefill_bucket
```

### 5.3 注意事项

- 两个 prefill 实例的 master 节点不同，需分别在各自 master 节点上启动
- 两个实例的 `MASTER_PORT` 不能相同（默认：5000 vs 5020）
- router 传入 `prefill_bucket` 参数时，自动注册两个 prefill 实例并启用 bucket 负载均衡策略
- router 不传 `prefill_bucket` 参数时，只注册一个 prefill 实例，使用 `--policy` 指定的策略
- 长请求实例的 `PP_SIZE` 可独立配置（默认：2）

### 5.4 Bucket 策略 CLI 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--prefill-policy bucket` | - | 按请求字符数分桶路由到不同 prefill 实例 |
| `--balance-abs-threshold` | 64 | 两实例字符数绝对差超过此值时触发负载均衡切换 |
| `--balance-rel-threshold` | 1.5 | 高负载实例字符数超过低负载的 1.5 倍时才认为不平衡（与绝对阈值同时满足） |
| `--bucket-adjust-interval-secs` | 5 | 每 5 秒根据历史负载自动调整分界线 |

#### 负载均衡策略

当请求到达时，先按字符数通过二分查找定位到对应的 boundary 分桶，选择对应的 prefill 实例。同时检查两个实例的累计字符数负载，当**同时满足**以下两个条件时，认为负载不平衡，改为路由到负载低的实例：

- 绝对差：`max_load - min_load > balance_abs_threshold`
- 相对差：`max_load > balance_rel_threshold × min_load`

两个条件同时满足才触发切换，避免负载轻微不均时频繁切换。

#### 分界线自动调整

初始分界线将字符长度范围 `[0, 4096]` 均分给各实例（2个实例时：实例1负责 `[0, 2047]`，实例2负责 `[2048, MAX]`）。后台线程每隔 `bucket_adjust_interval_secs` 秒执行一次自动调整：

1. 统计时间窗口内（`period = bucket_adjust_interval_secs × 1000` 毫秒）各请求的字符数分布
2. 根据历史负载重新划分各实例的 boundary 范围，使各实例负载更均衡
3. 如果负载变化不大（新负载 < 2倍旧负载 且 旧负载 < 2倍新负载），则跳过调整

如果请求大部分为短请求（< 1k 字符），实例1负载会偏高，分界线会自动往大调，最终收敛到接近 1k 的位置。
