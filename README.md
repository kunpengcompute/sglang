# SGLang鲲鹏优化版本参考实现

## Release Notes

- \[2026/09] 更新部署文档：新增多实例部署、分离tokenizer的PD分离部署方式及相关环境变量说明。
- \[2026/05] 本版本基于SGLang官方v0.5.11版本进行优化适配。

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

运行与编译还需依赖以下两个组件，具体安装步骤请前往对应仓库查看：

- **kutacc**：鲲鹏CPU通信与算子库（RDMA集合通信、shm allreduce等），使用<https://gitcode.com/zhengzhong722/kutacc/tree/br_sglang>分支。
- **kupl**：鲲鹏加速运行时（kupl多线程后端、SHM/SDMA等），使用<https://gitcode.com/kunpengcompute/kupl/tree/sglang_830>分支。

在PD（Prefill-Decode）分离部署模式下，需额外安装SGLang Model Gateway与Mooncake两个组件，以实现Prefill实例与Decode实例之间的请求分配以及KVCache数据传输：

- **SGLang Model Gateway**：安装方式请参考SGLang[官方文档](https://docs.sglang.com.cn/advanced_features/sgl_model_gateway.html)。
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

# 加载kupl环境变量（sgl-kernel的kupl_async模块依赖libkupl与kupl.h）
KUPL_PATH=/path-to-KUPL
export CPATH=${KUPL_PATH}/include:$CPATH
export LIBRARY_PATH=${KUPL_PATH}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${KUPL_PATH}/lib:$LD_LIBRARY_PATH

# 进入sgl-kernel子目录并安装内核模块
cd ../sgl-kernel
pip install -v . --no-build-isolation
```

### 2.3 PyTorch v2.9.0安装

SGLang鲲鹏优化版本需依赖通过毕昇编译器构建的PyTorch v2.9.0，并启用kupl多线程后端以获得最佳性能。安装前请先从 <https://gitcode.com/kunpengcompute/kunpeng-extension-for-pytorch/tree/main/thirdparty> 获取适配补丁`pytorch-v2.9.0-sglang-0830.patch`，然后按后续命令完成补丁应用。

```shell
git clone -b v2.9.0 --depth=1 --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
git apply pytorch-v2.9.0-sglang-0830.patch
```

## 3. SGLang部署和启动

### 3.1 DeepSeek模型使用

以DeepSeek V3 int8量化版本为例，native模式推荐配置为`DP=16`、`TP=16`（总共256个TP Worker），需要16个节点。部署配置分布在`scripts/cpu_kunpeng`目录下的多个文件中：

| 文件                                          | 说明                                              |
| ------------------------------------------- | ----------------------------------------------- |
| `runtime/env_base.sh`                       | 共享默认配置：路径、`INSTANCES`实例列表、TP/DP/EP/PP并行度、各类功能开关 |
| `runtime/env_prefill.sh`                    | prefill角色默认值：节点集、master地址、端口/NUMA自动推导           |
| `runtime/env_decode.sh`                     | decode角色默认值：节点集、master地址、端口/NUMA自动推导            |
| `.user_env.sh`                              | 集群级用户覆盖文件（先加载），可按角色参数分支配置                       |
| `runtime/.user_env_base.sh`                 | 可选的全角色共享覆盖文件（在`.user_env.sh`之后加载）               |
| `runtime/.user_env_prefill[_<instance>].sh` | prefill角色/实例覆盖文件                                |
| `runtime/.user_env_decode[_<instance>].sh`  | decode角色/实例覆盖文件                                 |

模型路径、conda环境名、native节点信息（`NATIVE_IP_SPEC`/`NATIVE_IP_FILE`、`NATIVE_MASTER_ADDR`）在`runtime/env_base.sh`中配置（或通过`.user_env.sh`覆盖）；prefill/decode节点集与并行度在对应角色文件中配置。完整配置加载顺序见3.3节。

由于V3量化版本权重加载较慢，可在配置文件中设置`LOAD_FORMAT`启用分片加载加速启动，支持`sharded_state`与`kunpeng_state`两种格式。首次使用前，需要用`scripts/cpu_kunpeng/model_processing/split_weights.py`脚本对原始权重进行预处理。

native模式启动和终止命令：

```shell
sh launch.sh native
sh stop.sh server native
```

PD分离模式的部署方式见3.4节。

### 3.2 非DeepSeek模型使用

使用非DeepSeek模型时，需要注释掉`srt/models/registry.py`中第98\~102行的条件判断，这段逻辑会跳过非DeepSeek模型依赖的加载，其目的是提升启动加载速度。也可以将条件判断的值改成具体所用模型对应的值，例如`sglang.srt.models.qwen3`。

### 3.3 部署模式

通过`./launch.sh <role> [instance]`选择部署角色：

- **native**（默认）：非PD分离，单集群全量服务。节点从`NATIVE_IP_SPEC`/`NATIVE_IP_FILE`读取，负载均衡`round_robin`。
- **prefill**：PD分离的prefill角色。节点从`PREFILL_IP_SPEC`/`PREFILL_IP_FILE`读取（默认值见`runtime/env_prefill.sh`）。
- **decode**：PD分离的decode角色。节点从`DECODE_IP_SPEC`/`DECODE_IP_FILE`读取（默认值见`runtime/env_decode.sh`）。
- **router**：PD分离的router节点，单节点运行SGLang Model Gateway（`sgl-model-gateway`，Rust二进制，默认取conda环境内的`bin/sgl-model-gateway`，可用`MODEL_GATEWAY_BIN`指定），默认策略`cache_aware`。
- **tokenizer**：在router节点启动某一侧的tokenizer HTTP server，用法`./launch.sh tokenizer <prefill|decode> [instance]`，PD分离模式下使用。
- **all**：一键部署，按`INSTANCES`列表依次拉起各实例server集群、各实例tokenizer、router。
- **update**：重新生成`.time_env.sh`并更新NUMA二进制副本。

脚本调用关系（`launch.sh`为纯任务分发器，实际逻辑位于子脚本）：

```
launch.sh（任务分发器：./launch.sh <role> [instance]）
  ├─→ runtime/launch_cluster.sh（prefill/decode/native：env加载、stop前置、SSH分发到各节点）
  │      └─→ server.sh（单节点进程启动）
  ├─→ runtime/launch_tokenizer.sh（tokenizer：SSH到router节点）
  │      └─→ runtime/server_tokenizer.sh（tokenizer HTTP server）
  └─→ runtime/launch_router.sh（router：等待全部后端就绪后启动Model Gateway）
         └─→ runtime/server_router.sh（SGLang Model Gateway）
```

配置加载顺序（`source env.sh <role> [instance]`）：

```
runtime/env_base.sh（共享默认值；内部依次加载.user_env.sh与runtime/.user_env_base.sh）
  └─ runtime/env_prefill.sh / runtime/env_decode.sh（角色默认值，${VAR:-...}回退，不覆盖已设置的值）
       └─ runtime/.user_env_<role>_<instance>.sh（指定实例时仅加载该实例文件）
          或 runtime/.user_env_<role>.sh（未指定实例时加载角色默认覆盖文件）
```

角色默认值：prefill侧`SGLANG_KUNPENG_MAX_SEQ_NUM=8`、`SGLANG_KUNPENG_MAX_CUR_LEN=576`；decode侧`SGLANG_KUNPENG_MAX_SEQ_NUM=64`、`SGLANG_KUNPENG_MAX_CUR_LEN=1`（开启MTP时自动取`SGLANG_SPECULATIVE_NUM_STEPS+1`）。

### 3.4 PD分离部署

PD分离模式下tokenizer默认与推理进程分离（`SGLANG_ENABLE_TOKENIZER_SEPERATE=1`，由`env_base.sh`按角色自动设置；native模式默认关闭）。请求路径为：客户端→SGLang Model Gateway（router节点）→各侧tokenizer HTTP server（router节点，prefill侧默认端口30001、decode侧默认30002）→prefill/decode集群。Model Gateway的后端地址是tokenizer端口而非推理master的30000端口，每个prefill后端同时携带该实例的bootstrap端口（默认9001）用于KV传输引导。

部署步骤：

1. 配置：在`runtime/env_base.sh`（或`.user_env.sh`）中设置`ROUTER_IP`、模型路径、conda环境等共享配置；在`runtime/.user_env_prefill.sh`与`runtime/.user_env_decode.sh`（或对应实例文件）中设置两侧节点集（`PREFILL_IP_SPEC`/`DECODE_IP_SPEC`）、master地址与并行度（`PREFILL_TP_SIZE`/`DECODE_TP_SIZE`等，未设置时回退到全局`TP_SIZE`/`DP_SIZE`）。
2. 一键启动（推荐，在任一节点执行）：

```shell
sh launch.sh all   # 按INSTANCES：各实例server集群→各实例tokenizer→router
```

1. 分步启动（集群命令在各master节点执行；tokenizer/router自动SSH到`ROUTER_IP`）：

```shell
sh launch.sh prefill            # prefill集群
sh launch.sh decode             # decode集群
sh launch.sh tokenizer prefill  # prefill侧tokenizer HTTP server
sh launch.sh tokenizer decode   # decode侧tokenizer HTTP server
sh launch.sh router             # 等待全部后端就绪后启动Model Gateway
```

1. 停止：

```shell
sh stop.sh all                 # router+全部tokenizer+按INSTANCES停止各实例server
sh stop.sh server prefill      # 支持实例名：sh stop.sh server decode 128p
sh stop.sh router
sh stop.sh tokenizer prefill   # 支持prefill|decode|all
```

多实例部署（同角色多实例、端口/NUMA自动错开、短长序列分组路由等）参见[多实例部署指南](scripts/cpu_kunpeng/runtime/README.md)。

### 3.5 环境变量说明

#### 3.5.1 功能类

**1. SGLANG\_ENABLE\_BINARY\_LAUNCH / SGLANG\_ENABLE\_NUMA\_DUPLICATION — 二进制启动模式**

```
export SGLANG_ENABLE_BINARY_LAUNCH=1      # 默认开
export SGLANG_ENABLE_NUMA_DUPLICATION=1   # 默认开
```

- **作用**：每个scheduler进程独立启动，使用`--tp-rank-in-node`区分rank号。`server.sh`遍历`RANK_IN_NODE`逐个启动进程，绑CPU核（`taskset -c`）并配置大页。关闭时仅有一个`python -m sglang.launch_server`进程。
- **SGLANG\_ENABLE\_NUMA\_DUPLICATION**：在上述基础上，每个NUMA节点使用`PYINSTALL_PATH/dist/sglang_server_tp{rank}/`下的独立预编译二进制（`SGLANG_ENABLE_BINARY_LAUNCH`开启前提下），实现各NUMA节点内存本地化，减少跨NUMA访存。关闭时所有rank共享一个二进制，同时影响`LD_LIBRARY_PATH`的设置逻辑。多副本的构建（PyInstaller打包与NUMA副本复制）请参考[PyInstaller打包工具说明](scripts/cpu_kunpeng/pyinstall/readme.md)。

**2. SGLANG\_ENABLE\_TOKENIZER\_SEPERATE — 分离Tokenizer**

```
export SGLANG_ENABLE_TOKENIZER_SEPERATE=1  # PD角色默认开，native默认关
```

- **作用**：将tokenizer从推理进程中拆出，由router节点上的独立HTTP server统一做tokenize/detokenize。
- **效果**：PD分离场景下由`launch.sh tokenizer`在router节点拉起各侧tokenizer HTTP server，Model Gateway将请求路由到tokenizer端口。native场景无需开启。

**3. SGLANG\_ENABLE\_HBW\_POOL / SGLANG\_KUNPENG\_SWAP\_\* — HBW内存管理**

```
export SGLANG_ENABLE_HBW_POOL=1            # 启用HBW KV cache分配，默认开
export SGLANG_KUNPENG_SWAP_KV_IN=0         # KV cache换入（DDR→HBM，SDMA异步拷贝）
export SGLANG_KUNPENG_SWAP_KV_OUT=0        # KV cache写回（HBM→DDR）
export SGLANG_KUNPENG_SWAP_KV_BLOCKWISE=0  # decode按块换入
export SGLANG_KUNPENG_SWAP_EXPERT=1        # 专家权重逐层换入（prefill默认1）
```

- **SGLANG\_ENABLE\_HBW\_POOL=1**：KV cache优先从高带宽内存分配，减少DRAM访存延迟。
- **SGLANG\_KUNPENG\_SWAP\_KV\_IN/OUT=1**：允许HBW页与DRAM之间通过SDMA异步迁移；`SWAP_KV_BLOCKWISE=1`时decode仅换入所需块。
- **相关设置**：`SGLANG_KUNPENG_WEIGTHS_HBW_POOL_SIZE_MB`、`SGLANG_KUNPENG_SWAP_MAX_KV_BLOCKS`、`SGLANG_KUNPENG_SDMA_MAX_EVENTS`、`SGLANG_KUNPENG_SDMA_THRESHOLD`。

**4. SGLANG\_ENABLE\_MTP — 多Token预测**

```
export SGLANG_ENABLE_MTP=0              # 默认关
export SGLANG_SPECULATIVE_NUM_STEPS=2   # 投机步数，默认2
```

- **作用**：启用DeepSeek模型的NEXTN投机解码。=1时通过`--speculative-algorithm NEXTN`开启多Token预测以加速推理；`SGLANG_SPECULATIVE_NUM_STEPS`指定投机步数；`SPECULATIVE_DRAFT_MODEL_PATH`可指定MTP草稿权重（未设置时回退到`MODEL_PATH`）。decode侧开启MTP时`SGLANG_KUNPENG_MAX_CUR_LEN`自动取`SGLANG_SPECULATIVE_NUM_STEPS+1`。

**5. SGLANG\_ENABLE\_GRAPH\_CAPTURE — 图捕获**

```
export SGLANG_ENABLE_GRAPH_CAPTURE=1  # 默认开
```

- **作用**：对forward计算做图捕获与重放，消除Python调度开销。相关设置：`SGLANG_KUNPENG_GRAPH_CACHE_SIZE`（捕获图缓存数量，默认10）、`SGLANG_KUNPENG_EXTEND_POWER_2_PADDING`（prefill图按2的幂对齐padding，默认开）、`SGLANG_ENABLE_GRAPH_PROFILE`（图性能统计）、`SGLANG_GRAPH_DEBUG_PRINT`（逐算子调试打印）。

**6. SGLANG\_ENABLE\_KUCCL — kuccl通信后端**

```
export SGLANG_ENABLE_KUCCL=0  # 默认关
```

- **作用**：=1时使用kuccl（UCX+UCG）集合通信后端替代gloo，需配置`KUCCL_PATH`。NUMA二进制副本模式下各rank自动使用NUMA本地的kuccl运行时插件副本。

**7. SGLANG\_KUNPENG\_PP\_LAYOUT — PP布局**

```
export SGLANG_KUNPENG_PP_LAYOUT=0  # 默认0
```

- **作用**：PP rank布局方式。`0`=node\_block，一个节点映射到一个PP stage（默认）；`1`=interleave，每个节点承载全部PP stage（节点内rank 0..7属PP0、8..15属PP1）。

**8. SGLANG\_KUNPENG\_RDMA\_ALLGATHER / RDMA\_BCAST / RDMA\_PP\_COMM — RDMA集合通信**

```
export SGLANG_KUNPENG_RDMA_ALLGATHER=1  # 默认开
export SGLANG_KUNPENG_RDMA_BCAST=1      # 默认开
export SGLANG_KUNPENG_RDMA_PP_COMM=1    # 默认开
```

- **作用**：scheduler间同步的allgather/broadcast走RDMA（kutacc实现），PP rank间的rid传输走RDMA。需使用br\_sglang分支的kutacc构建。

**9. SGLANG\_KUNPENG\_LC\_DP\_RANKS / LC\_MIN\_SEQ\_LEN — 长序列decode CP**

```
export SGLANG_KUNPENG_LC_DP_RANKS=""        # 运行长上下文CP的DP rank列表，如"14,15"
export SGLANG_KUNPENG_LC_MIN_SEQ_LEN=4096   # 长序列判定阈值
```

- **作用**：`input_len + max_new_tokens`大于等于阈值的请求路由到LC DP rank做上下文并行decode，其余请求走常规rank。两个变量在prefill与decode实例间必须配置一致。

#### 3.5.2 其他开关类

- **SGLANG\_USE\_CPU\_920F**：标识鲲鹏920F平台，触发fork启动、仅加载deepseek模型、CPU绑核等平台优化。
- **SGLANG\_USE\_CPU\_ENGINE**：强制走CPU推理引擎。
- **SGLANG\_SET\_CPU\_AFFINITY**：启用CPU绑核。
- **SGLANG\_LAUNCH\_HTTP\_ONLY**：仅启动HTTP server不做推理，router节点使用。需配合`SGLANG_ENABLE_TOKENIZER_SEPERATE`。
- **SGLANG\_SKIP\_HTTP**：跳过HTTP server启动，prefill/decode推理节点使用。需配合`SGLANG_ENABLE_TOKENIZER_SEPERATE`。
- **SGLANG\_KUNPENG\_PROFILE**：函数调用耗时打印，帮助定位性能瓶颈。
- **SGLANG\_KUNPENG\_PP\_PROFILE**：decode流水线性能分析。
- **SGLANG\_KUNPENG\_MOE\_FORCE\_LOAD\_BALANCE**：=1强制MoE负载均衡（仅用于性能测试，不保证正确性）。
- **SGLANG\_KUNPENG\_MOE\_SHUFFLE\_MODE**：动态冗余专家重排洗牌模式，0=轮询，1=随机。
- **SGLANG\_KUNPENG\_MOE\_TOKEN\_MULTIPLE**：MoE token填充倍数，默认2。
- **SGLANG\_KUNPENG\_SAVE\_AT\_TRACE**：=1记录专家激活数量。
- **SGLANG\_DISABLE\_RADIX\_CACHE**：=1时向启动参数追加`--disable-radix-cache`。
- **SGLANG\_SCHEDULER\_SKIP\_ALL\_GATHER**：scheduler跳过allgather同步。注意：=1在长prompt prefill场景可能影响正确性。
- **SGLANG\_TOKENIZER\_BACKEND**：tokenizer后端，`huggingface`（默认）或`fastokens`（需transformers>=5.12）。
- **SGLANG\_TOKENIZER\_TIMELINE\_LOG**：tokenizer侧跨进程batch时间线日志。
- **KUTACC\_ASYNC\_LAUNCH**：kutacc异步launch，默认1；=0时`KUPL_EXECUTOR_COUNT`需设为32。
- **KUPL\_SHM\_TYPE / KUPL\_SHM\_ON\_PACKAGE / KUPL\_SHM\_ENABLE\_HUGEPAGE**：kupl共享内存后端（sls）、package内存与大页开关。
- **SGLANG\_SET\_ZMQ\_CPU\_AFFINITY\_OFFSET**：ZMQ线程CPU亲和起始核。

#### 3.5.3 大小/阈值类

**共享内存池**

- **SGLANG\_KUNPENG\_PREFILL\_SHM\_SIZE\_MB**：prefill阶段共享内存池大小，默认476MB。
- **SGLANG\_KUNPENG\_DECODE\_SHM\_SIZE\_MB**：decode阶段共享内存池大小，默认100MB。

**HBW内存池**

- **SGLANG\_KUNPENG\_WEIGTHS\_HBW\_POOL\_SIZE\_MB**：权重缓存在高带宽内存中的预分配大小，默认3400MB（prefill侧3400、decode侧3900）。
- **SGLANG\_KUNPENG\_MEMORY\_ALIGNMENT**：HBW内存对齐字节数，默认4096。
- **SGLANG\_KUNPENG\_SWAP\_MAX\_KV\_BLOCKS**：按块换入时最大KV块数，默认512。

**SHM批处理容量**

- **SGLANG\_KUNPENG\_MAX\_SEQ\_NUM**：SHM预分配最大序列数。prefill默认8，decode默认64。
- **SGLANG\_KUNPENG\_MAX\_CUR\_LEN**：SHM预分配最大序列长度。prefill默认576，decode默认1（MTP时为投机步数+1）。
- **SGLANG\_KUNPENG\_MAX\_SEQ\_LEN**：最大序列长度，默认65536。

**SDMA传输**

- **SGLANG\_KUNPENG\_SDMA\_MAX\_EVENTS**：SDMA最大并发事件数，默认10。
- **SGLANG\_KUNPENG\_SDMA\_THRESHOLD**：SDMA传输阈值，超过此大小使用硬件DMA而非CPU memcpy，默认5。

**调度与并发**

- **CHUNKED\_PREFILL\_SIZE\_PER\_DP**：每DP rank的chunked prefill大小，默认4096（实际启动参数为该值×DP\_SIZE）。
- **STREAM\_INTERVAL**：流式输出刷新间隔token数（`--stream-interval`），默认1。
- **TOKENIZER\_WORKER\_NUM**：tokenizer worker数量，默认1。
- **SGLANG\_DISAGGREGATION\_THREAD\_POOL\_SIZE**：PD分离传输线程池大小，默认4。
- **SGLANG\_PP\_LAYER\_PARTITION**：decode PP各stage层数划分（如`"35,26"`），项数=decode PP size，总和=模型总层数。

**模型加载**

- **LOAD\_FORMAT**：权重加载格式，支持`sharded_state`与`kunpeng_state`，启用分片/预格式化加载加速启动（DeepSeek V3 INT8推荐）。

**多线程后端**

- **KUPL\_EXECUTOR\_BACKEND** / **KUPL\_EXECUTOR\_COUNT**：指定kupl多线程后端与线程数。常规PyTorch仅支持omp。`KUPL_EXECUTOR_COUNT`默认33（`KUTACC_ASYNC_LAUNCH=0`时需设为32）。

**其他**

- **DROP\_CACHES**：=1时stop.sh清理节点OS页缓存。
- **GEMM\_TILING\_PLAN\_FILE**：GEMM tiling计划文件路径。
- **SDMA\_KO\_PATH**：SDMA驱动ko文件路径。
- **KUCCL\_PATH**：kuccl安装路径（`SGLANG_ENABLE_KUCCL=1`时需要）。
- **LIBPTHREAD\_HOOK\_PATH**：libpthread\_hook.so路径（分离tokenizer模式使用）。

#### 3.5.4 路由策略类

- **ROUTER\_PREFILL\_POLICY**：Model Gateway的prefill路由策略。留空使用主策略（cache\_aware）；设为`bucket`启用bucket策略（要求`INSTANCES`含至少2个prefill条目）。
- **ROUTER\_BALANCE\_ABS\_THRESHOLD**：bucket策略下两实例累计字符数绝对差阈值，默认64。
- **ROUTER\_BALANCE\_REL\_THRESHOLD**：bucket策略下相对负载阈值，默认1.5。
- **ROUTER\_BUCKET\_ADJUST\_INTERVAL\_SECS**：bucket分界线自动调整间隔秒数，默认5。
- **ROUTER\_PREFILL\_SHORT\_COUNT**：分组模式短序列组prefill实例数，默认0（动态均衡模式）；大于0时启用短/长序列固定分组路由。
- **ROUTER\_PREFILL\_LENGTH\_THRESHOLD**：分组模式字符长度阈值，默认4096（按字符计，非token）。

策略细节与多实例路由说明参见[多实例部署指南](scripts/cpu_kunpeng/runtime/README.md)。

## 4. 正确性验证

验证脚本`scripts/cpu_kunpeng/curl.sh`支持多种功能，以下为常用用法。

### 4.1 基本用法

```shell
sh curl.sh                       # 基础推理（默认prompts/128.txt，32 req/s）
sh curl.sh -s -m 50              # 流式，50 tokens
sh curl.sh -n 10 -m 20           # 10个请求
sh curl.sh -s -f prompts/1k.txt  # 长prompt
sh curl.sh -i                    # 多轮交互式对话
```

### 4.2 路由测试

```shell
# 固定路由到指定DP rank
sh curl.sh -d 3 -n 10 -m 20
# 在rank区间内轮询（支持"0-15"、"0,2,5"等写法）
sh curl.sh -d 0-15 -n 10 -m 20
```

### 4.3 性能分析

```shell
# 开启profile模式（自动调用 start_profile / stop_profile）
sh curl.sh -p -s -f prompts/ragged.txt
```

### 4.4 参数说明

| 参数               | 说明                                                                | 默认值             |
| ---------------- | ----------------------------------------------------------------- | --------------- |
| `-p`             | 启用profile（start/stop）                                             | 关闭              |
| `-s`             | 启用流式输出                                                            | 关闭              |
| `-d RANK\|RANGE` | 固定路由到指定DP rank，或在rank集合内轮询（如`0-15`、`0,2,5`）                       | 服务端负载均衡         |
| `-n NUM`         | 请求数量（超过文件行数时循环使用prompt）                                           | 文件所有行           |
| `-m TOKENS`      | 每请求/每轮最大token数                                                    | 10（对话模式1024）    |
| `-r RATE`        | 发送速率（req/s，需大于0）                                                  | 32              |
| `-c CONC`        | 最大并发请求数                                                           | 等于请求数           |
| `-f FILE`        | prompt文件（每行一个，支持gen\_st\_prompts.py输出的JSON行）                      | prompts/128.txt |
| `-v`             | 详细统计（请求数>1时生效）：decode吞吐表、接受率（配合`-s`）、每请求时延明细                      | 关闭              |
| `-F`             | fake-transfer模式（注入bootstrap\_host=2.2.2.2，decode跳过KV传输，端口默认30002） | 关闭              |
| `-i`             | 多轮交互式对话模式                                                         | 关闭              |

通过`CURL_HOST`/`CURL_PORT`环境变量可指定目标地址与端口（默认本机网卡IP:30000）。根据请求的返回结果对正确性进行验证。

## 5. 多实例部署

在一个集群中同时部署多个prefill/decode实例（实例专属配置文件、端口/NUMA资源自动推导、短长序列分组路由、不同PP拓扑的prefill混布等）的完整说明，请参见[多实例部署指南](scripts/cpu_kunpeng/runtime/README.md)。
