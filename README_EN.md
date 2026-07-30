# SGLang Kunpeng Optimized Reference Implementation

## Release Notes

- \[2026/05] This release is based on SGLang official v0.5.11 with optimizations and adaptations.

## 1. Overview

The SGLang Kunpeng optimized version is a reference implementation for deploying the DeepSeek V3 model as a service, built upon the open-source high-performance inference framework SGLang and the Kunpeng platform. This implementation fully leverages the many-core parallelism and high-bandwidth interconnect capabilities of Kunpeng processors, combined with SGLang's KV cache management mechanism, PD (Prefill-Decode) disaggregated architecture, and efficient concurrency scheduling, to deliver low-latency, high-throughput request processing for Kunpeng ecosystem users.

### 1.1 Version Compatibility

- Hardware Platform
  - Kunpeng 920 Professional Edition
- System Requirements
  - openEuler 22.03 (LTS-SP4) AArch64

## 2. SGLang Installation and Deployment

### 2.1 SGLang Dependency Installation

Before installing SGLang components, you need to create and activate a Python virtual environment. The reference commands are as follows. For more information, refer to the [installation guide](https://docs.sglang.com.cn/platforms/cpu_server.html#install-from-source) provided by the SGLang community.

```shell
source ~/anaconda3/start_conda.sh
conda create -n sgl-cpu python=3.12 -y
conda activate sgl-cpu
```

In addition, compiling the SGLang kernel module requires the Bisheng Compiler. Please obtain the HPCKit installation package from <https://www.hikunpeng.com/developer/hpc/hpckit-download> and follow the official installation steps.

In PD (Prefill-Decode) disaggregated deployment mode, two additional components, SGLang Router and Mooncake, are required to handle request distribution and KVCache data transfer between Prefill and Decode instances:

- **SGLang Router**: Refer to the SGLang [official documentation](https://docs.sglang.com.cn/advanced_features/sgl_model_gateway.html) for installation instructions.
- **Mooncake**: Refer to the Mooncake official [GitHub repository](https://github.com/kvcache-ai/Mooncake#-quick-start) for installation instructions.

### 2.2 SGLang Application Installation

Before installing SGLang, activate the configured Python virtual environment. Then, obtain the SGLang source code from the installation path and execute the following commands to complete the build and installation:

```shell
# Install the SGLang main module (including CPU-related dependencies)
cd sglang/python
pip install --upgrade pip setuptools
pip install -e .
pip install torchvision==0.24.0 triton==3.5.0 --force-reinstall

# Load Bisheng Compiler environment variables
HPCKIT_PATH=/path-to-HPCKit
source ${HPCKIT_PATH}/latest/compiler/bisheng/env/setvars.sh
export CC=$(which clang)
export CXX=$(which clang++)

# Load KUTACC environment variables
KUTACC_PATH=/path-to-KUTACC
export KUTACC_LIB=${KUTACC_PATH}/install/lib
export KUTACC_INCLUDE=${KUTACC_PATH}/install/include

# Enter the sgl-kernel subdirectory and install the kernel module
cd ../sgl-kernel
pip install -v . --no-build-isolation
```

### 2.3 PyTorch v2.9.0 Installation

The SGLang Kunpeng optimized version requires PyTorch v2.9.0 built with the Bisheng Compiler, with the kupl multi-threading backend enabled for optimal performance. Before installation, obtain the adaptation patch from <https://gitcode.com/kunpengcompute/kunpeng-extension-for-pytorch/tree/main/thirdparty>, then apply the patch using the following commands.

```shell
git clone -b v2.9.0 --depth=1 --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
git apply pytorch-v2.9.0-kupl.patch
```

## 3. SGLang Deployment & Startup

### 3.1 Using DeepSeek Models

Taking the DeepSeek V3 int8 quantized version as an example, the recommended configuration is `DP=16`, `TP=16` (256 TP Workers in total), requiring 16 nodes. Configure `NATIVE_IP_SPEC`, `NATIVE_MASTER_ADDR`, the model path, the conda environment name, and other settings in the `env.sh` script under the `scripts/cpu_kunpeng` directory.

Since loading the V3 quantized weights is relatively slow, you can add `--load-format sharded_state` to the startup parameters to enable sharded weight loading for faster startup. Before the first use, preprocess the original weights using the `scripts/cpu_kunpeng/split_weights.py` script.

Startup and shutdown commands:

```shell
# native mode (non-PD disaggregated)
sh launch.sh native
sh stop.sh native

# PD disaggregated mode (run on prefill, decode, and router nodes respectively)
sh launch.sh prefill
sh launch.sh decode
sh launch.sh router
```

### 3.2 Using Non-DeepSeek Models

When using non-DeepSeek models, you need to comment out lines 98~99 in `srt/models/registry.py`. These two lines skip loading non-DeepSeek model dependencies to improve startup speed. Alternatively, you can change the condition value to the corresponding value of the specific model being used, for example, `sglang.srt.models.qwen3`.

### 3.3 Deployment Modes

Four deployment modes are available via `source env.sh [mode]`:

- **native** (default): Non-PD disaggregated, single-cluster full serving. Nodes read from `NATIVE_IP_SPEC`, load balancing via `round_robin`.
- **prefill**: PD disaggregation prefill role. Nodes read from `PREFILL_IP_SPEC`.
- **decode**: PD disaggregation decode role. Nodes read from `DECODE_IP_SPEC`.
- **router**: PD disaggregation router node, single-node running `sglang_router.launch_router`.

When `IS_PREFILL=1` (prefill mode), `SGLANG_KUNPENG_MAX_SEQ_NUM=4` and `SGLANG_KUNPENG_MAX_CUR_LEN=1024` are set automatically; when `IS_PREFILL=0` (decode/native), `SGLANG_KUNPENG_MAX_SEQ_NUM=128` and `SGLANG_KUNPENG_MAX_CUR_LEN=1` are set.

### 3.4 Environment Variables

#### 3.4.1 Functional

**1. SGLANG\_ENABLE\_BINARY\_LAUNCH / SGLANG\_ENABLE\_NUMA\_DUPLICATION — Binary Launch Mode**

```
export SGLANG_ENABLE_BINARY_LAUNCH=1      # default on
export SGLANG_ENABLE_NUMA_DUPLICATION=1   # default on
```

- **Effect**: Each scheduler process starts independently using `--tp-rank-in-node` to distinguish ranks. `server.sh` iterates `ATTN_TP_RANK` to launch processes one by one, pinning CPU cores (`taskset -c`) and configuring huge pages. When disabled, only a single `python -m sglang.launch_server` process is used.
- **SGLANG\_ENABLE\_NUMA\_DUPLICATION**: On top of the above, each NUMA node uses its own pre-built binary from `PYINSTALL_PATH/dist/sglang_server_tp{rank}/` (requires `SGLANG_ENABLE_BINARY_LAUNCH`), achieving per-NUMA memory locality and reducing cross-NUMA access. When disabled, all ranks share one binary and the `LD_LIBRARY_PATH` setup logic changes.

**2. SGLANG\_ENABLE\_TOKENIZER\_SEPERATE — Separate Tokenizer**

```
export SGLANG_ENABLE_TOKENIZER_SEPERATE=0  # default off
```

- **Effect**: Splits the tokenizer out of the inference process. When =1, triggers three sub-functions:

| Sub-mode | Condition |
|---|---|
| `is_tokenizer_separate()` | `=1` |
| `is_http_only()` | and `SGLANG_LAUNCH_HTTP_ONLY=1` |
| `is_skip_http()` | and `SGLANG_SKIP_HTTP=1` |

- **Related settings**: `SGLANG_LAUNCH_HTTP_ONLY=1` (router node runs HTTP only), `SGLANG_SKIP_HTTP=1` (inference nodes skip HTTP), `ROUTER_IP` (HTTP server binds to router IP instead of local IP).
- **Effect**: In PD disaggregation, the router node handles tokenize/detokenize centrally while inference nodes focus on forward. Not needed in non-PD (native) mode.

**3. SGLANG\_KUNPENG\_DISABLE\_MLA\_ALL2ALL — Disable MLA All2All**

```
export SGLANG_KUNPENG_DISABLE_MLA_ALL2ALL=1  # default on
```

- **Effect**: Controls the MLA attention communication strategy.
  - `=0` (not disabled): When TP=8 or 16, Q and attention_output use per-socket all2all (group_size=8), cross-socket merging via o_proj allreduce, reducing cross-socket SHM traffic.
  - `=1` (disabled): No per-socket all2all; attn_mqa computes on the full TP group, communication handled entirely by RowParallelLinear's allreduce.

**4. SGLANG\_ENABLE\_HBW\_POOL / SGLANG\_ENABLE\_HBW\_SWAP — HBW Memory Management**

```
export SGLANG_ENABLE_HBW_POOL=1   # enable HBW KV cache allocation
export SGLANG_ENABLE_HBW_SWAP=0   # disable HBW↔DRAM swap
```

- **SGLANG\_ENABLE\_HBW\_POOL=1**: KV cache allocated from High Bandwidth Memory first, reducing DRAM access latency.
- **SGLANG\_ENABLE\_HBW\_SWAP=1**: Enables asynchronous SDMA migration between HBW pages and DRAM, swapping out cold data under memory pressure.
- **Related settings**: `SGLANG_KUNPENG_WEIGTHS_HBW_POOL_SIZE_MB`, `SGLANG_KUNPENG_SDMA_MAX_EVENTS`, `SGLANG_KUNPENG_SDMA_THRESHOLD`.

**5. SGLANG\_ENABLE\_MTP — Multi-Token Prediction**

```
export SGLANG_ENABLE_MTP=0  # default off
```

- **Effect**: Enables DeepSeek model Multi-Token Prediction speculative decoding. When =1, enables multi-token prediction to accelerate inference.

#### 3.4.2 Other Toggles

- **SGLANG\_USE\_CPU\_920F**: Identifies Kunpeng 920F platform, triggering fork startup, loading only deepseek_v2 model, CPU affinity, etc.
- **SGLANG\_USE\_CPU\_ENGINE**: Forces CPU inference engine.
- **SGLANG\_SET\_CPU\_AFFINITY**: Enables CPU core binding.
- **SGLANG\_LAUNCH\_HTTP\_ONLY**: Starts HTTP server only (no inference), used by router node. Requires `SGLANG_ENABLE_TOKENIZER_SEPERATE`.
- **SGLANG\_SKIP\_HTTP**: Skips HTTP server startup, used by prefill/decode nodes. Requires `SGLANG_ENABLE_TOKENIZER_SEPERATE`.
- **SGLANG\_DISAGGREGATION\_FORCE\_QUERY\_PREFILL\_DP\_RANK**: Forces query prefill DP rank in PD disaggregation, bypassing `follow_bootstrap_room` routing.
- **SGLANG\_ENABLE\_TORCH\_COMPILE** / **TORCH\_COMPILE\_DISABLE**: Enables torch.compile (disabled in this version).
- **SGLANG\_LOG\_MS**: Adds millisecond timestamps to logs.
- **SGLANG\_KUNPENG\_PROFILE**: Prints function call durations to identify performance bottlenecks.
- **SGLANG\_ENABLE\_TP\_MEMORY\_INBALANCE\_CHECK**: TP memory capacity imbalance check (set to 0 to disable).

#### 3.4.3 Size / Threshold

**Shared Memory Pool**
- **SGLANG\_KUNPENG\_PREFILL\_SHM\_SIZE\_MB**: Prefill shared memory pool size, default 476MB.
- **SGLANG\_KUNPENG\_DECODE\_SHM\_SIZE\_MB**: Decode shared memory pool size, default 100MB.

**HBW Memory Pool**
- **SGLANG\_KUNPENG\_WEIGTHS\_HBW\_POOL\_SIZE\_MB**: Weight cache pre-allocation in HBW, default 3800MB.

**SHM Batch Capacity**
- **SGLANG\_KUNPENG\_MAX\_SEQ\_NUM**: Max sequences for SHM pre-allocation. Prefill default 4, decode default 128.
- **SGLANG\_KUNPENG\_MAX\_CUR\_LEN**: Max sequence length for SHM pre-allocation. Prefill default 1024, decode default 1.

**SDMA Transfer**
- **SGLANG\_KUNPENG\_SDMA\_MAX\_EVENTS**: Max SDMA concurrent events, default 10.
- **SGLANG\_KUNPENG\_SDMA\_THRESHOLD**: SDMA transfer threshold; above this size use hardware DMA instead of CPU memcpy, default 5.

**Timeout**
- **SGLANG\_WARMUP\_TIMEOUT**: Warmup timeout in seconds, default 1600.

**Model Loading**
- **LOAD\_FORMAT**: Weight loading format. Set to `sharded_state` for sharded loading to accelerate startup (recommended for DeepSeek V3 INT8).

**Multi-Thread Backend**
- **KUPL\_EXECUTOR\_BACKEND** / **KUPL\_EXECUTOR\_COUNT**: Specifies kupl multi-threading backend. Standard PyTorch only supports omp. `KUPL_EXECUTOR_COUNT` defaults to 32.

## 4. Correctness Verification

The verification script `scripts/cpu_kunpeng/curl.sh` supports multiple features. Common usage examples:

### 4.1 Basic Usage

```shell
sh curl.sh                    # basic inference
sh curl.sh -s -m 50           # streaming, 50 tokens
sh curl.sh -n 10 -m 20        # 10 requests
sh curl.sh -s -f prompts/1k.txt  # long prompt
```

### 4.2 Routing Test

```shell
# Route to a specific DP rank
sh curl.sh -d 3 -n 10 -m 20
```

### 4.3 Profiling

```shell
# Profile mode (automatically calls start_profile / stop_profile)
sh curl.sh -p -s -f prompts/ragged.txt
```

### 4.4 Options

| Option | Description | Default |
|---|---|---|
| `-p` | Enable profiling (start/stop) | off |
| `-s` | Enable streaming | off |
| `-d RANK` | Route to specific DP rank | - |
| `-n NUM` | Number of requests | all lines in file |
| `-m TOKENS` | Max tokens per request | 10 |
| `-f FILE` | Prompt file | prompts/5.txt |

Verify correctness based on the response returned by the request.

## 5. Prefill Multi-Instance Deployment

### 5.1 Configuration

Configure two prefill instances in `scripts/cpu_kunpeng/env.sh`:

```bash
# Default instance (short requests)
PREFILL_IP_SPEC="xxx.xxx.xxx. | 1-16"
PREFILL_MASTER_ADDR="xxx.xxx.xxx.1"
PREFILL_MASTER_PORT="5000"
PREFILL_PP_SIZE=1

# Long prompt instance
PREFILL_LONG_PROMPT_IP_SPEC="xxx.xxx.xxx. | 17-32"
PREFILL_LONG_PROMPT_MASTER_ADDR="xxx.xxx.xxx.17"
PREFILL_LONG_PROMPT_MASTER_PORT="5020"
PREFILL_LONG_PROMPT_PP_SIZE=2
```

### 5.2 Startup

#### 1. Single Instance Mode (Default)

```bash
# Prefill (run on prefill master node)
sh launch.sh prefill

# Decode (run on decode master node)
sh launch.sh decode

# Router (run on router node)
sh launch.sh router
```

#### 2. Dual Instance Mode (Bucket Load Balancing Strategy)

```bash
# Instance 1 (short request prefill, nodes 1~16, run on node 1)
sh launch.sh prefill

# Instance 2 (long request prefill, nodes 17~32, run on node 17)
sh launch.sh prefill long_prompt

# Decode (run on decode master node)
sh launch.sh decode

# Router (run on router node with bucket strategy enabled)
sh launch.sh router prefill_bucket
```

### 5.3 Notes

- The two prefill instances have different master nodes and must be started on their respective master nodes
- The `MASTER_PORT` of the two instances must differ (default: 5000 vs 5020)
- When the `prefill_bucket` parameter is passed to the router, it automatically registers two prefill instances and enables the bucket load balancing strategy
- When `prefill_bucket` is not passed, only one prefill instance is registered and the strategy specified by `--policy` is used
- The `PP_SIZE` of the long request instance can be configured independently (default: 2)

### 5.4 Bucket Strategy CLI Parameters

| Parameter | Default | Description |
|---|---|---|
| `--prefill-policy bucket` | - | Route by request character count to different prefill instances via bucket splitting |
| `--balance-abs-threshold` | 64 | Trigger load-balancing switch when the absolute difference in character counts between two instances exceeds this value |
| `--balance-rel-threshold` | 1.5 | Imbalance is only considered when the higher-load instance has >1.5x characters compared to the lower-load instance (must be met together with the absolute threshold) |
| `--bucket-adjust-interval-secs` | 5 | Automatically adjust boundaries based on historical load every 5 seconds |

#### Load Balancing Strategy

When a request arrives, it is first routed to the corresponding prefill instance based on character count using binary search to locate the matching boundary bucket. Simultaneously, the cumulative character counts of both instances are checked. When **both** of the following conditions are met, the load is considered unbalanced and the request is routed to the lower-load instance instead:

- Absolute difference: `max_load - min_load > balance_abs_threshold`
- Relative difference: `max_load > balance_rel_threshold × min_load`

Both conditions must be satisfied simultaneously to trigger the switch, preventing frequent switching due to minor load imbalances.

#### Boundary Auto-Adjustment

Initial boundaries evenly divide the character length range `[0, 4096]` across instances (2 instances: instance 1 handles `[0, 2047]`, instance 2 handles `[2048, MAX]`). A background thread runs auto-adjustment every `bucket_adjust_interval_secs` seconds:

1. Collect character count distribution of requests within the time window (`period = bucket_adjust_interval_secs × 1000` milliseconds)
2. Redistribute boundary ranges across instances based on historical load to balance workload
3. Skip adjustment if load change is minor (new load < 2x old load AND old load < 2x new load)

If most requests are short (< 1k characters), instance 1's load will be higher, causing the boundary to automatically shift upward and eventually converge near the 1k position.
