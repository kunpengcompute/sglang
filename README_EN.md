# SGLang Kunpeng Optimized Reference Implementation

## Release Notes

- \[2026/10] Added GLM-5.3 deployment guide (weight preprocessing, configuration, startup); added HiCache tiered KV cache with the mooncake L3 backend, and mooncake/store deployment roles.
- \[2026/09] Updated deployment docs: added multi-instance deployment, tokenizer-separate PD disaggregated deployment, and related environment variables.
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

Two more components are required for running and compilation; see the corresponding repositories for detailed installation steps:

- **kutacc**: Kunpeng CPU communication and operator library (RDMA collectives, shm allreduce, etc.); use the branch at <https://gitcode.com/zhengzhong722/kutacc/tree/br_sglang> (the RDMA allgather/broadcast features require a kutacc build from the sglang_930 branch, see Section 3.7.1 item 8).
- **kupl**: Kunpeng acceleration runtime (kupl multi-threading backend, SHM/SDMA, etc.); use the branch at <https://gitcode.com/kunpengcompute/kupl/tree/sglang_830>.

In PD (Prefill-Decode) disaggregated deployment mode, two additional components, SGLang Model Gateway and Mooncake, are required to handle request distribution and KVCache data transfer between Prefill and Decode instances:

- **SGLang Model Gateway**: Refer to the SGLang [official documentation](https://docs.sglang.com.cn/advanced_features/sgl_model_gateway.html) for installation instructions.
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

# Load KUPL environment variables (the kupl_async module of sgl-kernel
# depends on libkupl and kupl.h)
KUPL_PATH=/path-to-KUPL
export CPATH=${KUPL_PATH}/include:$CPATH
export LIBRARY_PATH=${KUPL_PATH}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${KUPL_PATH}/lib:$LD_LIBRARY_PATH

# Enter the sgl-kernel subdirectory and install the kernel module
cd ../sgl-kernel
pip install -v . --no-build-isolation
```

### 2.3 PyTorch v2.9.0 Installation

The SGLang Kunpeng optimized version requires PyTorch v2.9.0 built with the Bisheng Compiler, with the kupl multi-threading backend enabled for optimal performance. Before installation, obtain the adaptation patch `pytorch-v2.9.0-sglang-0830.patch` from <https://gitcode.com/kunpengcompute/kunpeng-extension-for-pytorch/tree/main/thirdparty>, then apply the patch using the following commands.

```shell
git clone -b v2.9.0 --depth=1 --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
git apply pytorch-v2.9.0-sglang-0830.patch
```

## 3. SGLang Deployment & Startup

### 3.1 Using DeepSeek Models

Taking the DeepSeek V3 int8 quantized version as an example, the recommended native configuration is `DP=16`, `TP=16` (256 TP Workers in total), requiring 16 nodes. Deployment configuration is distributed across multiple files under the `scripts/cpu_kunpeng` directory:

| File | Description |
|---|---|
| `runtime/env_base.sh` | Shared defaults: paths, `INSTANCES` list, TP/DP/EP/PP sizes, functional toggles |
| `runtime/env_native.sh` | Native role defaults: node set, SHM/HBW pools, chunked prefill, etc. (prefill-side values) |
| `runtime/env_prefill.sh` | Prefill role defaults: node set, master address, port/NUMA auto-derivation |
| `runtime/env_decode.sh` | Decode role defaults: node set, master address, port/NUMA auto-derivation |
| `runtime/env_mooncake.sh` / `runtime/env_store.sh` | HiCache mooncake-backend role defaults (store master / store service, see Section 3.6) |
| `.user_env.sh` | Cluster-level user override file (loaded first), may branch on the role argument |
| `runtime/.user_env_base.sh` | Optional all-role shared override file (loaded after `.user_env.sh`) |
| `runtime/.user_env_native[_<instance>].sh` | Native role/instance override files |
| `runtime/.user_env_prefill[_<instance>].sh` | Prefill role/instance override files |
| `runtime/.user_env_decode[_<instance>].sh` | Decode role/instance override files |

Configure the model path, conda environment name, and native node information (`NATIVE_IP_SPEC`/`NATIVE_IP_FILE`, `NATIVE_MASTER_ADDR`) in `runtime/env_base.sh` (or override via `.user_env.sh`); configure prefill/decode node sets and parallel sizes in the corresponding role files. See Section 3.4 for the full configuration loading order.

Since loading the V3 quantized weights is relatively slow, set `LOAD_FORMAT` in the configuration files to enable sharded loading for faster startup. Two formats are supported: `sharded_state` and `kunpeng_state`. Before the first use, preprocess the original weights using the `scripts/cpu_kunpeng/model_processing/split_weights_dsv3.py` script.

Native mode startup and shutdown commands:

```shell
sh launch.sh native
sh stop.sh server native
```

See Section 3.5 for PD disaggregated deployment.

### 3.2 Using GLM-5.3 Models

GLM-5.3 (`GlmMoeDsaForCausalLM`) reuses the DeepseekV2 model stack (MLA attention + DSA sparse-attention indexer + MoE), and `sglang.srt.models.glm4_moe` is already in the 920F model whitelist of `srt/models/registry.py` — no code changes are required. Deployment follows the same configuration files and launch commands as DeepSeek V3 (Section 3.1); the differences are the weight preprocessing script, the model paths, and the chat template/parsers.

**1. Weight preprocessing**

Before the first use, preprocess the original quantized GLM-5.3 weights with `scripts/cpu_kunpeng/model_processing/split_weights_glm5.py` (instead of `split_weights_dsv3.py`), then set `LOAD_FORMAT=kunpeng_state` to enable sharded loading. The script automatically recognizes both GLM-5 export naming conventions (`<module>` + `<module>.scale`) and the DeepSeek pipeline convention (`<module>.weight` + `<module>.weight_scale`), and both fused (`gateup_proj`/`gate_up_proj`) and separate (`gate_proj` + `up_proj`) gate/up exports (separate exports are fused into `[gate; up]` on the fly).

Non-MoE weights (one run, outputs per attention-TP rank):

```shell
cd scripts/cpu_kunpeng/model_processing
python split_weights_glm5.py \
    --model_dir /path-to-glm5.3-quantized \
    --output_dir /path-to-glm5.3-split \
    --attention_tp_size 16 \
    --num_hidden_layers 78
```

MoE expert weights (one run; add `--layer_range <start> <end>` to parallelize across machines, each writing a disjoint layer range):

```shell
python split_weights_glm5.py --moe \
    --model_dir /path-to-glm5.3-quantized \
    --output_dir /path-to-glm5.3-split \
    --num_hidden_layers 78
```

`--num_hidden_layers` is the number of main layers (78 for GLM-5.3, per its `config.json`). The MTP layer (layer 78) is redirected to a separate `{output_dir}_mtp/` directory so it can be loaded independently as the draft model.

GLM-5.3-specific splitting handled by the script:

- `q_a_proj` + `kv_a_proj_with_mqa` are fused into `fused_qkv_a_proj_with_mqa` and split along dim0 by `--socket_tp_size` (default 8).
- DSA indexer projections follow the model-side attention-TP split by default (`indexer.wq_b` split along dim0; `indexer.wk` / `indexer.weights_proj` split along dim1). Pass `--replicate-indexer` and set `SGLANG_KUNPENG_DISABLE_INDEXER_TP_SPLIT=1` on the serving side to replicate them on every rank instead.
- bf16 tensors without a scale (`kv_b_proj`, `indexer.wk`, `weights_proj`, `lm_head`, embeddings, etc.) are copied as-is.
- MoE experts take a byte-level fast path (sendfile) when the source shards are standard safetensors; expert output layout matches the DeepSeek script (`experts/layer_L/expert_E.safetensors`, `w13_weight` = fused `[gate; up]`).

**2. Configuration**

In `runtime/env_base.sh` (or `.user_env.sh`), point the paths to the split outputs and enable sharded loading:

```shell
MODEL_PATH="/path-to-glm5.3-split"
SPECULATIVE_DRAFT_MODEL_PATH="/path-to-glm5.3-split_mtp"
export LOAD_FORMAT="kunpeng_state"
```

Set the chat template and parsers for GLM in `.user_env.sh` as needed (see the `CHAT_TEMPLATE` / `TOOL_CALL_PARSER` / `REASONING_PARSER` notes in `runtime/env_base.sh`), e.g. `TOOL_CALL_PARSER=glm47`.

Parallel sizes (TP/DP/EP/PP), native/PD modes, and MTP speculative decoding (`SGLANG_ENABLE_MTP=1`, `SPECULATIVE_DRAFT_MODEL_PATH` pointing to the `_mtp` directory) are configured exactly as for DeepSeek V3 — see Section 3.1, Section 3.5, and Section 3.7.1.

**3. Startup**

Identical to DeepSeek V3:

```shell
sh launch.sh native          # native mode
sh stop.sh server native
# or the PD disaggregated flow (Section 3.5):
sh launch.sh all
```

### 3.3 Using Other Non-DeepSeek Models

When using non-DeepSeek models, you need to comment out the condition at lines 99~114 in `srt/models/registry.py`. This logic skips loading of models outside the whitelist to improve startup speed. Alternatively, you can change the condition value to the corresponding value of the specific model being used, for example, `sglang.srt.models.qwen3`. The whitelist already covers the DeepSeek family and GLM-5.3 (`sglang.srt.models.glm4_moe`, see Section 3.2 — no changes needed); the Qwen3.5 family (`qwen3_5`/`qwen3_5_mtp`) can be allowed by setting `SGLANG_KUNPENG_QWEN=1`. For single-node Qwen3.8 deployment, refer to the [Qwen3.8 deployment guide](docs/kunpeng/Qwen3.8_README.md).

### 3.4 Deployment Modes

Select the deployment role via `./launch.sh <role> [instance]`:

- **native** (default): Non-PD disaggregated, single-cluster full serving. Nodes read from `NATIVE_IP_SPEC`/`NATIVE_IP_FILE`, load balancing via `round_robin`.
- **prefill**: PD disaggregation prefill role. Nodes read from `PREFILL_IP_SPEC`/`PREFILL_IP_FILE` (defaults in `runtime/env_prefill.sh`).
- **decode**: PD disaggregation decode role. Nodes read from `DECODE_IP_SPEC`/`DECODE_IP_FILE` (defaults in `runtime/env_decode.sh`).
- **router**: PD disaggregation router node, single-node SGLang Model Gateway (`sgl-model-gateway`, Rust binary, resolved from `bin/sgl-model-gateway` in the conda env by default, overridable via `MODEL_GATEWAY_BIN`), default policy `cache_aware`.
- **tokenizer**: Start one side's tokenizer HTTP server on the router node, usage: `./launch.sh tokenizer <prefill|decode> [instance]`, used in PD disaggregation mode.
- **mooncake**: When the HiCache L3 backend is mooncake, run the mooncake store master on `MOONCAKE_MASTER_NODE` (default `ROUTER_IP`; binary resolved from `bin/mooncake_master` in the conda env by default, overridable via `MOONCAKE_MASTER_BIN`).
- **store**: Run the mooncake store service on `MOONCAKE_STORE_NODE` (the process that actually holds the L3 shared memory segment).
- **all**: One-shot deployment: per `INSTANCES` entry, launch each instance's server cluster, then each instance's tokenizer, then the router; when any prefill/native instance enables HiCache with the mooncake L3 backend, the mooncake store master and store service are launched first automatically.
- **update**: Regenerate `.time_env.sh` and refresh NUMA binary replicas.

Script call hierarchy (`launch.sh` is a pure task dispatcher; the real logic lives in sub-scripts):

```
launch.sh (dispatcher: ./launch.sh <role> [instance])
  ├─→ runtime/launch_cluster.sh (prefill/decode/native: env loading, stop-first, SSH fan-out to nodes)
  │      └─→ server.sh (single-node process launch)
  ├─→ runtime/launch_tokenizer.sh (tokenizer: SSH to the router node)
  │      └─→ runtime/server_tokenizer.sh (tokenizer HTTP server)
  ├─→ runtime/launch_mooncake.sh (mooncake: store master)
  ├─→ runtime/launch_store.sh (store: store service)
  └─→ runtime/launch_router.sh (router: wait for all backends, then launch the Model Gateway)
         └─→ runtime/server_router.sh (SGLang Model Gateway)
```

Configuration loading order (`source env.sh <role> [instance]`):

```
runtime/env_base.sh (shared defaults; loads .user_env.sh then runtime/.user_env_base.sh)
  └─ runtime/env_native.sh / runtime/env_prefill.sh / runtime/env_decode.sh / runtime/env_mooncake.sh / runtime/env_store.sh (role defaults, ${VAR:-...} fallbacks, never overwrite set values)
       └─ runtime/.user_env_<role>_<instance>.sh (only this instance file when an instance is specified)
          or runtime/.user_env_<role>.sh (role default file when no instance is given)
```

Role defaults: prefill sets `SGLANG_KUNPENG_MAX_SEQ_NUM=8` and `SGLANG_KUNPENG_MAX_CUR_LEN=576` (native uses the same prefill-side values); decode sets `SGLANG_KUNPENG_MAX_SEQ_NUM=64` and `SGLANG_KUNPENG_MAX_CUR_LEN=1` (automatically `SGLANG_SPECULATIVE_NUM_STEPS+1` when MTP is enabled).

### 3.5 PD Disaggregated Deployment

In PD disaggregation mode the tokenizer is separated from the inference processes by default (`SGLANG_ENABLE_TOKENIZER_SEPERATE=1`, set automatically per role by `env_base.sh`; off by default in native mode). The request path is: client → SGLang Model Gateway (router node) → per-side tokenizer HTTP servers (router node, default ports 30001 for prefill and 30002 for decode) → prefill/decode clusters. The Model Gateway's backend addresses are the tokenizer ports instead of the inference masters' port 30000, and each prefill backend also carries its instance's bootstrap port (default 9001) for KV transfer bootstrap.

Deployment steps:

1. Configuration: set `ROUTER_IP`, model path, conda environment, and other shared settings in `runtime/env_base.sh` (or `.user_env.sh`); set the node sets (`PREFILL_IP_SPEC`/`DECODE_IP_SPEC`), master addresses, and parallel sizes (`PREFILL_TP_SIZE`/`DECODE_TP_SIZE`, etc., falling back to the global `TP_SIZE`/`DP_SIZE` when unset) in `runtime/.user_env_prefill.sh` and `runtime/.user_env_decode.sh` (or the corresponding instance files).

2. One-shot launch (recommended, run on any node):

```shell
sh launch.sh all   # Per INSTANCES: each instance's servers → tokenizers → router (mooncake master + store launched first for the mooncake backend)
```

3. Step-by-step launch (cluster commands run on each master node; tokenizer/router SSH to `ROUTER_IP` automatically):

```shell
sh launch.sh prefill            # prefill cluster
sh launch.sh decode             # decode cluster
sh launch.sh tokenizer prefill  # prefill-side tokenizer HTTP server
sh launch.sh tokenizer decode   # decode-side tokenizer HTTP server
sh launch.sh router             # wait for all backends, then launch the Model Gateway
```

4. Shutdown:

```shell
sh stop.sh all                 # router + all tokenizers + per-INSTANCES servers (store + mooncake master stopped last for the mooncake backend)
sh stop.sh server prefill      # instance name supported: sh stop.sh server decode 128p
sh stop.sh router
sh stop.sh tokenizer prefill   # prefill|decode|all supported
```

For multi-instance deployment (same-role multi-instance, automatic port/NUMA staggering, short/long grouped routing, etc.), see the [multi-instance deployment guide](scripts/cpu_kunpeng/runtime/README.md).

### 3.6 HiCache (Tiered KV Cache) Deployment

HiCache splits the KV cache into tiers: the further back the tier, the larger the capacity and the higher the latency. Only the prefill/native roles participate (decode does not); the page granularity is 64 tokens, and 1 object in L3 = 1 page:

- **L1**: the main KV cache pool (DDR), i.e. the original main pool;
- **L2**: an extra DDR pool (host pool), sized by `KUNPENG_HICACHE_RATIO`/`KUNPENG_HICACHE_SIZE` (three-tier mode only);
- **L3**: the external storage backend, selected by `KUNPENG_HICACHE_BACKEND` and shared by all prefill nodes.

```
export ENABLE_KUNPENG_HICACHE=1   # master switch, default 0
export KUNPENG_HICACHE_L1L3=1     # default 1 = two-tier mode (L1+L3, L2 host pool removed in favor of a fixed-size flat transfer buffer); 0 = three-tier mode (L1+L2+L3)
```

- **mooncake backend (default)**: `launch.sh all` auto-detects "prefill/native instance with HiCache enabled and the mooncake backend" and launches the mooncake store master and store service first; `stop.sh all` stops them last. Main variables: `MOONCAKE_MASTER_NODE`/`MOONCAKE_MASTER_PORT` (default `ROUTER_IP:50061`), `MOONCAKE_STORE_NODE` (defaults to the master node), `MOONCAKE_STORE_PORT` (default 8081), `MOONCAKE_STORE_SEGMENT_SIZE` (default 64gb), `MOONCAKE_DEVICE` (RDMA device list), `MOONCAKE_ENABLE_SSD_OFFLOAD` (=1 enables SSD offload; requires `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`). The memory layout is automatically set to `page_first`.
- **file backend**: set `KUNPENG_HICACHE_BACKEND=file` and point `KUNPENG_HICACHE_L3_DIR` to a directory shared by all prefill nodes (empty = L3 off, L1↔L2 only); the memory layout is `layer_first`.
- **Other settings**: `KUNPENG_HICACHE_WRITE_POLICY` (write policy, default `write_through`), `KUNPENG_HICACHE_PREFETCH_POLICY` (L3 prefetch policy, default `wait_complete`, i.e. wait for the whole page load to finish to avoid recompute; `timeout`/`best_effort` give up early), `KUNPENG_HICACHE_RATIO`/`KUNPENG_HICACHE_SIZE` (L2 size, SIZE takes precedence over RATIO), `SGLANG_HICACHE_DEBUG` (=1 verbose transfer logs).

For per-tier variable details and tuning, see the [HiCache guide](docs/kunpeng/hicache.md).

### 3.7 Environment Variables

#### 3.7.1 Functional

**1. SGLANG\_ENABLE\_BINARY\_LAUNCH / SGLANG\_ENABLE\_NUMA\_DUPLICATION — Binary Launch Mode**

```
export SGLANG_ENABLE_BINARY_LAUNCH=1      # default on
export SGLANG_ENABLE_NUMA_DUPLICATION=1   # default on
```

- **Effect**: Each scheduler process starts independently using `--tp-rank-in-node` to distinguish ranks. `server.sh` iterates `RANK_IN_NODE` to launch processes one by one, pinning CPU cores (`taskset -c`) and configuring huge pages. When disabled, only a single `python -m sglang.launch_server` process is used.
- **SGLANG\_ENABLE\_NUMA\_DUPLICATION**: On top of the above, each NUMA node uses its own pre-built binary from `PYINSTALL_PATH/dist/sglang_server_tp{rank}/` (requires `SGLANG_ENABLE_BINARY_LAUNCH`), achieving per-NUMA memory locality and reducing cross-NUMA access. When disabled, all ranks share one binary and the `LD_LIBRARY_PATH` setup logic changes. For building the replicas (PyInstaller packaging and NUMA duplication), refer to the [PyInstaller packaging guide](scripts/cpu_kunpeng/pyinstall/readme.md).

**2. SGLANG\_ENABLE\_TOKENIZER\_SEPERATE — Separate Tokenizer**

```
export SGLANG_ENABLE_TOKENIZER_SEPERATE=1  # default on for PD roles, off for native
```

- **Effect**: Splits the tokenizer out of the inference processes; dedicated HTTP servers on the router node handle tokenize/detokenize centrally.
- **Result**: In PD disaggregation, `launch.sh tokenizer` starts per-side tokenizer HTTP servers on the router node, and the Model Gateway routes requests to the tokenizer ports. Not needed in native mode.

**3. SGLANG\_ENABLE\_HBW\_POOL / SGLANG\_KUNPENG\_SWAP\_\* — HBW Memory Management**

```
export SGLANG_ENABLE_HBW_POOL=1            # enable HBW KV cache allocation, default on
export SGLANG_KUNPENG_SWAP_KV_IN=0         # KV cache swap-in (DDR → HBM via SDMA)
export SGLANG_KUNPENG_SWAP_KV_OUT=0        # KV cache write-back (HBM → DDR)
export SGLANG_KUNPENG_SWAP_KV_BLOCKWISE=0  # block-wise swap-in in decode
export SGLANG_KUNPENG_SWAP_EXPERT=1        # layer-wise expert weight swap-in (prefill default 1)
```

- **SGLANG\_ENABLE\_HBW\_POOL=1**: KV cache allocated from High Bandwidth Memory first, reducing DRAM access latency.
- **SGLANG\_KUNPENG\_SWAP\_KV\_IN/OUT=1**: Enables asynchronous SDMA migration between HBW pages and DRAM; with `SWAP_KV_BLOCKWISE=1`, decode swaps in only the needed blocks.
- **Related settings**: `SGLANG_KUNPENG_WEIGHTS_HBW_POOL_SIZE_MB`, `SGLANG_KUNPENG_SWAP_MAX_KV_BLOCKS`, `SGLANG_KUNPENG_SDMA_MAX_EVENTS`, `SGLANG_KUNPENG_SDMA_THRESHOLD`.

**4. SGLANG\_ENABLE\_MTP — Multi-Token Prediction**

```
export SGLANG_ENABLE_MTP=0              # default off
export SGLANG_SPECULATIVE_NUM_STEPS=2   # speculative steps, default 2
```

- **Effect**: Enables DeepSeek model NEXTN speculative decoding. When =1, multi-token prediction is enabled via `--speculative-algorithm NEXTN` to accelerate inference; `SGLANG_SPECULATIVE_NUM_STEPS` sets the draft steps; `SPECULATIVE_DRAFT_MODEL_PATH` optionally points to the MTP draft weights (falls back to `MODEL_PATH` when unset). On the decode side with MTP enabled, `SGLANG_KUNPENG_MAX_CUR_LEN` automatically takes `SGLANG_SPECULATIVE_NUM_STEPS+1`.

**5. SGLANG\_ENABLE\_GRAPH\_CAPTURE — Graph Capture**

```
export SGLANG_ENABLE_GRAPH_CAPTURE=1  # default on
```

- **Effect**: Captures and replays forward computation graphs to eliminate Python scheduling overhead. Related settings: `SGLANG_KUNPENG_GRAPH_CACHE_SIZE` (captured graph cache count, default 10), `SGLANG_KUNPENG_EXTEND_POWER_2_PADDING` (prefill graph padding to power-of-2 sizes, default on), `SGLANG_GRAPH_DEBUG_PRINT` (per-op debug print). Per-op timing recording is request-driven via `/start_profile`/`/stop_profile`.

**6. SGLANG\_ENABLE\_KUCCL — kuccl Communication Backend**

```
export SGLANG_ENABLE_KUCCL=0  # default off
```

- **Effect**: When =1, uses the kuccl (UCX+UCG) collective communication backend instead of gloo; requires `KUCCL_PATH`. In NUMA binary replica mode, each rank automatically uses its NUMA-local copy of the kuccl runtime plugins.

**7. SGLANG\_KUNPENG\_PP\_LAYOUT — PP Layout**

```
export SGLANG_KUNPENG_PP_LAYOUT=0  # default 0
```

- **Effect**: PP rank layout. `0` = node_block: one node maps to one PP stage (default); `1` = interleave: every node hosts all PP stages (in-node ranks 0..7 belong to PP0, 8..15 to PP1).

**8. SGLANG\_KUNPENG\_RDMA\_ALLGATHER / RDMA\_BCAST / RDMA\_PP\_COMM — RDMA Collectives**

```
export SGLANG_KUNPENG_RDMA_ALLGATHER=1  # default on
export SGLANG_KUNPENG_RDMA_BCAST=1      # default on
export SGLANG_KUNPENG_RDMA_PP_COMM=1    # default on
```

- **Effect**: Scheduler-level allgather/broadcast synchronization runs over RDMA (kutacc implementation), and rid transfer between PP ranks runs over RDMA. Requires a kutacc build from the sglang_930 branch.

**9. SGLANG\_KUNPENG\_LC\_DP\_RANKS / LC\_MIN\_SEQ\_LEN — Long-Context Decode CP**

```
export SGLANG_KUNPENG_LC_DP_RANKS=""        # DP ranks running long-context CP, e.g. "14,15"
export SGLANG_KUNPENG_LC_MIN_SEQ_LEN=4096   # long-sequence threshold
```

- **Effect**: Requests with `input_len + max_new_tokens` greater than or equal to the threshold are routed to the LC DP ranks for context-parallel decode; other requests go to regular ranks. Both variables must be identical between prefill and decode instances.

**10. ENABLE\_KUNPENG\_HICACHE — Tiered KV Cache**

```
export ENABLE_KUNPENG_HICACHE=0  # default off
```

- **Effect**: When =1, enables the HiCache tiered KV cache (prefill/native roles only); the L3 backend is selected by `KUNPENG_HICACHE_BACKEND` (mooncake/file). See Section 3.6 and the [HiCache guide](docs/kunpeng/hicache.md).

#### 3.7.2 Other Toggles

- **SGLANG\_USE\_CPU\_920F**: Identifies Kunpeng 920F platform, triggering fork startup, loading only deepseek models, CPU affinity, and other platform optimizations.
- **SGLANG\_USE\_CPU\_ENGINE**: Forces CPU inference engine.
- **SGLANG\_SET\_CPU\_AFFINITY**: Enables CPU core binding.
- **SGLANG\_LAUNCH\_HTTP\_ONLY**: Starts HTTP server only (no inference), used by router node. Requires `SGLANG_ENABLE_TOKENIZER_SEPERATE`.
- **SGLANG\_SKIP\_HTTP**: Skips HTTP server startup, used by prefill/decode nodes. Requires `SGLANG_ENABLE_TOKENIZER_SEPERATE`.
- **SGLANG\_KUNPENG\_PROFILE**: Prints function call durations to identify performance bottlenecks.
- **SGLANG\_KUNPENG\_PP\_PROFILE**: Decode pipeline profiling.
- **SGLANG\_KUNPENG\_MOE\_FORCE\_LOAD\_BALANCE**: =1 forces MoE load balancing (perf-test only, correctness not preserved).
- **SGLANG\_KUNPENG\_MOE\_SHUFFLE\_MODE**: Dynamic redundant-expert remap shuffle mode, 0 = round-robin, 1 = random.
- **SGLANG\_KUNPENG\_MOE\_TOKEN\_MULTIPLE**: MoE token padding multiple, default 2.
- **SGLANG\_KUNPENG\_SAVE\_AT\_TRACE**: =1 records expert activation counts.
- **SGLANG\_DISABLE\_RADIX\_CACHE**: =1 appends `--disable-radix-cache` to server args.
- **SGLANG\_SCHEDULER\_SKIP\_ALL\_GATHER**: Scheduler skips allgather synchronization. WARNING: =1 can affect correctness in long-prompt prefill.
- **SGLANG\_TOKENIZER\_BACKEND**: Tokenizer backend, `huggingface` (default) or `fastokens` (requires transformers >= 5.12).
- **SGLANG\_TOKENIZER\_TIMELINE\_LOG**: Tokenizer-side cross-process batch timeline logging.
- **KUTACC\_ASYNC\_LAUNCH**: kutacc async kernel launch, default 1; when =0, `KUPL_EXECUTOR_COUNT` must be set to 32.
- **KUPL\_SHM\_TYPE / KUPL\_SHM\_ON\_PACKAGE / KUPL\_SHM\_ENABLE\_HUGEPAGE**: kupl SHM backend (sls), package memory, and hugepage toggles.
- **SGLANG\_SET\_ZMQ\_CPU\_AFFINITY\_OFFSET**: Starting core for ZMQ thread CPU affinity.
- **SGLANG\_KUNPENG\_QWEN**: =1 allows loading the Qwen3.5 family models (`qwen3_5`/`qwen3_5_mtp`) (see Section 3.3).
- **SGLANG\_KUNPENG\_DISABLE\_MLA\_ALL2ALL**: =1 disables the MLA alltoall path, default 0.
- **SGLANG\_KUNPENG\_ALLTOALL\_FENCE**: =1 appends an explicit `kupl_shm_fence` after the MLA shm alltoall, default 0.
- **SGLANG\_KUNPENG\_ALLREDUCE\_NAIVE**: =1 uses the non-kutacc naive shm allreduce, default 0.
- **SGLANG\_KUNPENG\_ENABLE\_SHM\_FENCE**: SHM fence toggle, default 0.
- **SGLANG\_ENABLE\_OVERLAP**: Scheduler overlap toggle, default 0; `SGLANG_ENABLE_OVERLAP_TRACE` is the corresponding trace toggle.
- **SGLANG\_KUNPENG\_HBW\_RESIDENT\_NON\_EXPERT\_WEIGHTS / \_EXPERT\_WEIGHTS / \_GRAPH**: HBW-residency toggles for non-expert/expert weights and graph weights. Non-expert and graph default 1; expert weights default 0 globally and 1 on the decode side.
- **SGLANG\_KUNPENG\_MOE\_PREFILL\_FORCE\_DDR**: =1 forces prefill MoE weights to DDR (for long-prompt prefill instances).
- **LONG\_PROMPT\_PREFILL\_INSTANCE**: =1 enables the long-prompt prefill instance preset (TP=16/DP=1/PP=16, chunk 2048, MoE forced to DDR).

#### 3.7.3 Size / Threshold

**Shared Memory Pool**

- **SGLANG\_KUNPENG\_PREFILL\_SHM\_SIZE\_MB**: Prefill shared memory pool size, default 476MB.
- **SGLANG\_KUNPENG\_DECODE\_SHM\_SIZE\_MB**: Decode shared memory pool size, default 100MB.

**HBW Memory Pool**

- **SGLANG\_KUNPENG\_WEIGHTS\_HBW\_POOL\_SIZE\_MB**: Weight cache pre-allocation in HBW, default 3400MB (prefill 3400, decode 3900).
- **SGLANG\_KUNPENG\_MEMORY\_ALIGNMENT**: HBW memory alignment in bytes, default 4096.
- **SGLANG\_KUNPENG\_SWAP\_MAX\_KV\_BLOCKS**: Max KV blocks for block-wise swap-in, default 512.

**SHM Batch Capacity**

- **SGLANG\_KUNPENG\_MAX\_SEQ\_NUM**: Max sequences for SHM pre-allocation. Prefill default 8, decode default 64.
- **SGLANG\_KUNPENG\_MAX\_CUR\_LEN**: Max sequence length for SHM pre-allocation. Prefill default 576, decode default 1 (`speculative steps + 1` with MTP).
- **SGLANG\_KUNPENG\_MAX\_SEQ\_LEN**: Max sequence length, default 65536.

**SDMA Transfer**

- **SGLANG\_KUNPENG\_SDMA\_MAX\_EVENTS**: Max SDMA concurrent events, default 10.
- **SGLANG\_KUNPENG\_SDMA\_THRESHOLD**: SDMA transfer threshold; above this size hardware DMA is used instead of CPU memcpy, default 5.

**Scheduling & Concurrency**

- **CHUNKED\_PREFILL\_SIZE\_PER\_DP**: Per-DP-rank chunked prefill size, default 4096 (actual server arg is this value × DP_SIZE).
- **STREAM\_INTERVAL**: Streaming output flush interval in tokens (`--stream-interval`), default 1.
- **TOKENIZER\_WORKER\_NUM**: Number of tokenizer workers, default 1.
- **SGLANG\_DISAGGREGATION\_THREAD\_POOL\_SIZE**: PD disaggregation transfer thread pool size, default 4.
- **SGLANG\_PP\_LAYER\_PARTITION**: Per-stage layer partition for decode PP (e.g. `"35,26"`); entry count = decode PP size, sum = total model layers.

**Model Loading**

- **LOAD\_FORMAT**: Weight loading format; supports `sharded_state` and `kunpeng_state` for sharded/pre-formatted loading to accelerate startup (recommended for DeepSeek V3 INT8).

**Multi-Thread Backend**

- **KUPL\_EXECUTOR\_BACKEND** / **KUPL\_EXECUTOR\_COUNT**: Specifies the kupl multi-threading backend and thread count. Standard PyTorch only supports omp. `KUPL_EXECUTOR_COUNT` defaults to 33 (set to 32 when `KUTACC_ASYNC_LAUNCH=0`).
- **KUTACC\_FUSEDMOE\_MULTIEXPT**: kutacc fusedmoe multi-expert parallelism group count (decode side). Default 0 = serial per-expert path (bit-exact); when set to G, the worker pool is split into G groups, each owning a share of the local experts.

**Others**

- **DROP\_CACHES**: =1 drops the node OS page cache in stop.sh.
- **GEMM\_TILING\_PLAN\_FILE**: GEMM tiling plan file path, default `scripts/cpu_kunpeng/configs/gemm_32_tiling.csv`.
- **SDMA\_KO\_PATH**: SDMA driver ko file path.
- **KUCCL\_PATH**: kuccl installation path (required when `SGLANG_ENABLE_KUCCL=1`).
- **LIBPTHREAD\_HOOK\_PATH**: libpthread_hook.so path (used in tokenizer-separate mode).

#### 3.7.4 Routing Policy

- **ROUTER\_PREFILL\_POLICY**: Model Gateway prefill routing policy. Empty = use the main policy (cache_aware); set to `bucket` to enable the bucket policy (requires at least 2 prefill entries in `INSTANCES`).
- **ROUTER\_BALANCE\_ABS\_THRESHOLD**: Bucket-policy absolute character-count difference threshold between two instances, default 64.
- **ROUTER\_BALANCE\_REL\_THRESHOLD**: Bucket-policy relative load threshold, default 1.5.
- **ROUTER\_BUCKET\_ADJUST\_INTERVAL\_SECS**: Bucket boundary auto-adjustment interval in seconds, default 5.
- **ROUTER\_PREFILL\_SHORT\_COUNT**: Number of prefill instances in the short-sequence group for grouped mode, default 0 (dynamic balancing mode); values > 0 enable fixed short/long grouped routing.
- **ROUTER\_PREFILL\_LENGTH\_THRESHOLD**: Grouped-mode character length threshold, default 4096 (counts characters, not tokens).

For policy details and multi-instance routing, see the [multi-instance deployment guide](scripts/cpu_kunpeng/runtime/README.md).

## 4. Correctness Verification

The verification script `scripts/cpu_kunpeng/curl.sh` supports multiple features. Common usage examples:

### 4.1 Basic Usage

```shell
sh curl.sh                       # basic inference (default prompts/128.txt, 32 req/s)
sh curl.sh -s -m 50              # streaming, 50 tokens
sh curl.sh -n 10 -m 20           # 10 requests
sh curl.sh -s -f prompts/1k.txt  # long prompt
sh curl.sh -J -f chat.jsonl      # chat mode: -f is a JSONL request-body file, posted to /v1/chat/completions
sh curl.sh -i                    # multi-turn interactive chat
```

### 4.2 Routing Test

```shell
# Pin every request to a specific DP rank
sh curl.sh -d 3 -n 10 -m 20
# Round-robin across a rank set (supports "0-15", "0,2,5", etc.)
sh curl.sh -d 0-15 -n 10 -m 20
```

### 4.3 Profiling

```shell
# Profile mode (automatically calls start_profile / stop_profile;
# in PD layout, sent to each side's tokenizer HTTP port)
sh curl.sh -p -s -f prompts/ragged.txt
```

### 4.4 Options

| Option | Description | Default |
|---|---|---|
| `-p` | Enable profiling (start/stop; sent to each side's tokenizer HTTP ports in PD layout) | off |
| `-s` | Enable streaming | off |
| `-d RANK\|RANGE` | Pin to a DP rank, or round-robin across a rank set (e.g. `0-15`, `0,2,5`) | server load balancing |
| `-n NUM` | Number of requests (prompts cycle when NUM exceeds the file size) | all lines in file |
| `-m TOKENS` | Max tokens per request / per turn | 10 (1024 in chat mode) |
| `-r RATE` | Send rate in req/s (must be > 0) | 32 |
| `-c CONC` | Max concurrent requests | equals request count |
| `-f FILE\|DIR` | Prompt source: a file (one prompt per line; JSON-string lines from gen_st_prompts.py supported) or a directory (each \*.txt/\*.md file inside as ONE whole-file prompt, sorted by name and cycled) | prompts/128.txt |
| `-J` | Chat mode: `-f` file is JSONL with one chat request body per line (`model`/`messages`), posted to `/v1/chat/completions` so the server renders the chat template | off |
| `-v` | Verbose stats (active when NUM > 1): decode throughput table, accept rate (with `-s`), per-request latency details | off |
| `-F` | Fake-transfer mode (injects bootstrap_host=2.2.2.2; decode skips KV transfer; port auto-derived from the decode entry in INSTANCES, `CURL_PORT` overrides) | off |
| `-i` | Multi-turn interactive chat mode | off |

The target address and port can be overridden via the `CURL_HOST`/`CURL_PORT` environment variables (default: local NIC IP:30000). Verify correctness based on the response returned by the request.

### 4.5 Benchmarks

`scripts/cpu_kunpeng/benchmark.sh` supports GSM8K and MMLU accuracy benchmarks plus aisbench performance runs (against the gateway port):

```shell
bash benchmark.sh gsm8k --max-new-tokens 128 --num-questions 50
bash benchmark.sh mmlu  --nsub 60 --parallel 32
```

Defaults for each benchmark can be overridden via environment variables; see the script header for details.

## 5. Multi-Instance Deployment

For deploying multiple prefill/decode instances in one cluster (per-instance configuration files, automatic port/NUMA resource derivation, short/long grouped routing, mixed prefill topologies with different PP sizes, etc.), see the [multi-instance deployment guide](scripts/cpu_kunpeng/runtime/README.md).
