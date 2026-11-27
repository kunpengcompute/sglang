# Qwen 920F 部署

统一入口为 `bash run.sh ACTION /绝对路径/config.sh`。配置和运行输出放在源码外的 `DEPLOY_DIR`。本示例是4个Prefill节点（TP16×PP4）和1个Decode节点（TP16），MTP3，支持文本及多模态。

## 1. 环境与配置

复制 `config.example.sh` 为自己的 `config.sh`，按标题从上到下修改。示例已填本集群路径；换机器至少核对以下内容。

| 配置 | 含义 |
|---|---|
| `SGLANG_SOURCE` / `DEPLOY_DIR` | 本次源码和新的部署输出目录，不覆盖旧release |
| `MODEL_PATH` / `SPECULATIVE_DRAFT_MODEL_PATH` | 主模型及Draft权重 |
| `PYTHON` / `CLIENT_PYTHON` | 推理Python 3.12环境 / 仅运行标准库控制脚本的Python |
| `CC` / `CXX` / `COMPILER_RUNTIME_LIB` | 配套毕昇编译器及运行库；libomp.so来自编译器环境 |
| `KUTACC_PREFIX` / `KUPL_PREFIX` | 对应SDK的include/lib前缀 |
| `TL_ROOT` / `TVM_LIBRARY_PATH` / `TVM_IMPORT_PYTHON_PATH` | Kunpeng TileLang及其TVM路径，启用TileLang时需要 |
| `PERF_TUNE` | 公共非root代理、驱动、大页和调频脚本目录 |
| `PD_NODE_HOSTS` / `PD_ADVERTISE_HOSTS` | 已申请节点及可达IP，先P后D；工具不申请节点 |
| `PD_SSH_OPTIONS` / `PD_IB_DEVICE` / `PD_NET_INTERFACE` | 可信SSH配置、RDMA设备和控制网卡 |

该核表要求aarch64、608 CPU、TP16；DDR NUMA0–15每NUMA 38 CPU，HBW NUMA16–31。同为920F但NUMA/CPU编号不同，需要调整核表。还需准备RDMA设备、驱动、cmake、ldd、taskset、numactl及已授权的公共非root机制。

通用服务参数使用公共名称，由启动工具转为原生SGLang命令行参数，没有新增模型前向接口：

| 配置 | SGLang接口 |
|---|---|
| `MODEL_PATH` | `--model-path` |
| `CONTEXT_LENGTH` | `--context-length` |
| `SERVER_PORT` / `DIST_PORT` | `--port` / `--nccl-port`及默认分布式初始化地址 |
| `MAX_RUNNING_REQUESTS` / `MAX_TOTAL_TOKENS` | `--max-running-requests` / `--max-total-tokens` |
| `TP_SIZE` / `PP_SIZE` / `NNODES` / `NODE_RANK` | 对应TP、PP和多节点参数 |
| `ATTENTION_BACKEND` / `TOOL_CALL_PARSER` / `RANDOM_SEED` | 对应backend、parser、seed参数 |

PD模式只填写 `PD_PREFILL_*` 和 `PD_DECODE_*` 的请求/KV容量；工具为各角色生成上表中的单节点参数。Router端口单独由 `PD_ROUTER_PORT` 配置。示例 `CONTEXT_LENGTH=262152` 包含256k输入输出总容量及8个内部MTP槽位，不表示256k输入再额外输出256k。

### TileLang配套环境

本次配套版本：

- KuTACC：`kutacc_20261104_edfe1e8fa`。
- [KuPL](https://atomgit.com/kunpengcompute/kupl)：`sglang_830`分支，`07c66678448819faae8c06a8eaf66d32145d8926`，库版本1.26.2.0，含SLS映射重试/错误清理修复。
- [Kunpeng TileLang](https://atomgit.com/mccllm/tilelang-kunpeng)：`sve_instruction_1116`分支，`efcb6418834a216f9fd0b59c1eef02ebcd4a150c`。
- 毕昇：HPCKit 26.0.RC1。Python环境还需SGLang依赖，以及TileLang要求的Cython、ml_dtypes、tornado等依赖。

TileLang是外部SDK；`run.sh build`编译SGLang C++扩展，不下载或重建TileLang。换环境时，先按该SDK安装说明准备匹配环境。对应原生构建方式为：

```bash
git clone --recursive --branch sve_instruction_1116 https://atomgit.com/mccllm/tilelang-kunpeng "$TL_ROOT"
git -C "$TL_ROOT" checkout efcb6418834a216f9fd0b59c1eef02ebcd4a150c
git -C "$TL_ROOT" submodule update --init --recursive
mkdir -p "$TL_ROOT/build"
cat > "$TL_ROOT/build/config.cmake" <<'EOF'
set(USE_LLVM OFF)
set(USE_CUDA OFF)
set(USE_ROCM OFF)
set(USE_ASCEND OFF)
set(USE_GTEST OFF)
set(USE_LIBBACKTRACE OFF)
EOF
cmake -S "$TL_ROOT" -B "$TL_ROOT/build" -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "$TL_ROOT/build" --target tilelang_module --parallel 32
"$PYTHON" -c 'import site,os; from pathlib import Path; Path(site.getsitepackages()[0], "tilelang_kunpeng.pth").write_text(os.environ["TL_ROOT"]+"\n")'
```

在已分配计算资源上编译SDK，不在登录节点编译。最后一行将SDK登记到选定Python环境，供NUMA副本正常导入；仅填写TL_ROOT不能代替这一步。启用 `SGLANG_KUNPENG_GDN_VERIFY_TILELANG=1` 后，pack/check会核对TileLang/TVM依赖、`T.sve`及原生库内容。`SGLANG_KUNPENG_TL_OP_PRECOMPILE=1`复用上游预编译流程，实际核验TileLang的`target="pto"`编译入口；它不是TVM的同名target kind。batch=1沿用上游配置的C++回退。关闭TileLang时不要求这些SDK文件。

`TILELANG_CACHE_DIR`必须为当前节点可写目录，示例使用本地 `/tmp/${USER}_qwen_tilelang`。同节点TP ranks共享缓存，不同节点各自缓存；可按部署版本设置目录。此集群共享ODC目录出现过缓存文件锁长等待，故不作为默认缓存位置。源码/SDK仍可放共享目录。

## 2. build → pack → start

```bash
SOURCE=/home/share/lidonghua/workspaces_inference/inference_work/qwen36_27b_bf16/sglang_20261125_upstream348_delivery
DEPLOY=/home/share/lidonghua/workspaces_inference/inference_work/qwen36_27b_bf16/deployments/qwen38_upstream348_20261125_01
TOOL="$SOURCE/scripts/cpu_kunpeng/qwen/deploy"
CONFIG="$DEPLOY/config.sh"
mkdir -p "$DEPLOY"
cp "$TOOL/config.example.sh" "$CONFIG"
# 核对路径、已申请节点、可信SSH配置、端口和容量。
bash "$TOOL/run.sh" build "$CONFIG"
bash "$TOOL/run.sh" pack  "$CONFIG"
RUN="$DEPLOY/release/run.sh"
bash "$RUN" plan   "$CONFIG"
bash "$RUN" check  "$CONFIG"
bash "$RUN" start  "$CONFIG"
```

- `build`：从源码编译sgl_kernel，输出 `DEPLOY_DIR/build`，记录 `KERNEL_BUILD.json`。
- `pack`：冻结应用、算子、工具和依赖记录到 `release`，生成16份rank私有NUMA镜像到 `numa_images`。输出为目录；不打包整套SDK或模型权重。
- `plan`：仅展示拓扑和角色覆盖，不联网、不分配资源。
- `check`：检查发布、SSH、依赖和网络，不等同整模验收。
- `start`：使用已冻结发布启动；进入READY前进行真实预热。不自动重新编译或申请节点。

另开终端执行 `bash "$RUN" status "$CONFIG"` 或 `bash "$RUN" stop "$CONFIG"`。Ctrl-C同样触发本次服务收尾；资源恢复确认后再释放调度作业。

修改Python/部署脚本后，在新DEPLOY_DIR中可将 `KERNEL_BUILD_RECEIPT` 指向旧构建记录并重新pack；原生算子编译输入变化则须重新build。已冻结release和NUMA镜像不原地覆盖。SDK版本变化也需重新build/pack。

同一共享环境的多个节点可复用一份pack，各节点的进程、实例状态和日志独立。不同Python/SDK/系统ABI、绝对路径或NUMA布局，需要重新build/pack，不能只修改config后搬用旧NUMA镜像。

单节点可设置 `DEPLOY_MODE=single`，另填 `SHM_MIB`、`MAX_RUNNING_REQUESTS`、`MAX_TOTAL_TOKENS`；`SERVER_HOST`默认0.0.0.0，拓扑默认TP16/PP1。PD节点配置由工具生成，不必手工填写这些值。

## 3. SHM、HBW和权限

对外SHM仅保留两个角色配置：

```bash
export PD_PREFILL_SHARED_HBW_SHM_MIB=384
export PD_DECODE_SHARED_HBW_SHM_MIB=128
```

工具生成每个节点的 `SHM_MIB`，统一推导KuPL窗口大小和私有池预算。底层 `SGLANG_KUNPENG_*SHM*` 字段由工具生成，不再在总配置中重复填写。数值单位均为MiB、每rank计。

| 每rank | Prefill | Decode |
|---|---:|---:|
| 主HBW预算 | 4000MiB | 4000MiB |
| KuPL共享窗口 | 384MiB | 128MiB |
| 私有HBW池 | 3616MiB | 3872MiB |
| 最大请求 / KV token容量 | 32 / 524544 | 8 / 262720 |
| Target换入 / 双镜像 | 关闭 / 关闭 | 开启 / 开启 |
| Prefill工作区HBW | 开启 | 关闭 |
| Draft驻留 | 关闭 | 开启 |

KuPL使用 `KUPL_SHM_TYPE=sls`、`KUPL_SHM_ON_PACKAGE=y`、`KUPL_SHM_ENABLE_HUGEPAGE=y` 原生HBW大页窗口。内部保留的“split”字段只表示共享/私有容量拆分，不再由Qwen自行创建共享映射。除4000MiB主预算外，两个显式同步窗口及三个KuPL内部fence窗口另需10MiB。自动预留为每HBW NUMA **2005个2MiB大页，即4010MiB**；两种角色相同。

非root配置统一为 `SGLANG_ENABLE_NOROOT_USER=1` 和 `SGLANG_KUNPENG_NOROOT_PROXY_PATH="$PERF_TUNE"`，按公共顺序执行SDMA驱动、ZCopy驱动、大页权限和SDMA阈值脚本，再准备本次大页与频率。节点必须已分配给当前服务，不能与其他模型共享执行驱动重载。

`HBM_HUGEPAGE_MODE=ensure`按需补大页；`check`仅核验。`CPU_FREQ_MODE=apply`设置目标频率，`keep`保持当前值。当前计算核目标2000MHz，恢复值1550MHz是本集群配置，换机器需确认。停止时恢复本次记录的资源设置。

## 4. 配置迁移与历史开关

通用参数去除旧QWEN前缀，例如 `QWEN_CONTEXT_LENGTH→CONTEXT_LENGTH`、`QWEN_HTTP_PORT→SERVER_PORT`、`QWEN_MAX_RUNNING_REQUESTS→MAX_RUNNING_REQUESTS`、`QWEN_TP_SIZE→TP_SIZE`、`MODEL→MODEL_PATH`。PD请求容量继续使用 `PD_PREFILL_*` / `PD_DECODE_*`，不再同时填写通用基值。新版本请从新模板生成配置，不混用旧config；旧release仍使用自己的旧配置。

保留当前有效取值，仅将六个历史开关改为功能命名；以下名称均带 `SGLANG_KUNPENG_` 前缀：

| 旧后缀 | 新后缀 |
|---|---|
| `P2_NORM_REDUCE_KEEP_SILU` | `QWEN_GDN_NORM_REDUCE_KEEP_SILU` |
| `P2_REFERENCE_VECTOR_WIDTH` | `QWEN_GDN_NORM_VECTOR_WIDTH` |
| `P2_AR_SMALLROW_COPY` | `QWEN_AR_SMALLROW_COPY` |
| `P2_GDN_HEAD_PARALLEL` | `QWEN_GDN_HEAD_PARALLEL` |
| `P20_GDN_POSTCONV_STRIDED` | `QWEN_GDN_POSTCONV_STRIDED` |
| `P2_DECODE_CONV_REQUESTS` | `QWEN_GDN_CONV_REQUEST_PARALLEL` |

删除无消费者的旧配置项：typed-gate、MTP-MLP-tail、Prefill resident/stream及GQA stream/prefetch/layer-pipeline。没有删除仍被调用的算子、图注册符号或修改计算精度。模型专用选项继续保留Qwen标签；上游TileLang开关保持原名。

## 5. 日志与目录

`DEPLOY_DIR`中，`build`为编译结果，`release`为冻结发布，`numa_images`为可复用副本；`instances/<实例ID>`保存配置、PID和资源恢复记录；`cluster/runs/<集群ID>`保存拓扑及各角色配置。

持续日志放 `logs/instances`、`logs/driver` 及对应集群运行目录。默认每流16MiB、2份备份（最多48MiB），不记录请求正文，不启用profile/replay；SDK自行写出的文件和一次性build/pack日志另计。`LOG_KEEP_DAYS=7` / `LOG_KEEP_RUNS=5`为历史策略，不自动删除发布：

```bash
bash "$RUN" prune "$CONFIG"          # 预览
bash "$RUN" prune "$CONFIG" --apply  # 清理已结束且收尾确认的旧日志
```

当前活动、资源未确认恢复的实例不会清理；release、镜像、STATE/PLAN/配置和构建记录均保留。
