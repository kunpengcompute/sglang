# Qwen 920F：源码构建、发布和服务启动

入口统一为 `bash run.sh ACTION /绝对路径/config.sh`。`config.sh` 放在源码之外；运行输出放在 `DEPLOY_DIR`，不写回 SGLang 源码。支持文本和多模态，MTP 由同一配置控制。

## 1. 换一台同型号机器，需要填写什么

先复制 `config.example.sh` 为自己的 `config.sh`。配置已经集中分为路径、日志、HBW/频率、模型参数、PD 拓扑和角色预算。

| 配置 | 需要准备的信息 |
|---|---|
| `SGLANG_SOURCE`、`DEPLOY_DIR` | 本次源码目录、新的发布输出目录；不要指向旧实验的 release |
| `PYTHON`、`CLIENT_PYTHON` | 推理 Python 3.12 环境；标准库控制脚本的 Python |
| `CC`、`CXX`、`COMPILER_RUNTIME_LIB` | 配套毕昇 clang/clang++ 和运行库目录；`libomp.so` 等来自该环境，不是 Qwen 算子单独携带的包 |
| `KUTACC_PREFIX`、`KUPL_PREFIX` | 匹配的 SDK 前缀，均包含 include/lib；相同库名不代表相同版本 |
| `EXTRA_LIBRARY_PATH` | RDMA/系统运行库的补充目录；其余缺失项通过 deps/verify 报告定位 |
| `MODEL` | Qwen3.8 权重目录，包含 config、量化描述、tokenizer 和完整权重 |
| `PERF_TUNE` | 已获授权的 root_proxy 和 hugepage/调频/SDMA helper 所在目录 |
| `PD_NODE_HOSTS` 或 `PD_NODE_FILE` | 当前已分配的计算节点，先 P 后 D；工具不申请节点 |
| `PD_ADVERTISE_HOSTS`、`PD_IB_DEVICE`、`PD_NET_INTERFACE` | 节点间可达地址、RDMA 设备、控制网卡；端口和 SSH 用户也在配置中 |

本 NUMA 配方要求 aarch64、TP16、608 CPU：DDR NUMA0–15，每 NUMA 38 CPU，HBW NUMA16–31；每物理节点一个 PP stage。相同 920F 芯片但 CPU/NUMA 编号或可用核范围不同，仍需要适配核表，不能仅替换路径。

运行还需要对应 Python 依赖、RDMA 驱动/设备、`cmake`、`ldd`、`taskset`、`numactl` 和已授权的资源设置权限。部署包不负责安装驱动或替用户获得 root 权限。

## 2. 首次：build → pack

```bash
SOURCE=/共享绝对路径/sglang_20261120_r9_clean
DEPLOY=/共享绝对路径/deployments/qwen38_r9_20261120_clean
TOOL="$SOURCE/scripts/cpu_kunpeng/qwen/deploy"
CONFIG="$DEPLOY/config.sh"
mkdir -p "$DEPLOY"
cp "$TOOL/config.example.sh" "$CONFIG"
# 填写上述环境信息，并把当前节点清单写入配置指定的文件。
bash "$TOOL/run.sh" build "$CONFIG"
bash "$TOOL/run.sh" pack  "$CONFIG"
```

- `build`：从选定源码编译 sgl_kernel，输出到 `DEPLOY_DIR/build`；编译记录为 `KERNEL_BUILD.json`。
- `pack`：校验源码、构建产物和依赖，冻结 Python 应用、原生库和工具到 `release`，制作16份 rank 私有 NUMA 镜像到 `numa_images`。输出是目录，不是自动生成的压缩包。
- 修改`python/`下的模型/框架代码或部署脚本后，可在新的`DEPLOY_DIR`配置`KERNEL_BUILD_RECEIPT`指向原构建记录，直接pack；允许未提交的Python调试改动，打包时记录当前commit及修改状态。
- 修改`sgl-kernel/csrc`、`sgl-kernel/include`或`sgl-kernel/python`后须重新build。新构建记录按文件内容核对编译输入；旧记录兼容原算子目录干净、仅Python或部署脚本变化的情况。SDK和DSO校验保持。
- 不覆盖已有 build、release 或 NUMA 镜像。升级代码或 SDK 使用新的 `DEPLOY_DIR`。

`release/app` 不塞入整套 SDK 或模型权重。NUMA 镜像会复制运行所需的部分 ELF 和依赖，纯 Python 内容仍可能链接到本机环境；这与把整个 HPCKit/SDK 打成压缩包不同。

## 3. 服务启动

后续直接使用这次冻结发布中的入口，避免工具与应用版本混用：

```bash
RUN="$DEPLOY/release/run.sh"
bash "$RUN" plan   "$CONFIG"  # PD：只打印计划，不联网、不分配资源
bash "$RUN" check  "$CONFIG"  # PD：核查发布、SSH、网络和 router；不等同整模验收
bash "$RUN" start  "$CONFIG"  # 前台运行，等待 READY
```

另开终端管理：

```bash
bash "$RUN" status "$CONFIG"
bash "$RUN" stop   "$CONFIG"
```

`start` 复用已打包应用，不重新编译。单节点使用 `QWEN_DEPLOY_MODE=single`，可通过 `request/bench` 发请求；PD 使用 `QWEN_DEPLOY_MODE=pd`，客户端访问输出的 router URL。PD 的首个 stage 提供 HTTP，后续 stage 等待实际 scheduler 就绪；router 完成真实 P/D 预热后接入。

默认 PD 拓扑为物理4P+4D：一个 TP16×PP4 的 P 实例，层数20/16/16/12；四个 TP16×PP1 的 D 实例。不是把四个 P 当成四套完整模型。

- `PD_PREFILL_NODES` / `PD_DECODE_NODES`：物理节点数量。
- `PD_*_NNODES` / `PD_*_PP_SIZE`：每逻辑实例占用节点数，本配方要求二者相同。
- 逻辑副本数为 `NODES / NNODES`，必须整除；D 为 PP1 时，D 节点数就是 D 副本数。
- MTP 由 `SGLANG_ENABLE_MTP`、`SGLANG_SPECULATIVE_NUM_STEPS` 控制；PD 示例使用 MTP3。PP 分层、请求容量或角色优化配方变化，需要相应验证。

正常 Ctrl-C 也执行本次收尾。确认服务/资源清理后再释放调度作业；工具不会替调度器释放节点。

## 4. HBW、大页与选频属于启动流程

| 每 rank 配置 | Prefill | Decode |
|---|---:|---:|
| HBW 总预算 | 4000 MiB | 4000 MiB |
| 共享通信 / 私有 HBW | 384 / 3616 MiB | 128 / 3872 MiB |
| 最大请求 / KV token 容量 | 32 / 524544 | 8 / 262720 |
| Target K/V 换入 / 双缓冲 | 关闭 / 关闭 | 开启 / 开启 |
| Prefill 工作区放 HBW | 是 | 否 |
| Draft 镜像 / V2 驻留 | 关闭 | 开启 |

共享通信区包含在4000 MiB总预算内。大页自动预留覆盖私有 HBW，并加2页余量；共享区另外核对普通空闲容量。默认对应 P 每 HBW NUMA1810个2MiB页，D为1938页。

`start` 的顺序是：校验发布与依赖 → 取得本节点所有权 → 记录原资源设置 → 按需补大页 → SDMA 设置/读回 → 按配置调频 → 启动模型 → 真实预热、采样并按配置选核 → READY。

- `HBM_HUGEPAGE_MODE=ensure`：不足时通过已有 helper 补齐；`check`：只核查，由管理员预先准备。
- `CPU_FREQ_MODE=apply`：申请目标计算核2GHz；`keep`：保持原频率，但仍采样。
- `CPU_REMAP_SAME_NUMA`：控制是否在同 NUMA 内换到合适的核；不改变 TP/PP。
- `CPU_FREQ_RESTORE_MHZ`：无法读取可靠原标称频率时使用管理员确认值，不能把其他机器的恢复值照搬。
- 调频异常会记录告警后继续服务；HBW不足、依赖错误、SDMA设置失败则启动失败。正常停止按原记录恢复本次调整，大页/频率恢复不确定会保留待处理记录。

GDN/MTP checkpoint HBW、短 query Flash-Decoding、直接 AllReduce、有效 Draft V2 驻留配方继续使用。未获采纳的 Prefill 驻留和精细 SDMA 流水线保持关闭。

## 5. 多节点能否复用同一份 pack

**同一共享环境可以。** 所有节点访问相同的 `DEPLOY_DIR/release` 和 `numa_images`；每次启动创建独立的实例、进程、状态和日志。host、PP rank、角色预算等运行信息也不同，并非只有日志文件名变化。

需要保持配套的 Python/SDK/系统 ABI 和同样的绝对路径、NUMA 布局。工具核对依赖内容与实际加载路径，防止同名不同版本的库混用。

**不同安装路径或不同环境不能直接搬旧 NUMA 镜像。** 镜像中含 Python prefix、绝对路径和链接；在另一环境填写配置后，使用源码重新执行 build/pack。这样生成属于该环境的镜像。仅改 config 后沿用旧 `numa_images` 不受支持。

## 6. 日志位置、大小和历史保留

```text
DEPLOY_DIR/
├── config.sh、KERNEL_BUILD.json、PACKAGE.json
├── build/                         编译产物
├── release/                       冻结应用和工具
├── numa_images/rank0 ... rank15/   可复用的NUMA镜像
├── instances/<时间_节点_端口_PID_ID>/
│   ├── STATE.json、environment.json、config.sh
│   └── 资源设置、频率摘要、PID和清理记录
├── logs/
│   ├── instances/<实例ID>/         rank0...15.log、launcher.log
│   ├── driver/                    run.sh入口输出
│   ├── build/                     一次性编译诊断
│   └── pack/                      一次性镜像制作诊断
└── cluster/runs/<集群ID>/          PLAN/STATE、角色配置、router和SSH日志
```

配置中的生产默认值：

```bash
export QWEN_LOG_LEVEL=warning
export QWEN_LOG_MAX_MIB=16
export QWEN_LOG_BACKUPS=2
export QWEN_FREQUENCY_LOG=summary
export QWEN_LOG_KEEP_DAYS=7
export QWEN_LOG_KEEP_RUNS=5
```

rank、launcher、router、SSH和driver持续日志都轮转。每条流最多 `16 × (1+2) = 48 MiB`，文件名为 `rank0.log`、`.log.1`、`.log.2`；单节点16 rank+launcher+start-driver最多约864 MiB。4P4D连同控制/路由约7.2 GiB活跃日志；调试full频率日志另增加每节点最多48 MiB。

默认不记录请求正文，不启用 profile/replay；频率仅保存覆盖更新的摘要。`QWEN_FREQUENCY_LOG=full` 才持续记录有界JSONL。设置 `QWEN_LOG_LEVEL=info` 可增加诊断信息。SDK自己写到其他位置的文件、显式benchmark结果、一次性build/pack/helper诊断不包含在持续流上限内。

文件轮转不会自动删除历次运行目录。定期执行：

```bash
bash "$RUN" prune "$CONFIG"          # 预览文件与预计释放字节
bash "$RUN" prune "$CONFIG" --apply  # 按配置真正清理
```

该策略按节点保留最近5个已结束日志集合，且最长7天；超出任一条件可清理。集群日志和完成的driver日志分别计数。只处理已确认停止/资源收尾的旧日志，跳过当前、活动、未确认实例及未闭合driver；不删除STATE/PLAN/配置/构建记录、release、镜像或其他部署。可以把 `prune --apply` 放入日常运维任务；只轮转而从不清理历史，仍会累计磁盘占用。

一次性build/pack日志随发布保留；旧发布目录的归档由版本保留策略处理，不由日志清理命令自动删除。

## 验证范围

本轮执行本地语法、路由/角色配置、日志轮转、历史日志保护及代码等价检查。没有重建920F DSO、制作新NUMA镜像或启动真实多节点服务；源码整理后应重新build/pack验证，不复用旧原生库。父R9真实验证为4P1D，且保留一项C8单token差异待查；不得把本轮本地验证视为4P4D性能或一致性验收。
