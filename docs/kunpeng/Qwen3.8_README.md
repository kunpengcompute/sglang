# Qwen3.8-27B 鲲鹏 920F SGLang 部署指南

> 面向鲲鹏 920F（aarch64）单节点的 Qwen3.8-27B-W8A8 推理服务部署与性能验证。
> 本文档用环境变量代替具体机器路径，实际执行时请替换为你的目录。

## 目录

1. [源码与环境准备](#1-源码与环境准备)
2. [SGLang 安装](#2-sglang-安装)
3. [HBM 与大页准备](#3-hbm-与大页准备)
4. [部署工具与配置](#4-部署工具与配置)
5. [构建、打包与启动](#5-构建打包与启动)
6. [请求、测试与停止](#6-请求测试与停止)

---

## 1. 源码与环境准备

### 源码仓库

| 组件 | 仓库 | 分支 |
| --- | --- | --- |
| SGLang | `https://gitcode.com/kunpengcompute/sglang.git` | `v0.5.11` |
| KuTACC | `https://gitcode.com/zhengzhong722/kutacc.git` | `sglang_930`（按 README 安装） |

### 模型权重下载

从 ModelScope 获取 Qwen3.8-27B-W8A8 量化权重（即后续 config.sh 中 `MODEL` 要填写的目录）。

模型页：<https://modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8/files>

```bash
# 方式一：ModelScope CLI（需 pip install modelscope）
modelscope download --model Eco-Tech/Qwen3.8-27B-w8a8 --local_dir <模型存放目录>

# 方式二：git clone（权重为 LFS 大文件，需先安装 git-lfs）
git clone https://www.modelscope.cn/Eco-Tech/Qwen3.8-27B-w8a8.git <模型存放目录>
```

> **注意**：本指南配套的是 **2026-08-17 之前发布**的权重版本；该日期之后仓库如有更新，请回退到该日期之前的版本再使用：
>
> ```bash
> cd <模型存放目录>
> git log --until=2026-08-17 --oneline    # 查看该日期之前的提交
> git checkout <该日期之前的 commit>
> ```

### 环境变量

```bash
# 激活 conda 环境（请替换为你的 anaconda 安装路径与环境名）
source <anaconda3 安装路径>/start_anaconda.sh
conda activate <你的环境名>

# 编译器与依赖库路径
export HPCKIT_PATH="<HPCKit 安装路径>"
export KUTACC_PATH="<kutacc 安装路径>"
export KUPL_PATH="<kupl 安装路径>"
source ${HPCKIT_PATH}/latest/compiler/bisheng/env/setvars.sh

# 动态库搜索路径
export LD_LIBRARY_PATH=/usr/lib64/libibverbs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${KUPL_PATH}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${KUTACC_PATH}/lib:$LD_LIBRARY_PATH

# KuTACC 库与头文件
export KUTACC_LIB=${KUTACC_PATH}/lib
export KUTACC_INCLUDE=${KUTACC_PATH}/include

# KuPL 头文件与链接
export CPATH=${KUPL_PATH}/include:$CPATH
export INCLUDE=${KUPL_PATH}/include:$INCLUDE
export LIBRARY_PATH=${KUPL_PATH}/lib:$LIBRARY_PATH
```

> **注意**：所有 `export` 均为当前 shell 会话生效，建议写入启动脚本或在同一个会话中完成后续步骤。

---

## 2. SGLang 安装

> **必须先装好对应版本的 KuTACC 与 KUPL，再按下面的顺序安装：`KUPL 版 PyTorch → sgl-kernel → python`。**

### 2.1 编译安装 KUPL 版 PyTorch（2.9.0）

> 必须使用打上 kunpeng 补丁、带 **kupl 多线程后端** 的 **PyTorch 2.9.0**（KUPL 版 PyTorch），不要用官方 PyPI 版本替换，否则 sgl-kernel/KuTACC 无法对位加载姿态。
>
> 补丁来源：`kunpeng-extension-for-pytorch` 仓库中的 `pytorch-v2.9.0-sglang-0830.patch`（适配 sglang）：
>
> ```text
> https://gitcode.com/kunpengcompute/kunpeng-extension-for-pytorch/blob/main/thirdparty/pytorch-v2.9.0-sglang-0830.patch
> ```

**第一步：从 GitHub 下载 PyTorch 2.9.0 源码并打补丁**

```bash
git clone -b v2.9.0 --depth=1 --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive

# 先从上方链接下载补丁文件并放到 pytorch 源码根目录，再执行应用
git apply pytorch-v2.9.0-sglang-0830.patch
```

**第二步：执行编译脚本**

在打完补丁的 PyTorch 源码根目录下执行以下脚本进行编译（尖括号内容替换为实际路径）：

```bash
#!/bin/bash

# 1. 初始化 Anaconda 环境
source <anaconda3 安装路径>/start_anaconda.sh
conda activate <你的环境名>

# 2. 设置 HPCKit 路径
HPCKIT_PATH="<HPCKit 安装路径>"

# 3. 加载毕昇编译器环境变量
source ${HPCKIT_PATH}/latest/compiler/bisheng/env/setvars.sh

# 4. 配置编译器
export CC=$(which clang)
export CXX=$(which clang++)
export FC=$(which flang)

# 5. KUPL 相关路径配置
export KUPL_PATH="<kupl 安装路径>"
export LD_LIBRARY_PATH=${KUPL_PATH}/lib:$LD_LIBRARY_PATH
export CPATH=${KUPL_PATH}/include:$CPATH
export INCLUDE=${KUPL_PATH}/include:$INCLUDE
export LIBRARY_PATH=${KUPL_PATH}/lib:$LIBRARY_PATH

# 6. 关闭不适用的后端和测试选项（CPU-only 构建）
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_LAPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_CUDA=0
export USE_CUDNN=0
export USE_CUDSS=0
export USE_NCCL=0
export USE_CUPTI_SO=0
export USE_CUSPARSELT=0
export USE_NVRTC=0
export USE_CUFILE=0
export USE_ROCM=0
export USE_RCCL=0
export USE_MAGMA=0
export USE_FBGEMM=0

# 7. 构建配置
export BUILD_TEST=0
export DEBUG=0
export USE_HBW=1
export USE_OPENMP=1
export USE_DISTRIBUTED=1
#export CXXFLAGS="-fopenmp"

# 8. 设置 OpenBLAS 路径
export OpenBLAS_HOME="<OpenBLAS 安装路径>"

# 9. 设置链接库
export CXXFLAGS="-lkupl"

# 10. 执行构建
python setup.py bdist_wheel
```

**第三步：安装编译产物**

编译完成后，whl 产物位于 PyTorch 源码目录的 `dist/` 下（内有多个以 torch 开头的 wheel 文件）：

```bash
cd dist

# --force-reinstall 强制覆盖已安装的官方 torch；
# torch*.whl 是 shell 通配符，会把目录下所有以 torch 开头的 wheel 一起安装
pip install torch*.whl --force-reinstall
```

### 2.2 安装 sgl-kernel

```bash
cd sgl-kernel
cp pyproject_cpu.toml pyproject.toml
```

修改 `build.sh` 中的 pip 安装命令（约第 28 行），由：

```bash
pip install -v ./ --no-build-isolation --no-deps
```

改为：

```bash
pip install -v ./ --no-build-isolation --no-deps \
  --config-settings=cmake.define.SGLANG_KUNPENG_ALLOW_MISSING_OPTIONAL_KUTACC_APIS=ON
```

然后执行：

```bash
sh build.sh
```

### 2.3 安装 Python 包

```bash
cd python
cp pyproject_cpu.toml pyproject.toml
# --no-deps 跳过依赖解析与安装，避免 pip 用官方版本覆盖已装好的 KUPL 版 torch
pip install -v -e . --no-deps
```

---

## 3. HBM 与大页准备

### 准备工作目录与工具

将 `performance_tune` 与 `qwen_tools` 两个工具目录复制到你的工作路径下，然后设置目录变量：

```bash
export WORK="<工作目录>"
export TOOLS="$WORK/qwen_tools"
export PERFORMANCE_TUNE="$WORK/performance_tune"
```

### 保存资源状态并配置大页

```bash
# 保存当前资源占用快照
python "$TOOLS/resource_state.py" save --before "$WORK/resources-before.json"

# 记录原有 HBM 大页数量（用于后续恢复）
python "$TOOLS/resource_state.py" restore-count \
  --before "$WORK/resources-before.json" > "$WORK/original-hbm-pages.txt"

# 配置 HBM 大页（8 个参数的含义见下方说明）
cd "$PERFORMANCE_TUNE" && sh hugepage.sh 0 0 0 0 0 1810 0 0

# 设置 CPU 频率为 2.00GHz（0-607 号核；失败需多运行几次）
cd "$PERFORMANCE_TUNE" && sh set_cpu_frequency.sh 0-607 2.00GHz
```

> **参数说明**：`hugepage.sh` 的 8 个参数依次为 node0~15（DDR）与 node16~31（HBM）两组 NUMA 的 4 种页大小（64KiB / 2MiB / 32MiB / 1GiB）预留页数，按给定值直接设置（不是增量累加）。本例第 6 个参数 1810 是每个 HBM 节点预留的 2MiB 大页数，换算方式：每 rank HBM 总预算 4000MiB − 共享 SHM 384MiB = 私有 HBW 池 3616MiB，按 2MiB 向上取整得 1808 页、再留 2 页余量。
>
> **注意**：
> - 大页数须与 4.2 节 config.sh 的 HBM 划分配套：共享 SHM 为 384MiB 时对应 1810 页，为 768MiB 时对应 1616 页；
> - 为 0 的参数会将对应组/页大小的预留清零，节点上已有其他大页预留时，请对照 `$WORK/original-hbm-pages.txt` 中的原值填写，避免误清；
> - `set_cpu_frequency.sh` 的核号范围（0-607）请按实际节点情况确认。

---

## 4. 部署工具与配置

### 4.1 部署工具简介

部署与启动使用仓库内置的部署工具 `scripts/cpu_kunpeng/qwen/deploy`，构建、打包、NUMA 副本制作、资源准备与启停均由工具完成：

- 适用 Qwen（非 MTP / MTP）单机 920F **TP16**：DDR NUMA0~15 各 38 个 CPU，HBM NUMA16~31；
- 工具只管理部署，不改模型、算子和框架；
- **`start` 自动准备大页、HBM/SHM 预算并设置 CPU 频率**（经 `PERF_TUNE` 指向的 performance_tune/root_proxy 执行；第 3 节的手工流程仅用于管理员预准备或问题排查）。

> 完整说明（部署目录结构、多节点复用、部署工具自身升级等）见
> [`scripts/cpu_kunpeng/qwen/deploy/README.md`](../../scripts/cpu_kunpeng/qwen/deploy/README.md)。

### 4.2 准备部署目录与 config.sh

```bash
export TOOL=<sglang 源码目录>/scripts/cpu_kunpeng/qwen/deploy
export DEPLOY=<部署目录>   # 用版本/依赖/拓扑命名（如 qwen_20261106_920f_tp16），不含节点名

mkdir -p "$DEPLOY"
cp "$TOOL/config.example.sh" "$DEPLOY/config.sh"
```

编辑 `$DEPLOY/config.sh`，主要内容：

- **路径**：`SGLANG_SOURCE`（sglang 源码目录）、`PYTHON`（配套 Python 解释器）、`CC`/`CXX`（毕昇编译器）、`KUTACC_PREFIX`、`KUPL_PREFIX`、`COMPILER_RUNTIME_LIB`（毕昇运行库）、`MODEL`（Qwen3.8-27B-w8a8 权重目录，获取方式见第 1 节「模型权重下载」）、`PERF_TUNE`（已授权的 performance_tune 目录，即第 3 节的 `$PERFORMANCE_TUNE`）；
- **服务参数**：`QWEN_HOST`、`QWEN_HTTP_PORT`（config 示例默认 31086，本文示例统一用 30000）、`QWEN_DIST_PORT`（示例默认 29986）、`QWEN_CONTEXT_LENGTH`、`QWEN_MAX_TOTAL_TOKENS`、`QWEN_MAX_RUNNING_REQUESTS`；
- **资源与调频**（示例已给出默认值）：`SGLANG_KUNPENG_WEIGHTS_HBW_POOL_SIZE_MB=4000`、`SGLANG_KUNPENG_PREFILL/DECODE_SHM_SIZE_MB=384`、`HBM_HUGEPAGE_MODE=ensure`、`HBM_HUGEPAGE_MB_PER_NODE=auto`（对应每 HBM 节点 1810 个 2MiB 大页，换算见第 3 节）、`CPU_FREQ_MODE=apply`、`CPU_FREQ_TARGET_MHZ=2000`、`CPU_FREQ_TOLERANCE_MHZ=20`；无法可靠识别原标称频率时，按管理员确认值填写 `CPU_FREQ_RESTORE_MHZ`；
- **MTP**：`SGLANG_ENABLE_MTP=0` 关闭；开启设 `1` 并配置 `SGLANG_SPECULATIVE_NUM_STEPS`（默认 3）；共享 checkpoint 时 `SPECULATIVE_DRAFT_MODEL_PATH` 可留空；
- **默认配方**（`config.example.sh` 已配好，一般无需改动）：prefill 走 KuTACC（`SGLANG_KUNPENG_QWEN_PREFILL_BACKEND=kutacc`）、M4096（`SGLANG_KUNPENG_COLLECTIVE_MAX_TOKENS=4096`）、M96（`KUTACC_INT8_LARGE_M_MACRO_M96=1`）、图捕获与 CPU 绑核开启、**默认关闭 thinking**（`SGLANG_QWEN_DEFAULT_THINKING=0`）。

---

## 5. 构建、打包与启动

### 5.1 首次构建与打包

```bash
bash "$TOOL/run.sh" build "$DEPLOY/config.sh"
bash "$TOOL/run.sh" pack "$DEPLOY/config.sh"
```

- `build` 用 CMake 从源码生成 `sgl_kernel`；`pack` 校验构建记录、源码、SDK 与 DSO，冻结应用，并在相同 920F 拓扑的节点上制作 16 份 NUMA 副本（`numa_images/rank0..rank15`）；
- `start` 不执行编译、pack 或镜像复制；已有同版本、同依赖的完整构建记录时，可在 config.sh 填 `KERNEL_BUILD_RECEIPT` 跳过 build、只 pack；
- 源码或原生依赖更新时**换一个新的部署目录**重新 build/pack，已有 release 与副本不会被覆盖。

### 5.2 启动服务

在已分配的独占 920F 计算节点上执行：

```bash
bash "$TOOL/run.sh" verify "$DEPLOY/config.sh"
bash "$TOOL/run.sh" start "$DEPLOY/config.sh"
```

`start` 内部流程（大页、调频、选核换核均自动完成）：

```text
取得本节点 TP16 管理锁、创建独立实例
  → 核验冻结应用/运行库/拓扑及现有副本
  → 保存本次配置与资源快照
  → 按配置准备大页、HBM/SHM 预算及频率
  → 用已有 NUMA 副本启动 16 个 rank
  → 短请求、选频/同 NUMA 换核、预热
  → READY 并记录频率状态
  → stop 后清理自有进程/SHM、恢复本次资源改动
```

**出现 READY 后即可进入第 6 节测试**；READY 与频率验收分开记录，调频/换核异常只告警、不阻断服务。

> **注意**：
> - 同一部署目录在同一主机不允许同时启动两个完整 TP16 管理进程，不同主机互不阻塞；
> - 后续启动相同 release 只需 `start`，实例记录（配置快照、状态、资源与测试结果）保存在 `$DEPLOY/instances/<实例ID>/`；
> - 未出现 READY 时，先查看 `$DEPLOY/logs/driver/`（各命令输出）与 `$DEPLOY/logs/instances/<实例ID>/`（launcher/rank 日志）；
> - 异常杀死管理进程时文件锁会自动释放，但服务与资源是否已清理须依据实例记录确认，不能仅凭锁释放判断。

---

## 6. 请求、测试与停止

READY 后在同一节点另开终端（设置相同的 `TOOL`/`DEPLOY`）执行：

```bash
# 短请求验证（服务可用性）
bash "$TOOL/run.sh" request "$DEPLOY/config.sh"

# 基准测试：输入/输出 token 数、并发与预热取 config.sh 的 QWEN_TEST_* 配置，
# 示例默认 1024×1024、并发 1
bash "$TOOL/run.sh" bench "$DEPLOY/config.sh"

# 查看实例状态
bash "$TOOL/run.sh" status "$DEPLOY/config.sh"

# 停止服务；stop 会清理自有进程/SHM 并恢复本次资源改动（大页/频率）
bash "$TOOL/run.sh" stop "$DEPLOY/config.sh"
```

**curl 直接请求示例**（OpenAI 兼容接口；端口取 config.sh 的 `QWEN_HTTP_PORT`，下例为 30000）：

```bash
# 查询服务实际模型名
curl --noproxy '*' http://127.0.0.1:30000/v1/models

# 非流式对话请求
curl --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<上一步返回的模型名>",
    "messages": [{"role": "user", "content": "你好，简单介绍一下你自己"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

**预期返回**（`/v1/models`，`data[0].id` 即请求体中要填的模型名）：

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3.8-27B-w8a8",
      "object": "model",
      "created": 1760000000,
      "owned_by": "sglang",
      "root": null,
      "parent": null,
      "max_model_len": 69632
    }
  ]
}
```

**预期返回**（`/v1/chat/completions`，HTTP 200）：

```json
{
  "id": "chatcmpl-1a2b3c",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "Qwen3.8-27B-w8a8",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！我是 Qwen3.8-27B-W8A8……"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 64,
    "total_tokens": 72
  }
}
```

> 以上 JSON 中 `id`、`created`、`usage` 数值与生成正文均为示例，以实际返回为准。判定成功的基本条件是 **HTTP 200 且 `choices[0].message.content` 非空**（因 `max_tokens` 截断时 `finish_reason` 为 `"length"`）；开启 thinking（见下方说明）后 `message` 中还会多出 `reasoning_content` 字段；流式响应为 SSE 格式，`object` 为 `chat.completion.chunk`，逐块输出 `delta` 内容。

> 流式输出在请求体加 `"stream": true`；服务默认关闭 thinking，需要开启时在请求体加 `"chat_template_kwargs": {"enable_thinking": true}`。

**说明**：

- `request`/`bench` 须在实例所在节点执行，避免把另一节点的 localhost 误认作服务；跨节点发请求请用 HTTP 客户端指定实际节点地址；
- `status`/`stop` 支持显式指定实例（`--instance 实例ID`），共享文件系统下可从登录节点查看状态或写入协作停止标记；
- MTP 开/关两种模式的测试入口均支持；
- 命令输出与各实例日志在 `$DEPLOY/logs/` 下（`driver/` 各命令输出，`instances/<实例ID>/` 请求测试与调频日志）；
- 频率验收结果记录在实例目录（如 `frequency_admission.json`），`performance_condition_met` 为 True 才可标注采样期间约 2 GHz（1.98～2.02 GHz，频率记录须覆盖 512 个计算核）；首次图捕获属于冷态，比较暖态性能前先做同负载预热。

---

## 附：完整流程速览

```
环境变量 → KuTACC/KUPL → KUPL 版 PyTorch → sgl-kernel → python 包
    → config.sh → build → pack → verify/start（大页/调频/选核换核/预热）
    → request/bench/status → stop（清理进程并恢复资源）
```
