# Qwen3.8-27B 鲲鹏 920F SGLang 部署指南

> 面向鲲鹏 920F（aarch64）单节点的 Qwen3.8-27B-W8A8 推理服务部署与性能验证。
> 本文档用环境变量代替具体机器路径，实际执行时请替换为你的目录。

## 目录

1. [源码与环境准备](#1-源码与环境准备)
2. [SGLang 安装](#2-sglang-安装)
3. [HBM 与大页准备](#3-hbm-与大页准备)
4. [计算节点：NUMA 副本生成与启动](#4-计算节点numa-副本生成与启动)
5. [功能与基本性能测试](#5-功能与基本性能测试)
6. [停止并释放资源](#6-停止并释放资源)

---

## 1. 源码与环境准备

### 源码仓库

| 组件 | 仓库 | 分支 |
| --- | --- | --- |
| SGLang | `https://gitcode.com/kunpengcompute/sglang.git` | `v0.5.11` |
| KuTACC | `https://gitcode.com/zhengzhong722/kutacc.git` | `sglang_930`（按 README 安装） |

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

> **必须先装好对应版本的 KuTACC，再按下面的顺序安装：`kutacc → sgl-kernel → python`。**

### 2.1 安装 sgl-kernel

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

### 2.2 安装 Python 包

```bash
cd python
cp pyproject_cpu.toml pyproject.toml
pip install -v -e .
```

### 2.3 安装补丁版 PyTorch（2.9.0）

> 必须使用打上 kunpeng 补丁的 **PyTorch 2.9.0**，不要用官方 PyPI 版本替换，否则 sgl-kernel/KuTACC 无法对位加载姿态。
>
> 补丁来源：`kunpeng-extension-for-pytorch` 仓库中的 `pytorch-v2.9.0-sglang-0830.patch`（适配 sglang）：
>
> ```text
> https://gitcode.com/kunpengcompute/kunpeng-extension-for-pytorch/blob/main/thirdparty/pytorch-v2.9.0-sglang-0830.patch
> ```
>
> 需要先把该补丁打到 PyTorch 2.9.0 源码上并重新构建出 whl；下面仅给出安装这些已构建 whl 的步骤。

```bash
# 进入构建出的 whl 所在目录（目录内有多个以 torch 开头的 wheel 文件）
cd <pytorch-whl 所在目录>

# --force-reinstall 强制覆盖已安装的官方 torch；
# torch*.whl 是 shell 通配符，会把目录下所有以 torch 开头的 wheel 一起安装
pip install torch*.whl --force-reinstall
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

# 配置 HBM 大页（参数含义见脚本说明）
cd "$PERFORMANCE_TUNE" && sh hugepage.sh 0 0 0 0 0 1616 0 0

# 设置 CPU 频率为 2.00GHz（0-607 号核；失败需多运行几次）
cd "$PERFORMANCE_TUNE" && sh set_cpu_frequency.sh 0-607 2.00GHz
```

> **提示**：`hugepage.sh` 的具体参数与 `set_cpu_frequency.sh` 的核号范围请按实际节点情况确认。

---

## 4. 计算节点：NUMA 副本生成与启动

### 4.1 定义路径变量

```bash
export PYTHON=<你的 python3.12 解释器路径>
export IMAGES="$WORK/numa_images"
export RUN="$WORK/service"
export MODEL_PATH=<Qwen3.8-27B-w8a8 模型路径>
export SGLANG_SOURCE=<sglang 源码目录>
```

### 4.2 生成 NUMA 副本

```bash
python "$TOOLS/prepare_numa_images.py" \
  --python "$PYTHON" --kutacc-prefix "$KUTACC_PATH" \
  --kupl-prefix "$KUPL_PATH" --output "$IMAGES"
```

### 4.3 启动服务

```bash
python "$TOOLS/launch.py" --source-root "$SGLANG_SOURCE" \
  --model "$MODEL_PATH" --images "$IMAGES" --run "$RUN" \
  --host 0.0.0.0 --port 30000 --dist-port 29926 \
  > "$WORK/launcher.log" 2>&1 &

# 查看 rank0 日志并确认就绪
tail -f "$RUN/rank0.log"

# 就绪验证：HTTP 200 后才可测试
curl --noproxy '*' -f http://127.0.0.1:30000/health
```

> **注意事项**：
> - 模型加载期间可能尚未生成 rank 日志或 `/health` 尚未成功，请查看 `launcher.log` 与 `service/rank*.log`；
> - **HTTP 200 后才开始测试**；
> - HTTP 与分布式两个端口均需空闲。

### 4.4 启动器默认配置说明

启动器读取 `qwen_tools/environment.json`，默认参数：

- **TP16**、多 NUMA 原生库副本；
- prefill 走 KuTACC、M4096、M96 开关；
- HBM 4000 / 共享 768 / arena 864 MiB；
- **MTP 关闭**；
- 工具 parser 为 `qwen`，reasoning parser 为 `qwen3`；
- 通过外部模板**默认关闭 thinking**，请求可传 `chat_template_kwargs.enable_thinking=true` 开启。

> **重要**：修改启动配置要改 JSON，单纯 `export` 同名变量不会覆盖它。
> `prepare_numa_images.py` 已负责副本，无需再开另一套 NUMA 复制。
> **重编 SO 后必须重新制作镜像**；镜像仍依赖原 Python 环境，**不能删除该环境**。

### 4.5 服务就绪后选核与重映射

```bash
python "$TOOLS/compute_frequency.py" plan \
  --samples 10 --interval 0.2 --output "$WORK/compute_plan.json"

python "$TOOLS/compute_frequency.py" apply \
  --plan "$WORK/compute_plan.json" --evidence-dir "$RUN" --receipt "$WORK/remap.json"
```

> 工具只在同 NUMA 内调整已初始化 worker 的位置，不负责系统调频；
> 替代核须属于作业允许范围；**重启后需要重新 apply**。

---

## 5. 功能与基本性能测试

### 5.1 接口与工具调用验证

```bash
# 确认模型已加载
curl --noproxy '*' -f http://127.0.0.1:30000/v1/models

# 工具调用完整往返（forced 工具选择、关闭 thinking）
python "$TOOLS/check_tool_call.py" --ip 127.0.0.1 --port 30000 \
  --tool-choice forced --thinking off --output "$WORK/tool-call.json"
```

### 5.2 1024×1024 基准测试（1/8 并发，各两轮）

```bash
python - <<'PY_BENCH'
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, os.environ['TOOLS'])
import runtime_frequency
work = Path(os.environ['WORK'])
sampler = runtime_frequency.start(work/'frequency.jsonl', work/'compute_plan.json')
try:
    subprocess.run([sys.executable, os.environ['TOOLS']+'/bench_1k1k.py',
        '--ip', '127.0.0.1', '--port', '30000', '--times', '2',
        '--concurrency', '1', '8', '--output-dir', str(work/'bench')], check=True)
finally:
    report = runtime_frequency.stop(sampler, work/'frequency.json')
    print('请求期2GHz采样条件满足：', report['performance_condition_met'])
PY_BENCH
```

**结果解读**：

- 终端打印各轮 TTFT（ms）与 TPOT（ms/token）；
- 原始结果在 `$WORK/bench/summary.json` 与 `summary.csv`；
- **仅对成功且长度完整的结果取平均**；
- 首次图捕获属于冷态，比较暖态性能前**先做同负载预热**；
- 频率记录须覆盖 512 个计算核，缺失/越界均为 0，才可标注采样期间约 2 GHz（1.98～2.02 GHz）。

---

## 6. 停止并释放资源

```bash
# 优雅停止启动器
kill -TERM "$LAUNCHER_PID"
wait "$LAUNCHER_PID"

# 按保存的快照清理占用的资源
python "$TOOLS/resource_state.py" cleanup --before "$WORK/resources-before.json"
```

---

## 附：完整流程速览

```
环境变量 → KuTACC → sgl-kernel → python 包 → 厂商 PyTorch
    → 大页/频率 → NUMA 副本 → 启动 → 校验 → 选核重映射 → 功能/性能测试 → 停止清理
```