# PyInstaller 打包指南

使用 PyInstaller 将 SGLang 服务打包为独立的可执行目录，无需依赖完整 Conda 环境即可运行。

## 目录结构

```
pyinstall/
├── pyinstall.sh       # 首次打包脚本（完整打包）
├── updata.sh          # 增量更新脚本（仅更新 Python 源代码）
├── updata.spec        # 增量更新的 PyInstaller 配置文件
├── numa_duplication.sh# 复制打包产物用于 NUMA 多实例部署
└── readme.md
```

## 使用流程

### 1. 配置环境变量

首次使用前，需要配置 `scripts/cpu_kunpeng/env.sh` 中的 4 个路径变量。

参考 [env.sh](file:///m:/code/sglang/scripts/cpu_kunpeng/env.sh) 示例：

```bash
# 鲲鹏 CPU 工具链路径
HPCKIT_PATH=/path/to/HPCKit_26.0.RC1/HPCKit

# kutacc 库路径
KUTACC_PATH=/path/to/kutacc/kutacc-630

# SGLang 源码路径
SGLANG_PATH=/path/to/sglang/sglang-0.5.11-open

# Conda 环境路径
CONDA_ENV_PATH=/path/to/anaconda3/envs/$CONDA_ENV_NAME
```

> **注意**：确保这些路径指向正确的目录，否则打包过程可能因找不到依赖库而失败。

### 2. 首次打包

确保在 Conda 目标环境中已安装所有依赖（包括 `sglang`、`sgl_kernel`、`torch`、`triton` 等），然后执行：

```bash
cd scripts/cpu_kunpeng/pyinstall
bash pyinstall.sh
```

打包完成后，产物位于 `dist/sglang_server/`，可直接执行：

```bash
./dist/sglang_server/sglang_server
```

### 3. 增量更新（仅更新源代码）

如果只修改了 SGLang 的 Python 源代码（未新增依赖库），无需重新执行完整的 `pyinstall.sh`，使用增量更新脚本更快：

```bash
bash updata.sh
```

`updata.sh` 基于 `updata.spec` 配置，只重新打包变化的 Python 文件，并输出到 `dist/sglang_server_update/`。

### 4. （可选）NUMA 多实例部署

如需在多 NUMA 节点上分别启动服务实例，可用 `numa_duplication.sh` 将打包产物复制为多份：

```bash
bash numa_duplication.sh
```

该脚本会将 `dist/sglang_server` 重命名为 `dist/sglang_server_tp0`，并复制出 `dist/sglang_server_tp1` 至 `dist/sglang_server_tp15` 共 16 份，供不同 NUMA 节点使用。

## 注意事项

- `pyinstall.sh` 会将 Conda 环境中的标准库完整复制到打包产物中，以确保在没有标准库的环境中也能运行。
- 如果新增了 Python 依赖（如通过 `pip install` 新安装了库），需要重新执行 `pyinstall.sh` 完整打包。
- `updata.spec` 中的 `sglang/launch_server.py` 路径基于示例配置，需确认与实际 `SGLANG_PATH` 一致。# SGLang PyInstaller 打包工具

本工具用于将 SGLang 及其依赖打包为独立可执行文件，方便在鲲鹏 CPU 环境上部署。

---

## 目录结构

```
pyinstall/
├── pyinstall.sh        # 首次打包脚本（全量打包）
├── updata.sh           # 增量更新脚本（仅更新源代码）
├── updata.spec         # 增量更新的 PyInstaller spec 文件
├── numa_duplication.sh # 生成 NUMA 多副本（0~15）
└── readme.md           # 本文件
```

---

## 一、首次使用：配置环境路径

在执行打包前，需先编辑 `../env.sh`，配置以下路径变量：

| 变量 | `env.sh` | 说明 |
|---|---|---|
| `HPCKIT_PATH`  | HPCKit 安装路径 |
| `OpenBLAS_PATH` | OpenBLAS 安装路径 |
| `KUTACC_PATH` | kutacc 安装路径 |
| `SGLANG_PATH` | SGLang 源码根目录 |
| `CONDA_ENV_PATH` | Conda 虚拟环境路径（由 `CONDA_ENV_NAME` 拼接） |

修改后执行 `pyinstall.sh`：

```bash
cd scripts/cpu_kunpeng/pyinstall
bash pyinstall.sh
```

打包完成后，输出位于 `dist/sglang_server/`，可直接执行：

```bash
./dist/sglang_server/sglang_server
```

---

## 二、后续仅更新源代码

如果只修改了 SGLang 或 sgl-kernel 源码，无需重新全量打包，只需运行 `updata.sh` 快速更新打包产物。

1. 确保 `env.sh` 中的 `SGLANG_PATH` 已指向最新的源码路径
2. 执行增量更新：

```bash
bash updata.sh
```

> **注意**：`updata.sh` 会通过 `updata.spec` 重新编译 Python 字节码并打包，速度比全量打包快得多。如需更新依赖库（如 PyTorch、Triton 等），仍需重新执行 `pyinstall.sh` 全量打包。

---

## 三、NUMAT 多副本部署

生成 `sglang_server_tp0` ~ `sglang_server_tp15` 共 16 份副本，用于不同 NUMA 节点绑定，后续若更新依赖库，需执行 `numa_duplication.sh` 更新所有副本依赖。

```bash
bash numa_duplication.sh
```

位于 `pyinstall/dist/` 目录下。
