# SGLang PyInstaller 打包工具

将 SGLang 及其依赖打包为独立的目录结构，无需完整 Conda 环境即可在多节点、多 NUMA 上部署。

---

## 目录结构

```
pyinstall/
├── pyinstall.sh        # 首次全量打包脚本
├── update.sh           # 增量更新脚本（拷贝源码，不重新 PyInstaller）
├── numa_duplication.sh # 将打包产物复制为 NUMA 多副本（0~15）
└── readme.md
```

---

## 一、首次使用

### 1. 配置环境变量

编辑 `scripts/cpu_kunpeng/env.sh`，按实际环境设置以下路径：

| 变量 | 说明 |
|---|---|
| `HPCKIT_PATH` | HPCKit 安装路径 |
| `OpenBLAS_PATH` | OpenBLAS 安装路径 |
| `KUTACC_PATH` | kutacc 安装路径 |
| `SGLANG_PATH` | SGLang 源码根目录 |
| `CONDA_BASE_PATH` | Conda 安装路径 |
| `CONDA_ENV_NAME` | Conda 虚拟环境名称 |
| `CONDA_ENV_PATH` | Conda 虚拟环境路径 |
| `PYINSTALL_PATH` | pyinstall 脚本所在目录 |

示例：

```bash
export HPCKIT_PATH="/path/to/HPCKit_26.0.RC1/HPCKit"
export OpenBLAS_PATH="/path/to/openblas"
export KUTACC_PATH="/path/to/kutacc/kutacc-630"
export SGLANG_PATH="/path/to/sglang/sglang-0.5.11-open"
export CONDA_ENV_PATH="$CONDA_BASE_PATH/envs/$CONDA_ENV_NAME"
export PYINSTALL_PATH="$SGLANG_PATH/scripts/cpu_kunpeng/pyinstall"
```

### 2. 安装 PyInstaller

首次执行打包前，需要通过 pip 安装 PyInstaller：

```bash
pip install pyinstaller
```

### 3. 执行全量打包

```bash
cd scripts/cpu_kunpeng/pyinstall
bash pyinstall.sh
```

`pyinstall.sh` 会依次完成：
1. 读取 `env.sh` 中的环境变量
2. 生成 `sglang_server.spec`（包括依赖的 `.so` 库、数据文件、hidden-import）
3. 自动修改 spec 将 `sglang`/`sgl_kernel` 移出 PYZ（确保更新时可直接替换源码）
4. 执行 `pyinstaller` 全量打包
5. 自动执行 `numa_duplication.sh` 生成 NUMA 多副本

打包产物位于 `dist/sglang_server_tp0` ~ `dist/sglang_server_tp15`，可直接执行：

```bash
./dist/sglang_server_tp0/sglang_server
```

---

## 二、后续源码更新

代码更新已集成到 `scripts/cpu_kunpeng/launch.sh` 启动流程中。当环境变量 `SGLANG_ENABLE_BINARY_LAUNCH=1` 时，每次调用 `launch.sh` 都会自动执行增量更新：

```bash
# 启动 prefill/decode 节点时，自动先更新打包产物中的源码
export SGLANG_ENABLE_BINARY_LAUNCH=1
bash launch.sh native
```

`launch.sh` 中的自动更新逻辑（`sh ./pyinstall/update.sh`）会直接拷贝最新源码覆盖各 NUMA 副本的 `_internal` 目录，**无需重新执行 PyInstaller**，速度远快于全量打包。

也可以手动执行增量更新：

```bash
# 更新所有（sglang + sgl_kernel）
bash update.sh

# 只更新 sglang 源码
bash update.sh sglang

# 只更新 sgl_kernel 源码
bash update.sh kernel
```

> **注意**：增量更新只替换 Python 源码（`.py` 文件）。如果依赖库发生变化（如 PyTorch、Triton 升级），仍需重新执行 `pyinstall.sh` 全量打包或手动更新。

---

## 三、依赖更新与 NUMA 副本重置

当依赖库（如 `torch`、`triton`、`sgl_kernel` 的 `_internal` 目录中除 `.py` 外的 `.so` 文件等）发生变化时，需按以下步骤处理：

### 1. 重新全量打包 or 手动更新 tp0

- 执行 `pyinstall.sh` 全量打包即可自动覆盖所有副本。
- 或仅手动替换 `dist/sglang_server_tp0/_internal/` 中的依赖文件。

### 2. 执行 NUMA 副本复制

```bash
bash numa_duplication.sh
```

该脚本将 `dist/sglang_server` 重命名为 `dist/sglang_server_tp0`，然后复制出 `sglang_server_tp1` ~ `sglang_server_tp15` 共 16 份。

> `pyinstall.sh` 末尾已自动调用 `numa_duplication.sh`，全量打包后无需手动执行。

---

## 四、使用场景速查

| 场景 | 操作 | 耗时 |
|---|---|---|
| 首次部署 | 配置 `env.sh` → `bash pyinstall.sh` | 较长 |
| 修改了 `.py` 源码 | 直接调用 `launch.sh`（自动增量更新）或 `bash update.sh` | 较短 |
| 新增/升级了 pip 依赖 | `bash pyinstall.sh` 全量打包 | 较长 |
| 只想重新分发 NUMA 副本 | `bash numa_duplication.sh` | 较短 |

---

## 五、注意事项

- `pyinstall.sh` 会将 Conda 环境中的标准库完整复制到 `_internal` 中，确保打包产物可在无标准库的目标机器上运行。
- `update.sh` 直接拷贝源码替换所有 NUMA 副本的 `_internal/sglang` 和 `_internal/sgl_kernel`，不重新编译 `.pyc`，因此速度极快。
- 如果新增了 Python 依赖（`hidden-import` 或 `.so` 库），需要同步更新 `pyinstall.sh` 中的 `--add-binary` 和 `--hidden-import` 参数，然后执行全量打包。
- `numa_duplication.sh` 的副本数量固定为 16 个（tp0 ~ tp15），如需调整，可手动修改脚本中的 `seq` 范围。