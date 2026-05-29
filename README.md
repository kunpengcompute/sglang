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

在正式安装SGLang相关组件之前，需先完成Python虚拟环境的创建与Torch库的安装，命令如下所示，更多信息可参考SGLang官方社区提供的[安装指南](https://docs.sglang.com.cn/platforms/cpu_server.html#install-from-source)。

```shell
source ~/anaconda3/start_conda.sh
conda create -n sgl-cpu python=3.12 -y
conda activate sgl-cpu

pip install --upgrade pip setuptools
conda install -y tbb libnuma numactl

pip install torch==2.9.0 torchvision==0.24.0 triton==3.5.0 --force-reinstall
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
cp pyproject_cpu.toml pyproject.toml
pip install --upgrade pip setuptools
pip install -e "python[all]"
pip install vllm

# 加载毕昇编译器环境变量
HPCKIT_PATH=/path-to-HPCKit
source ${HPCKIT_PATH}/latest/compiler/bisheng/env/setvars.sh
export CC=$(which clang)
export CXX=$(which clang++)

# 进入sgl-kernel子目录并安装内核模块
cd ../sgl-kernel
cp pyproject_cpu.toml pyproject.toml
pip install -v . --no-build-isolation
```

## 3. 非PD分离场景启动过程

### 3.1 DeepSeek模型使用

以DeepSeek V3 int8量化版本为例，推荐配置为`DP=16`、`TP=16`（总共256个TP Worker），需要16个节点。在`scripts/cpu_kunpeng`目录下的`env.sh`脚本中配置`NATIVE_IP_SPEC`、`NATIVE_MASTER_ADDR`以及模型路径、conda环境名等信息。

由于V3量化版本权重加载较慢，可以在启动参数中添加`--load-format sharded_state`，从而支持权重分片加载以加速启动。首次使用前，需要用`scripts/cpu_kunpeng/split_weights.py`脚本对原始权重进行预处理。

启动和终止命令如下：

```shell
sh launch.sh native
sh stop.sh native
```

### 3.2 非DeepSeek模型使用

使用非DeepSeek模型时，需要注释掉`srt/models/registry.py`中第98\~99行，这两行会跳过非DeepSeek模型依赖的加载，其目的是提升启动加载速度。也可以将条件判断的值改成具体所用模型对应的值，例如`sglang.srt.models.qwen3`。

### 3.3 环境变量说明

- **KUPL\_EXECUTOR\_BACKEND** / **KUPL\_EXECUTOR\_COUNT**：仅在`TORCH_USE_KUPL`设置为`1`时生效，需要使用以kupl作为多线程后端的PyTorch。常规安装的PyTorch仅支持omp多线程后端。
- **SGLANG\_ENABLE\_BINARY\_LAUNCH**：用于优化启动和多线程绑核，默认开启。开启后每个scheduler进程独立启动，通过`--tp-rank-in-node`参数区分rank号。
- **SGLANG\_KUNPENG\_PROFILE**：开启函数调用开销打印，默认关闭。开启后会在标准输出中打印每个函数的调用时间，帮助定位性能瓶颈。

## 4. 正确性验证

非PD分离场景下的正确性验证，可通过向主节点的指定端口发送curl请求来验证。验证脚本示例如下：

```shell
export PORT=30000
MODEL_PATH=/path-to-model

time curl -s http://localhost:$PORT/v1/completions  \
  -H "Content-Type: application/json"   \
  -d '{
    "model": "$MODEL_PATH",
    "prompt": [
      "What is the capital of France?"
    ],
    "max_tokens": 64,
    "temperature": 0.01
  }'
```

根据请求的返回结果对正确性进行验证。
