# SGLang鲲鹏优化版本参考实现


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

# 进入sgl-kernel子目录并安装内核模块
cd ../sgl-kernel
cp pyproject_cpu.toml pyproject.toml
pip install .
```

### 2.3 DeepSeek-V3-Sample安装

该优化方案对接DeepSeek鲲鹏分布式推理参考实现，具体安装过程参考gitcode[安装指导文档](https://gitcode.com/kunpengcompute/DeepSeek-V3-Sample)。

## 3. SGLang PD分离启动过程
目前PD分离场景下的Prefill Worker和Decode Worker均需配置为`DP=16`且`TP=16`（每个节点上1个DP，每个DP内16个TP Worker），即使用32个节点来启动一组PD分离实例。

### 3.1 SGLang PD分离启动

在`scripts/cpu_kunpeng`目录下的`env.sh`脚本中配置`PREFILL_IP_SPEC`、`DECODE_IP_SPEC`等信息，并修改模型路径和conda环境名。然后执行`launch_sglang.sh`脚本启动多节点Prefill Worker和Decode Worker：

```shell
bash launch_sglang.sh prefill
bash launch_sglang.sh decode
```

### 3.2 启动DeepSeek-V3-Sample

Server模式下的启动配置参考[安装指导文档](https://gitcode.com/kunpengcompute/DeepSeek-V3-Sample)中的2.5小节。

### 3.4 启动SGLang Router

在完成SGLang和DeepSeek-V3-Sample启动后，执行如下命令启动Router。Router建议部署在独立的节点上。

```shell
python -m sglang_router.launch_router --model-path /path-to-model --pd-disaggregation --prefill http://$PREFILL_MASTER_ADDR:30001 9001 --decode http://$DECODE_MASTER_ADDR:30001 --policy cache_aware --prefill-policy cache_aware --health-check-interval-secs 10000 --queue-timeout-secs 10000 --request-timeout-secs 10000 --health-check-timeout-secs 10000
```

## 4. SGLang PD分离正确性验证

SGLang PD分离的正确性验证，可通过在Router所在节点上运行curl测试脚本完成。脚本示例如下：

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