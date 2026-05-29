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

Before installing SGLang components, you need to create a Python virtual environment and install the Torch library. The commands are as follows. For more information, refer to the [installation guide](https://docs.sglang.com.cn/platforms/cpu_server.html#install-from-source) provided by the SGLang community.

```shell
source ~/anaconda3/start_conda.sh
conda create -n sgl-cpu python=3.12 -y
conda activate sgl-cpu

pip install --upgrade pip setuptools
conda install -y tbb libnuma numactl

pip install torch==2.9.0 torchvision==0.24.0 triton==3.5.0 --force-reinstall
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
cp pyproject_cpu.toml pyproject.toml
pip install --upgrade pip setuptools
pip install -e "python[all]"
pip install vllm

# Load Bisheng Compiler environment variables
HPCKIT_PATH=/path-to-HPCKit
source ${HPCKIT_PATH}/latest/compiler/bisheng/env/setvars.sh
export CC=$(which clang)
export CXX=$(which clang++)

# Enter the sgl-kernel subdirectory and install the kernel module
cd ../sgl-kernel
cp pyproject_cpu.toml pyproject.toml
pip install -v . --no-build-isolation
```

## 3. Non-PD Disaggregated Startup

### 3.1 Using DeepSeek Models

Taking the DeepSeek V3 int8 quantized version as an example, the recommended configuration is `DP=16`, `TP=16` (256 TP Workers in total), requiring 16 nodes. Configure `NATIVE_IP_SPEC`, `NATIVE_MASTER_ADDR`, the model path, the conda environment name, and other settings in the `env.sh` script under the `scripts/cpu_kunpeng` directory.

Since loading the V3 quantized weights is relatively slow, you can add `--load-format sharded_state` to the startup parameters to enable sharded weight loading for faster startup. Before the first use, preprocess the original weights using the `scripts/cpu_kunpeng/split_weights.py` script.

The startup and shutdown commands are as follows:

```shell
sh launch.sh native
sh stop.sh native
```

### 3.2 Using Non-DeepSeek Models

When using non-DeepSeek models, you need to comment out lines 98~99 in `srt/models/registry.py`. These two lines skip loading non-DeepSeek model dependencies to improve startup speed. Alternatively, you can change the condition value to the corresponding value of the specific model being used, for example, `sglang.srt.models.qwen3`.

### 3.3 Environment Variables

- **KUPL_EXECUTOR_BACKEND** / **KUPL_EXECUTOR_COUNT**: Only effective when `TORCH_USE_KUPL` is set to `1`. Requires a PyTorch build that uses kupl as the multi-threading backend. The standard PyTorch installation only supports the omp multi-threading backend.
- **SGLANG_ENABLE_BINARY_LAUNCH**: Used to optimize startup and multi-thread core binding. Enabled by default. When enabled, each scheduler process starts independently, distinguished by the `--tp-rank-in-node` parameter.
- **SGLANG_KUNPENG_PROFILE**: Enables function call overhead logging. Disabled by default. When enabled, the call time of each function is printed to stdout to help identify performance bottlenecks.

## 4. Correctness Verification

In non-PD disaggregated scenarios, correctness can be verified by sending a curl request to the specified port on the master node. An example verification script is as follows:

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

Verify correctness based on the response returned by the request.
