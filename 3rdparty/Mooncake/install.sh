#!/usr/bin/env bash

# ---------- 配置区 ----------
# 一般配置项
WORKSPACE=${WORKSPACE:-/tmp/mooncake}

# Conda 相关
USE_CONDA=${USE_CONDA:-true}
CONDA_ROOT=  # 留空时在 activate_conda_env 中从环境变量自动推导
CONDA_NAME=mooncake

# 本地 RDMA 包（环境对 RDMA 有特殊需求时使用）：两个路径同时配置后，
# Conda 模式将跳过 rdma-core 安装，并在 dependencies.sh 完成后拷贝到 conda 环境中
RDMA_CORE_INCLUDE=${RDMA_CORE_INCLUDE:-/usr/include/infiniband}
RDMA_CORE_LIB=${RDMA_CORE_LIB:-/usr/lib64}
# 只拷贝与模式匹配的库文件，避免把整个系统 lib 目录拷进 conda 环境
RDMA_CORE_LIB_NAMES=(libibverbs.so* librdmacm.so*)

# ---------- 其他变量 ----------
MOONCAKE_BASE_COMMIT=919ee81e5eb2891cfc79703762c6422f10ed9bac

# 本脚本与 Patch 同目录，基于脚本真实路径（解析符号链接）拼接 Patch 路径
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
THIS_DIR="${SCRIPT_DIR}/"
# sglang 根目录：默认按 <脚本目录>/../.. 推导，可用环境变量覆盖
ROOT_DIR=${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}

PATCH_BIND_NAME=0001-cpu-bind.patch
PATCH_CONDA_NAME=0002-conda-install.patch
PATCH_BIND_PATH="${SCRIPT_DIR}/${PATCH_BIND_NAME}"
PATCH_CONDA_PATH="${SCRIPT_DIR}/${PATCH_CONDA_NAME}"

# ---------- 输出与工具函数区 ----------
# 非终端（重定向/管道）时禁用颜色
if [ -t 1 ]; then
    GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'
else
    GREEN=''; RED=''; YELLOW=''; BLUE=''; NC=''
fi

# 分节标题（蓝色）
print_section() { echo -e "\n${BLUE}=== $1 ===${NC}"; }
# 一般信息（无着色）
print_info()    { echo -e "$1"; }
# 成功信息（绿色 ✓）
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
# 警告信息（黄色 !）
print_warn()    { echo -e "${YELLOW}! $1${NC}"; }
# 错误信息（红色 ✗，输出到 stderr）并退出
print_error()   { echo -e "${RED}✗ ERROR: $1${NC}" >&2; exit 1; }

# 检测返回码：上一条命令返回码非 0 则报错退出；$1 为错误描述（可省略）
check_success() {
    local ret=$?
    if [ "${ret}" -ne 0 ]; then
        print_error "${1:-上一步命令执行失败（返回码 ${ret}）}"
    fi
}

# 激活 Conda 环境：环境不存在则新建；CONDA_ROOT 未配置时从环境自动推导
activate_conda_env() {
    # 推导 conda 安装根目录：优先 CONDA_EXE，其次 PATH 中的 conda，最后报错
    if [ -z "${CONDA_ROOT}" ]; then
        if [ -n "${CONDA_EXE}" ]; then
            CONDA_ROOT=$(cd "$(dirname "$(dirname "${CONDA_EXE}")")" && pwd)
        elif command -v conda >/dev/null 2>&1; then
            CONDA_ROOT=$(conda info --base)
        else
            print_error "未检测到 Conda 且未配置 CONDA_ROOT，请安装 Conda 或设置 CONDA_ROOT 后重试"
        fi
    fi

    # 非交互式脚本中需先加载 conda 的 shell 函数，conda activate 才是可用命令
    # shellcheck disable=SC1091
    if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook)"
    fi
    check_success "加载 conda shell 函数失败"

    if conda env list | awk '{print $1}' | grep -qx "${CONDA_NAME}"; then
        print_info "检测到 Conda 环境 ${CONDA_NAME}，激活使用"
        conda activate "${CONDA_NAME}"
        check_success "激活 Conda 环境 ${CONDA_NAME} 失败"
    else
        print_info "未检测到 Conda 环境 ${CONDA_NAME}，新建环境"
        conda create -y -n "${CONDA_NAME}"
        check_success "创建 Conda 环境 ${CONDA_NAME} 失败"
        conda activate "${CONDA_NAME}"
        check_success "激活新建的 Conda 环境 ${CONDA_NAME} 失败"
    fi
}

# ---------- 执行区 ----------
# 本地 RDMA 包开关：Conda 模式且两个路径配置并有效时，拷贝系统 RDMA 包；
# 路径缺失/无效时退化为 conda 安装 rdma-core
USE_LOCAL_RDMA=false
RDMA_VIA_CONDA=false
if [ "${USE_CONDA}" = "true" ]; then
    if [ -n "${RDMA_CORE_INCLUDE}" ] && [ -n "${RDMA_CORE_LIB}" ]; then
        if [ -d "${RDMA_CORE_INCLUDE}" ] && [ -d "${RDMA_CORE_LIB}" ]; then
            if [ "${#RDMA_CORE_LIB_NAMES[@]}" -gt 0 ]; then
                USE_LOCAL_RDMA=true
            fi
        else
            print_warn "RDMA_CORE_INCLUDE/LIB 目录不存在（${RDMA_CORE_INCLUDE} / ${RDMA_CORE_LIB}），退化为 conda 安装 rdma-core"
        fi
    fi
    if [ "${USE_LOCAL_RDMA}" = "false" ]; then
        RDMA_VIA_CONDA=true
        print_info "RDMA 提供方式：conda-forge 安装 rdma-core（如需系统 RDMA，请配置有效的 RDMA_CORE_INCLUDE/RDMA_CORE_LIB）"
    fi
fi

# 将关键配置项打印（mooncake 下载到哪儿、是否使用 conda、conda 环境名等）
print_section "配置预览"
print_info "  WORKSPACE            = ${WORKSPACE}"
print_info "  MOONCAKE_BASE_COMMIT = ${MOONCAKE_BASE_COMMIT}"
print_info "  USE_CONDA            = ${USE_CONDA}"
print_info "  CONDA_ROOT           = ${CONDA_ROOT:-<自动推导>}"
print_info "  CONDA_NAME           = ${CONDA_NAME}"
print_info "  ROOT_DIR             = ${ROOT_DIR}"
print_info "  RDMA_CORE_INCLUDE    = ${RDMA_CORE_INCLUDE:-<未配置>}"
print_info "  RDMA_CORE_LIB        = ${RDMA_CORE_LIB:-<未配置>}"
print_info "  RDMA_CORE_LIB_NAMES  = ${RDMA_CORE_LIB_NAMES[*]:-<未配置>}"
print_info "  USE_LOCAL_RDMA       = ${USE_LOCAL_RDMA}"
print_info "  RDMA_VIA_CONDA       = ${RDMA_VIA_CONDA}"
print_info "  ${PATCH_BIND_NAME}   = ${PATCH_BIND_PATH}"
print_info "  ${PATCH_CONDA_NAME}  = ${PATCH_CONDA_PATH}"
print_info ""

if [ "${USE_LOCAL_RDMA}" = "true" ]; then
    print_warn "将使用本地 RDMA 包：include=${RDMA_CORE_INCLUDE}，lib=${RDMA_CORE_LIB}"
    print_info "0002 patch 已移除 rdma-core 的 conda 安装，依赖安装完成后拷贝到 conda 环境中"
fi

print_section "拉取 Mooncake 代码"
mkdir -p "${WORKSPACE}"
check_success "创建工作目录 ${WORKSPACE} 失败"
cd "${WORKSPACE}"
check_success "进入工作目录 ${WORKSPACE} 失败"

if [ -d Mooncake/.git ]; then
    print_info "Mooncake 已存在，跳过 clone"
else
    git clone https://github.com/kvcache-ai/Mooncake.git -b main Mooncake
    check_success "clone Mooncake 失败"
fi
cd Mooncake
check_success "进入 Mooncake 目录失败"

# 统一行尾为 LF，如果工作区检出为 CRLF 可能导致 Patch 上下文匹配失败。仓库级配置只影响本仓库，不影响全局。
git config core.autocrlf false
git config core.eol lf
if grep -q $'\r' dependencies.sh 2>/dev/null; then
    print_info "工作区检出为 CRLF 行尾，按 LF 重新检出"
    git checkout -f -- .
fi

print_section "应用 Patch"
if [ ! -f "${PATCH_BIND_PATH}" ]; then
    print_error "未找到 ${PATCH_BIND_NAME}：${PATCH_BIND_PATH}"
fi
if [ ! -f "${PATCH_CONDA_PATH}" ]; then
    print_error "未找到 ${PATCH_CONDA_NAME}：${PATCH_CONDA_PATH}"
fi

# 清空未提交的内容
git checkout .

if [ "$(git rev-parse HEAD)" != "${MOONCAKE_BASE_COMMIT}" ]; then
    git checkout "${MOONCAKE_BASE_COMMIT}"
    check_success "切换到基线 commit ${MOONCAKE_BASE_COMMIT} 失败"
fi
print_info "apply ${PATCH_BIND_NAME}"
git apply "${PATCH_BIND_PATH}"
check_success "应用 ${PATCH_BIND_NAME} 失败"
print_success "${PATCH_BIND_NAME} 应用成功"

if [ "${USE_CONDA}" = "true" ]; then
    print_info "apply ${PATCH_CONDA_NAME}"
    git apply "${PATCH_CONDA_PATH}"
    check_success "应用 ${PATCH_CONDA_NAME} 失败"
    print_success "${PATCH_CONDA_NAME} 应用成功"
else
    print_info "USE_CONDA=false，跳过 ${PATCH_CONDA_NAME}"
fi

print_section "安装依赖"
if [ "${USE_CONDA}" = "true" ]; then
    activate_conda_env
    # 让编译/链接能解析 conda 环境里的库和头文件（这里显式导出，避免依赖重新激活环境）
    export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
    export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include:${CPLUS_INCLUDE_PATH:-}"
else
    print_warn "USE_CONDA=false：以系统包管理器安装依赖（需要 root 权限）"
    if [ "$(id -u)" -ne 0 ]; then
        print_warn "当前非 root 用户，建议使用 sudo 重新执行本脚本"
    fi
fi

bash dependencies.sh
check_success "dependencies.sh 执行失败"

# 本地 RDMA 包：dependencies.sh 完成后拷贝到 conda 环境
if [ "${USE_LOCAL_RDMA}" = "true" ]; then
    print_section "拷贝本地 RDMA 包到 Conda 环境"
    # include：整个目录拷入（如 /usr/include/infiniband → include/infiniband/），
    # 保持 #include <infiniband/verbs.h> 的路径结构
    print_info "  ${RDMA_CORE_INCLUDE} → ${CONDA_PREFIX}/include/"
    cp -r "${RDMA_CORE_INCLUDE}" "${CONDA_PREFIX}/include/"
    check_success "拷贝 RDMA include 文件失败"
    # lib：只拷与 RDMA_CORE_LIB_NAMES 匹配的文件，避免污染 conda 环境的 lib
    _rdma_lib_copied=0
    shopt -s nullglob
    for pattern in "${RDMA_CORE_LIB_NAMES[@]}"; do
        for lib_file in "${RDMA_CORE_LIB}"/${pattern}; do
            print_info "  ${lib_file} → ${CONDA_PREFIX}/lib/"
            cp -r "${lib_file}" "${CONDA_PREFIX}/lib/"
            check_success "拷贝 ${lib_file} 失败"
            _rdma_lib_copied=$((_rdma_lib_copied + 1))
        done
    done
    if [ "${_rdma_lib_copied}" -eq 0 ]; then
        print_error "RDMA_CORE_LIB 下未匹配到任何库文件（模式：${RDMA_CORE_LIB_NAMES[*]}）"
    fi
    print_success "本地 RDMA 包已拷贝到 Conda 环境"
fi

# RDMA 提供方式的退化路径：本地无系统 RDMA 时用 conda 安装 rdma-core
if [ "${RDMA_VIA_CONDA}" = "true" ]; then
    print_section "安装 rdma-core 到 Conda 环境"
    conda install -y -n "${CONDA_NAME}" -c conda-forge rdma-core
    check_success "conda 安装 rdma-core 失败"
    print_success "rdma-core 已安装到 Conda 环境"
fi

print_section "编译安装 Mooncake"
mkdir -p build
cd build
check_success "进入 build 目录失败"

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_UNIT_TESTS=OFF -DENABLE_DEBUG_SYMBOLS=OFF"
if [ "${USE_CONDA}" = "true" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}"
fi
cmake .. ${CMAKE_ARGS}
check_success "cmake 配置失败"

make -j"$(nproc)"
check_success "make 编译失败"

make install
check_success "make install 失败"

print_success "Mooncake 安装完成"
print_info "Mooncake安装位置：$(python3 -c "import mooncake; print(mooncake.__path__[0])")"
