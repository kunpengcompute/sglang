cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sgl_kernel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Platform-specific library directory
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(PLAT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PLAT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|ppc64")
    set(PLAT_LIB_DIR "/usr/lib/powerpc64le-linux-gnu")
else()
    set(PLAT_LIB_DIR "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
endif()
link_directories(${PLAT_LIB_DIR})

# Conda library path support
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Using Conda lib dir: ${CONDA_LIB_DIR}")
    link_directories(${CONDA_LIB_DIR})
    set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
    include_directories(${CONDA_INCLUDE_DIR})

    # Look for libnuma in Conda's lib directory
    find_library(NUMA_LIB numa HINTS "${CONDA_LIB_DIR}")
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    else()
        message(FATAL_ERROR "libnuma not found in Conda environment at ${CONDA_LIB_DIR}\n"
                            "Please install it using: conda install libnuma numactl\n")
    endif()
else()
    if(DEFINED ENV{VIRTUAL_ENV})
        set(VENV_LIB_DIR "$ENV{VIRTUAL_ENV}/lib")
        message(STATUS "Using venv lib dir: ${VENV_LIB_DIR}")
        link_directories(${VENV_LIB_DIR})
        set(VENV_INCLUDE_DIR "$ENV{VIRTUAL_ENV}/include")
        include_directories(${VENV_INCLUDE_DIR})
    endif()
    # Look for libnuma in system env paths
    find_library(NUMA_LIB numa)
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    else()
        message(FATAL_ERROR "libnuma not found in system environment\n"
                            "Please install it using: apt-get install libnuma numactl\n")
    endif()
endif()

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
# Exclude kupl_async_kunpeng.cpp from common_ops (it's a separate module)
list(FILTER SOURCES EXCLUDE REGEX "kupl_async_kunpeng\\.cpp$")

if(NOT DEFINED ENV{SGLANG_CPU_FP8_CVT_FTZ})
    set(ENV{SGLANG_CPU_FP8_CVT_FTZ} "1")
endif()

if("$ENV{SGLANG_CPU_FP8_CVT_FTZ}" STREQUAL "1")
    message(STATUS "Enabling macro: SGLANG_CPU_FP8_CVT_FTZ")
    add_compile_definitions(SGLANG_CPU_FP8_CVT_FTZ)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_compile_options(
        -O3
        -Wno-unknown-pragmas
        -march=armv8.2-a+sve+fp16+bf16
        -fopenmp
    )
else()
    add_compile_options(
        -O3
        -Wno-unknown-pragmas
        -march=native
        -fopenmp
    )
endif()
add_compile_options(-Wno-unused-value)
add_compile_options(-Wno-vla-cxx-extension)

find_library(TORCH_PYTHON_LIB torch_python
    PATHS ${TORCH_INSTALL_PREFIX}/lib
    NO_DEFAULT_PATH)
if(NOT TORCH_PYTHON_LIB)
    message(FATAL_ERROR "libtorch_python not found in ${TORCH_INSTALL_PREFIX}/lib")
endif()

set(KUTACC_LIB "$ENV{KUTACC_LIB}/libkutacc.so")
set(KUTACC_INCLUDE "$ENV{KUTACC_INCLUDE}")
Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB} ${NUMA_LIB} ${KUTACC_LIB})
target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS} ${KUTACC_INCLUDE})

install(TARGETS common_ops
    LIBRARY DESTINATION sgl_kernel
)

# === pg_helper: standalone pybind11 module for ProcessGroup pointer extraction ===
# This module bypasses the torch dispatcher schema (which does not support
# ProcessGroup as a parameter type) by using pybind11 directly.
set(PG_HELPER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../pg_helper/pg_helper.cpp")
Python_add_library(pg_helper MODULE WITH_SOABI ${PG_HELPER_SOURCES})
target_link_libraries(pg_helper PRIVATE 
    ${TORCH_LIBRARIES}
)
target_include_directories(pg_helper PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

install(TARGETS pg_helper
    LIBRARY DESTINATION sgl_kernel
)

# === kupl_async: pybind11 module for async task submission to KUPL executor ===
set(KUPL_ASYNC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpu_kunpeng/kupl_async_kunpeng.cpp")
Python_add_library(kupl_async MODULE WITH_SOABI ${KUPL_ASYNC_SOURCES})
set_target_properties(kupl_async PROPERTIES OUTPUT_NAME "_kupl_async")
target_link_libraries(kupl_async PRIVATE
    ${TORCH_LIBRARIES}
    kupl
)
target_include_directories(kupl_async PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python_INCLUDE_DIRS}
    ${KUTACC_INCLUDE}
)

install(TARGETS kupl_async
    LIBRARY DESTINATION sgl_kernel
)

