# kunpeng/patches

本目录存放针对外部项目（未内置在本仓库）的补丁，基于固定的上游基线 commit 应用。

| Patch | 目标项目 | 上游基线 commit | 说明 |
|-------|----------|-----------------|------|
| `0001-cpu-bind-transfer-engine-threads-to-offset-cpu-of-affi.patch` | Mooncake (`mooncake-transfer-engine`) | `919ee81e5eb2891cfc79703762c6422f10ed9bac` | 支持 `SGLANG_SET_MOONCAKE_CPU_AFFINITY_OFFSET`，将 transfer engine 线程绑定到亲和性 NUMA 节点上的偏移 CPU |

## 详细介绍

### Mooncake绑核功能补丁

```bash
# 在 Mooncake checkout 上应用（建议用 git am 保留 commit 信息）
git checkout 919ee81e5eb2891cfc79703762c6422f10ed9bac
git am <本目录>/0001-cpu-bind-transfer-engine-threads-to-offset-cpu-of-affi.patch

# 或只改工作区
git apply <本目录>/0001-cpu-bind-transfer-engine-threads-to-offset-cpu-of-affi.patch
```

完整的编译流程请参阅[内部Wiki](https://wiki.huawei.com/domains/110323/wiki/201487/WIKI2026081112248980)。