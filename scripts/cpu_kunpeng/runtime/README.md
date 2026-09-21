# 多实例部署指南（cpu_kunpeng / PD 分离）

本文档介绍如何在一个集群中同时部署多个 prefill/decode 实例（例如普通 prefill 与 pp16 长序列 prefill 混布），以及实例相关的端口、NUMA、日志、路由规则的自动推导方式。

## 1. 核心概念：INSTANCES 列表

所有实例在 `runtime/.user_env_base.sh` 的 `INSTANCES` 中声明（默认值在 `runtime/env_base.sh`，`.user_env_base.sh` 覆盖之）：

```bash
export INSTANCES="prefill,prefill_pp16,decode_64p"
```

- 逗号分隔，每项为 `<role>` 或 `<role>_<instance>`；
- 实例名（`<instance>`，如 `pp16`、`64p`）只用于选择该实例的专属 env 覆盖文件与日志目录，任意命名均可；
- 同一角色可有多个实例（如上：两个 prefill 实例）。

`./launch.sh all` 会按此列表顺序依次拉起：每个实例的 server 集群 → 每个实例的 tokenizer → router。

## 2. 实例专属配置文件

配置加载顺序（`env.sh <role> <instance>`）：

```
env_base.sh（含 .user_env_base.sh，所有角色共享）
  └─ env_<role>.sh（角色默认值，${VAR:-...} 回退）
       └─ .user_env_<role>_<instance>.sh   ← 指定了实例时，只加载该实例文件
          或 .user_env_<role>.sh           ← 未指定实例时，加载角色默认文件
```

要点：

- **实例文件是"只此一份"的**：带实例启动时不再加载 `.user_env_<role>.sh`；
- 节点集（`<ROLE>_IP_SPEC` / `<ROLE>_IP_FILE`）、并行度（TP/DP/EP/PP）、master 地址等每实例必配项都写在实例文件里；
- `.user_env_base.sh` 是全角色共享的，**只能放对所有实例都成立的全局项**，不要放会因实例而异的变量。

### 现有实例文件示例

| 文件 | 用途 |
|---|---|
| `.user_env_prefill.sh` | 普通 prefill 实例（pp=1，短序列组） |
| `.user_env_prefill_pp16.sh` | pp16 prefill 实例（长序列组） |
| `.user_env_decode_64p.sh` / `_128p.sh` / `_32p.sh` / `_64p_2.sh` | 不同规模的 decode 实例 |

## 3. 每实例资源的自动推导

同一节点上/路由节点上的多实例资源按条目位置自动错开，**无需手动分配**（实例 env 文件也可显式覆盖）：

| 资源 | prefill 实例 | decode 实例 |
|---|---|---|
| tokenizer HTTP 端口 | `30001 + 全局条目序号` | 同左 |
| bootstrap 端口 | `9001 + prefill 侧序号` | 固定 `9001` |
| NUMA base | `4 × prefill 侧序号` | `8 + 4 × decode 侧序号` |
| master 端口 | `5000 + 100 × prefill 侧序号` | `5010 + 100 × decode 侧序号` |

以 `INSTANCES="prefill,prefill_pp16,decode_64p"` 为例：

| 条目 | 全局序号 | tokenizer 端口 | bootstrap 端口 | NUMA base | master 端口 |
|---|---|---|---|---|---|
| prefill | 0 | 30001 | 9001 | 0 | 5000 |
| prefill_pp16 | 1 | 30002 | 9002 | 4 | 5100 |
| decode_64p | 2 | 30003 | 9001 | 8 | 5010 |

日志目录同样按实例隔离：`$LOG_BASE_DIR/<日期>/<role>[_<instance>]/<时间>/`。

## 4. 路由：分组模式（短/长序列固定分组）

`runtime/.user_env_base.sh` 中：

```bash
export ROUTER_PREFILL_POLICY=bucket
export ROUTER_PREFILL_SHORT_COUNT=1        # 前 N 个 prefill 实例 = 短序列组
export ROUTER_PREFILL_LENGTH_THRESHOLD=8192  # 字符阈值（≈1.5-2 字符/token）
```

- 分组按 **prefill 实例的 URL（即 tokenizer 端口）升序** 排列，等价于 INSTANCES 中的先后顺序；
- 长度 < 阈值的请求发给前 `SHORT_COUNT` 个 prefill 实例，其余发给后面的实例；
- `SHORT_COUNT=0` 时为动态均衡模式（传递 balance/bucket 调参项）；
- 因此**实例在 INSTANCES 中的顺序决定了谁是短序列组**：想让 pp16 大实例收长序列，把它排在普通 prefill 之后即可。

## 5. 不同 PP 拓扑的 prefill 混布（重点）

支持 `prefill(pp=1)` 与 `prefill_pp16(pp=16)` 混布、decode 为 pp2 的拓扑。KV 传输路径按 bootstrap server 自动区分：

- decode 连普通 prefill（pp=1）：走 legacy 路径（`info.pp_size == 1` 分支），单个 prefill rank 同时服务两个 decode stage；
- decode 连 pp16 prefill：走 **PP 层区间映射（layer-interval mapping）**，按层 overlap 选择 prefill rank 并校验覆盖完整性，跨界 rank 自动分流/跳过。

### 必须注意的环境变量

1. **`PREFILL_PP_SIZE=16` + `DECODE_PP_SIZE=2` 必须同时出现在需要启用层映射的进程里**
   （`disaggregation/common/conn.py` 的 `_pp_layer_mapping_enabled` 要求两值同时满足）。
   只写在**两侧对应的实例文件**中（`prefill_pp16` 实例补 `DECODE_PP_SIZE=2`；`decode` 实例补 `PREFILL_PP_SIZE=16`），**严禁放进 `.user_env_base.sh`**——否则普通 prefill 实例的 `PP_SIZE` 会被污染成 16，且其 pp1→pp2 的 legacy 传输路径会被错误切换。

2. **`SGLANG_PP_LAYER_PARTITION`（如 `"35,26"`）只在 decode 实例文件中设置**
   约束：项数 = `DECODE_PP_SIZE`，总和 = 模型总层数（DeepSeek-R1 为 61），且所有 decode 节点一致。若泄漏到 prefill（pp16）进程，`len(partitions)=2 != pp_size=16` 会启动即崩。decode 侧改这个值是安全的——层区间选择、覆盖校验、prefill 侧切片全部由它动态推导，无需改其他配置。

3. 每个 prefill 实例有独立 bootstrap server（端口见第 3 节），decode 侧按 bootstrap 地址分别维护连接信息，互不干扰。

## 6. 常用命令

```bash
# 全量拉起（按 INSTANCES 顺序：各实例 server → 各实例 tokenizer → router）
./launch.sh all

# 单独拉起某个实例（第二个参数为实例名，可省略 = 默认实例）
./launch.sh prefill pp16
./launch.sh decode 64p

# 单独拉起某实例的 tokenizer（<prefill|decode> + 实例名）
./launch.sh tokenizer prefill pp16

# router / native
./launch.sh router
./launch.sh native

# 停止（注意：stop_server.sh 会杀死该实例节点集上的全部 sglang 进程）
./stop.sh decode 64p
```

## 7. 部署检查清单

- [ ] `INSTANCES` 中每个带实例名的条目都有对应的 `.user_env_<role>_<instance>.sh`；
- [ ] **各实例节点集互不重叠**（`stop_server.sh` 按节点集清理，重叠会导致误杀；IP_SPEC 按 `10.36.173. | a-b` 范围写法配置）；
- [ ] 同实例所有节点加载同一份实例 env（实例文件改动需同步到该实例全部节点）；
- [ ] 混布 pp16 prefill 时：`prefill_pp16` 实例文件有 `DECODE_PP_SIZE=2`，decode 实例文件有 `PREFILL_PP_SIZE=16`，且二者均不在 `.user_env_base.sh`；
- [ ] `SGLANG_PP_LAYER_PARTITION` 只出现在 decode 实例文件中，项数/总和正确；
- [ ] 路由分组顺序符合预期（INSTANCES 前部 = 短序列组）；
- [ ] 每实例 tokenizer 端口、master 端口未被同节点其他服务占用（自动推导已错开，显式覆盖时注意避让）。
