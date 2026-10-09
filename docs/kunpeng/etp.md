# Kunpeng ETP（Expert Tensor Parallel，专家张量并行）

## 1. 概述

ETP在EP（专家并行）之上把每个专家的权重沿intermediate维切分到一组rank上，让热点专家的计算由整组分摊。面向920F decode场景：标准EP下热点expert单rank峰值计算远超稳态负载，TPOT由最慢rank决定；ETP把热点分摊并行度从1提升到moe_tp_size，同时每rank专家权重持有量不变（42MiB）。

DSV3 INT8、16节点decode的三档拓扑：

| 拓扑 | ep | etp(moe_tp) | attn_tp | dp | 组构成 | 每组专家数 | 热点分摊 |
|---|---|---|---|---|---|---|---|
| 非ETP | 256 | 1 | 16 | 16 | 每rank一组 | 1 | 1 |
| etp8 | 32 | 8 | 16 | 16 | 半节点（同socket）一组 | 8 | 8 |
| etp16 | 16 | 16 | 16 | 16 | 整节点（跨socket）一组 | 16 | 16 |

attn部分始终按attn_tp=tp/dp切分，与MoE切分独立，可组合出"attn按16切+MoE按8切"。etp8组内SHM访问全为socket本地；etp16组内必有跨socket访问。

## 2. 工作原理

### 并行度与分组

- moe_tp_size=tp_size/ep_size（映射到sglang的moe_tp机制，moe_dp=1）
- ETP组=节点内连续moe_tp_size个rank，leader=intra_rank%moe_tp_size==0
- 每组共持num_experts/ep_size个专家，每rank持有每个本地专家的1/moe_tp_size inter份额（gateup N、down K同比例缩小）

### 每层MoE数据流

```
[层边界] hidden: 组内各rank冗余持完整batch（DP-attention语义）
  → topk（组内冗余计算，结果逐位相同）
  → etp_remap_topk_ids：逻辑专家id → (leader rank, 本地槽位)   ← 路由表重映射
  → dispatch_send（全员duty切片，表驱动单播leader，kutacc代码零改动）
  → [仅leader] dispatch_recv
  → dispatch_share（pull-A）：leader压缩dense+推送段元数据；peer自旋等序号
  → gateup/silu/down：组内各rank相同dense输入 × 各自1/etp权重分片 → partial
  → etp_reduce（行所有权）：全组各认领1/etp行段求和，写入leader的moe_down
  → [仅leader] combine_send
  → combine_recv（源端加权求和，与非ETP完全相同）
  → mul_scalar_add(shared) → 下层attention的attn-TP allreduce
```

两个设计要点：

- dispatch/combine的kutacc RDMA协议零改动。仅靠路由表重映射，收/发侧角色自然集中到leader：peer的槽位bitmap全零（一个completion都不poll），src_info恒为-1（零iov）。
- combine_recv后组内不需要allreduce：attn-TP组的token维是同一份数据的冗余持有而非划分；MoE层内token维经dispatch暂时物化为划分，combine时还原为源rank持有，源rank恰好就是原本冗余持有它的那个组。

### dispatch_share（pull-A模式）

etp_dense_buf是组内SHM对称分配，peer天然可寻址leader副本，因此不需要广播数据：

1. leader把有效slot行压缩进自己的dense_buf（唯一一次数据拷贝，kutacc线程池按行块并行）；
2. 仅向peer推送段元数据（experts_offset+total，几十字节）；
3. peer等序号后，gateup的pack直接读leader的dense：C++侧etp_redirect_dense按地址区间把act/scale指针重定向到leader副本。Python侧act/scale视图仍指向本rank的dense_buf（永不写入，但SHM对称分配必须保留），图注册与算子schema零改动。

### etp_reduce（行所有权）

reduce-scatter式全组参与：rank gid认领互不相交的行段[gid*chunk,(gid+1)*chunk)，读该段全部moe_tp_size份partial（fp32累加、bf16写回），写入leader的moe_down切片；仅leader等齐全组段完成序号后返回（combine_send语义不变）。相比leader-only归约，读带宽由全组节点分摊而非压leader单点。求和顺序固定（组内编号升序）。

### 同步协议（etp_seq，int64[3]，SHM对称）

| 槽位 | 含义 | 写者 |
|---|---|---|
| [0] | dense压缩+元数据发布序号 | leader |
| [1] | down GEMM完成序号 | 每个rank写自己的 |
| [2] | 行段归约完成序号 | 每个rank写自己的 |

单调计数、release/acquire序、publish-then-wait、全组lockstep。不使用kupl_shm_fence（它是全节点集合屏障，与自旋等待互等会死锁）。图重放安全：每次replay重新执行op体，序号继续推进。

### 无撕裂保证

leader为下层覆写dense前，必经本层reduce等齐全组[2]（peer的down晚于gateup），故peer的gateup读不会被下层压缩写撕裂；peer的partial被他人读后，其下次覆写（下层down GEMM）排在leader下层bcast之后，而那又排在leader本层reduce返回之后。

## 3. 开启与配置

### 部署实例

runtime实例文件（如`.user_env_decode_32p-etp8.sh`）：

```bash
export DECODE_TP_SIZE=256
export DECODE_EP_SIZE=32              # < TP_SIZE 即启用ETP；moe_tp = 256/32 = 8
export DECODE_DP_SIZE=16              # attn_tp = 256/16 = 16，与MoE切分独立
export DECODE_PP_SIZE=1
```

启动：`INSTANCES="prefill,decode_32p-etp8"` + `./launch.sh all`（或`./launch.sh decode 32p-etp8`）。

### 权重预切分（强烈建议）

运行时切分存在etp倍读放大（整专家读入再丢弃其余分片，etp16实测约6.5min）。离线预切分后loader自动探测目录走fast path（presharded，跳过运行时切片）：

```bash
python scripts/cpu_kunpeng/model_processing/split_weights_dsv3.py \
    --model_dir <model_path> --etp_shard --etp_size 8
# 产出 <model_path>/experts_etp8/layer_{l}/ep{g}_tp{r}.safetensors
# 每文件=组g的本地专家各1/8分片，16进程按(层,组)并行，无读写放大
# MTP draft模型（experts/下仅layer_0）对<mtp_model_path>再跑一次，脚本按目录自动适配
```

磁盘代价约等于原experts目录。目录不存在时loader回落运行时切分，行为兼容。

### 环境变量与内存预算

| 变量 | 说明 |
|---|---|
| DECODE_EP_SIZE | 小于TP_SIZE即启用ETP，moe_tp=tp/ep |
| SGLANG_KUNPENG_ETP_DENSE_MAX | dense行上限；默认dp×max_tokens_per_mb×min(topk,n_local)，MTP按steps+1线性放大；偏小运行时报ETP dense buffer overflow |
| SGLANG_KUNPENG_DECODE_SHM_SIZE_MB | decode SHM池，ETP扩容主要项（etp8+MTP2约256；etp16+MTP2需320且SHM走DDR，即unset KUPL_SHM_ON_PACKAGE） |
| DECODE_WEIGTHS_HBW_POOL_SIZE_MB | 权重HBW池（etp8/etp16配3850/3830） |

推荐配置（MTP=1，即SGLANG_ENABLE_MTP=1 + SGLANG_SPECULATIVE_NUM_STEPS=1，取自runtime实例文件实测值）：

| 实例 | SGLANG_KUNPENG_DECODE_SHM_SIZE_MB | DECODE_WEIGTHS_HBW_POOL_SIZE_MB |
|---|---|---|
| decode_32p-etp8（ep32/dp16，16节点） | 160 | 3850 |
| decode_64p-etp8（ep32/dp32×pp2，32节点） | 200 | 3800 |

MTP steps每+1，dense相关SHM按steps+1线性放大（见SGLANG_KUNPENG_ETP_DENSE_MAX），池大小需相应上调（如32p-etp8在MTP=2时配256）。

主要SHM项（组内对称分配，get_peer_shm_baseptr按相同偏移互访）：

- dispatch_recv_buf（槽位空间）=n_local×comm×max_dispatch×(hidden+4)，decode侧在HBW池
- etp_dense_buf=dense_max×(hidden+4)
- combine_send_buf=dense_max×hidden×2（bf16，leader归约结果的宿主）

注意：graph runner会把SHM池剩余整体alloc作capture workspace，池容量需为此留余量。

### 启动约束（server_args._validate_kunpeng_etp自动校验）

- tp%ep==0且moe_tp>1；moe_dp==1；仅decode角色
- moe_tp整除每节点rank数（组不出节点）
- 无EPLB、冗余专家、静态路由、FORCE_LOAD_BALANCE、fused shared experts（后者自动关闭并打日志）

## 4. 性能与调优（32p-etp8实测，DSV3，单请求1kin/128out/MTP1）

| 阶段 | etp_share(μs) | etp_reduce(μs) | MoE总时间 | vs非ETP基线 |
|---|---|---|---|---|
| 非ETP基线（ep256） | — | — | 59.24ms | — |
| 初始（push广播+串行归约） | 51.0 | 77.2 | 69.69ms | +17.7% |
| reduce线程池并行 | 51.0 | 26.8 | 65.45ms | +10.5% |
| share改pull-A | 6.5 | 29.2 | 62.69ms | +5.8% |
| reduce行所有权 | 6.9 | 18.4 | 62.07ms | +4.8% |

剩余回退主要在gateup的GEMM形状（8专家串行8波×每波barrier）。

### multiexpt专家分组（kutacc侧，实验特性）

fusedmoe_*_multiexpt_parallel把工作池分成G组×T线程（T=池/G），组以stride-G认领专家并发处理，波数与barrier数按每组专家数成比例下降。ne==2保留dualexst路径（128p冗余专家场景）。

开关KUTACC_FUSEDMOE_MULTIEXPT=\<组数\>（默认0=串行路径，bit-exact回退）：

| 组数G（etp8, gateup N=512/K=7168, down N=7168/K=256） | gateup | down | gateup波数 |
|---|---|---|---|
| 0（默认） | 串行 | 串行 | 8 |
| 2 | T=16, tile (256,896)* | T=16, tile (448,256)* | 4 |
| 4 | T=8, tile (512,896)* | T=8, tile (896,256)* | 2 |
| 8 | 不满足（T=4不被bk=8整除），回落串行 | T=4, tile (1792,256)* | — |

*组内tile由kutacc内核自行推导（multiexpt_group_tile：保持CSV条目的tile_k、重算tile_n使组内blocks恰等于T），**不依赖也不修改CSV**——串行路径要求同形状CSV条目blocks==池大小（32），同一(N,K)键无法同时服务两条路径（曾因在CSV里加multiexpt档位覆盖串行条目，导致串行路径4×冗余计算、gateup 137.6→785μs/层，2026-10-08实测踩坑后改为内核内推导）。tile_k保持不变确保k-partial舍入与串行路径bit一致；tile_n变化是结果中性的。

串行路径防呆：kutacc在fusedmoe分派处校验CSV条目blocks==池大小——blocks>池直接abort（会静默漏算、结果错误），blocks<池打警告（冗余计算、结果正确但成倍变慢）。

tile_k敏感性警告：任何tile_k变化会改变k-partial的bf16舍入边界，数值末位漂移会显著拉低MTP接受率（实测(256,1792)使接受率1.87→1.46）。调tiling时保持tile_k不变，并把MTP接受率作为回归守卫。

## 5. 关键代码位置

- 约束校验：`python/sglang/srt/server_args.py`（_validate_kunpeng_etp）
- 状态捕获、buffer布局、pull-A集成：`python/sglang/srt/layers/moe/token_dispatcher/kunpeng.py`
- 路由表重映射调用：`python/sglang/srt/layers/moe/topk.py`
- 权重1/N切分与预切分fast path：`python/sglang/srt/model_loader/loader.py`
- dense sizing与reduce插入：`python/sglang/srt/layers/moe/ep_moe/layer.py`
- 图算子注册：`python/sglang/srt/graph/adapters.py`
- ETP算子（dispatch_share/reduce/重定向）：`sgl-kernel/csrc/cpu/cpu_kunpeng/moe/kunpeng_moe.cpp`
- 图算子C++适配：`sgl-kernel/csrc/cpu/cpu_kunpeng/adapters/moe_etp.cpp`
- multiexpt并行与MR注册参数化：kutacc仓库`src/core/matmul/linear_moe.cpp`
- 离线预切分：`scripts/cpu_kunpeng/model_processing/split_weights_dsv3.py`（--etp_shard）
- GEMM tiling：`scripts/cpu_kunpeng/configs/dsv3_32_tiling.csv`
- 部署实例：`scripts/cpu_kunpeng/runtime/.user_env_decode_32p-etp8.sh`（etp16同理）
