
> 适用：Kunpeng CPU + PP（pipeline parallel）disagg-decode 场景。
>
> 相关文件：
> - 打点实现：[pp_perf.py](../../python/sglang/srt/utils/pp_perf.py)（`sglang.srt.utils.pp_perf`）
> - 开关：由 `env.sh` 中的 `SGLANG_KUNPENG_PP_PROFILE` 环境变量控制
> - 本文档：描述 PP 主流程、每个打点点位对应的耗时含义，以及如何使用。

---

## 1. 如何开关（默认关闭，零开销）

在 `scripts/cpu_kunpeng/env.sh` 中：

```bash
export SGLANG_KUNPENG_PP_PROFILE=0   # 0 = 关闭（默认）；1 = 打开 PP 打点
```

- **关闭（=0，默认）**：`@Kunpeng_PP_Profiler(...)` 装饰器直接返回原始函数，不打点、不包包装、不记时——热路径上**完全没有性能开销**。
  配合 `event_loop_pp_disagg_decode` 里的 `pp_perf_start/pp_perf_report` 也是空操作。
- **打开（=1）**：每个有请求（`cur_batch` 非空）的 mb 迭代末尾，打印一棵阶段耗时树（见 §5）。

> 由于是否打点是在 `import` 时就按环境变量判定的，必须在**进程启动前**设置好该变量（`env.sh` 在启动脚本里已自动 source）。

---

## 2. 打点 API 与层级

**装饰器**，按真实的调用栈自动形成父子层级：

```python
from sglang.srt.utils.pp_perf import Kunpeng_PP_Profiler

@Kunpeng_PP_Profiler(depth=2, name="收请求")
def recv_requests(self): ...
```

- `depth`：记录到**该函数下多少层**被装饰的子调用。`depth=2` = 记录本函数 + 最多 2 层被装饰的子函数。
- `name`：日志里显示的 span 名（不填则用函数名）。
- 层级来源：装饰器会记录"当前线程调用栈"，被装饰函数 A 的体内调用另一个被装饰函数 B 时，B 就自动成为 A 的子节点。
- 顶层 span：事件循环里被直接调用的被装饰方法（调用栈为空）会成为 mb 迭代的**一级节点**；未被子节点覆盖的剩余时间记为 `[self]`，窗口 total 减去所有一级 span 之和记为 `[other]`。

---

## 3. PP 主流程（每个阶段打点的是什么）

事件循环 `event_loop_pp_disagg_decode` 每个 mb 迭代按下面顺序执行。**加粗函数**为已打点（被装饰）的阶段函数；该行即是它对应的耗时。

### 3.1 PP0（首 rank）
```
pp_perf_start("mb{i}", tag="pp0")            # 开窗口
① 收请求       recv_requests / process_input_requests      ZMQ 非阻塞收 tokenizer/RPC 请求 + 分发
   commit_comm_work                           等上一轮"请求转发 isend"等通信完成
② PD 共识      pd_retract / pd_prealloc / pd_transfer     KV retract/prealloc/transfer 的 rids 共识（含 poll + allreduce）
③ 调度批       get_batch (→ update_running / mlp_sync / pd_prepare)   取 decode batch、DP allgather、prepare_for_decode
④ 收 proxy     (PP0 无)
⑤ 提前发输出    send_output_pre (async depth>0 时，launch 前)
   commit_comm_work                           等上轮 proxy 发送完成
⑥ 模型执行     launch_batch (→ run_batch → forward_batch → model_runner.forward)     计算大头
⑦ 共识发送     send_consensus / send_release              把本轮共识转发给下一 rank
⑧ 共识同步     recv_pyobj / proc_queue / commit_comm_work 收上一级共识 + 本地处理 + 等发送完成
⑨ 结果处理     process_batch_result (→ stream_output)     copy/sample 转 list、per-req 完成处理、KV 释放、流式输出
⑩ 转发         send_pyobj / send_proxy                    转发请求/rid 与 hidden proxy 到下一 rank
pp_perf_report(...)                            # 关窗口并打印
```

### 3.2 PP1（末 rank，PP=2 时）
```
pp_perf_start("mb{i}", tag="pp1")
① 收请求       recv_requests (阻塞收) / process_input_requests
② PD 共识      pd_retract / pd_prealloc / pd_transfer    （非首 rank 会 recv_prev_rids 与本地取交集）
③ 调度批       get_batch (→ update_running / mlp_sync / pd_prepare)
④ 收 proxy     recv_proxy                                  从 PP0 收 hidden proxy 张量（RDMA / gloo）
⑤ 提前发输出    send_output_pre
⑥ 模型执行     launch_batch (→ run_batch → model_runner.forward → sample)
⑦ 共识发送     send_consensus / send_release              末 rank 真正发出共识（回环给 PP0）
⑧ 共识同步     recv_pyobj / proc_queue / commit_comm_work
⑨ 结果处理     process_batch_result (→ stream_output)
pp_perf_report(...)
```

### 3.3 PP0 / PP1 差异对照
| 阶段 | PP0（first rank） | PP1（last rank） |
|---|---|---|
| 收请求 | ZMQ 非阻塞收 | 从 PP0 阻塞收 |
| PD 共识 | 本地 `get_rids` | `recv_prev_rids` + 本地取交集 |
| recv_proxy | 无 | 有（收 proxy 张量） |
| send_consensus | 转发上一轮共识 | 真正发出共识（回环给 PP0） |
| send_pyobj / send_proxy | 有 | 无 |
| 采样 sample | 无 | 有 |

---

## 4. 打点点位 → 耗时含义

以下点位均为装饰器打点（对应 §3 中加粗名称）。`commit_comm_work` 每次迭代出现多次、按同名在日志里聚合成一行，表示该迭代累加的所有提交等待（含跨 rank 等 ack）。

| 点位（name） | 对应函数 | 耗时含义 | 出现时机 |
|---|---|---|---|
| `recv_requests` | scheduler.recv_requests | 收一波请求 | 每个 mb |
| `proc_input` | scheduler.process_input_requests | 请求分发/入队 | 每个 mb |
| `commit_comm_work` | _pp_commit_comm_work | 等各类 PP isend/ack 完成（送请求、送 proxy、送共识） | 多处聚合 |
| `pd_retract` | _pp_pd_get_retract_ids | retract 共识 rids（含 poll+allreduce/收 prev） | 每个 mb |
| `pd_prealloc` | _pp_pd_get_prealloc_ids | prealloc 共识 rids | 每个 mb |
| `pd_transfer` | _pp_pd_get_decode_transferred_ids | transfer 共识 rids | 每个 mb |
| `recv_proxy` | _pp_recv_proxy_tensors | 收 proxy 张量 | (仅PP1) 有 proxy 时 |
| `send_output_pre` | _pp_commit_send_output… | 提前发/收输出并预处理 | async depth >/== 0 |
| `send_res` | _pp_send_recv_and_preprocess… | 输出张量 send/recv + 组结果 | 提前发输出内部 |
| `send_consensus` | _pp_pd_send_consensus_bootstrapped_ids | 发 retract/prealloc 共识 | 每个 mb |
| `send_release` | _pp_pd_send_consensus_release_ids | 发 release 共识 | 每个 mb |
| `recv_pyobj` | _pp_recv_pyobj_from_prev_stage | 收上一级共识 rid | 有下一 mb 数据时 |
| `process_retract/prealloc/decode_transfer_queue` | process_*_queue | 处理收到的共识队列 | 有下一 mb 数据时 |
| `get_batch` | get_next_disagg_decode_batch_to_run | 取 decode batch（prebuilt 合并+update_running+mlp_sync） | 每个 mb |
| `get_new_prebuilt` | get_new_prebuilt_batch | 造 fake prefill batch | 有 waiting 时 |
| `proc_prebuilt` | process_batch_result_prebuilt | 处理上一批 prebuilt 结果 | 有 prebuilt 时 |
| `update_running` | update_running_batch | filter/check_mem/retract/prepare_for_decode | running 非空时 |
| `pd_prepare` | ScheduleBatch.prepare_for_decode | KV 页分配、seq_len 自增等 | decode batch 时 |
| `mlp_sync` | maybe_prepare_mlp_sync_batch | DP allgather MLP 同步准备 | 每个 mb |
| `prepare_mlp_sync` | prepare_mlp_sync_batch_raw | allgather 本体 | MLP 同步时 |
| `launch_batch` | _pp_launch_batch | 整轮模型执行（含 run_batch→forward） | 有 cur_batch 时 |
| `run_batch` | scheduler.run_batch | 调度并执行 batch | launch 内部 |
| `forward_batch` | tp_worker.forward_batch_generation | 模型 worker 前向 + 采样 | launch 内部 |
| `model_runner.forward` | ModelRunner.forward | 模型 forward 分发 | 前向内部 |
| `forward_decode` / `forward_extend` / `forward_idle` | ModelRunner.forward_* | 各 forward 模式本体（decode 大头/图执行） | 按 forward_mode |
| `process_batch_result` | process_batch_result_decode | copy 同步、token 转 list、per-req 完成处理、KV 释放、stats | 有下一 mb 结果时 |
| `stream_output` | stream_output(_generation) | 流式输出/送 detokenizer | 结果处理内部 |
| `send_pyobj` | _pp_send_pyobj_to_next_stage | 转发请求/rid | (仅PP0) |
| `send_proxy` | _pp_send_dict_to_next_stage | 发 proxy 张量 | (仅PP0) 有 proxy 时 |

---

## 5. 日志输出示例与阅读

开启后，有真实请求的 mb 迭代末尾打印一棵树：

```
[2026-08-19 11:44:12.216] [pp0][mb0] loop start
[2026-08-19 11:44:12.216] [pp0][mb0] flow total=106.08ms
[2026-08-19 11:44:12.216] [pp0][mb0] |-- launch_batch                   49.39ms cpu: 43.11ms
[2026-08-19 11:44:12.216] [pp0][mb0] |   |-- run_batch                   49.06ms cpu: 42.90ms
[2026-08-19 11:44:12.216] [pp0][mb0] |   |   |-- model_runner.forward    48.55ms cpu: 42.60ms
[2026-08-19 11:44:12.216] [pp0][mb0] |   |   |   |-- forward_decode      48.10ms cpu: 42.10ms
[2026-08-19 11:44:12.216] [pp0][mb0] |-- recv_requests                    1.02ms cpu:  0.98ms
...
[2026-08-19 11:44:12.216] [pp0][mb0] loop end
```

阅读要点：
1. 每行 `ms` 为墙钟耗时，`cpu:` 为该线程 CPU 制时间；墙钟 ≫ CPU 说明该点主要在**等待**（跨 rank 通信/等 ack）。
2. 每个父节点下未被更细子节点覆盖的时间为 `[self]`（残留 >0.05ms 才打印）；窗口 `total` 减所有一级 span 之和为 `[other]`。
3. **深度控制**：想展开更多层，就把对应父级装饰器的 `depth` 调大（例如 `launch_batch` 想看到 `sample`、图执行等就改成 `depth=3`）。
4. **真实 token 间隔**（与打点树的"窗口总耗时"口径不同）：用日志时间戳——同一 slot 相邻两条 `loop start` 行的时间戳差。