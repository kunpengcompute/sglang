# Tokenizer 时间线分析（跨进程性能打点）

## 概述

Tokenizer 时间线面向 PD 分离 + tokenizer 分离部署（router 节点上跑 HTTP server / detokenizer / tok worker，计算节点上跑 scheduler）：在 decode 输出链路的每个环节打 wall-clock 时间戳，由 tok worker 逐 batch 追加写入 JSONL 文件，配合离线分析脚本定位 TPOT 异常的瓶颈环节。

特性：跨进程跨机时间线（scheduler → detokenizer → tok worker）、batch 级与 request 级两种记录、per-request TPOT 分解（恒等式逐项拆解）、与 aisbench 客户端指标的对照口径。

## 工作原理

一个 batch 的输出沿链路传递，各环节在消息对象上原地打戳（`time.time()`，秒级 Unix 时间戳）：

```
计算节点                                router 节点
scheduler ──sched_send──> detokenizer ──dtok_recv/dtok_send──> tok worker
                          (dtok_proc)                          (tok_recv/tok_send)
```

| 时间戳 | 打点位置 | 含义 |
|---|---|---|
| `sched_send` | `scheduler_output_processor_mixin.py`，构造 `BatchTokenIDOutput` 时 | scheduler 发出该 batch（idle batch 不打） |
| `dtok_recv` / `dtok_send` | `cpu_kunpeng/managers/detokenizer_mixin.py`（由 `detokenizer_manager.py` 的 `event_loop` hook 调用） | detokenizer 收到 / 处理完发出 |
| `tok_recv` / `tok_send` | `cpu_kunpeng/managers/tokenizer_mixin.py`（由 `tokenizer_manager.py` 的 `_handle_batch_output` hook 调用） | tok worker 收到 / 处理完（组装 meta_info、拼文本） |

时间戳字段定义在 `io_struct.py` 的 `BatchTokenIDOutput` / `BatchStrOutput` 上，随消息经 ZMQ 传递；tok worker 处理完后把完整记录写 JSONL。

由此定义四个 pipeline 段（分析脚本按此统计）：

| 段 | 计算 | 跨机 |
|---|---|---|
| `sched->dtok` | `dtok_recv − sched_send` | 是（受 NTP 时钟偏移影响） |
| `dtok_proc` | `dtok_send − dtok_recv` | 否（router 节点内） |
| `dtok->tok` | `tok_recv − dtok_send` | 否（含排队） |
| `tok_proc` | `tok_send − tok_recv` | 否 |

### TPOT 分解恒等式

单请求视角下，第 n 轮 TPOT（相邻两次 `tok_send` 之差）可精确拆解为各段增量之和：

```
TPOT_n = tok_send(n+1) − tok_send(n)
       = Δsched_send + Δ(sched->dtok) + Δdtok_proc + Δ(dtok->tok) + Δtok_proc
```

`Δsched_send` 即 scheduler 迭代周期；`Δ(dtok->tok)` 持续为正说明 worker 接收队列在积压，为负说明在消化。batch 记录中的 `rids` 字段支持按请求筛选回放轮次（trace 模式）。

## 开启与配置

环境变量（router 与计算节点都要生效）：

- `SGLANG_TOKENIZER_TIMELINE_LOG`（默认 0）：总开关。计算节点上控制 `sched_send` 打点，router 节点上控制 detok/tok 打点与 JSONL 写入
- `SGLANG_TOKENIZER_TIMELINE_PATH`（默认空）：日志路径；空 = `/tmp/sglang_tokenizer_timeline_{pid}.jsonl`

注意：

- 与 `--enable-request-time-stats-logging` 无关。该 server 参数只控制 scheduler/tok 侧的文本日志；JSONL 记录仅由上面的环境变量控制
- **两节点代码必须同步**：时间戳字段加在 `io_struct.py` 的消息对象上，pickle 反序列化要求两端一致；router 节点代码过旧时 `sched_send` 等字段会丢失（分析脚本会提示 timestamps absent）
- 只统计非 idle batch（`bs=0` 心跳不写记录、不打 `sched_send`）

## 分析脚本

```bash
cd scripts/cpu_kunpeng/analysis

# 全局统计：默认取 /tmp 下最新 timeline 文件
python analyze_tokenizer_timeline.py [jsonl_file]

# 单请求追踪：逐轮 TPOT 分解；不带 RID 随机选一个可追踪请求
python analyze_tokenizer_timeline.py --trace [RID] [--limit N]
```

### 全局模式输出

- per-request 统计（`type=req` 记录的 ttft_ms / tpot_server_ms 等字段的 P50/P95/P99）
- 各段延迟总览 + early/middle/late 三阶段 P50（观察随时间的劣化）
- per-DP 的 `dtok->tok`（含 dtok_proc）统计
- TPOT 估计：per-DP 相邻 `tok_send` 间隔（按 bs 加权），附 `sched_P50`（scheduler 周期）与 `added_P50`（管线额外增加的间隔）

### trace 模式输出

- 逐轮明细表：`sched_send`（wall-clock，可与 server log 对时）+ 五个分量增量 + TPOT
- 分量统计：各分量 avg/P50/P95 及占 TPOT 比例
- identity check：`|Σ分量 − TPOT|` 最大值，应约为 0（验证分解完整）

### 结果落盘

```
outputs/<YYYYMMDD_HHMMSS>/global/global_stats.xlsx   # 5 张表各一个 sheet；无 openpyxl 时回退为 5 个 CSV
outputs/<YYYYMMDD_HHMMSS>/trace/{trace_detail.csv, trace_stats.csv}
```

## 数据格式

batch 记录（每 batch 一行）：

```json
{"type": "batch", "ts": "15:21:17.520", "dp": 0, "bs": 1,
 "rids": ["..."], "sched_send": 1786951277.510, "dtok_recv": 1786951277.517,
 "dtok_send": 1786951277.518, "tok_recv": 1786951277.520,
 "tok_send": 1786951277.521}
```

request 记录（请求完成时一行，时间均在 tok worker 进程内，`perf_counter` 单调时钟）：

| 字段 | 含义 |
|---|---|
| `ttft_ms` | 请求进入 → 首个含该请求的 batch 到达 worker（服务端 TTFT） |
| `tokenize_ms` / `dispatch_ms` | 入口分词耗时 / 分词完成到发往 scheduler |
| `tpot_server_ms` | worker 接收侧平均 token 到达间隔 ≈ aisbench TPOT 的下界 |
| `first_token_lag_ms` | 首 batch 到达 worker → 最后一个 chunk 发出 |
| `tok_proc_avg_ms` | 每轮 batch 在 worker 处理函数内的平均耗时 |
| `e2e_ms` | 服务端端到端 |

历史版本字段名 `worker_recv` / `worker_send` / `worker_proc_avg_ms`（worker→tok 改名前）分析脚本自动兼容。

## 判读方法

- `tpot_server_ms` ≈ aisbench TPOT → 损耗在 serving pipeline 内部；再看 trace 分量定位到段
- `tpot_server_ms` 明显更低 → 损耗在 worker 处理（`tok_proc_avg_ms`）或 HTTP 返回侧
- 全局模式 `dtok->tok` late 阶段暴涨 → tok worker 消费速率跟不上（单 worker 排队积压）
- `d_queue` 持续为正 → 队列持续积压；单轮尖峰（P95 高 avg 低）→ 间歇性卡顿
- `d_sched` 占比高 → 瓶颈在计算侧（scheduler 每轮迭代本身）

## 关键代码位置

router 侧（detok/tok 打点与 JSONL 写入）通过 mixin 挂载（对齐 `hardware_backend/mlx/scheduler_mixin.py` 模式）：通用 manager 类里只保留 no-op hook 调用，实际实现放在 `cpu_kunpeng/managers/` 下，仅当 `is_cpu_920f()` 为真时条件 import 生效；非鲲鹏平台走 no-op stub，行为零变化。

- 环境变量注册：`python/sglang/srt/environ.py`（`SGLANG_TOKENIZER_TIMELINE_LOG` / `SGLANG_TOKENIZER_TIMELINE_PATH`）
- 时间戳字段：`python/sglang/srt/managers/io_struct.py`（`BatchTokenIDOutput` / `BatchStrOutput`）
- scheduler 打点：`python/sglang/srt/managers/scheduler_output_processor_mixin.py`
- detokenizer 打点实现：`python/sglang/srt/hardware_backend/cpu_kunpeng/managers/detokenizer_mixin.py`（`_timeline_stamp_recv` / `_timeline_stamp_send`；挂载点在 `detokenizer_manager.py` 的 `event_loop`）
- tok worker 打点与 JSONL 写入实现：`python/sglang/srt/hardware_backend/cpu_kunpeng/managers/tokenizer_mixin.py`（`_timeline_batch_enter` / `_on_chunk` / `_on_finish` / `_batch_exit` 与 `_dump_timeline_record` / `_dump_timeline_request_record`；挂载点在 `tokenizer_manager.py` 的 `_handle_batch_output`）
- 分析脚本：`scripts/cpu_kunpeng/analysis/analyze_tokenizer_timeline.py`

mixin 实现说明：mixin 无法覆写类体内已定义的方法（类体优先于基类查找），因此通用文件中的挂载点调用的是新命名的 hook 方法，由条件 import 决定走鲲鹏实现还是 no-op stub。部署时需确保 pyinstall 打包包含 `hardware_backend/cpu_kunpeng/managers/` 目录（与 attention 等目录同等对待）。
