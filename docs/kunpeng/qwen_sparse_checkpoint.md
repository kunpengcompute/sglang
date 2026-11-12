# Qwen 稀疏 Mamba Checkpoint（extra_buffer）

## 功能

Qwen3.8 的 GDN 线性注意力层不能像全注意力 KV 那样按任意前缀复用：默认只有序列末尾或 decode 的 256-token 边界存有一份状态。

开启本特性后，prefill 会每隔 `stride` 个 token 顺带保存一份 GDN 的 Conv + SSM 状态（checkpoint）。Radix Cache 因此可以命中长 prompt 内部的这些位置：

- 前缀命中粒度从 256 token 降到 64 token；
- 相同前缀的后续请求少算 prefill；
- checkpoint 在同一次 GDN 计算中产出，不额外重算。

适用平台：Kunpeng 920F + Qwen3.8，且 `SGLANG_KUNPENG_QWEN=1`。

## 最佳配置

前提：`sgl-kernel` 已包含 `qwen36_gdn_prefill_parallel_with_multi_track_workspace` 符号；多 NUMA 副本需要同步更新，否则启动会直接报 `ValueError`。

```bash
# 服务参数
--page-size 64
--chunked-prefill-size 2048
--max-running-requests 16
--mamba-scheduler-strategy extra_buffer
--mamba-checkpoint-stride-pages 1      # stride = 1 × 64 = 64 token
--mamba-track-interval 256
--max-mamba-cache-size 128             # 必须显式设置，见下文
```

```bash
# 环境变量
export SGLANG_USE_CPU_920F=1
export SGLANG_KUNPENG_QWEN=1
export SGLANG_ENABLE_HBW_POOL=1
export SGLANG_KUNPENG_MAX_SEQ_NUM=2048
export SGLANG_KUNPENG_COLLECTIVE_MAX_TOKENS=4096
export SGLANG_KUNPENG_EXTEND_POWER_2_PADDING=0

# 需要 prefill 图回放时再加
export SGLANG_ENABLE_GRAPH_CAPTURE=1
export SGLANG_KUNPENG_ENABLE_PREFILL_GRAPH=1
export SGLANG_KUNPENG_QWEN_PREFILL_BACKEND=kutacc
export SGLANG_KUNPENG_QWEN_SHARED_PREFILL_KV=1
```

## 为什么这么配置

| 配置 | 原因 |
| --- | --- |
| `--mamba-checkpoint-stride-pages 1` | `page_size=64` 时 stride=64，命中粒度最细。stride 必须是 `max(64, page_size)` 的整数倍；128 个 slot 下 checkpoint 占用较快，若出现频繁淘汰或 `Not enough space for mamba cache`，改成 4（stride=256）可把槽位消耗降到 1/4 |
| `--chunked-prefill-size 2048` | 图捕获最多 64 行，`行数 = batch tokens / stride + batch_size`。2048/64 + 16 = 48 ≤ 64，stride=64 时仍能进图；若 chunk=4096，则 64 + 16 = 80 超限，只能回退 eager |
| `--mamba-track-interval 256` | decode 侧的粗粒度兜底。必须是 page_size 的倍数，MTP 下还要 ≥ draft tokens；256 能被 64 整除，checkpoint 网格对齐 |
| `--page-size 64` | 与 `FLA_CHUNK_SIZE=64` 一致，checkpoint 只能落在 64-token 边界 |
| `MAX_SEQ_NUM=2048` | 让所有 prefill batch 落进同一个 SHM small 缓存。超过该值会走 large 分支逐级分配，图捕获占满池后容易出现 `Not enough shared memory ... remain 0`；设为 chunk 大小即可，不要盲目调大 |
| `COLLECTIVE_MAX_TOKENS=4096` | 单批 collective 行数上限，与图捕获的 4096 token 上限对齐 |
| `EXTEND_POWER_2_PADDING=0` | 上游 padding 与 MTP 同时开启时 target/draft 行数不一致，保持关闭 |
| `--max-mamba-cache-size 128` | 必须显式设置。不设时按 `--mamba-full-memory-ratio 0.9` 自动计算，得到的 slot 数会超出 HBW 池最大连续空闲块，启动阶段直接报 `HBW pool exhausted`。128 是当前验证值；调大前先确认 HBW 池有空闲块，调小会减少可缓存的 checkpoint 数 |

## 怎么确认生效

```bash
grep -i "\[graph\]" <log> | tail -20
grep -i "cached-token" <log> | tail -20
```

- `[graph] captured mode=EXTEND total_tokens=N`：该形状已捕获，后续直接回放；
- `[graph] decline ... reason=checkpoint_plan_unavailable`：该形状行数超限或窗口越界，回退 eager，结果仍正确；
- `#cached-token: N`：实际命中长度。841 token 的 prefill、stride 64 时最大对齐点是 832。

## 已知限制

- 短 batch 没有 64 对齐点，不产生 checkpoint，也不新写 Radix Cache；
- stride 越小，checkpoint slot 越多，容量不足会频繁淘汰，命中率下降；
- `no_buffer` 模式不支持本特性；
- padding 与 MTP 不能同时开启。
