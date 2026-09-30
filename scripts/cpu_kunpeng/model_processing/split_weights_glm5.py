"""GLM-5 (GlmMoeDsaForCausalLM) 权重切分脚本 — kunpeng CPU 部署离线预处理.

输出布局与 split_weights.py (DeepSeek) 完全一致, 供 KunpengStateLoader 加载:
  非 MoE:  {output_dir}/tp{atp}/model-rank-{r}-part-0.safetensors
           MTP 层单独输出到 {output_dir}_mtp/tp{atp}/ (key 重映射为 draft 模型参数名)
  MoE:     {output_dir}/experts/layer_{L}/expert_{E}.safetensors
           (w13_weight = gateup_proj 原样, w2_weight = down_proj, 含 *_scale)

与 DeepSeek 版的差异:
  1. 输入为已量化的 GLM-5 导出, 张量命名自动识别两种约定:
       <module>       + <module>.scale        (GLM-5 导出约定, 即权重清单字面格式)
       <module>.weight + <module>.weight_scale (DeepSeek 管线约定)
     scale 可缺省 (kv_b_proj / indexer.wk / weights_proj / lm_head / embed 等为 bf16).
   2. gate/up 两种导出约定均支持, 自动识别 (WeightResolver.resolve_gateup):
        gateup_proj / gate_up_proj          (融合导出, 布局 [gate; up])
        gate_proj + up_proj                 (分开导出, 脚本融合为 [gate; up])
      TP 切分时对前后两半分别切再拼接, 输出重命名为 gate_up_proj
      (与 sglang 模型参数名 MergedColumnParallelLinear 对齐). 专家的 w13
      同理 (w13_weight = [gate; up] 融合结果, 原样不切分).
  3. indexer.* (wq_b/wk/weights_proj/k_norm) 均为 ReplicatedLinear
     (见 nsa_indexer.py), 不切分, 原样复制到每个 rank.
  4. q_a_proj + kv_a_proj_with_mqa 融合为 fused_qkv_a_proj_with_mqa 并按
      socket_tp 切分 dim0 (920F 上该模块为 ColumnParallelLinear(tp_size=
      socket_tp_size), 见 deepseek_v2.py DeepseekV2AttentionMLA).

输入分片发现 (build_weight_map):
  优先使用 model.safetensors.index.json; 不存在时使用目录下任意
  *.safetensors.index.json; 都没有时直接扫描 *.safetensors 分片
  (如 quant_model_weights-00001-of-00008.safetensors) 解析文件头
  构建 key -> 文件名映射, 无需索引文件.
"""

import argparse
import json
import os
import re
import shutil
import struct
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


# ============================================================================
# 公共工具函数
# ============================================================================


def load_shards_into_memory(keys, weight_map, model_dir="./", use_tqdm=True, num_readers=4):
    if isinstance(keys, str):
        keys = [keys]
    needed_files = set(weight_map[k] for k in keys if k in weight_map)
    combined_data = {}
    if not needed_files:
        return combined_data

    def load_one(f):
        # 按需取 key: 避免整个分片 (含无关的专家权重) 全量物化
        return _load_shard_tensors(os.path.join(model_dir, f), keys)

    # 并行预取: NFS/RAID 等聚合带宽远高于单流, 顺序读会卡在单流上限
    with ThreadPoolExecutor(max_workers=min(num_readers, len(needed_files))) as pool:
        for d in tqdm(
            pool.map(load_one, needed_files),
            total=len(needed_files),
            desc="加载分片文件到内存",
            disable=not use_tqdm,
        ):
            combined_data.update(d)
    return combined_data


def _natural_key(name: str):
    """文件名自然排序 key (数字段按数值比较)."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def _safetensors_header(path: str) -> Dict[str, Dict]:
    """解析 safetensors 文件头: {key: {dtype, shape, begin, end}} (不读张量数据).

    begin/end 为张量数据在文件中的绝对字节偏移 (data_offsets 是相对数据区
    起始的偏移, 需加上 8 + header_len).
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    data_base = 8 + n
    return {
        k: {
            "dtype": v["dtype"],
            "shape": v["shape"],
            "begin": data_base + v["data_offsets"][0],
            "end": data_base + v["data_offsets"][1],
        }
        for k, v in header.items()
        if k != "__metadata__"
    }


def _safetensors_meta(path: str) -> Dict[str, int]:
    """解析 safetensors 文件头: {tensor key: 数据字节数} (不读张量数据)."""
    return {k: m["end"] - m["begin"] for k, m in _safetensors_header(path).items()}


def _safetensors_keys(path: str) -> List[str]:
    return list(_safetensors_meta(path).keys())


def _load_shard_tensors(path: str, keys):
    """按需加载 shard 中指定 key 的张量.

    优先用 safetensors.safe_open 逐张量取 (不物化分片里其余张量);
    无 safe_open (如 mock 环境) 时退回 load_file 全量加载后筛选.
    """
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            have = set(f.keys())
            return {k: f.get_tensor(k) for k in keys if k in have}
    except ImportError:
        pass
    shard_data = load_file(path)
    out = {k: shard_data[k] for k in keys if k in shard_data}
    del shard_data
    return out


# 字节级直拷: MoE 专家输出 = 源张量字节的原样搬运 (w13 = gate 字节 + up 字节),
# 无需 torch 张量化. sendfile 内核态拷贝 (读写合一、释放 GIL、可真并行),
# 不可用时退回 read/write (同样释放 GIL).
_COPY_CHUNK = 32 * 1024 * 1024


def _copy_range(dst, src, begin: int, n: int):
    sent = 0
    use_sendfile = hasattr(os, "sendfile")
    while sent < n:
        if use_sendfile:
            try:
                sent += os.sendfile(
                    dst.fileno(), src.fileno(), begin + sent,
                    min(_COPY_CHUNK, n - sent),
                )
                continue
            except OSError:
                use_sendfile = False  # 该文件系统不支持 sendfile -> read/write
        src.seek(begin + sent)
        buf = src.read(min(_COPY_CHUNK, n - sent))
        if not buf:
            raise IOError(f"short read from {src.name} @ {begin + sent}")
        dst.write(buf)
        sent += len(buf)


def _write_safetensors_bytes(out_path: str, entries) -> int:
    """字节级构造 safetensors 文件, 返回数据字节数.

    entries: [(key, dtype, shape, ranges)]; ranges: [(src_path, begin, nbytes)],
    顺序拼接即为该张量的数据 (torch.cat(dim=0) 的行主序字节等价).
    """
    header = {}
    off = 0
    for key, dtype, shape, ranges in entries:
        nbytes = sum(n for _, _, n in ranges)
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [off, off + nbytes],
        }
        off += nbytes
    hj = json.dumps(header, separators=(",", ":"))
    hj += " " * ((-len(hj)) % 8)  # 数据区 8 字节对齐, 与官方序列化一致
    hb = hj.encode("utf-8")
    # 必须无缓冲: 缓冲文件对象 + os.sendfile 混用会把缓冲里的头部
    # (dst.write) 留到 close 时才 flush, 而 sendfile 已直写 fd 把数据
    # 放在文件开头 -> 输出变成 [数据][头部] 而非 [头部][数据]。
    with open(out_path, "wb", buffering=0) as dst:
        dst.write(struct.pack("<Q", len(hb)))
        dst.write(hb)
        for _, _, _, ranges in entries:
            for src_path, begin, n in ranges:
                with open(src_path, "rb") as src:
                    _copy_range(dst, src, begin, n)
    return off


class _ByteLevelUnsupported(Exception):
    pass


def _build_byte_level_plans(model_dir, weight_map, plans):
    """为全部专家构建字节级写出计划.

    任一源 key 的 safetensors 头不可得 (如 mock 环境) 时返回 None,
    由调用方回退到张量级路径.
    """
    headers = {}

    def meta_of(key):
        f = weight_map.get(key)
        if f is None:
            return None
        if f not in headers:
            try:
                headers[f] = _safetensors_header(os.path.join(model_dir, f))
            except Exception:
                headers[f] = None
        h = headers[f]
        if h is None or key not in h:
            return None
        return f, h[key]

    def direct(out_key, ck):
        if ck is None:
            return None
        m = meta_of(ck)
        if m is None:
            raise _ByteLevelUnsupported(ck)
        f, meta = m
        return (
            out_key, meta["dtype"], meta["shape"],
            [(os.path.join(model_dir, f), meta["begin"], meta["end"] - meta["begin"])],
        )

    def concat(out_key, ck1, ck2, base):
        m1, m2 = meta_of(ck1), meta_of(ck2)
        if m1 is None or m2 is None:
            raise _ByteLevelUnsupported(f"{base} gate/up")
        f1, d1 = m1
        f2, d2 = m2
        if d1["dtype"] != d2["dtype"] or list(d1["shape"][1:]) != list(d2["shape"][1:]):
            raise ValueError(f"{base}: gate/up dtype/shape 不匹配, 无法按字节拼接")
        shape = [d1["shape"][0] + d2["shape"][0]] + list(d1["shape"][1:])
        ranges = [
            (os.path.join(model_dir, f1), d1["begin"], d1["end"] - d1["begin"]),
            (os.path.join(model_dir, f2), d2["begin"], d2["end"] - d2["begin"]),
        ]
        return out_key, d1["dtype"], shape, ranges

    byte_plans = {}
    for dest, plan in plans.items():
        layer, _ = dest
        prefix = plan["prefix"]
        k = f"model.layers.{layer}.mlp.experts"
        g = plan["gateup"]
        try:
            entries = []
            if g[0] == "fused":
                wk, sk = g[1]
                e = direct(f"{k}.w13_weight", wk)
                if e is not None:
                    entries.append(e)
                e = direct(f"{k}.w13_weight_scale", sk)
                if e is not None:
                    entries.append(e)
            else:
                (wg, sg), (wu, su) = g[1], g[2]
                entries.append(concat(f"{k}.w13_weight", wg, wu, prefix))
                if sg is not None or su is not None:
                    if sg is None or su is None:
                        # scale 不对称: 交给张量路径报详细错误
                        raise _ByteLevelUnsupported(f"{prefix} scale 不对称")
                    entries.append(concat(f"{k}.w13_weight_scale", sg, su, prefix))
            e = direct(f"{k}.w2_weight", plan["w_down"])
            if e is None:
                raise _ByteLevelUnsupported(plan["w_down"])
            entries.append(e)
            e = direct(f"{k}.w2_weight_scale", plan["s_down"])
            if e is not None:
                entries.append(e)
        except _ByteLevelUnsupported:
            return None
        byte_plans[dest] = entries
    return byte_plans


def build_weight_map(model_dir: str) -> Dict[str, str]:
    """构建 key -> 分片文件名 映射.

    查找优先级:
      1. model.safetensors.index.json (标准 HF 索引)
      2. 目录下任意 *.safetensors.index.json
         (如 quant_model_weights.safetensors.index.json)
      3. 无索引: 直接扫描 *.safetensors 分片
         (如 quant_model_weights-00001-of-00008.safetensors),
         逐个解析文件头收集 key; 非标准格式时退回 load_file.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        for f in sorted(os.listdir(model_dir)):
            if f.endswith(".safetensors.index.json"):
                index_path = os.path.join(model_dir, f)
                break
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        print(f"使用索引文件: {os.path.basename(index_path)}")
        return weight_map

    def _nat_key(name: str):
        return _natural_key(name)

    shards = sorted(
        (f for f in os.listdir(model_dir) if f.endswith(".safetensors")), key=_nat_key
    )
    if not shards:
        raise FileNotFoundError(
            f"目录 {model_dir} 下既无 *.safetensors.index.json 索引, "
            f"也无 *.safetensors 分片文件"
        )

    weight_map = {}
    for f in shards:
        path = os.path.join(model_dir, f)
        try:
            keys = _safetensors_keys(path)
        except Exception:
            # 非 safetensors 布局或解析失败: 退回完整加载取 key
            keys = list(load_file(path).keys())
        for k in keys:
            weight_map[k] = f
    print(f"未找到索引文件, 已扫描 {len(shards)} 个分片构建 weight_map")
    return weight_map


def _extract_layer_idx(key: str) -> int:
    """从 key 中提取层索引, 如 'model.layers.5.self_attn...' -> 5"""
    prefix = "model.layers."
    if not key.startswith(prefix):
        return -1
    rest = key[len(prefix) :]
    idx_str = rest.split(".")[0]
    try:
        return int(idx_str)
    except ValueError:
        return -1


def split_tensor_by_rank(
    tensor: torch.Tensor,
    split_mode: str,
    dim: int,
    rank: int,
    attention_tp_size: int,
    socket_tp_size: int = 8,
) -> torch.Tensor:
    if split_mode == "none":
        return tensor.clone()

    if split_mode == "attention_tp":
        tp_size, tp_rank = attention_tp_size, rank % attention_tp_size
    elif split_mode == "socket_tp":
        tp_size, tp_rank = socket_tp_size, rank % socket_tp_size
    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r}")

    total = tensor.shape[dim]
    if total % tp_size != 0:
        raise ValueError(
            f"tensor.dim({dim})={total} 不能被 tp_size={tp_size} 整除 "
            f"(split_mode={split_mode})"
        )
    chunk = total // tp_size
    start, end = tp_rank * chunk, (tp_rank + 1) * chunk

    if dim == 0:
        return tensor[start:end].clone()
    if dim == 1:
        return tensor[:, start:end].clone()
    raise ValueError(f"Unsupported dim: {dim!r}")


def split_and_concat_by_rank(
    tensors,
    split_mode: str,
    dim: int,
    rank: int,
    attention_tp_size: int,
    socket_tp_size: int = 8,
) -> torch.Tensor:
    chunks = [
        split_tensor_by_rank(
            t, split_mode, dim, rank, attention_tp_size, socket_tp_size
        )
        for t in tensors
    ]
    return torch.cat(chunks, dim=dim)


def _rename_mtp_key(key: str) -> str:
    """HF key -> draft 模型 state_dict key for MTP layers.

    与 DeepseekModelNextN (deepseek_nextn.py) 的参数名对齐:
      model.layers.N.enorm/hnorm/eh_proj/shared_head.* -> model.*
      model.layers.N.self_attn/mlp/input_layernorm/post_attention_layernorm
        -> model.decoder.*
    """
    prefix = "model.layers."
    if not key.startswith(prefix):
        return key
    rest = key[len(prefix) :]
    parts = rest.split(".", 1)
    if len(parts) < 2:
        return key
    _, suffix = parts

    for pattern in ("enorm", "hnorm", "eh_proj", "shared_head"):
        if suffix.startswith(pattern):
            return f"model.{suffix}"

    for pattern in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm"):
        if suffix.startswith(pattern):
            return f"model.decoder.{suffix}"

    return key


# ============================================================================
# GLM-5 输入命名解析
# ============================================================================


def to_module_base(key: str) -> str:
    """把 checkpoint 张量 key 归一化为模块路径 base.

    支持的输入约定 (自动识别):
      X.weight + X.weight_scale   (DeepSeek 管线约定)
      X.weight + X.weight.scale
      X          + X.scale        (GLM-5 导出约定, 按权重清单字面格式)
    """
    k = key
    if k.endswith("_scale"):
        k = k[: -len("_scale")]
    if k.endswith(".scale"):
        k = k[: -len(".scale")]
    if k.endswith(".weight"):
        k = k[: -len(".weight")]
    return k


class WeightResolver:
    """按模块 base 解析 (weight_key, scale_key); scale 可缺省 (bf16 权重)."""

    def __init__(self, weight_map: Dict[str, str]):
        self.weight_map = weight_map

    def resolve(self, base: str) -> Tuple[Optional[str], Optional[str]]:
        weight_key = None
        for cand in (base + ".weight", base):
            if cand in self.weight_map:
                weight_key = cand
                break
        if weight_key is None:
            return None, None

        scale_key = None
        for cand in (weight_key + "_scale", weight_key + ".scale", base + ".scale"):
            if cand in self.weight_map and cand != weight_key:
                scale_key = cand
                break
        return weight_key, scale_key

    def load_plan(self, bases) -> List[str]:
        """返回加载 bases 全部 (weight + scale) 所需的 checkpoint key 列表."""
        keys = []
        for b in bases:
            w, s = self.resolve(b)
            if w is None:
                raise KeyError(f"在 index 中找不到模块权重: {b}")
            keys.append(w)
            if s is not None:
                keys.append(s)
        return keys

    def resolve_gateup(self, prefix: str):
        """解析某 MLP 模块前缀下的 gate/up 权重来源.

        自动识别两种导出约定:
          融合: prefix.gateup_proj / prefix.gate_up_proj
                -> ("fused", (w_key, s_key))
          分开: prefix.gate_proj + prefix.up_proj (脚本融合为 [gate; up])
                -> ("split", (gw_key, gs_key), (uw_key, us_key))
          都找不到 -> (None,) (调用方负责带上下文报错)
        """
        for name in ("gateup_proj", "gate_up_proj"):
            w, s = self.resolve(f"{prefix}.{name}")
            if w is not None:
                return "fused", (w, s)
        gate = self.resolve(f"{prefix}.gate_proj")
        up = self.resolve(f"{prefix}.up_proj")
        if gate[0] is not None and up[0] is not None:
            return "split", gate, up
        return (None,)

    def gateup_error(self, prefix: str) -> KeyError:
        """构造带诊断信息 (该前缀下实际存在的 key) 的解析失败异常."""
        have = sorted(k for k in self.weight_map if k.startswith(prefix + "."))
        return KeyError(
            f"无法解析 {prefix} 的 gate/up 权重: 支持 {prefix}.gateup_proj "
            f"(融合导出) 或 {prefix}.gate_proj + {prefix}.up_proj "
            f"(分开导出, 脚本自动融合为 [gate; up]). "
            f"该前缀下实际存在的 key: {have}"
        )


def fuse_gate_up(gate_w, gate_s, up_w, up_s, base: str):
    """融合分开导出的 gate/up 为 [gate; up] 布局 (含 per-row scale)."""
    if gate_w.shape[0] != up_w.shape[0]:
        raise ValueError(
            f"{base}: gate 行数 {gate_w.shape[0]} != up 行数 {up_w.shape[0]}"
        )
    if (gate_s is None) != (up_s is None):
        raise KeyError(
            f"{base}: gate/up 的 scale 必须同时存在或同时缺失 "
            f"(gate_scale={'有' if gate_s is not None else '无'}, "
            f"up_scale={'有' if up_s is not None else '无'})"
        )
    w = torch.cat([gate_w, up_w], dim=0)
    s = torch.cat([gate_s, up_s], dim=0) if gate_s is not None else None
    return w, s


# ============================================================================
# MoE 专家权重切分
# ============================================================================


def split_moe_experts(
    model_dir: str,
    output_dir: str,
    num_hidden_layers: Optional[int] = None,
    mtp_output_dir: Optional[str] = None,
    num_writers: int = 8,
    num_readers: int = 4,
    prefetch_bytes: int = 16 * 1024**3,
    layer_range: Optional[Tuple[int, int]] = None,
):
    """单遍分片扫描切分 MoE 专家 (读/写全并行流水线).

    优先走字节级直拷路径 (源 key 的 safetensors 头全部可解析时):
    专家输出 = 源张量字节原样搬运 (w13 = gate + up 字节拼接), sendfile
    内核态拷贝, 不经 torch/safetensors API, 释放 GIL 可真并行.
    否则回退到张量级预取流水线.

    layer_range: (start, end) 左闭右开, 只处理该范围内的层 (原始层号,
    含 MTP 层), 供多机分工 (各机写各自层, 输出目录不冲突).

      1. 先为每个 (layer, expert) 构建加载计划 (gateup/down 的 key 清单)
      2. 按 key 所属分片分组, reader 线程池预取后续分片 (每个 shard 只读
         一次, 且只取需要的 key); 预取窗口按字节数受限 (默认 16GB)
      3. 某专家的 key 全部到齐 -> 立即构建 state_dict 提交后台线程写出
    """
    t_start = time.time()
    weight_map = build_weight_map(model_dir)
    resolver = WeightResolver(weight_map)

    os.makedirs(output_dir, exist_ok=True)
    if mtp_output_dir is not None:
        os.makedirs(mtp_output_dir, exist_ok=True)

    # 1) 逐专家构建加载计划
    layer_to_experts = defaultdict(set)
    for key in weight_map.keys():
        if ".mlp.experts." not in key:
            continue
        base = to_module_base(key)
        parts = base.split(".")
        layer_to_experts[int(parts[2])].add(int(parts[parts.index("experts") + 1]))

    if layer_range is not None:
        lo, hi = layer_range
        keep = sorted(l for l in layer_to_experts if lo <= l < hi)
        print(f"层范围过滤: [{lo}, {hi}), 保留 {len(keep)} 层 (原始层号)")
        layer_to_experts = defaultdict(
            set, {l: layer_to_experts[l] for l in keep}
        )

    plans = {}  # (layer, expert) -> plan
    owner = {}  # checkpoint key -> (layer, expert)
    for layer in sorted(layer_to_experts.keys()):
        for expert in sorted(layer_to_experts[layer]):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            gateup = resolver.resolve_gateup(prefix)
            if gateup[0] is None:
                raise resolver.gateup_error(prefix)
            w_down_key, s_down_key = resolver.resolve(f"{prefix}.down_proj")
            if w_down_key is None:
                raise KeyError(f"缺少 {prefix}.down_proj")

            plan_keys = []
            for wk, sk in (
                (gateup[1],) if gateup[0] == "fused" else (gateup[1], gateup[2])
            ):
                plan_keys.append(wk)
                if sk is not None:
                    plan_keys.append(sk)
            plan_keys.append(w_down_key)
            if s_down_key is not None:
                plan_keys.append(s_down_key)

            if num_hidden_layers is not None and layer >= num_hidden_layers:
                layer_dir = os.path.join(
                    mtp_output_dir, f"layer_{layer - num_hidden_layers}"
                )
            else:
                layer_dir = os.path.join(output_dir, f"layer_{layer}")

            dest = (layer, expert)
            plans[dest] = {
                "prefix": prefix,
                "gateup": gateup,
                "w_down": w_down_key,
                "s_down": s_down_key,
                "keys": plan_keys,
                "out": os.path.join(layer_dir, f"expert_{expert}.safetensors"),
            }
            for k in plan_keys:
                if owner.get(k, dest) != dest:
                    raise KeyError(f"key {k} 被多个专家引用")
                owner[k] = dest

    # 2) 按 key 所属分片分组, 每个 shard 只加载一次
    shard_keys = defaultdict(list)
    for k, dest in owner.items():
        shard_keys[weight_map[k]].append(k)
    shard_order = sorted(shard_keys.keys(), key=_natural_key)

    for plan in plans.values():
        os.makedirs(os.path.dirname(plan["out"]), exist_ok=True)

    # ---- 字节级直拷快速路径 ----
    byte_plans = _build_byte_level_plans(model_dir, weight_map, plans)
    if byte_plans is not None:
        t0 = time.time()
        total = sum(
            n for entries in byte_plans.values() for _, _, _, rs in entries
            for _, _, n in rs
        )
        print(
            f"MoE 专家总数: {len(plans)}, 字节级直拷路径 (sendfile, "
            f"{num_writers} 写线程), 数据量 {total / 1024**3:.1f} GiB"
        )
        max_inflight = num_writers * 4
        with ThreadPoolExecutor(max_workers=num_writers) as pool:
            futures = []
            for dest, entries in tqdm(
                byte_plans.items(), desc="写出专家文件"
            ):
                futures.append(
                    pool.submit(_write_safetensors_bytes, plans[dest]["out"], entries)
                )
                # 限制在途任务数, 控制并发打开的文件描述符
                while len(futures) >= max_inflight:
                    futures.pop(0).result()
            for fu in futures:
                fu.result()
        elapsed = time.time() - t0
        print(
            f"MoE 切分完成: 共写出 {len(plans)} 个专家文件, 耗时 "
            f"{elapsed / 60:.1f} 分钟, 拷贝 {total / 1024**3:.1f} GiB "
            f"({total / 1024**2 / max(elapsed, 1e-9):.0f} MB/s)"
        )
        return

    # ---- 张量级回退路径 ----

    # 预取窗口按字节预算控制: 从文件头读取各 key 的数据字节数
    shard_size = {}
    for shard in shard_order:
        try:
            meta = _safetensors_meta(os.path.join(model_dir, shard))
            shard_size[shard] = sum(meta.get(k, 0) for k in shard_keys[shard])
        except Exception:
            shard_size[shard] = 0  # 未知大小: 仅按窗口内分片数受限

    total_read_bytes = sum(shard_size.values())
    print(
        f"MoE 专家总数: {len(plans)}, 涉及分片: {len(shard_order)} "
        f"(单遍加载, 预取 {num_readers} 线程 / 写出 {num_writers} 线程)"
    )
    if total_read_bytes:
        print(f"待读取专家数据量: {total_read_bytes / 1024**3:.1f} GiB")

    def build_state_dict(dest, tensors):
        layer, _ = dest
        plan = plans[dest]
        if plan["gateup"][0] == "fused":
            wk, sk = plan["gateup"][1]
            w13 = tensors[wk]
            s13 = tensors[sk] if sk is not None else None
        else:
            _, (wg, sg), (wu, su) = plan["gateup"]
            w13, s13 = fuse_gate_up(
                tensors[wg],
                tensors[sg] if sg is not None else None,
                tensors[wu],
                tensors[su] if su is not None else None,
                plan["prefix"],
            )
        sd = {
            f"model.layers.{layer}.mlp.experts.w13_weight": w13,
            f"model.layers.{layer}.mlp.experts.w2_weight": tensors[plan["w_down"]],
        }
        if s13 is not None:
            sd[f"model.layers.{layer}.mlp.experts.w13_weight_scale"] = s13
        if plan["s_down"] is not None:
            sd[
                f"model.layers.{layer}.mlp.experts.w2_weight_scale"
            ] = tensors[plan["s_down"]]
        return sd

    # 3) 预取流水线: reader 线程池提前加载后续分片, 主线程按序消费并派发写出
    pending = {dest: {} for dest in plans}
    remaining = {dest: len(plan["keys"]) for dest, plan in plans.items()}
    max_inflight = num_writers * 4

    shard_iter = iter(shard_order)
    inflight = deque()  # (shard, 预估字节, future)
    inflight_bytes = 0

    read_bytes_done = 0
    write_bytes_total = 0

    def _tensor_bytes(t) -> int:
        try:
            return t.numel() * t.element_size()
        except Exception:
            return 0

    with ThreadPoolExecutor(max_workers=num_readers) as read_pool, \
            ThreadPoolExecutor(max_workers=num_writers) as write_pool:
        futures = []

        def submit_write(dest, sd):
            futures.append(write_pool.submit(save_file, sd, plans[dest]["out"]))

        def fill():
            nonlocal inflight_bytes
            while inflight_bytes < prefetch_bytes and len(inflight) < 16:
                try:
                    shard = next(shard_iter)
                except StopIteration:
                    return
                sz = shard_size.get(shard, 0)
                fut = read_pool.submit(
                    _load_shard_tensors,
                    os.path.join(model_dir, shard),
                    shard_keys[shard],
                )
                inflight.append((shard, sz, fut))
                inflight_bytes += sz

        fill()
        for _ in tqdm(range(len(shard_order)), desc="处理 MoE 分片"):
            shard, sz, fut = inflight.popleft()
            inflight_bytes -= sz

            tensors = fut.result()
            read_bytes_done += sz
            for k in shard_keys[shard]:
                dest = owner[k]
                pending[dest][k] = tensors[k]
                remaining[dest] -= 1
            del tensors

            ready = [d for d, r in remaining.items() if r == 0]
            for dest in ready:
                sd = build_state_dict(dest, pending.pop(dest))
                del remaining[dest]
                write_bytes_total += sum(_tensor_bytes(t) for t in sd.values())
                submit_write(dest, sd)
                # 限制在途任务数, 控制未写出的专家张量驻留内存
                while len(futures) >= max_inflight:
                    futures.pop(0).result()

            fill()

        for fu in futures:
            fu.result()

    if remaining:
        sample = sorted(remaining.keys())[:5]
        raise RuntimeError(f"有 {len(remaining)} 个专家的 key 未集齐: {sample}")

    elapsed = time.time() - t_start
    stats = f"MoE 切分完成: 共写出 {len(plans)} 个专家文件, 耗时 {elapsed/60:.1f} 分钟"
    if read_bytes_done:
        stats += (
            f", 读取 {read_bytes_done/1024**3:.1f} GiB"
            f" ({read_bytes_done/1024**2/max(elapsed,1e-9):.0f} MB/s)"
        )
    if write_bytes_total:
        stats += f", 写出 {write_bytes_total/1024**3:.1f} GiB"
    print(stats)


# ============================================================================
# Non-MoE 权重切分
# ============================================================================


def split_non_moe_weights(
    model_dir: str,
    output_dir: str,
    attention_tp_size: int = 16,
    socket_tp_size: int = 8,
    ranks=None,
    num_hidden_layers=None,
    mtp_output_dir: Optional[str] = None,
    num_readers: int = 4,
):
    if ranks is None:
        ranks = list(range(attention_tp_size))
    weight_map = build_weight_map(model_dir)
    resolver = WeightResolver(weight_map)

    os.makedirs(output_dir, exist_ok=True)
    if mtp_output_dir is not None:
        os.makedirs(mtp_output_dir, exist_ok=True)

    # 按 base 去重分类 (scale key 与 weight key 归一化到同一 base)
    non_layer_bases = []
    layer_bases = []
    mtp_bases = []
    offset_keys = []
    seen = set()
    for key in weight_map.keys():
        # weight_offset (非对称量化): CPU W8A8 为对称量化不消费 offset,
        # 单独收集, 加载后校验是否全零 (非零则丢弃会影响精度)
        if key.endswith(".weight_offset"):
            offset_keys.append(key)
            continue
        base = to_module_base(key)
        if base in seen:
            continue
        seen.add(base)
        if resolver.resolve(base)[0] is None:
            continue
        if ".mlp.experts." in base:
            continue  # MoE 专家由 --moe 模式处理
        if "layers." not in base:
            non_layer_bases.append(base)
            continue
        layer_idx = _extract_layer_idx(base)
        if layer_idx < 0:
            continue
        # 当指定 num_hidden_layers 时, MTP 层的权重单独处理
        if num_hidden_layers is not None and layer_idx >= num_hidden_layers:
            mtp_bases.append(base)
        else:
            layer_bases.append(base)

    # 存在 q_a_proj (q_lora_rank 路径) 时, 与 kv_a_proj_with_mqa 融合后按 socket 切
    fused_qkva = any(
        b.endswith("self_attn.q_a_proj") for b in layer_bases + mtp_bases
    )
    if fused_qkva:
        all_bases_set = set(layer_bases + mtp_bases)
        for b in layer_bases + mtp_bases:
            if b.endswith("self_attn.q_a_proj"):
                kv_base = b.replace(
                    "self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"
                )
                if kv_base not in all_bases_set:
                    raise ValueError(
                        f"存在 {b} 但缺少配对的 {kv_base}, "
                        f"无法融合 fused_qkv_a_proj_with_mqa"
                    )
        print("检测到 self_attn.q_a_proj: 融合为 fused_qkv_a_proj_with_mqa 并按 socket_tp 切分")

    all_weights = load_shards_into_memory(
        resolver.load_plan(non_layer_bases + layer_bases + mtp_bases) + offset_keys,
        weight_map,
        model_dir,
        num_readers=num_readers,
    )

    if offset_keys:
        nonzero = [k for k in offset_keys if not torch.all(all_weights[k] == 0)]
        if nonzero:
            print(
                f"警告: {len(nonzero)}/{len(offset_keys)} 个 weight_offset 非零 "
                f"(非对称量化); CPU W8A8 为对称量化, 丢弃 offset 会影响精度! "
                f"示例: {nonzero[:3]}"
            )
        else:
            print(f"全部 {len(offset_keys)} 个 weight_offset 为零, 安全丢弃")
        for k in offset_keys:
            del all_weights[k]

    def get(base):
        w_key, s_key = resolver.resolve(base)
        return all_weights[w_key], (all_weights[s_key] if s_key else None)

    def split_for_rank(tensor, split_mode, dim, rank):
        return split_tensor_by_rank(
            tensor, split_mode, dim, rank, attention_tp_size, socket_tp_size
        )

    def split_and_concat_for_rank(tensors, split_mode, dim, rank):
        return split_and_concat_by_rank(
            tensors, split_mode, dim, rank, attention_tp_size, socket_tp_size
        )

    def _store_replicate(target_dict, base, w, s):
        target_dict[base + ".weight"] = split_for_rank(w, "none", 0, rank)
        if s is not None:
            target_dict[base + ".weight_scale"] = split_for_rank(s, "none", 0, rank)

    def _store_col_parallel(target_dict, base, w, s):
        target_dict[base + ".weight"] = split_for_rank(w, "attention_tp", 0, rank)
        if s is not None:
            target_dict[base + ".weight_scale"] = split_for_rank(
                s, "attention_tp", 0, rank
            )

    def _store_row_parallel(target_dict, base, w, s):
        target_dict[base + ".weight"] = split_for_rank(w, "attention_tp", 1, rank)
        if s is not None:
            # per-row scale 不随输入维切分, 每个 rank 持有完整 scale
            target_dict[base + ".weight_scale"] = split_for_rank(s, "none", 0, rank)

    def _store_fused_gateup(target_dict, base, w, s):
        # gateup_proj 布局 [gate; up]: 前后两半分别按 rank 切再拼接,
        # 输出重命名 gate_up_proj (模型参数名)
        if w.shape[0] % 2 != 0:
            raise ValueError(f"{base} 行数 {w.shape[0]} 不是偶数, 非法 gateup 布局")
        half = w.shape[0] // 2
        if base.endswith("gateup_proj"):
            new_base = base[: -len("gateup_proj")] + "gate_up_proj"
        else:  # gate_proj (与 up_proj 融合后传入)
            new_base = base[: -len("gate_proj")] + "gate_up_proj"
        target_dict[new_base + ".weight"] = split_and_concat_for_rank(
            [w[:half], w[half:]], "attention_tp", 0, rank
        )
        if s is not None:
            target_dict[new_base + ".weight_scale"] = split_and_concat_for_rank(
                [s[:half], s[half:]], "attention_tp", 0, rank
            )

    def _process_layer_base(base, target_dict, rank):
        """对单个 layer 模块 base 做切分, 结果写入 target_dict."""
        w, s = get(base)

        # bias 类张量 (k_norm.bias / e_score_correction_bias):
        # to_module_base 不剥 .bias 后缀, base 即完整 key, 按原名原样写出
        if base.endswith(".bias") or base.endswith("_bias"):
            target_dict[base] = w
            return

        # shared experts / dense mlp: gateup 融合导出, 或 gate/up 分开导出
        # (自动融合为 [gate; up] 后切分, 输出统一重命名 gate_up_proj)
        if base.endswith("mlp.shared_experts.gateup_proj") or base.endswith(
            "mlp.gateup_proj"
        ):
            _store_fused_gateup(target_dict, base, w, s)
            return

        elif base.endswith("mlp.shared_experts.gate_proj") or base.endswith(
            "mlp.gate_proj"
        ):
            up_base = base[: -len("gate_proj")] + "up_proj"
            if resolver.resolve(up_base)[0] is None:
                raise KeyError(f"缺少配对的 {up_base}")
            up_w, up_s = get(up_base)
            w, s = fuse_gate_up(w, s, up_w, up_s, base)
            _store_fused_gateup(target_dict, base, w, s)
            return

        elif base.endswith("mlp.shared_experts.up_proj") or base.endswith(
            "mlp.up_proj"
        ):
            gate_base = base[: -len("up_proj")] + "gate_proj"
            if resolver.resolve(gate_base)[0] is None:
                raise KeyError(f"缺少配对的 {gate_base}")
            return  # 已在对应 gate_proj 分支融合处理

        elif base.endswith("mlp.shared_experts.down_proj") or base.endswith(
            "mlp.down_proj"
        ):
            _store_row_parallel(target_dict, base, w, s)
            return

        # attention (MLA)
        elif any(
            base.endswith(k)
            for k in (
                "self_attn.q_proj",
                "self_attn.q_b_proj",
                "self_attn.kv_b_proj",
            )
        ):
            _store_col_parallel(target_dict, base, w, s)
            return

        elif base.endswith("self_attn.o_proj"):
            _store_row_parallel(target_dict, base, w, s)
            return

        elif base.endswith("self_attn.q_a_proj"):
            if fused_qkva:
                return  # 与 kv_a_proj_with_mqa 一起融合处理
            _store_replicate(target_dict, base, w, s)
            return

        elif base.endswith("self_attn.kv_a_proj_with_mqa"):
            if not fused_qkva:
                _store_replicate(target_dict, base, w, s)
                return
            qa_base = base.replace(
                "self_attn.kv_a_proj_with_mqa", "self_attn.q_a_proj"
            )
            qa_w, qa_s = get(qa_base)
            if (qa_s is None) != (s is None):
                raise ValueError(
                    f"{qa_base} 与 {base} 的 scale 必须同时存在或同时缺失"
                )
            new_base = base.replace(
                "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
            )
            fused_w = torch.cat([qa_w, w], dim=0)
            target_dict[new_base + ".weight"] = split_for_rank(
                fused_w, "socket_tp", 0, rank
            )
            if s is not None:
                fused_s = torch.cat([qa_s, s], dim=0)
                target_dict[new_base + ".weight_scale"] = split_for_rank(
                    fused_s, "socket_tp", 0, rank
                )
            return

        # MTP 专属: eh_proj (ColumnParallelLinear), shared_head.head (若有)
        elif base.endswith("eh_proj") or base.endswith("shared_head.head"):
            _store_col_parallel(target_dict, base, w, s)
            return

        # 其余 (norm / mlp.gate / indexer.* / enorm / hnorm / shared_head.norm)
        # 均为 replicated: indexer 的 wq_b/wk/weights_proj 见 nsa_indexer.py
        else:
            _store_replicate(target_dict, base, w, s)

    for rank in tqdm(ranks, desc="处理rank"):
        rank_weights = {}
        mtp_rank_weights = {}

        for base in non_layer_bases:
            w, s = get(base)
            if base.endswith("embed_tokens") or base == "lm_head":
                # embed / lm_head 亦写入 MTP 侧 (loader 会跳过 draft 中不存在的 key)
                for d in (rank_weights, mtp_rank_weights):
                    _store_col_parallel(d, base, w, s)
            else:
                _store_replicate(rank_weights, base, w, s)

        for base in tqdm(layer_bases, desc=f"处理keys(rank={rank})", leave=False):
            _process_layer_base(base, rank_weights, rank)

        for base in tqdm(mtp_bases, desc=f"处理MTP keys(rank={rank})", leave=False):
            _process_layer_base(base, mtp_rank_weights, rank)

        out_layer_path = os.path.join(
            output_dir, f"model-rank-{rank}-part-0.safetensors"
        )
        save_file(rank_weights, out_layer_path)
        del rank_weights

        if mtp_output_dir is not None and mtp_rank_weights:
            mtp_layer_path = os.path.join(
                mtp_output_dir, f"model-rank-{rank}-part-0.safetensors"
            )
            mtp_weights_renamed = {
                _rename_mtp_key(k): v for k, v in mtp_rank_weights.items()
            }
            save_file(mtp_weights_renamed, mtp_layer_path)
        del mtp_rank_weights


# ============================================================================
# CLI 入口
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GLM-5 (GlmMoeDsaForCausalLM) 权重切分脚本 "
        "(支持 MoE 专家切分与 非MoE 权重切分, 输入需为已融合 gateup 的量化导出)"
    )

    # 通用参数
    parser.add_argument(
        "--model_dir", type=str, default="./", help="原始模型权重及索引文件所在目录"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data", help="切分后保存输出目录"
    )
    parser.add_argument(
        "--moe",
        action="store_true",
        default=False,
        help="是否切分 MoE 专家权重 (True) 或非 MoE 权重 (False)",
    )

    parser.add_argument(
        "--attention_tp_size", type=int, default=16, help="Attention 张量并行大小"
    )
    parser.add_argument(
        "--socket_tp_size",
        type=int,
        default=8,
        help="Socket 张量并行大小 (用于 fused_qkv_a 切分)",
    )

    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help="隐藏层数量，用于区分 MTP 层与普通层的边界；不传时表示无 MTP 层，所有权重保存到同一目录",
    )

    parser.add_argument(
        "--num_writers",
        type=int,
        default=8,
        help="MoE 专家文件后台写出线程数 (默认 8)",
    )

    parser.add_argument(
        "--readers",
        type=int,
        default=4,
        help="分片并行预取线程数 (默认 4); NFS/RAID 可调大, 单流本地盘可设 1",
    )

    parser.add_argument(
        "--layer_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="仅处理 [START, END) 范围的层 (原始层号, 含 MTP 层); "
        "多机分工时各机各写一段, 输出目录不冲突",
    )

    args = parser.parse_args()

    if not args.moe:
        print("=================== 非 MoE 切分配置 ===================")
        print(f"模型目录 (model_dir)      : {args.model_dir}")
        print(f"输出目录 (output_dir)     : {args.output_dir}")
        print(f"Attention TP 大小         : {args.attention_tp_size}")
        print(f"Socket TP 大小            : {args.socket_tp_size}")
        if args.num_hidden_layers is not None:
            print(f"隐藏层数量                : {args.num_hidden_layers}")
            print(
                f"MTP 输出目录              : {args.output_dir}_mtp/tp{args.attention_tp_size}"
            )
        print("========================================================\n")

        output_dir = args.output_dir + f"/tp{args.attention_tp_size}"
        mtp_output_dir = (
            args.output_dir + f"_mtp/tp{args.attention_tp_size}"
            if args.num_hidden_layers is not None
            else None
        )

        os.makedirs(output_dir, exist_ok=True)
        if mtp_output_dir is not None:
            os.makedirs(mtp_output_dir, exist_ok=True)

        split_non_moe_weights(
            args.model_dir,
            output_dir,
            args.attention_tp_size,
            args.socket_tp_size,
            ranks=list(range(args.attention_tp_size)),
            num_hidden_layers=args.num_hidden_layers,
            mtp_output_dir=mtp_output_dir,
            num_readers=args.readers,
        )
    else:
        print("=================== MoE 专家切分配置 ===================")
        print(f"模型目录 (model_dir)      : {args.model_dir}")
        print(f"输出目录 (output_dir)     : {args.output_dir}")
        if args.num_hidden_layers is not None:
            print(f"隐藏层数量                : {args.num_hidden_layers}")
            print(f"MTP 输出目录              : {args.output_dir}_mtp/experts")
        print("========================================================\n")

        output_dir = args.output_dir + "/experts"
        mtp_output_dir = (
            args.output_dir + "_mtp/experts"
            if args.num_hidden_layers is not None
            else None
        )

        os.makedirs(output_dir, exist_ok=True)
        if mtp_output_dir is not None:
            os.makedirs(mtp_output_dir, exist_ok=True)

        split_moe_experts(
            args.model_dir,
            output_dir,
            num_hidden_layers=args.num_hidden_layers,
            mtp_output_dir=mtp_output_dir,
            num_writers=args.num_writers,
            num_readers=args.readers,
            layer_range=tuple(args.layer_range) if args.layer_range else None,
        )

    # 将非 .safetensors 文件（config.json、tokenizer 等）复制到输出目录；
    # 若启用 MTP 分离，也复制一份到 MTP 输出目录，保证独立加载时元数据完整
    mtp_meta_output_dir = (
        args.output_dir + "_mtp" if args.num_hidden_layers is not None else None
    )
    if mtp_meta_output_dir is not None:
        os.makedirs(mtp_meta_output_dir, exist_ok=True)

    for file in os.listdir(args.model_dir):
        if os.path.splitext(file)[1] in (".safetensors",):
            continue
        # kunpeng_state is a plain I8+FP32-scale format; the ModelSlim quant
        # metadata from the source checkpoint is not applicable and would make
        # ModelConfig detect quant_method=modelslim (an NPU quant) at load.
        if file in ("quant_model_description.json",):
            continue
        # Split-generated outputs must never be clobbered by source dirs of the
        # same name (e.g. a ModelSlim source `experts/` would otherwise be
        # copytree'd over the freshly split per-expert files, and the serving
        # load would read source-format files -> safetensors header too large).
        if file == "experts" or file.startswith("tp"):
            continue
        src = os.path.join(args.model_dir, file)
        for dst_dir in (args.output_dir, mtp_meta_output_dir):
            if dst_dir is None:
                continue
            dst = os.path.join(dst_dir, file)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)
