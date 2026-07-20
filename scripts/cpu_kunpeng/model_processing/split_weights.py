import json
import os
import shutil
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# ============================================================================
# 公共工具函数
# ============================================================================


def load_shards_into_memory(keys, weight_map, model_dir="./", use_tqdm=True):
    if isinstance(keys, str):
        keys = [keys]
    needed_files = set(weight_map[k] for k in keys if k in weight_map)
    combined_data = {}
    for f in tqdm(needed_files, desc="加载分片文件到内存", disable=not use_tqdm):
        shard_data = load_file(os.path.join(model_dir, f))
        for k in keys:
            if k in shard_data:
                combined_data[k] = shard_data[k]
        del shard_data
    return combined_data


def _extract_layer_idx(key: str) -> int:
    """从 key 中提取层索引，如 'model.layers.5.self_attn...' -> 5"""
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
    """HF key -> draft model state_dict key for MTP layers."""
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
# MoE 专家权重切分
# ============================================================================


def split_moe_experts(model_dir: str, output_dir: str):

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"找不到索引文件: {index_path}")

    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    layer_to_expert_keys = defaultdict(lambda: defaultdict(list))
    for key in weight_map.keys():
        if "mlp.experts." in key:
            parts = key.split(".")
            layer_idx = int(parts[2])
            expert_idx = int(parts[parts.index("experts") + 1])
            layer_to_expert_keys[layer_idx][expert_idx].append(key)

    sorted_layers = sorted(layer_to_expert_keys.keys())

    for layer in tqdm(sorted_layers, desc="处理 MoE 层"):
        layer_dir = os.path.join(output_dir, f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)

        expert_dict = layer_to_expert_keys[layer]
        sorted_experts = sorted(expert_dict.keys())

        all_keys = []
        for expert_keys in expert_dict.values():
            all_keys.extend(expert_keys)

        moe_weights = load_shards_into_memory(
            all_keys, weight_map, model_dir, use_tqdm=False
        )

        for expert in sorted_experts:
            expert_state_dict = {}

            up_tensor = moe_weights[
                f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"
            ]
            up_scale_tensor = moe_weights[
                f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight_scale"
            ]
            gate_tensor = moe_weights[
                f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
            ]
            gate_scale_tensor = moe_weights[
                f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight_scale"
            ]
            down_tensor = moe_weights[
                f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
            ]
            down_scale_tensor = moe_weights[
                f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight_scale"
            ]

            w13_tensor = torch.cat([gate_tensor, up_tensor], dim=0)
            w13_scale_tensor = torch.cat([gate_scale_tensor, up_scale_tensor], dim=0)
            w2_tensor = down_tensor
            w2_scale_tensor = down_scale_tensor

            expert_state_dict[
                "model.layers.{}.mlp.experts.w13_weight".format(layer)
            ] = w13_tensor
            expert_state_dict[
                "model.layers.{}.mlp.experts.w13_weight_scale".format(layer)
            ] = w13_scale_tensor
            expert_state_dict["model.layers.{}.mlp.experts.w2_weight".format(layer)] = (
                w2_tensor
            )
            expert_state_dict[
                "model.layers.{}.mlp.experts.w2_weight_scale".format(layer)
            ] = w2_scale_tensor

            out_file_path = os.path.join(layer_dir, f"expert_{expert}.safetensors")
            save_file(expert_state_dict, out_file_path)

        del moe_weights


# ============================================================================
# Non-MoE 权重切分
# ============================================================================


def split_non_moe_weights(
    model_dir: str,
    output_dir: str,
    attention_tp_size: int = 16,
    socket_tp_size=8,
    ranks=None,
    num_hidden_layers=61,
):
    if ranks is None:
        ranks = list(range(attention_tp_size))
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"找不到索引文件: {index_path}")

    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    non_layer_keys = []
    layer_keys = []

    for key in weight_map.keys():
        if "layers." not in key:
            non_layer_keys.append(key)
        elif "mlp.experts." not in key:
            layer_idx = _extract_layer_idx(key)
            if layer_idx < 0:
                continue
            layer_keys.append(key)

    all_keys = non_layer_keys + layer_keys
    fused_qkva = "model.layers.0.self_attn.q_a_proj.weight" in layer_keys
    all_weights = load_shards_into_memory(all_keys, weight_map, model_dir)

    def split_for_rank(tensor, split_mode, dim, rank):
        return split_tensor_by_rank(
            tensor,
            split_mode,
            dim,
            rank,
            attention_tp_size,
            socket_tp_size,
        )

    def split_and_concat_for_rank(tensors, split_mode, dim, rank):
        return split_and_concat_by_rank(
            tensors,
            split_mode,
            dim,
            rank,
            attention_tp_size,
            socket_tp_size,
        )

    for rank in tqdm(ranks, desc="处理rank"):
        rank_weights = {}

        for key in non_layer_keys:
            tensor = all_weights[key]
            if "embed_tokens" in key:
                rank_weights[key] = split_for_rank(tensor, "attention_tp", 0, rank)
            elif "lm_head" in key:
                rank_weights[key] = split_for_rank(tensor, "attention_tp", 0, rank)
            else:
                rank_weights[key] = split_for_rank(tensor, "none", 0, rank)

        for key in tqdm(layer_keys, desc=f"处理keys(rank={rank})", leave=False):
            if "_scale" in key:
                continue

            # shared experts
            if "shared_experts.gate_proj.weight" in key:
                up_key = key.replace("gate_proj", "up_proj")
                tensor = all_weights[key]
                tensor_scale = all_weights[key + "_scale"]
                up_tensor = all_weights[up_key]
                up_tensor_scale = all_weights[up_key + "_scale"]
                new_key = key.replace("gate_proj", "gate_up_proj")

                rank_weights[new_key] = split_and_concat_for_rank(
                    [tensor, up_tensor], "attention_tp", 0, rank
                )
                rank_weights[new_key + "_scale"] = split_and_concat_for_rank(
                    [tensor_scale, up_tensor_scale], "attention_tp", 0, rank
                )

            elif "shared_experts.up_proj.weight" in key:
                continue

            elif "shared_experts.down_proj.weight" in key:
                tensor = all_weights[key]
                scale_tensor = all_weights[key + "_scale"]

                rank_weights[key] = split_for_rank(tensor, "attention_tp", 1, rank)
                rank_weights[key + "_scale"] = split_for_rank(
                    scale_tensor, "none", 0, rank
                )

            # mlp
            elif "mlp.down_proj.weight" in key:
                tensor = all_weights[key]
                scale_tensor = all_weights[key + "_scale"]
                rank_weights[key] = split_for_rank(tensor, "attention_tp", 1, rank)
                rank_weights[key + "_scale"] = split_for_rank(
                    scale_tensor, "none", 0, rank
                )

            elif "mlp.gate_proj.weight" in key:
                up_key = key.replace("gate_proj", "up_proj")
                tensor = all_weights[key]
                tensor_scale = all_weights[key + "_scale"]
                up_tensor = all_weights[up_key]
                up_scale_tensor = all_weights[up_key + "_scale"]
                new_key = key.replace("gate_proj", "gate_up_proj")

                rank_weights[new_key] = split_and_concat_for_rank(
                    [tensor, up_tensor], "attention_tp", 0, rank
                )
                rank_weights[new_key + "_scale"] = split_and_concat_for_rank(
                    [tensor_scale, up_scale_tensor], "attention_tp", 0, rank
                )

            elif "mlp.up_proj.weight" in key:
                continue

            # attention
            elif any(
                k in key
                for k in [
                    "self_attn.q_proj.weight",
                    "self_attn.q_b_proj.weight",
                    "self_attn.kv_b_proj.weight",
                ]
            ):
                tensor = all_weights[key]
                scale_tensor = all_weights[key + "_scale"]
                rank_weights[key] = split_for_rank(tensor, "attention_tp", 0, rank)
                rank_weights[key + "_scale"] = split_for_rank(
                    scale_tensor, "attention_tp", 0, rank
                )

            elif "self_attn.o_proj.weight" in key:
                tensor = all_weights[key]
                scale_tensor = all_weights[key + "_scale"]
                rank_weights[key] = split_for_rank(tensor, "attention_tp", 1, rank)
                rank_weights[key + "_scale"] = split_for_rank(
                    scale_tensor, "none", 0, rank
                )

            elif "self_attn.q_a_proj.weight" in key:
                if fused_qkva:
                    continue
                tensor = all_weights[key]
                scale_tensor = all_weights[key + "_scale"]
                rank_weights[key] = split_for_rank(tensor, "none", 0, rank)
                rank_weights[key + "_scale"] = split_for_rank(
                    scale_tensor, "none", 0, rank
                )

            elif "self_attn.kv_a_proj_with_mqa.weight" in key:
                if not fused_qkva:
                    tensor = all_weights[key]
                    scale_tensor = all_weights[key + "_scale"]
                    rank_weights[key] = split_for_rank(tensor, "none", 0, rank)
                    rank_weights[key + "_scale"] = split_for_rank(
                        scale_tensor, "none", 0, rank
                    )
                else:
                    qa_key = key.replace(
                        "self_attn.kv_a_proj_with_mqa", "self_attn.q_a_proj"
                    )
                    tensor = all_weights[key]
                    qa_tensor = all_weights[qa_key]
                    scale_tensor = all_weights[key + "_scale"]
                    qa_scale_tensor = all_weights[qa_key + "_scale"]
                    new_key = key.replace(
                        "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                    )

                    fused_tensor = torch.cat([qa_tensor, tensor], dim=0)
                    fused_scale = torch.cat([qa_scale_tensor, scale_tensor], dim=0)
                    rank_weights[new_key] = split_for_rank(
                        fused_tensor, "socket_tp", 0, rank
                    )
                    rank_weights[new_key + "_scale"] = split_for_rank(
                        fused_scale, "socket_tp", 0, rank
                    )

            elif "eh_proj" in key:
                tensor = all_weights[key]
                rank_weights[key] = split_for_rank(tensor, "attention_tp", 1, rank)

            else:
                tensor = all_weights[key]
                rank_weights[key] = split_for_rank(tensor, "none", 0, rank)

        key_num = len(rank_weights.keys())
        weights_saved = {}
        _keys = list(rank_weights.keys())
        for k in _keys:
            v = rank_weights.pop(k)
            if _extract_layer_idx(k) < num_hidden_layers:
                weights_saved[k] = v
            else:
                weights_saved[_rename_mtp_key(k)] = v
        assert len(weights_saved.keys()) == key_num

        out_layer_path = os.path.join(
            output_dir, f"model-rank-{rank}-part-0.safetensors"
        )
        save_file(weights_saved, out_layer_path)
        del weights_saved


# ============================================================================
# CLI 入口
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepSeek 权重切分脚本 (支持 MoE 专家切分与 非MoE 权重切分)"
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
        help="Socket 张量并行大小 (用于 qkva 切分)",
    )

    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=61,
        help="隐藏层数量，用于区分 MTP 层与普通层的边界",
    )

    args = parser.parse_args()

    if not args.moe:
        print("=================== 非 MoE 切分配置 ===================")
        print(f"模型目录 (model_dir)      : {args.model_dir}")
        print(f"输出目录 (output_dir)     : {args.output_dir}")
        print(f"Attention TP 大小         : {args.attention_tp_size}")
        print(f"Socket TP 大小            : {args.socket_tp_size}")
        print(f"隐藏层数量                : {args.num_hidden_layers}")
        print("========================================================\n")

        output_dir = args.output_dir + f"/tp{args.attention_tp_size}"

        os.makedirs(output_dir, exist_ok=True)

        split_non_moe_weights(
            args.model_dir,
            output_dir,
            args.attention_tp_size,
            args.socket_tp_size,
            ranks=list(range(args.attention_tp_size)),
            num_hidden_layers=args.num_hidden_layers,
        )
    else:
        print("=================== MoE 专家切分配置 ===================")
        print(f"模型目录 (model_dir)      : {args.model_dir}")
        print(f"输出目录 (output_dir)     : {args.output_dir}")
        print("========================================================\n")

        output_dir = args.output_dir + "/experts"

        os.makedirs(output_dir, exist_ok=True)

        split_moe_experts(
            args.model_dir,
            output_dir,
        )

    for file in os.listdir(args.model_dir):
        if os.path.splitext(file)[1] in (".safetensors",):
            continue
        src = os.path.join(args.model_dir, file)
        dst = os.path.join(args.output_dir, file)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
