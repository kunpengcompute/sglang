import json
import os
import shutil
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def split_tensor_by_rank(
    tensor: torch.Tensor,
    split_mode: str,
    dim: int,
    rank: int,
    attention_tp_size: int,
    global_tp_size: int,
    socket_tp_size: int = 8,
) -> torch.Tensor:
    """
    根据切分方式和维度对 tensor 进行切分，返回当前 rank 对应的切片。

    Args:
        tensor: 待切分的张量
        split_mode: 切分方式
            "none"         - 不切分，返回完整副本
            "attention_tp" - 按 attention_tp_size 切分 (实际 rank = rank % attention_tp_size)
            "global_tp"    - 按 global_tp_size 切分 (实际 rank = rank)
            "socket_tp"    - 按 socket_tp_size 切分 (实际 rank = rank % socket_tp_size)
        dim: 切分维度 (0 或 1)；split_mode 为 "none" 时忽略
        rank: 当前全局 rank
        attention_tp_size: attention 张量并行大小
        global_tp_size: 全局张量并行大小
        socket_tp_size: socket 张量并行大小

    Returns:
        切分后 (或完整副本) 的 tensor
    """
    if split_mode == "none":
        return tensor.clone()

    if split_mode == "attention_tp":
        tp_size, tp_rank = attention_tp_size, rank % attention_tp_size
    elif split_mode == "global_tp":
        tp_size, tp_rank = global_tp_size, rank
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
    global_tp_size: int,
    socket_tp_size: int = 8,
) -> torch.Tensor:
    """
    对多个 tensor 分别按相同切分方式和维度切分，再在 dim 上 cat 起来。

    用于 gate_proj/up_proj 这类需要 "分别切分配对、再拼接" 的场景，
    与 "先 cat 再整体切分" 语义不同：本函数保证每个 rank 拿到的是
    每个 tensor 的第 rank 块的拼接，gate 与 up 块始终配对。

    Args:
        tensors: 待切分的张量列表 (按拼接顺序)
        其余参数同 split_tensor_by_rank

    Returns:
        拼接后的 tensor
    """
    chunks = [
        split_tensor_by_rank(
            t, split_mode, dim, rank, attention_tp_size, global_tp_size, socket_tp_size
        )
        for t in tensors
    ]
    return torch.cat(chunks, dim=dim)


def load_shards_into_memory(keys, weight_map, model_dir="./", use_tqdm=True):
    if isinstance(keys, str):
        keys = [keys]
    needed_files = set(weight_map[k] for k in keys if k in weight_map)
    # print(f"   正在加载分片文件{needed_files}到内存以处理当前层的 {len(keys)} 个权重 Key...")
    combined_data = {}
    for f in tqdm(needed_files, desc="加载分片文件到内存", disable=not use_tqdm):
        shard_data = load_file(os.path.join(model_dir, f))
        for k in keys:
            if k in shard_data:
                combined_data[k] = shard_data[k]
        del shard_data
    return combined_data


def split_moe_experts(model_dir: str, output_dir: str, global_tp_size: int = 64):

    # 加载权重索引
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"找不到索引文件: {index_path}")

    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    # 1. 筛选出所有与专家 (mlp.experts) 相关的 Key，并按“层”归类
    moe_expert_to_keys = defaultdict(list)
    for key in weight_map.keys():
        if "mlp.experts." in key:
            # 提取层号，例如 "model.layers.1.mlp.experts.0..." -> "layers.1"
            parts = key.split(".")
            layer_idx = parts[parts.index("experts") + 1]
            moe_expert_to_keys[f"experts.{layer_idx}"].append(key)

    # 2. 按层升序处理
    sorted_moe_experts = sorted(
        list(moe_expert_to_keys.keys()), key=lambda x: int(x.split(".")[1])
    )

    for expert in tqdm(sorted_moe_experts, desc="处理 MoE 专家"):
        keys_in_experts = moe_expert_to_keys[expert]
        layers = set([i.split(".")[2] for i in keys_in_experts])
        experts_state_dict = {}

        moe_weights = load_shards_into_memory(
            keys_in_experts, weight_map, model_dir, use_tqdm=False
        )

        for layer in sorted([int(i) for i in layers]):
            up_tensor = moe_weights[f"model.layers.{layer}.mlp.{expert}.up_proj.weight"]
            up_scale_tensor = moe_weights[
                f"model.layers.{layer}.mlp.{expert}.up_proj.weight_scale"
            ]
            gate_tensor = moe_weights[
                f"model.layers.{layer}.mlp.{expert}.gate_proj.weight"
            ]
            gate_scale_tensor = moe_weights[
                f"model.layers.{layer}.mlp.{expert}.gate_proj.weight_scale"
            ]
            down_tensor = moe_weights[
                f"model.layers.{layer}.mlp.{expert}.down_proj.weight"
            ]
            down_scale_tensor = moe_weights[
                f"model.layers.{layer}.mlp.{expert}.down_proj.weight_scale"
            ]

            w13_tensor = torch.cat([gate_tensor, up_tensor], dim=0)
            w13_scale_tensor = torch.cat([gate_scale_tensor, up_scale_tensor], dim=0)
            w2_tensor = down_tensor
            w2_scale_tensor = down_scale_tensor

            # 改成3维,第0维是1
            experts_state_dict[f"model.layers.{layer}.mlp.experts.w13_weight"] = (
                w13_tensor.unsqueeze(0)
            )
            experts_state_dict[f"model.layers.{layer}.mlp.experts.w13_weight_scale"] = (
                w13_scale_tensor.unsqueeze(0)
            )
            experts_state_dict[f"model.layers.{layer}.mlp.experts.w2_weight"] = (
                w2_tensor.unsqueeze(0)
            )
            experts_state_dict[f"model.layers.{layer}.mlp.experts.w2_weight_scale"] = (
                w2_scale_tensor.unsqueeze(0)
            )

        out_file_path = os.path.join(
            output_dir,
            f"model-rank-{expert.split('.')[1]}-part-1.safetensors.safetensors",
        )
        save_file(experts_state_dict, out_file_path)

        del moe_weights, experts_state_dict


def split_non_moe_weights(
    model_dir: str,
    output_dir: str,
    attention_tp_size: int = 16,
    global_tp_size=64,
    socket_tp_size=8,
    dp_lm_head=True,
    dp_mlp=True,
    ranks=None,
):
    """
    处理单个核心层的非MOE部分权重切分与存储。

    Args:
        ranks: 需要处理的 rank 列表。为 None 时处理全部 global_tp_size 个 rank。
               传入子集可降低峰值内存占用，调用方需自行分批遍历全部 rank。
    """
    if ranks is None:
        ranks = list(range(global_tp_size))
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
            layer_keys.append(key)

    all_keys = non_layer_keys + layer_keys
    fused_qkva = "model.layers.0.self_attn.q_a_proj.weight" in layer_keys
    all_weights = load_shards_into_memory(all_keys, weight_map, model_dir)

    def split_for_rank(tensor, split_mode, dim, rank):
        """对指定 rank 切分 tensor。"""
        return split_tensor_by_rank(
            tensor,
            split_mode,
            dim,
            rank,
            attention_tp_size,
            global_tp_size,
            socket_tp_size,
        )

    def split_and_concat_for_rank(tensors, split_mode, dim, rank):
        """对指定 rank 分别切分多个 tensor 后 cat。"""
        return split_and_concat_by_rank(
            tensors,
            split_mode,
            dim,
            rank,
            attention_tp_size,
            global_tp_size,
            socket_tp_size,
        )

    for rank in tqdm(ranks, desc="处理rank"):
        rank_weights = {}

        for key in non_layer_keys:
            tensor = all_weights[key]
            if "embed_tokens" in key:
                rank_weights[key] = split_for_rank(tensor, "attention_tp", 0, rank)
            elif "lm_head" in key:
                rank_weights[key] = split_for_rank(
                    tensor, "attention_tp" if dp_lm_head else "global_tp", 0, rank
                )
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

                # gate 与 up 必须分别切分配对后再 cat，不能先 cat 再切
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
                mode = "attention_tp" if dp_mlp else "global_tp"
                rank_weights[key] = split_for_rank(tensor, mode, 1, rank)
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

                # gate 与 up 必须分别切分配对后再 cat，不能先 cat 再切
                mode = "attention_tp" if dp_mlp else "global_tp"
                rank_weights[new_key] = split_and_concat_for_rank(
                    [tensor, up_tensor], mode, 0, rank
                )
                rank_weights[new_key + "_scale"] = split_and_concat_for_rank(
                    [tensor_scale, up_scale_tensor], mode, 0, rank
                )

            elif "mlp.up_proj.weight" in key:
                continue

            # attention
            # "layernorm.weight"   不切分

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

            else:
                tensor = all_weights[key]
                rank_weights[key] = split_for_rank(tensor, "none", 0, rank)

        out_layer_path = os.path.join(
            output_dir, f"model-rank-{rank}-part-0.safetensors.safetensors"
        )
        save_file(rank_weights, out_layer_path)
        del rank_weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepSeek MoE 16-Worker 高性能并行切分脚本"
    )

    parser.add_argument(
        "--model_dir", type=str, default="./", help="原始模型权重及索引文件所在目录"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data", help="切分后保存输出目录"
    )
    parser.add_argument(
        "--attention_tp_size", type=int, default=16, help="Attention 张量并行大小"
    )
    parser.add_argument(
        "--global_tp_size",
        type=int,
        default=64,
        help="总 Global/MoE 并行大小 (对应专家数)",
    )
    parser.add_argument(
        "--disable_dp_lm_head",
        action="store_true",
        default=False,
        help="禁用 dp_lm_head 模式 (默认启用)",
    )
    parser.add_argument(
        "--disable_dp_mlp",
        action="store_true",
        default=False,
        help="禁用 dp_mlp 模式 (默认启用)",
    )
    parser.add_argument(
        "--socket_tp_size",
        type=int,
        default=8,
        help="Socket 张量并行大小 (用于 qkva 切分)",
    )

    args = parser.parse_args()
    args.dp_lm_head = not args.disable_dp_lm_head
    args.dp_mlp = not args.disable_dp_mlp

    print("=================== 🚀 多进程并发配置 ===================")
    print(f"模型目录 (model_dir)      : {args.model_dir}")
    print(f"输出目录 (output_dir)     : {args.output_dir}")
    print(f"总计 Rank 数量 (Total)    : {args.global_tp_size}")
    print(f"Attention TP 大小 (Attention TP Size) : {args.attention_tp_size}")
    print(f"Global TP 大小 (Global TP Size)       : {args.global_tp_size}")
    print(f"Socket TP 大小 (Socket TP Size)       : {args.socket_tp_size}")
    print(f"DP LM Head 模式 (DP LM Head)         : {args.dp_lm_head}")
    print(f"DP MLP 模式 (DP MLP)                 : {args.dp_mlp}")
    print("========================================================\n")

    os.makedirs(args.output_dir, exist_ok=True)

    split_moe_experts(
        args.model_dir,
        args.output_dir,
        args.global_tp_size,
    )

    split_non_moe_weights(
        args.model_dir,
        args.output_dir,
        args.attention_tp_size,
        args.global_tp_size,
        args.socket_tp_size,
        args.dp_lm_head,
        args.dp_mlp,
    )

    # Copy metadata files to output directory
    for file in os.listdir(args.model_dir):
        if os.path.splitext(file)[1] in (".bin", ".pt", ".safetensors"):
            continue

        if os.path.isdir(os.path.join(args.model_dir, file)):
            shutil.copytree(
                os.path.join(args.model_dir, file), os.path.join(args.output_dir, file)
            )
        else:
            shutil.copy(os.path.join(args.model_dir, file), args.output_dir)
