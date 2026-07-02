import json
import os
import shutil
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


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
    dp_lm_head=True,
    dp_mlp=True,
):
    """
    处理单个核心层的非MOE部分权重切分与存储。
    """
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

    global_rank_weights = [{} for _ in range(global_tp_size)]

    for key in non_layer_keys:
        tensor = all_weights[key]
        if "embed_tokens" in key:
            chunk = tensor.shape[0] // attention_tp_size
            for rank in range(global_tp_size):
                attention_tp_rank = rank % attention_tp_size
                global_rank_weights[rank][key] = tensor[
                    attention_tp_rank * chunk : (attention_tp_rank + 1) * chunk
                ].clone()
        elif "lm_head" in key:
            if dp_lm_head:
                chunk = tensor.shape[0] // attention_tp_size
                for rank in range(global_tp_size):
                    attention_tp_rank = rank % attention_tp_size
                    global_rank_weights[rank][key] = tensor[
                        attention_tp_rank * chunk : (attention_tp_rank + 1) * chunk
                    ].clone()
            else:
                chunk = tensor.shape[0] // global_tp_size
                for rank in range(global_tp_size):
                    global_rank_weights[rank]["lm_head.weight"] = tensor[
                        rank * chunk : (rank + 1) * chunk
                    ].clone()
        else:
            for rank in range(global_tp_size):
                global_rank_weights[rank][key] = tensor.clone()

    for key in tqdm(layer_keys, desc="处理keys"):
        if "_scale" in key:
            continue

        # moe
        # "mlp.gate.weight"   # 不切分

        # shared experts
        if "shared_experts.gate_proj.weight" in key:
            up_key = key.replace("gate_proj", "up_proj")
            tensor = all_weights[key]
            tensor_scale = all_weights[key + "_scale"]
            up_tensor = all_weights[up_key]
            up_tensor_scale = all_weights[up_key + "_scale"]

            g_chunk = tensor.shape[0] // attention_tp_size
            u_chunk = up_tensor.shape[0] // attention_tp_size
            new_key = key.replace("gate_proj", "gate_up_proj")
            for rank in range(global_tp_size):
                attention_tp_rank = rank % attention_tp_size
                gate_chunk = tensor[
                    attention_tp_rank * g_chunk : (attention_tp_rank + 1) * g_chunk
                ]
                gate_scale_chunk = tensor_scale[
                    attention_tp_rank * g_chunk : (attention_tp_rank + 1) * g_chunk
                ]
                up_chunk_tensor = up_tensor[
                    attention_tp_rank * u_chunk : (attention_tp_rank + 1) * u_chunk
                ]
                up_scale_chunk = up_tensor_scale[
                    attention_tp_rank * u_chunk : (attention_tp_rank + 1) * u_chunk
                ]
                global_rank_weights[rank][new_key] = torch.cat(
                    [gate_chunk, up_chunk_tensor], dim=0
                ).clone()
                global_rank_weights[rank][new_key + "_scale"] = torch.cat(
                    [gate_scale_chunk, up_scale_chunk], dim=0
                ).clone()

        elif "shared_experts.up_proj.weight" in key:
            continue

        elif "shared_experts.down_proj.weight" in key:
            tensor = all_weights[key]
            scale_tensor = all_weights[key + "_scale"]

            chunk = tensor.shape[1] // attention_tp_size
            for rank in range(global_tp_size):
                attention_tp_rank = rank % attention_tp_size
                global_rank_weights[rank][key] = tensor[
                    :, attention_tp_rank * chunk : (attention_tp_rank + 1) * chunk
                ].clone()
                global_rank_weights[rank][key + "_scale"] = scale_tensor.clone()

            if any(
                k in key
                for k in ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]
            ):
                tensor = all_weights[key]
                for rank in range(global_tp_size):
                    global_rank_weights[rank][key] = tensor.clone()

        # mlp
        elif "mlp.down_proj.weight" in key:
            tensor = all_weights[key]
            scale_tensor = all_weights[key + "_scale"]

            if dp_mlp:
                chunk = tensor.shape[1] // attention_tp_size
                for rank in range(global_tp_size):
                    attention_tp_rank = rank % attention_tp_size
                    global_rank_weights[rank][key] = tensor[
                        :,
                        attention_tp_rank * chunk : (attention_tp_rank + 1) * chunk,
                    ].clone()
                    global_rank_weights[rank][key + "_scale"] = scale_tensor.clone()
            else:
                chunk = tensor.shape[1] // global_tp_size
                for rank in range(global_tp_size):
                    global_rank_weights[rank][key] = tensor[
                        :, rank * chunk : (rank + 1) * chunk
                    ].clone()
                    global_rank_weights[rank][key + "_scale"] = scale_tensor.clone()

        elif "mlp.gate_proj.weight" in key:
            up_key = key.replace("gate_proj", "up_proj")
            tensor = all_weights[key]
            tensor_scale = all_weights[key + "_scale"]
            up_tensor = all_weights[up_key]
            up_scale_tensor = all_weights[up_key + "_scale"]
            new_key = key.replace("gate_proj", "gate_up_proj")

            if dp_mlp:
                g_chunk = tensor.shape[0] // attention_tp_size
                u_chunk = up_tensor.shape[0] // attention_tp_size

                for rank in range(global_tp_size):
                    attention_tp_rank = rank % attention_tp_size
                    gate_chunk = tensor[
                        attention_tp_rank * g_chunk : (attention_tp_rank + 1) * g_chunk
                    ]
                    gate_scale_chunk = tensor_scale[
                        attention_tp_rank * g_chunk : (attention_tp_rank + 1) * g_chunk
                    ]
                    up_chunk_tensor = up_tensor[
                        attention_tp_rank * u_chunk : (attention_tp_rank + 1) * u_chunk
                    ]
                    up_scale_chunk = up_scale_tensor[
                        attention_tp_rank * u_chunk : (attention_tp_rank + 1) * u_chunk
                    ]
                    global_rank_weights[rank][new_key] = torch.cat(
                        [gate_chunk, up_chunk_tensor], dim=0
                    ).clone()
                    global_rank_weights[rank][new_key + "_scale"] = torch.cat(
                        [gate_scale_chunk, up_scale_chunk], dim=0
                    ).clone()
            else:

                g_chunk = tensor.shape[0] // global_tp_size
                u_chunk = up_tensor.shape[0] // global_tp_size

                for rank in range(global_tp_size):
                    gate_chunk = tensor[rank * g_chunk : (rank + 1) * g_chunk]
                    gate_scale_chunk = tensor_scale[
                        rank * g_chunk : (rank + 1) * g_chunk
                    ]
                    up_chunk_tensor = up_tensor[rank * u_chunk : (rank + 1) * u_chunk]
                    up_scale_chunk = up_scale_tensor[
                        rank * u_chunk : (rank + 1) * u_chunk
                    ]

                    global_rank_weights[rank][new_key] = torch.cat(
                        [gate_chunk, up_chunk_tensor], dim=0
                    ).clone()
                    global_rank_weights[rank][new_key + "_scale"] = torch.cat(
                        [gate_scale_chunk, up_scale_chunk], dim=0
                    ).clone()

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
            chunk = tensor.shape[0] // attention_tp_size

            for rank in range(global_tp_size):
                attention_tp_rank = rank % attention_tp_size
                rank_weight = tensor[
                    attention_tp_rank * chunk : (attention_tp_rank + 1) * chunk
                ].clone()
                rank_scale = scale_tensor[
                    attention_tp_rank * chunk : (attention_tp_rank + 1) * chunk
                ].clone()
                global_rank_weights[rank][key] = rank_weight
                global_rank_weights[rank][key + "_scale"] = rank_scale

        elif "self_attn.o_proj.weight" in key:
            tensor = all_weights[key]
            scale_tensor = all_weights[key + "_scale"]

            chunk = tensor.shape[1] // attention_tp_size
            for rank in range(global_tp_size):
                attention_tp_rank = rank % attention_tp_size
                global_rank_weights[rank][key] = tensor[
                    :, attention_tp_rank * chunk : (attention_tp_rank + 1) * chunk
                ].clone()
                global_rank_weights[rank][key + "_scale"] = scale_tensor.clone()

        elif "self_attn.kv_a_proj_with_mqa.weight" in key:
            if not fused_qkva:
                for rank in range(global_tp_size):
                    global_rank_weights[rank][key] = all_weights[key].clone()
                    global_rank_weights[rank][key + "_scale"] = all_weights[
                        key + "_scale"
                    ].clone()
            else:
                qa_key = key.replace(
                    "self_attn.kv_a_proj_with_mqa", "self_attn.q_a_proj"
                )
                tensor = all_weights[key]
                qa_tensor = all_weights[qa_key]
                scale_tensor = all_weights[key + "_scale"]
                qa_scale_tensor = all_weights[qa_key + "_scale"]
                new_key = key.replace("kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa")

                fused_tensor = torch.cat([qa_tensor, tensor], dim=0)
                fused_scale = torch.cat([qa_scale_tensor, scale_tensor], dim=0)
                chunk = fused_tensor.shape[0] // attention_tp_size
                for rank in range(global_tp_size):
                    attention_tp_rank = rank % attention_tp_size
                    global_rank_weights[rank][new_key] = fused_tensor[
                        attention_tp_rank * chunk : (attention_tp_rank + 1) * chunk
                    ].clone()
                    global_rank_weights[rank][new_key + "_scale"] = fused_scale[
                        attention_tp_rank * chunk : (attention_tp_rank + 1) * chunk
                    ].clone()

        else:
            tensor = all_weights[key]
            for rank in range(global_tp_size):
                global_rank_weights[rank][key] = tensor.clone()

    for rank in tqdm(range(global_tp_size), desc="保存权重到文件"):
        out_layer_path = os.path.join(
            output_dir, f"model-rank-{rank}-part-0.safetensors.safetensors"
        )
        save_file(global_rank_weights[rank], out_layer_path)


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

    args = parser.parse_args()
    args.dp_lm_head = not args.disable_dp_lm_head
    args.dp_mlp = not args.disable_dp_mlp

    print("=================== 🚀 多进程并发配置 ===================")
    print(f"模型目录 (model_dir)      : {args.model_dir}")
    print(f"输出目录 (output_dir)     : {args.output_dir}")
    print(f"总计 Rank 数量 (Total)    : {args.global_tp_size}")
    print(f"Attention TP 大小 (Attention TP Size) : {args.attention_tp_size}")
    print(f"Global TP 大小 (Global TP Size)       : {args.global_tp_size}")
    print(f"DP LM Head 模式 (DP LM Head)         : {args.dp_lm_head}")
    print(f"DP MLP 模式 (DP MLP)                 : {args.dp_mlp}")
    print("========================================================\n")

    os.makedirs(args.output_dir, exist_ok=True)

    split_non_moe_weights(
        args.model_dir,
        args.output_dir,
        args.attention_tp_size,
        args.global_tp_size,
        args.dp_lm_head,
        args.dp_mlp,
    )

    split_moe_experts(
        args.model_dir,
        args.output_dir,
        args.global_tp_size,
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
