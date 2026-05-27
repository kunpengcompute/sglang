import os
import shutil
import json
import torch
from safetensors.torch import load_file, save_file
from collections import defaultdict


def load_shards_into_memory(keys, weight_map, model_dir='./'):
    if isinstance(keys,str):
        keys = [keys]
    needed_files = set(weight_map[k] for k in keys if k in weight_map)
    # print(f"   正在加载分片文件{needed_files}到内存以处理当前层的 {len(keys)} 个权重 Key...")
    combined_data = {}
    for f in needed_files:
        shard_data = load_file(os.path.join(model_dir, f))
        for k in keys:
            if k in shard_data:
                combined_data[k] = shard_data[k]
        del shard_data
    return combined_data


def split_global_weights(model_dir: str, output_dir: str, tp_size: int = 4):
    """
    处理全局通用权重 (Embedding, LM_Head 等) 并按 Tensor Parallel 存储。
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"找不到索引文件: {index_path}")

    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    non_layer_keys = []
    for key in weight_map.keys():
        if "layers." not in key:
            non_layer_keys.append(key)
    
    print("正在处理全局通用权重 (Embedding, LM_Head 等)...")
    global_weights = load_shards_into_memory(non_layer_keys, weight_map,model_dir)
    rank_globals = [{} for _ in range(tp_size)]
    
    for key in non_layer_keys:
        tensor = global_weights[key]
        if "embed_tokens" in key :
            # 按第 0 维进行切分 (Row / Column Parallel 视具体框架实现而定，此处延续原逻辑)
            chunk = tensor.shape[0] // tp_size
            for rank in range(tp_size):
                rank_globals[rank][key] = tensor[rank * chunk : (rank + 1) * chunk].clone()
        elif  "lm_head" in key:
            continue
        else:
            # 不需要切分的权重，所有 rank 复制一份
            for rank in range(tp_size):
                rank_globals[rank][key] = tensor.clone()
                
    for rank in range(tp_size):
        os.makedirs(os.path.join(output_dir, f"rank_{rank}"), exist_ok=True)
        save_path = os.path.join(output_dir, f"rank_{rank}", "global_weights.safetensors")
        save_file(rank_globals[rank], save_path)
        
    del global_weights, rank_globals

def split_layer_weights(model_dir: str, model:str, output_dir: str, tp_size: int = 4, global_tp_size=256):
    """
    处理单个核心层（Layer）的非MOE部分权重切分与存储。
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"找不到索引文件: {index_path}")

    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    layer_to_keys = defaultdict(list)
    for key in weight_map.keys():
        if "layers." in key:
            # 提取层号，例如 "model.layers.1.mlp..." -> "layers.1"
            parts = key.split(".")
            layer_idx = parts[parts.index("layers") + 1]
            layer_to_keys[f"layers.{layer_idx}"].append(key)
            
    sorted_layers = sorted(list(layer_to_keys.keys()), key=lambda x: int(x.split(".")[1]))
    for layer_name in sorted_layers:
        keys_in_layer = layer_to_keys[layer_name]
        rank_weights = [{} for _ in range(tp_size)]
        global_rank_weights = [{} for _ in range(global_tp_size)]


        keys_in_layer = [i for i in keys_in_layer if 'mlp.experts.' not in i] 
        
        print(f"正在处理核心层: {layer_name} ...")

    
        for key in keys_in_layer:
            if "_scale" in key:
                continue  # scale 权重会在对应主权重分支中一起处理，不单独切分
            
            # --- 1. 广播/复制 权重 (Layernorm, Router 等) ---
            if "layernorm.weight" in key:
                tensor = load_shards_into_memory(key, weight_map,model_dir)[key]
                for rank in range(tp_size): 
                    rank_weights[rank][key] = tensor.clone()

            elif any(k in key for k in [   "mlp.gate.weight","mlp.gate.e_score_correction_bias"]):
                tensor = load_shards_into_memory(key, weight_map,model_dir)[key]
                for rank in range(tp_size): 
                    rank_weights[rank][key] = tensor.clone()


            # --- 2. 按第 0 维切分 (Column Parallel) ---
            elif any(k in key for k in ["self_attn.q_proj.weight",  "self_attn.q_b_proj.weight","self_attn.kv_b_proj.weight"]):
                tensor_dict = load_shards_into_memory([key,key+"_scale"], weight_map,model_dir)
                tensor = tensor_dict[key]
                scale_tensor = tensor_dict[key+ "_scale"] 
                chunk = tensor.shape[0] // tp_size
                for rank in range(tp_size): 
                    rank_weights[rank][key] = tensor[rank * chunk : (rank + 1) * chunk].clone()
                    rank_weights[rank][key + "_scale"] = scale_tensor[rank * chunk : (rank + 1) * chunk].clone()

            # --- 3. 按第 1 维切分 (Row Parallel) ---
            elif "self_attn.o_proj.weight" in key:
                tensor_dict = load_shards_into_memory([key,key+"_scale"], weight_map,model_dir)
                tensor = tensor_dict[key]
                scale_tensor = tensor_dict[key + "_scale"]
                chunk = tensor.shape[1] // tp_size
                for rank in range(tp_size): 
                    rank_weights[rank][key] = tensor[:, rank * chunk : (rank + 1) * chunk].clone()
                    rank_weights[rank][key + "_scale"] = scale_tensor.clone() # Row Parallel 的 scale 不切分，所有 rank 共享同一份

            elif any(k in key for k in [ "mlp.down_proj.weight", "shared_experts.down_proj.weight"]):
                tensor_dict = load_shards_into_memory([key,key+"_scale"], weight_map,model_dir)
                tensor = tensor_dict[key]
                scale_tensor = tensor_dict[key + "_scale"]
                chunk = tensor.shape[1] // global_tp_size
                for rank in range(global_tp_size): 
                    global_rank_weights[rank][key] = tensor[:, rank * chunk : (rank + 1) * chunk].clone()
                    global_rank_weights[rank][key + "_scale"] = scale_tensor.clone() # Row Parallel 的 scale 不切分，所有 rank 共享同一份

            # --- 4. 融合门控并切分 (gate_proj + up_proj -> gate_up_proj) ---
            elif "mlp.gate_proj.weight" in key or "shared_experts.gate_proj.weight" in key:
                up_key = key.replace("gate_proj", "up_proj")
                layer_weights = load_shards_into_memory([key, up_key, key + "_scale", up_key + "_scale",], weight_map,model_dir)
                tensor = layer_weights[key]
                tensor_scale = layer_weights[key + "_scale"]
                up_tensor = layer_weights[up_key]
                up_tensor_scale = layer_weights[up_key + "_scale"]

                
                g_chunk = tensor.shape[0] // global_tp_size
                u_chunk = up_tensor.shape[0] // global_tp_size
                
                for rank in range(global_tp_size):
                    new_key = key.replace("gate_proj", "gate_up_proj")
                    gate_chunk = tensor[rank * g_chunk : (rank + 1) * g_chunk]
                    gate_scale_chunk = tensor_scale[rank * g_chunk : (rank + 1) * g_chunk]
                    up_chunk_tensor = up_tensor[rank * u_chunk : (rank + 1) * u_chunk]
                    up_scale_chunk = up_tensor_scale[rank * u_chunk : (rank + 1) * u_chunk]
                    global_rank_weights[rank][new_key] = torch.cat([gate_chunk, up_chunk_tensor], dim=0).clone()
                    global_rank_weights[rank][new_key + "_scale"] = torch.cat([gate_scale_chunk, up_scale_chunk], dim=0).clone()
                    
            elif "mlp.up_proj.weight" in key or "shared_experts.up_proj.weight" in key:
                # 已在 gate_proj 分支中合并处理，此处跳过
                continue

            elif "self_attn.q_a_proj.weight" in key:
                continue
            
            elif "self_attn.kv_a_proj_with_mqa.weight" in key :
                if model == "v2_lite":
                    tensor_dict = load_shards_into_memory([key,key+"_scale"], weight_map,model_dir)
                    for rank in range(tp_size): 
                        rank_weights[rank][key] = tensor_dict[key].clone()
                        rank_weights[rank][key+"_scale"] = tensor_dict[key+"_scale"].clone()
                elif model == "v3":
                    qa_key = key.replace("self_attn.kv_a_proj_with_mqa", "self_attn.q_a_proj")
                    tensor_dict = load_shards_into_memory([key,key+"_scale", qa_key,qa_key + "_scale"], weight_map,model_dir)
                    tensor = tensor_dict[key]
                    qa_tensor = tensor_dict[qa_key]
                    scale_tensor = tensor_dict[key+ "_scale"] 
                    qa_scale_tensor = tensor_dict[qa_key + "_scale"]
                    new_key = key.replace("kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa")
                    for rank in range(tp_size): 
                        # ReplicatedLinear, 直接将 q_a_proj 和 kv_a_proj_with_mqa 的权重在第 0 维上拼接后复制给所有 rank
                        rank_weights[rank][new_key] = torch.cat([qa_tensor, tensor], dim=0).clone()
                        rank_weights[rank][new_key + "_scale"] = torch.cat([qa_scale_tensor, scale_tensor], dim=0).clone()
            else:
                print(f"remaining key {key}")
        
        for rank in range(tp_size):
            os.makedirs(os.path.join(output_dir, f"rank_{rank}"), exist_ok=True)
            out_layer_path = os.path.join(output_dir, f"rank_{rank}", f"{layer_name}.safetensors")
            save_file(rank_weights[rank], out_layer_path)

        for rank in range(global_tp_size):
            os.makedirs(os.path.join(output_dir, "global", f"rank_{rank}"), exist_ok=True)
            out_layer_path = os.path.join(output_dir, "global",f"rank_{rank}", f"{layer_name}.safetensors")
            save_file(global_rank_weights[rank], out_layer_path)

        del rank_weights
        

 
def split_lm_head_experts(model_dir: str, output_dir: str, tp_size: int = 2):
    """
    专门处理 MoE 专家权重的切分函数。
    1. 自动识别模型中包含的所有 MoE 核心层。
    2. 将离散的 64 个专家打包成 3D 张量。
    3. 按照 Tensor Parallel (Column/Row) 规则切分并独立保存。
    """
    # 加载权重索引
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"找不到索引文件: {index_path}")
        
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    tensor = load_shards_into_memory(["lm_head.weight"], weight_map,model_dir)["lm_head.weight"]

    out_dict = defaultdict(dict)
    chunk = tensor.shape[0] // tp_size
    for rank in range(tp_size):
        out_dict[rank]["lm_head.weight"] = tensor[rank * chunk : (rank + 1) * chunk].clone()
    
    os.makedirs(os.path.join(output_dir, f"lm_head"), exist_ok=True)
    for rank in range(tp_size):
        save_path = os.path.join(output_dir, f"lm_head", f"{rank}.safetensors")
        save_file(out_dict[rank], save_path)

  
    
def split_moe_experts(model_dir: str, output_dir: str, tp_size: int = 2 ):
    """
    专门处理 MoE 专家权重的切分函数。
    1. 自动识别模型中包含的所有 MoE 核心层。
    2. 将离散的 64 个专家打包成 3D 张量。
    3. 按照 Tensor Parallel (Column/Row) 规则切分并独立保存。
    """
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
    sorted_moe_experts = sorted(list(moe_expert_to_keys.keys()), key=lambda x: int(x.split(".")[1]))

    new_index = { "metadata": index_data.get("metadata", {}), "weight_map": {} }
    
    os.makedirs(os.path.join(output_dir, f"moe"), exist_ok=True)

    for expert in sorted_moe_experts:
        print(f"==============================")
        print(f"🚀 正在单独处理 MoE 专家: {expert} ...")
        keys_in_experts = moe_expert_to_keys[expert]
        layers = set([i.split(".")[2] for i in keys_in_experts])
        experts_state_dict = {}

        moe_weights = load_shards_into_memory(keys_in_experts, weight_map, model_dir)

        for layer in sorted([int(i) for i in layers]):
            up_tensor = moe_weights[f'model.layers.{layer}.mlp.{expert}.up_proj.weight']
            up_scale_tensor = moe_weights[f'model.layers.{layer}.mlp.{expert}.up_proj.weight_scale']
            gate_tensor = moe_weights[f'model.layers.{layer}.mlp.{expert}.gate_proj.weight']
            gate_scale_tensor = moe_weights[f'model.layers.{layer}.mlp.{expert}.gate_proj.weight_scale']
            down_tensor = moe_weights[f'model.layers.{layer}.mlp.{expert}.down_proj.weight']
            down_scale_tensor = moe_weights[f'model.layers.{layer}.mlp.{expert}.down_proj.weight_scale']
            
            w13_tensor = torch.cat([gate_tensor, up_tensor], dim=0)
            w13_scale_tensor = torch.cat([gate_scale_tensor, up_scale_tensor], dim=0)
            w2_tensor = down_tensor
            w2_scale_tensor = down_scale_tensor

            # 改成3维,第0维是1
            experts_state_dict[f'model.layers.{layer}.mlp.experts.w13_weight'] = w13_tensor.unsqueeze(0)
            experts_state_dict[f'model.layers.{layer}.mlp.experts.w13_weight_scale'] = w13_scale_tensor.unsqueeze(0)
            experts_state_dict[f'model.layers.{layer}.mlp.experts.w2_weight'] = w2_tensor.unsqueeze(0)
            experts_state_dict[f'model.layers.{layer}.mlp.experts.w2_weight_scale'] = w2_scale_tensor.unsqueeze(0)

        out_file_path = os.path.join(output_dir, f"moe", f"{expert}.safetensors")
        save_file(experts_state_dict, out_file_path)
        
        new_index["weight_map"].update({k: f"moe/{expert}.safetensors" for k in experts_state_dict})

        
        del moe_weights, experts_state_dict

    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSeek MoE 专家权重张量并行切分工具")
    
    # 添加脚本参数
    parser.add_argument("--model_dir", type=str,  default='./',
                        help="原始 DeepSeek 模型权重及索引文件所在的目录路径")
    parser.add_argument("--model", type=str,  default='v2_lite', 
                        help="原始 DeepSeek 模型权重及索引文件所在的目录路径")
    parser.add_argument("--output_dir", type=str, default="./data", 
                        help="切分后权重的保存输出目录 (默认: ./data)")
    parser.add_argument("--tp_size", type=int, default=16, 
                        help="张量并行大小 (Tensor Parallel Size, 默认: 16)")
    parser.add_argument("--global_tp_size", type=int, default=64, 
                        help="张量并行大小 (Tensor Parallel Size, 默认: 64)")
    
    args = parser.parse_args()
    
    print("================ 参数配置 ================")
    print(f"模型目录 (model_dir)  : {args.model_dir}")
    print(f"输出目录 (output_dir) : {args.output_dir}")
    print(f"并行大小 (tp_size)    : {args.tp_size}")
    print("=========================================\n")

    split_global_weights(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        tp_size=args.tp_size,
    )

    # split_layer_weights(
    #     model_dir=args.model_dir,
    #     model = args.model,
    #     output_dir=args.output_dir,
    #     tp_size=args.tp_size,
    #     global_tp_size = args.global_tp_size
    # )

    # split_lm_head_experts(
    #     model_dir=args.model_dir,
    #     output_dir=args.output_dir,
    #     tp_size=args.global_tp_size,
    # )

    
    # split_moe_experts(
    #     model_dir=args.model_dir,
    #     output_dir=args.output_dir,
    #     tp_size=args.tp_size,
    # )

    # # Copy metadata files to output directory
    # for file in os.listdir(args.model_dir):
    #     if file == "model.safetensors.index.json":
    #         continue
    #     if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
    #         if os.path.isdir(os.path.join(args.model_dir, file)):
    #             shutil.copytree(
    #                 os.path.join(args.model_dir, file), os.path.join(args.output_dir, file)
    #             )
    #         else:
    #             shutil.copy(os.path.join(args.model_dir, file), args.output_dir)


# python split_weights.py --model_dir /path-to-deepseek-v2-lite  --output_dir ./data --tp_size 16 --global_tp_size 64

# python split_weights.py --model_dir /path-to-deepseek-v3-int8 --model v3 --output_dir ./data
