import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def weight_quant(tensor: torch.Tensor):
    if tensor.dim() != 2:
        raise ValueError(
            f"Weight quant only supports 2D tensor, but got {tensor.shape}"
        )
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    if scale.shape != (tensor.shape[0], 1):
        raise ValueError(
            f"Scale shape {scale.shape} does not match tensor shape {tensor.shape}"
        )
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def copy_file(src_dir, dst_dir, filename):
    """
    Copy a file with given filename from src_dir to dst_dir.
    Assumes both src_dir and dst_dir already exist.
    """
    src_file = os.path.join(src_dir, filename)
    dst_file = os.path.join(dst_dir, filename)

    shutil.copy2(src_file, dst_file)  # preserves metadata


def copy_metadata_files(src_dir, dst_dir):
    all_files = os.listdir(src_dir)
    metadata_files = [f for f in all_files if not f.endswith("safetensors")]
    metadata_files = [
        f for f in metadata_files if os.path.isfile(os.path.join(src_dir, f))
    ]
    for file in metadata_files:
        copy_file(src_dir, dst_dir, file)


def main(bf16_path, int8_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(int8_path, exist_ok=True)

    copy_metadata_files(bf16_path, int8_path)

    model_index_file = os.path.join(int8_path, "model.safetensors.index.json")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    quant_layer_types = [
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "q_proj",
        "o_proj",
        "down_proj",
        "gate_proj",
        "up_proj",
    ]

    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()
    quant_count = 0
    new_weight_map = {}
    for safetensor_file in safetensor_files:
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device="cpu")
        new_state_dict = {}
        for weight_name, weight in tqdm(state_dict.items()):
            weight_name_split = weight_name.split(".")
            quant_type = weight_name_split[-2] if len(weight_name_split) > 2 else "None"
            if quant_type in quant_layer_types:
                if weight.element_size() != 2:
                    raise ValueError(
                        f"Weight quant only supports 2-byte tensor, but got {weight_name} with element size {weight.element_size()}"
                    )
                quant_count += 1
                int8_weight, scale_inv = weight_quant(weight)
                new_state_dict[weight_name] = int8_weight
                new_scale_name = weight_name + "_scale"
                new_state_dict[new_scale_name] = scale_inv

                new_weight_map[weight_name] = file_name
                new_weight_map[new_scale_name] = file_name
            else:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
        new_safetensor_file = os.path.join(int8_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
    print(f"{quant_count} weights are quantized.")

    # modify model.safetensors.index.json
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    model_index["weight_map"] = new_weight_map
    with open(model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"model.safetensors.index.json modified and saved to {model_index_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True)
    parser.add_argument("--output-int8-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_bf16_hf_path, args.output_int8_hf_path)
    print("done")
