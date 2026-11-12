#!/usr/bin/env python3
"""Patch a preprocessed GLM-5 weight tree with the shared expert file.

Flash-comm co-locates the shared expert as logical expert ``n_routed`` (256
for GLM-5): the runtime loader reads it from the same per-expert files as
the routed experts (``experts/layer_L/expert_256.safetensors``).  Trees
preprocessed before flash-comm only contain ``expert_0..255``, so loading
fails with "Cannot find expert weight file .../expert_256.safetensors".

This script patches the tree IN PLACE from the ORIGINAL checkpoint's
``model.layers.L.mlp.shared_experts.*`` tensors, reusing the resolution and
gate/up fusion logic of ``split_weights_glm5.py`` — the emitted file format
(2D tensors, keys ``model.layers.L.mlp.experts.{w13_weight,w13_weight_scale,
w2_weight,w2_weight_scale}``) matches what the byte-level split path writes.

Trees patched (MTP tree only when it exists):
    {output_dir}/experts/layer_{L}/expert_{ID}.safetensors
    {output_dir}_mtp/experts/layer_{L-nhl}/expert_{ID}.safetensors

Usage:
    python add_shared_expert_glm5.py \
        --model_dir /path/to/original_checkpoint \
        --output_dir /root/pacific_ext/sglang/models/glm_5_3_shareded_1015 \
        --num_hidden_layers 78
"""

import argparse
import os
import re
import sys
from glob import glob

from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import split_weights_glm5 as sw


def shared_expert_id(layer_dir: str) -> int:
    """Derive the shared-expert id from the existing routed expert files."""
    ids = [
        int(m.group(1))
        for f in glob(os.path.join(layer_dir, "expert_*.safetensors"))
        if (m := re.match(r"expert_(\d+)\.safetensors$", os.path.basename(f)))
    ]
    if not ids:
        raise RuntimeError(f"no routed expert files in {layer_dir}")
    return max(ids) + 1


def build_shared_sd(tensors, gateup, w_down_key, s_down_key, src_layer, prefix):
    if gateup[0] == "fused":
        wk, sk = gateup[1]
        w13 = tensors[wk]
        s13 = tensors[sk] if sk is not None else None
    else:
        _, (wg, gs), (wu, us) = gateup
        w13, s13 = sw.fuse_gate_up(
            tensors[wg],
            tensors[gs] if gs is not None else None,
            tensors[wu],
            tensors[us] if us is not None else None,
            prefix,
        )
    sd = {
        f"model.layers.{src_layer}.mlp.experts.w13_weight": w13.contiguous(),
        f"model.layers.{src_layer}.mlp.experts.w2_weight": tensors[
            w_down_key
        ].contiguous(),
    }
    if s13 is not None:
        sd[f"model.layers.{src_layer}.mlp.experts.w13_weight_scale"] = (
            s13.contiguous()
        )
    if s_down_key is not None:
        sd[f"model.layers.{src_layer}.mlp.experts.w2_weight_scale"] = tensors[
            s_down_key
        ].contiguous()
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="原始 checkpoint 目录")
    ap.add_argument(
        "--output_dir", required=True, help="预处理输出目录 (含 experts/)"
    )
    ap.add_argument(
        "--num_hidden_layers",
        type=int,
        default=78,
        help="用于 _mtp 树的层号偏移 (GLM-5 = 78)",
    )
    ap.add_argument(
        "--layer_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="仅处理 [START, END) 范围的层 (原始层号)",
    )
    ap.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    args = ap.parse_args()

    weight_map = sw.build_weight_map(args.model_dir)
    resolver = sw.WeightResolver(weight_map)

    trees = [(args.output_dir.rstrip("/"), 0)]
    mtp_root = args.output_dir.rstrip("/") + "_mtp"
    if os.path.isdir(os.path.join(mtp_root, "experts")):
        trees.append((mtp_root, args.num_hidden_layers))
        print(f"检测到 MTP 树: {mtp_root}")

    written, skipped = 0, 0
    for root, offset in trees:
        base = os.path.join(root, "experts")
        for layer_dir in sorted(glob(os.path.join(base, "layer_*"))):
            if not os.path.isdir(layer_dir):
                continue
            src_layer = int(os.path.basename(layer_dir)[len("layer_") :]) + offset
            if args.layer_range and not (
                args.layer_range[0] <= src_layer < args.layer_range[1]
            ):
                continue

            sid = shared_expert_id(layer_dir)
            out_path = os.path.join(layer_dir, f"expert_{sid}.safetensors")
            if os.path.exists(out_path) and not args.force:
                skipped += 1
                continue

            prefix = f"model.layers.{src_layer}.mlp.shared_experts"
            gateup = resolver.resolve_gateup(prefix)
            if gateup[0] is None:
                print(
                    f"skip layer {src_layer}: checkpoint has no "
                    f"mlp.shared_experts"
                )
                continue
            w_down_key, s_down_key = resolver.resolve(f"{prefix}.down_proj")
            if w_down_key is None:
                raise KeyError(f"缺少 {prefix}.down_proj")

            pairs = (
                (gateup[1],) if gateup[0] == "fused" else (gateup[1], gateup[2])
            )
            keys = []
            for wk, sk in pairs:
                keys.append(wk)
                if sk is not None:
                    keys.append(sk)
            keys.append(w_down_key)
            if s_down_key is not None:
                keys.append(s_down_key)

            tensors = sw.load_shards_into_memory(
                keys, weight_map, args.model_dir
            )
            sd = build_shared_sd(
                tensors, gateup, w_down_key, s_down_key, src_layer, prefix
            )

            # The FusedMoE slot dims follow the routed experts; a shared
            # expert with a different intermediate size would be silently
            # truncated by the loader, so fail loudly here instead.
            ref = load_file(
                os.path.join(layer_dir, f"expert_{sid - 1}.safetensors")
            )
            ref_w13 = next(v for k, v in ref.items() if k.endswith("w13_weight"))
            ref_w2 = next(v for k, v in ref.items() if k.endswith("w2_weight"))
            new_w13 = next(v for k, v in sd.items() if k.endswith("w13_weight"))
            new_w2 = next(v for k, v in sd.items() if k.endswith("w2_weight"))
            if list(new_w13.shape) != list(ref_w13.shape) or list(
                new_w2.shape
            ) != list(ref_w2.shape):
                raise RuntimeError(
                    f"layer {src_layer}: shared expert shape mismatch "
                    f"(shared w13={list(new_w13.shape)} w2={list(new_w2.shape)} "
                    f"vs routed w13={list(ref_w13.shape)} "
                    f"w2={list(ref_w2.shape)}); the FusedMoE slot cannot "
                    f"hold it"
                )

            save_file(sd, out_path)
            written += 1
            print(
                f"layer {src_layer}: wrote {out_path} "
                f"(w13 {list(new_w13.shape)}, {len(sd)} tensors)"
            )

    print(
        f"\n完成: 写出 {written} 个共享专家文件, 跳过已存在 {skipped} 个"
        + (" (--force 可覆盖)" if skipped else "")
    )


if __name__ == "__main__":
    main()
