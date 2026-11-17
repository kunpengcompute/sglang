#!/usr/bin/env python3
"""合并 Kunpeng 热度并扣除 padding token 的人为热度。

支持两种输入：
- ``--input-dir``：auto-dump 目录，累加所有 ``expert_distribution_recorder_rank_*.pt``；
- ``--input-file``：HTTP 单次 dump 文件，或已经合并过的单文件。

每个输入带 ``pad_logical_count``，求和后从 ``logical_count`` 中减去，得到
只含真实 token 的热度。
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="累加 logical_count 并扣除 padding token 的热度，生成静态 "
        "EPLB 可导入的热度文件。"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        type=Path,
        help="auto-dump 目录（递归查找 expert_distribution_recorder_rank_*.pt）",
    )
    source.add_argument(
        "--input-file",
        type=Path,
        help="HTTP 单次 dump 文件（或本脚本产出、带 padding_subtracted 标记的合并文件）",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--keep-padding",
        action="store_true",
        help="不扣除 padding 热度（仅调试用）",
    )
    args = parser.parse_args()

    if args.input_file is not None:
        if not args.input_file.is_file():
            raise SystemExit(f"输入文件不存在: {args.input_file}")
        paths = [args.input_file]
    else:
        paths = sorted(args.input_dir.rglob("expert_distribution_recorder_rank_*.pt"))
        if not paths:
            raise SystemExit(f"未在 {args.input_dir} 下找到 rank 本地热度文件")

    total = None
    total_pad = None
    files_with_pad = 0
    files_pre_subtracted = 0
    for path in paths:
        data = torch.load(path, weights_only=True)
        count = data["logical_count"]
        if count.ndim == 2:
            count = count.unsqueeze(0)
        if data.get("padding_subtracted"):
            files_pre_subtracted += 1
        if total is None:
            total = count.clone()
        else:
            if count.shape != total.shape:
                raise ValueError(f"热度形状不一致: {path}: {count.shape} != {total.shape}")
            total += count

        pad = data.get("pad_logical_count")
        if pad is not None:
            if pad.ndim == 2:
                pad = pad.unsqueeze(0)
            if total_pad is None:
                total_pad = pad.clone()
            else:
                if pad.shape != total_pad.shape:
                    raise ValueError(
                        f"padding 热度形状不一致: {path}: {pad.shape} != {total_pad.shape}"
                    )
                total_pad += pad
            files_with_pad += 1

    if files_pre_subtracted:
        if files_pre_subtracted != len(paths):
            raise SystemExit(
                "输入中混有已扣除和未扣除 padding 的文件，请统一来源"
            )
        if not args.keep_padding:
            print("输入已标记 padding_subtracted，跳过重复扣除")
        do_subtract = False
    else:
        do_subtract = not args.keep_padding and total_pad is not None
        if do_subtract:
            if files_with_pad != len(paths):
                print(
                    f"WARNING: 只有 {files_with_pad}/{len(paths)} 个输入文件包含 "
                    "pad_logical_count，padding 扣除不完整（混用了旧版本 dump？）"
                )
            if total_pad.shape != total.shape:
                raise ValueError(
                    f"padding 与热度形状不一致: {total_pad.shape} != {total.shape}"
                )
            before = total.clone()
            out_dtype = total.dtype
            total = total - total_pad
            neg = total < 0
            n_neg = int(neg.sum().item())
            if n_neg:
                # 各 rank 的最后一次 dump 时刻不同，可能出现局部负值；夹到 0 并告警。
                worst = int(total.min().item())
                print(
                    f"WARNING: 扣除 padding 后有 {n_neg} 个 cell 为负值"
                    f"（最小值 {worst}），已夹到 0；通常表示各 rank dump 窗口不一致"
                )
                total = torch.clamp(total, min=0)
            total = total.to(out_dtype)
            print(
                f"已扣除 padding 热度: total_before={before.sum().item()}, "
                f"padding={total_pad.sum().item()}, total_after={total.sum().item()}"
            )

    output = {
        "rank": 0,
        "logical_count": total,
        "average_utilization_rate_over_window": None,
        "source_files": len(paths),
    }
    if total_pad is not None:
        output["pad_logical_count"] = total_pad
    if do_subtract or files_pre_subtracted:
        output["padding_subtracted"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"已处理 {len(paths)} 个输入文件，写入 {args.output}")
    print(f"logical_count shape={tuple(total.shape)}, total={total.sum().item()}")


if __name__ == "__main__":
    main()
