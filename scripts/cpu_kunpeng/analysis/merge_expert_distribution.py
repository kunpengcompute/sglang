#!/usr/bin/env python3
"""合并 Kunpeng 各 rank 自动导出的本地专家热度文件。"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="按全部文件累加 logical_count，生成静态 EPLB 可导入的热度文件。"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input_dir.rglob("expert_distribution_recorder_rank_*.pt"))
    if not paths:
        raise SystemExit(f"未在 {args.input_dir} 下找到 rank 本地热度文件")

    total = None
    for path in paths:
        data = torch.load(path, weights_only=True)
        count = data["logical_count"]
        if count.ndim == 2:
            count = count.unsqueeze(0)
        if total is None:
            total = count.clone()
        else:
            if count.shape != total.shape:
                raise ValueError(f"热度形状不一致: {path}: {count.shape} != {total.shape}")
            total += count

    output = {
        "rank": 0,
        "logical_count": total,
        "average_utilization_rate_over_window": None,
        "source_files": len(paths),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"已合并 {len(paths)} 个 rank 累计快照到 {args.output}")
    print(f"logical_count shape={tuple(total.shape)}, total={total.sum().item()}")


if __name__ == "__main__":
    main()
