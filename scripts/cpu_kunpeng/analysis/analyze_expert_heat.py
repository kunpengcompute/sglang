"""按层分析合并后的专家热度，输出CSV与热力图。

输入端为 merge_expert_distribution.py 的产物：含 ``logical_count``（形状
``(L, E)`` 或 ``(1, L, E)``，已扣除padding）与可选的 ``pad_logical_count``。

输出（默认与输入同目录、同前缀）：
  <stem>_layer_summary.csv   每层总量/均值/极值/CV/max_mean_ratio/TopK热点专家
  <stem>_layer_expert.csv    层×专家 原始计数矩阵
  <stem>_heatmap.png         左：log1p(count)，右：count/本层均值
"""

import argparse
import csv
from pathlib import Path

import torch


def _squeeze_layers(count: torch.Tensor) -> torch.Tensor:
    while count.ndim > 2:
        count = count.sum(dim=0)
    return count.to(torch.float64).cpu()


def _load(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        count = data.get("logical_count")
        pad = data.get("pad_logical_count")
    else:
        count, pad = data, None
    if count is None:
        raise SystemExit(f"{path} 中未找到 logical_count")
    count = _squeeze_layers(count)
    if pad is not None:
        pad = _squeeze_layers(pad)
    return count, pad


def _parse_draft_row(value: str, num_layers: int):
    if value is None or value.lower() in ("", "none"):
        return None
    if value.lower() == "last":
        return num_layers - 1
    idx = int(value)
    if not 0 <= idx < num_layers:
        raise SystemExit(f"--draft-row {idx} 超出层数范围 [0, {num_layers})")
    return idx


def _row_type(layer: int, draft_row) -> str:
    return "draft" if draft_row is not None and layer == draft_row else "target"


def _write_summary(path: Path, count: torch.Tensor, topk: int, draft_row):
    num_layers, num_experts = count.shape
    k = min(topk, num_experts)
    top_values, top_indices = torch.topk(count, k=k, dim=1)
    mean = count.mean(dim=1)
    std = count.std(dim=1, unbiased=False)
    max_value = count.max(dim=1).values
    min_value = count.min(dim=1).values
    total = count.sum(dim=1)
    ratio = max_value / mean.clamp(min=1.0)
    cv = std / mean.clamp(min=1.0)
    nonzero = (count > 0).sum(dim=1)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer",
                "row_type",
                "total",
                "mean",
                "std",
                "cv",
                "max",
                "min",
                "max_mean_ratio",
                "nonzero_experts",
            ]
            + [f"top{i + 1}" for i in range(k)]
        )
        for layer in range(num_layers):
            hot = ";".join(
                f"e{int(idx)}:{int(val)}"
                for val, idx in zip(top_values[layer], top_indices[layer])
            )
            writer.writerow(
                [
                    layer,
                    _row_type(layer, draft_row),
                    int(total[layer]),
                    f"{mean[layer]:.1f}",
                    f"{std[layer]:.1f}",
                    f"{cv[layer]:.4f}",
                    int(max_value[layer]),
                    int(min_value[layer]),
                    f"{ratio[layer]:.3f}",
                    int(nonzero[layer]),
                ]
                + hot.split(";")
            )
    return total, mean, max_value, ratio, cv


def _write_matrix(path: Path, count: torch.Tensor, draft_row):
    num_layers, num_experts = count.shape
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "row_type"] + [f"e{i}" for i in range(num_experts)])
        for layer in range(num_layers):
            writer.writerow(
                [layer, _row_type(layer, draft_row)]
                + [int(v) for v in count[layer]]
            )


def _print_overview(count: torch.Tensor, pad, total, max_value, ratio, cv, draft_row):
    num_layers, num_experts = count.shape
    grand_total = int(total.sum())
    print(f"logical_count: shape=({num_layers}, {num_experts}), total={grand_total}")
    if pad is not None:
        pad_total = int(pad.sum())
        denom = grand_total + pad_total
        ratio_pad = pad_total / denom if denom else 0.0
        print(f"padding: {pad_total} ({ratio_pad:.2%} of raw heat), 已在merge时扣除")
    else:
        print("padding: 文件未包含 pad_logical_count")

    order = torch.argsort(ratio, descending=True)
    print("\n最热层（按 max/本层均值）:")
    for idx in order[:5].tolist():
        print(
            f"  layer {idx:>3} [{_row_type(idx, draft_row)}]  "
            f"total={int(total[idx])}  max={int(max_value[idx])}  "
            f"max/mean={ratio[idx]:.2f}  cv={cv[idx]:.3f}"
        )
    print("\n最均匀层（按 max/本层均值）:")
    for idx in order[-3:].tolist():
        print(
            f"  layer {idx:>3} [{_row_type(idx, draft_row)}]  "
            f"total={int(total[idx])}  max={int(max_value[idx])}  "
            f"max/mean={ratio[idx]:.2f}  cv={cv[idx]:.3f}"
        )
    if draft_row is not None:
        d = draft_row
        print(
            f"\ndraft行 layer {d}: total={int(total[d])}  max={int(max_value[d])}  "
            f"max/mean={ratio[d]:.2f}  cv={cv[d]:.3f}"
        )

    expert_total = count.sum(dim=0)
    top_values, top_indices = torch.topk(expert_total, k=min(10, num_experts))
    print("\n全局最热专家（跨层累计占比）:")
    for val, idx in zip(top_values.tolist(), top_indices.tolist()):
        share = val / grand_total if grand_total else 0.0
        print(f"  e{idx:<3} count={int(val):>12}  share={share:.2%}")


def _plot(count: torch.Tensor, mean: torch.Tensor, path: Path, draft_row, title: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过热力图（pip install matplotlib 后可重跑）")
        return

    num_layers, num_experts = count.shape
    log_count = torch.log1p(count)
    ratio = count / mean.unsqueeze(1).clamp(min=1.0)

    height = max(6.0, num_layers * 0.2)
    fig, axes = plt.subplots(1, 2, figsize=(18, height), constrained_layout=True)
    images = [
        (log_count, "log1p(count)", "viridis", None, None),
        (ratio, "count / layer mean", "magma", 0.0, 2.0),
    ]
    y_labels = [
        f"{i}*" if draft_row is not None and i == draft_row else str(i)
        for i in range(num_layers)
    ]
    y_step = max(1, num_layers // 25)
    for ax, (matrix, subtitle, cmap, vmin, vmax) in zip(axes, images):
        image = ax.imshow(
            matrix.numpy(),
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(subtitle)
        ax.set_xlabel("logical expert")
        ax.set_ylabel("layer")
        ax.set_yticks(range(0, num_layers, y_step))
        ax.set_yticklabels(y_labels[::y_step], fontsize=7)
        ax.set_xticks(range(0, num_experts, max(1, num_experts // 8)))
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    if draft_row is not None:
        for ax in axes:
            ax.axhline(draft_row, color="cyan", linewidth=0.6, linestyle="--")
    fig.suptitle(title + ("  (* = draft row)" if draft_row is not None else ""))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="按层分析专家热度：输出汇总CSV、层×专家矩阵CSV和热力图。"
    )
    parser.add_argument("--input", type=Path, required=True, help="合并后的 .pt 文件")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认与输入同目录",
    )
    parser.add_argument("--topk", type=int, default=10, help="每层输出的热点专家数")
    parser.add_argument(
        "--draft-row",
        default="none",
        help="draft行位置：none、last 或具体层号（仅影响标注，不改变数据）",
    )
    parser.add_argument("--no-plot", action="store_true", help="只输出CSV，不画热力图")
    args = parser.parse_args()

    count, pad = _load(args.input)
    num_layers = count.shape[0]
    draft_row = _parse_draft_row(args.draft_row, num_layers)

    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    summary_path = out_dir / f"{stem}_layer_summary.csv"
    matrix_path = out_dir / f"{stem}_layer_expert.csv"
    plot_path = out_dir / f"{stem}_heatmap.png"

    total, mean, max_value, ratio, cv = _write_summary(
        summary_path, count, args.topk, draft_row
    )
    _write_matrix(matrix_path, count, draft_row)
    _print_overview(count, pad, total, max_value, ratio, cv, draft_row)
    if not args.no_plot:
        _plot(count, mean, plot_path, draft_row, args.input.name)

    print(f"\n输出: {summary_path}")
    print(f"输出: {matrix_path}")
    if not args.no_plot:
        print(f"输出: {plot_path}")


if __name__ == "__main__":
    main()
