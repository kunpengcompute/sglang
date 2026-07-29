import argparse
import io
import os

import pandas as pd


def parse_mixed_profile(filepath):
    """从单个文件中提取并解析 ## extend 和 ## decode 两个段落的数据"""
    extend_lines = []
    decode_lines = []
    current_section = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # 匹配段落标记
            if stripped.startswith("## extend") or stripped.startswith("# extend"):
                current_section = "extend"
                continue
            elif stripped.startswith("## decode") or stripped.startswith("# decode"):
                current_section = "decode"
                continue

            if current_section == "extend":
                extend_lines.append(line)
            elif current_section == "decode":
                decode_lines.append(line)

    def process_df(lines):
        if not lines:
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO("".join(lines)))
        # 重命名第一列为 operator
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "operator"})
        df["operator"] = df["operator"].fillna("unknown").astype(str).str.strip()
        return df

    return process_df(extend_lines), process_df(decode_lines)


def merge_section(df1, df2, name1, name2):
    """对比两份数据的同一个阶段（如对比 run1 的 extend 与 run2 的 extend）"""
    if df1.empty and df2.empty:
        return pd.DataFrame()

    # 对齐算子名称（Outer Join，防止漏算独特算子）
    merged = pd.merge(
        df1, df2, on="operator", how="outer", suffixes=(f"_{name1}", f"_{name2}")
    )

    # 填充类型与空值
    num_cols = ["count", "min_us", "max_us", "avg_us", "total_ms"]
    for col in num_cols:
        c1, c2 = f"{col}_{name1}", f"{col}_{name2}"
        if c1 in merged.columns:
            merged[c1] = pd.to_numeric(merged[c1], errors="coerce").fillna(0)
        if c2 in merged.columns:
            merged[c2] = pd.to_numeric(merged[c2], errors="coerce").fillna(0)

    # 计算差异 (File2 - File1)
    col_t1, col_t2 = f"total_ms_{name1}", f"total_ms_{name2}"
    col_a1, col_a2 = f"avg_us_{name1}", f"avg_us_{name2}"

    merged[f"total_ms_diff({name2}-{name1})"] = (merged[col_t2] - merged[col_t1]).round(
        3
    )
    merged[f"avg_us_diff({name2}-{name1})"] = (merged[col_a2] - merged[col_a1]).round(1)

    # 分离 total 汇总行
    is_total = merged["operator"].str.lower() == "total"
    df_ops = merged[~is_total].copy()
    df_total = merged[is_total].copy()

    # 按最高耗时降序排序
    df_ops["_max_t"] = df_ops[[col_t1, col_t2]].max(axis=1)
    df_ops = df_ops.sort_values(by="_max_t", ascending=False).drop(columns=["_max_t"])

    result_df = pd.concat([df_ops, df_total], ignore_index=True)

    # 调整列顺序
    cols = ["operator"]
    for m in ["count", "total_ms", "percent", "avg_us", "min_us", "max_us"]:
        c1, c2 = f"{m}_{name1}", f"{m}_{name2}"
        if c1 in result_df.columns and c2 in result_df.columns:
            cols.extend([c1, c2])
            if m == "total_ms":
                cols.append(f"total_ms_diff({name2}-{name1})")
            elif m == "avg_us":
                cols.append(f"avg_us_diff({name2}-{name1})")

    remaining = [c for c in result_df.columns if c not in cols]
    return result_df[cols + remaining]


def main():
    parser = argparse.ArgumentParser(
        description="比较两份包含 ## extend 和 ## decode 的 Profiling 文件"
    )
    parser.add_argument("file1", help="第一个 Profiling 文件 (例如 run1.txt)")
    parser.add_argument("file2", help="第二个 Profiling 文件 (例如 run2.txt)")
    parser.add_argument(
        "-o",
        "--output_prefix",
        default="compare",
        help="输出文件名前缀 (默认: compare)",
    )
    parser.add_argument(
        "--name1", default="run1", help="文件1在列名中的别名 (默认: run1)"
    )
    parser.add_argument(
        "--name2", default="run2", help="文件2在列名中的别名 (默认: run2)"
    )

    args = parser.parse_args()

    # 1. 解析两个文件
    ext1, dec1 = parse_mixed_profile(args.file1)
    ext2, dec2 = parse_mixed_profile(args.file2)

    # 2. 分别比较 Extend 阶段和 Decode 阶段
    res_extend = merge_section(ext1, ext2, args.name1, args.name2)
    res_decode = merge_section(dec1, dec2, args.name1, args.name2)

    # 3. 导出结果
    out_ext_path = f"{args.output_prefix}_extend_summary.csv"
    out_dec_path = f"{args.output_prefix}_decode_summary.csv"

    res_extend.to_csv(out_ext_path, index=False, encoding="utf-8-sig")
    res_decode.to_csv(out_dec_path, index=False, encoding="utf-8-sig")

    print(f"对比完成！")
    print(f" -> Extend 阶段对比已生成: {out_ext_path}")
    print(f" -> Decode 阶段对比已生成: {out_dec_path}")


if __name__ == "__main__":
    main()
