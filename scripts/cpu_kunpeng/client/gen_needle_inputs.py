#!/usr/bin/env python3
"""生成 GLM 大海捞针（JSON 干草堆）测试输入。

按目标 token 数生成 prompt：先校准「每个 key-value 对」的 token 开销，
再线性外推 + 迭代修正 pair 数量，使实际 token 数贴近目标。
生成结果的 token 数绝不超过目标（宁可少一对），pair 数量没有预设上限。
"""

import os
import sys
import json
import math
import uuid
import random
import argparse
from transformers import AutoTokenizer

# ================= 配置 =================
CALIB_PAIRS = 64          # 校准用的样本对数（小样本，编码很快）
TOLERANCE = 0.001         # 允许「低于目标」的相对比例（0.1%），绝不允许超出
MAX_ROUNDS = 12           # 线性逼近 / 边界探测的最大轮数
# =======================================


def _rand_uuid(rng: random.Random) -> str:
    """用受 seed 控制的随机源生成 uuid，保证同一 seed 结果可复现。"""
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


class Haystack:
    """惰性生成、前缀固定的 JSON 干草堆。

    token 数随 pair 数近似线性，逼近目标时要求「取前 n 对」的内容稳定；
    若每次重新随机生成整堆，uuid 长度波动会引入噪声，导致难以收敛。
    """

    def __init__(self, seed: int):
        rng = random.Random(seed)
        self._needle_frac = rng.random()  # needle 在干草堆中的相对位置
        self._rng = random.Random(seed + 1)
        self._pairs = []

    def _ensure(self, n: int) -> None:
        while len(self._pairs) < n:
            self._pairs.append((_rand_uuid(self._rng), _rand_uuid(self._rng)))

    def slice(self, n: int) -> dict:
        self._ensure(n)
        return dict(self._pairs[:n])

    def needle_key(self, n: int) -> str:
        self._ensure(n)
        return self._pairs[int(self._needle_frac * n)][0]


def build_user_content(json_data: dict, needle_key: str) -> str:
    """JSON 用 indent=2 保持多行缩进。"""
    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
    tail = (
        f'\nQ: \nKey: "{needle_key}"\n'
        f'The value associated with the specified key is: '
    )
    return f"JSON data:\n{json_str}{tail}"


def build_messages(json_data: dict, needle_key: str) -> list:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": build_user_content(json_data, needle_key)},
    ]


def render_prompt(tokenizer, json_data: dict, needle_key: str) -> str:
    messages = build_messages(json_data, needle_key)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def encoded_len(tokenizer, prompt: str) -> int:
    """统计 token 数，明确禁止按 model_max_length 截断超长 prompt。"""
    try:
        tokens = tokenizer.encode(prompt, truncation=False)
    except TypeError:  # 少数自定义 tokenizer 的 encode 不接受该关键字
        tokens = tokenizer.encode(prompt)
    return len(tokens)


def generate_one(
    tokenizer,
    target_tokens: int,
    output_dir: str,
    seed: int = 42,
    tolerance: float = TOLERANCE,
) -> tuple:
    """生成指定目标 token 长度的一个 prompt 文件。

    返回 (文件路径, 是否达标)；无法满足「不超限」时返回 (None, False) 且不写文件。
    pair 数按 token 数动态外推，无硬编码上限；实际 token 数恒 <= target_tokens。
    """
    if target_tokens <= 0:
        raise ValueError(f"目标 token 数必须为正: {target_tokens}")

    haystack = Haystack(seed)
    counts = {}  # n -> token 数，避免重复编码同一规模

    def render(n: int) -> str:
        return render_prompt(tokenizer, haystack.slice(n), haystack.needle_key(n))

    def measure(n: int) -> tuple:
        n = max(1, int(n))
        if n not in counts:
            counts[n] = encoded_len(tokenizer, render(n))
        return n, counts[n]

    # 1) 用两个小样本校准「每对平均 token 数」与「固定开销」
    _, c1 = measure(CALIB_PAIRS)
    _, c2 = measure(2 * CALIB_PAIRS)
    slope = (c2 - c1) / CALIB_PAIRS
    if slope <= 0:
        raise RuntimeError(
            "token 数未随 pair 数增长，请检查 tokenizer 是否对长文本做了截断"
        )
    base = c1 - slope * CALIB_PAIRS  # chat template / 大括号 / 问题尾巴等固定开销

    # 2) 线性外推 + 迭代修正。token 数 ≈ base + n * 每对平均 token 数，
    #    一般 2~3 轮即可收敛；每轮用实测的全局平均斜率修正，抗 uuid 长度噪声。
    #    一对约 slope 个 token，精度不可能优于这个粒度，容差至少要覆盖它。
    n = max(1, round((target_tokens - base) / slope))
    tol = max(1.0, target_tokens * tolerance, slope)
    stop_at = max(1.0, slope)  # 尽量收敛到「一对」以内，而非刚进容差就停
    best_n, best_err = None, None
    for _ in range(MAX_ROUNDS):
        n, count = measure(n)
        err = count - target_tokens
        if best_err is None or abs(err) < abs(best_err):
            best_n, best_err = n, err
        if abs(err) <= stop_at:
            break

        cur_slope = (count - base) / n
        if cur_slope <= 0:
            cur_slope = slope
        delta = (target_tokens - count) / cur_slope
        step = int(round(delta)) if abs(delta) >= 1 else (1 if delta > 0 else -1)
        nxt = max(1, n + step)
        # 到最短 prompt、或回到已测过且更差的点时停止，避免来回震荡
        if nxt == n or (nxt in counts and abs(counts[nxt] - target_tokens) >= abs(err)):
            break
        n = nxt

    # 3) 钳制到「绝不超过目标」。count(n) 随 n 单调递增，先找到相邻边界
    #    (lo_ok 不超限, hi_bad 已超限)，再二分取最大的不超限 pair 数。
    def clamp_to_target(n: int, c: int) -> tuple:
        # 向下退到不超限
        while c > target_tokens and n > 1:
            step = max(1, math.ceil((c - target_tokens) / slope))
            n = max(1, n - step)
            _, c = measure(n)
        if c > target_tokens:  # 连最短 prompt 都超限，无解
            return n, c

        lo_n, lo_c = n, c
        hi_n = None
        for _ in range(MAX_ROUNDS):
            # 故意多跨一对越过目标，制造一个已知的超限上界
            step = max(1, int((target_tokens - lo_c) / slope) + 1)
            nxt = lo_n + step
            _, c2 = measure(nxt)
            if c2 > target_tokens:
                hi_n = nxt
                break
            lo_n, lo_c = nxt, c2

        if hi_n is not None:
            while hi_n - lo_n > 1:
                mid = (lo_n + hi_n) // 2
                _, mc = measure(mid)
                if mc <= target_tokens:
                    lo_n, lo_c = mid, mc
                else:
                    hi_n = mid
        return lo_n, lo_c

    best_n, _ = clamp_to_target(best_n, counts[best_n])

    # 4) 最终复核：对真正要写出去的字符串再数一次，超限就逐对回退。
    #    render 是确定性的，通常一次即通过；这一步保证产出绝不超限。
    prompt = render(best_n)
    count = encoded_len(tokenizer, prompt)
    tries = 0
    while count > target_tokens and best_n > 1 and tries < MAX_ROUNDS:
        best_n -= 1
        prompt = render(best_n)
        count = encoded_len(tokenizer, prompt)
        tries += 1
    if count > target_tokens:
        print(
            f"[ERROR] 目标 {target_tokens} tokens，但最短 prompt 也有 "
            f"{count} tokens，无法生成不超限的文件",
            file=sys.stderr,
        )
        return None, False

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"needle_{target_tokens}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    short = target_tokens - count  # 恒 >= 0
    pct = 100.0 * short / target_tokens
    ok = short <= max(tol, target_tokens * 0.01)
    print(
        f"[目标 {target_tokens:>7}] 实际 {count:>7} tokens "
        f"(-{short}, -{pct:.3f}%), {best_n} 对, "
        f"{'OK' if ok else '未达标'}, 文件: {out_path}"
    )
    if not ok:
        print(
            f"[ERROR] 比目标少 {short} tokens；请检查 tokenizer 是否截断长文本",
            file=sys.stderr,
        )
    elif pct > 1.0 and short <= slope:
        print(
            f"[WARN] 目标较小，一对约 {slope:.0f} tokens，已取不超过目标的最长长度",
            file=sys.stderr,
        )
    elif short > tol:
        print(f"[WARN] 比容差 {tol:.0f} tokens 少得更多，但已在 1% 以内", file=sys.stderr)
    return out_path, ok


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成 GLM-5.3 大海捞针测试输入（JSON 干草堆）；"
                    "实际 token 数绝不超过目标，可略少"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="GLM-5.3 模型路径或 HF 名称，例如 /data/models/GLM-5.3",
    )
    parser.add_argument(
        "--length", "-l",
        type=int,
        nargs="+",
        required=True,
        help="目标 token 长度，可传多个，例如 -l 8192 32768 131072 800000",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./needle_inputs",
        help="输出目录，例如 ./needle_inputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE,
        help=f"允许低于目标的最大比例（默认 {TOLERANCE}，即 "
             f"{TOLERANCE * 100:.1f} percent，实际下限为一对的 token 数）；"
             f"token 数永不超过目标",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tolerance <= 0:
        sys.exit("--tolerance 必须大于 0")

    print(f"加载 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )

    failed = []
    for target in args.length:
        _, ok = generate_one(
            tokenizer=tokenizer,
            target_tokens=target,
            output_dir=args.output_dir,
            seed=args.seed,
            tolerance=args.tolerance,
        )
        if not ok:
            failed.append(target)

    if failed:
        sys.exit(f"以下长度未达标: {failed}")


if __name__ == "__main__":
    main()
