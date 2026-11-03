import os
import json
import uuid
import random
import argparse
from transformers import AutoTokenizer

# ================= 配置 =================
NUM_PAIRS = 2000          # 初始 pair 数，二分会在此基础上调整
TOLERANCE = 0.05          # token 数误差容忍度（5%）
# =======================================


def gen_uuid() -> str:
    return str(uuid.uuid4())


def build_json_haystack(num_pairs: int) -> dict:
    return {gen_uuid(): gen_uuid() for _ in range(num_pairs)}


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


def generate_one(tokenizer, target_tokens: int, output_dir: str, seed: int = 42) -> str:
    """生成指定目标 token 长度的一个 prompt 文件，返回文件路径。"""
    random.seed(seed)

    json_data = build_json_haystack(NUM_PAIRS)
    needle_key = random.choice(list(json_data.keys()))
    prompt = render_prompt(tokenizer, json_data, needle_key)
    token_count = len(tokenizer.encode(prompt))

    # 二分调整 pair 数量逼近目标 token 数
    lo, hi = 1, NUM_PAIRS * 4
    for _ in range(15):
        if abs(token_count - target_tokens) < target_tokens * TOLERANCE:
            break
        if token_count > target_tokens:
            hi = len(json_data)
        else:
            lo = len(json_data)

        new_num = (lo + hi) // 2
        if new_num == len(json_data) or new_num < 1:
            break

        json_data = build_json_haystack(new_num)
        needle_key = random.choice(list(json_data.keys()))
        prompt = render_prompt(tokenizer, json_data, needle_key)
        token_count = len(tokenizer.encode(prompt))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"needle_{target_tokens}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"[目标 {target_tokens:>7}] 实际 {token_count:>7} tokens, "
          f"{len(json_data)} 对, 文件: {out_path}")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成 GLM-5.3 大海捞针测试输入（JSON 干草堆）"
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
        help="目标 token 长度，可传多个，例如 -l 8192 32768 131072",
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
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"加载 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )

    for target in args.length:
        generate_one(
            tokenizer=tokenizer,
            target_tokens=target,
            output_dir=args.output_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()