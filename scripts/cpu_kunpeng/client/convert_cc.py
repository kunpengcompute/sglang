import os
import json
import argparse
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="将 Claude Code JSONL 轨迹转换为模型输入 Prompt")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--trace", "-t", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--length", "-l", type=int, default=None)
    parser.add_argument("--no-system", action="store_true", help="丢弃 system 消息")
    return parser.parse_args()


def _text_of(content) -> str:
    """str / [{type:text,...}] / [{type:thinking,...}] / [{type:tool_use,...}] -> 纯文本"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for c in content:
        if not isinstance(c, dict):
            continue
        ct = c.get("type")
        if ct == "text":
            t = c.get("text", "")
            if t:
                parts.append(t)
        elif ct == "thinking" and c.get("thinking"):
            parts.append(f"<think>{c['thinking']}</think>")
        elif ct == "tool_use":
            name = c.get("name", "tool")
            try:
                args = json.dumps(c.get("input", {}), ensure_ascii=False)
            except Exception:
                args = str(c.get("input"))
            parts.append(f"```bash\n{name} {args}\n```")
        elif ct == "tool_result":
            inner = _text_of(c.get("content"))
            if c.get("is_error"):
                parts.append(f"Tool Result (error):\n{inner}" if inner else "Tool Result (error)")
            else:
                parts.append(f"Tool Result:\n{inner}" if inner else "Tool Result")
    return "\n".join(p for p in parts if p)


def parse_claude_trace(filepath: str) -> list:
    """
    按时间序解析 Claude Code JSONL，返回 messages: [{role, content}]。
    - system 来自 api-request-shape 的 system 数组（取第一条即可，Claude 的 system 稳定）
    - 对话按事件顺序：user / assistant / attachment(renderedRole=user)
    - 跳过 mode/permission-mode/file-history-snapshot/atis-latch/
      api-request/api-request-blob/tools-used/last-prompt/cost-state
    """
    system_parts = []
    messages = []
    current_assistant_parts = []

    def flush_assistant():
        if current_assistant_parts:
            text = "\n".join(p for p in current_assistant_parts if p)
            if text:
                if messages and messages[-1]["role"] == "assistant":
                    messages[-1]["content"] += "\n" + text
                else:
                    messages.append({"role": "assistant", "content": text})
            current_assistant_parts.clear()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            et = ev.get("type")

            # ---------- system 来源 ----------
            if et == "api-request-shape":
                shape = ev.get("shape") or {}
                for s in shape.get("system", []):
                    if isinstance(s, dict) and s.get("type") == "text":
                        t = s.get("text", "")
                        if t:
                            system_parts.append(t)
                # 只取第一条 shape 即可，Claude 的 system 在会话内稳定
                continue

            # ---------- 对话事件 ----------
            if et == "user":
                flush_assistant()
                msg = ev.get("message", {}) or {}
                txt = _text_of(msg.get("content"))
                if txt.strip():
                    if messages and messages[-1]["role"] == "user":
                        messages[-1]["content"] += "\n" + txt
                    else:
                        messages.append({"role": "user", "content": txt})

            elif et == "assistant":
                msg = ev.get("message", {}) or {}
                content_arr = msg.get("content", [])
                if isinstance(content_arr, str):
                    if content_arr.strip():
                        current_assistant_parts.append(content_arr.strip())
                elif isinstance(content_arr, list):
                    for c in content_arr:
                        if not isinstance(c, dict):
                            continue
                        ct = c.get("type")
                        if ct == "thinking" and c.get("thinking"):
                            current_assistant_parts.append(f"<think>{c['thinking']}</think>")
                        elif ct == "text":
                            t = c.get("text", "")
                            if t:
                                current_assistant_parts.append(t)
                        elif ct == "tool_use":
                            name = c.get("name", "tool")
                            try:
                                args = json.dumps(c.get("input", {}), ensure_ascii=False)
                            except Exception:
                                args = str(c.get("input"))
                            current_assistant_parts.append(f"```bash\n{name} {args}\n```")

            elif et == "attachment":
                role = ev.get("renderedRole")
                if role not in ("user", "system"):
                    continue
                rendered = ev.get("rendered") or []
                for r in rendered:
                    if isinstance(r, dict):
                        t = r.get("content", "")
                        if t and t.strip():
                            if role == "system":
                                system_parts.append(t)
                            else:
                                flush_assistant()
                                if messages and messages[-1]["role"] == "user":
                                    messages[-1]["content"] += "\n" + t
                                else:
                                    messages.append({"role": "user", "content": t})

            # 其他 type 全部跳过

    flush_assistant()

    # 对话应以 user 开头
    while messages and messages[0]["role"] != "user":
        messages.pop(0)

    out = []
    if system_parts:
        out.append({"role": "system", "content": "\n\n".join(system_parts)})
    out.extend(messages)
    return out


def truncate_messages_forward(messages: list, max_tokens: int, tokenizer) -> list:
    if not messages:
        return messages
    kept = []
    for msg in messages:
        kept.append(msg)
        prompt = tokenizer.apply_chat_template(kept, tokenize=False, add_generation_prompt=True)
        if len(tokenizer.encode(prompt)) > max_tokens:
            kept.pop()
            break
    return kept


def main():
    args = parse_args()

    print(f"加载 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"解析轨迹文件: {args.trace}")
    messages = parse_claude_trace(args.trace)

    if args.no_system:
        messages = [m for m in messages if m["role"] != "system"]

    if not messages:
        print("错误: 未从轨迹中解析到任何有效消息。")
        return

    original_len = len(messages)

    if args.length is not None:
        print(f"执行截断: 目标上限 {args.length} tokens")
        messages = truncate_messages_forward(messages, args.length, tokenizer)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    token_count = len(tokenizer.encode(prompt))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(prompt)

    print("转换完成！")
    if args.length is not None:
        print(f"原始消息条数: {original_len} -> 截断后保留: {len(messages)}")
    print(f"最终 Token 数: {token_count}")
    print(f"文件已保存至: {args.output}")


if __name__ == "__main__":
    main()