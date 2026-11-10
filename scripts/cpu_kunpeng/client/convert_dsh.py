import os
import json
import argparse
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="将 DSH JSONL 轨迹转换为模型输入 Prompt（支持从头严格截断）")
    parser.add_argument("--model", "-m", type=str, required=True, help="模型路径或 HuggingFace 名称")
    parser.add_argument("--trace", "-t", type=str, required=True, help="输入的 DSH JSONL 轨迹文件路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出的 txt 文件路径")
    parser.add_argument("--length", "-l", type=int, default=None, help="可选：目标 Token 长度上限。如果指定，超出则直接回退截断。")
    return parser.parse_args()


def parse_dsh_trace(filepath: str) -> list:
    """解析 DSH JSONL 事件流，将其还原成标准的 messages 格式。"""
    messages = []
    current_assistant_parts = []

    def flush_assistant():
        if current_assistant_parts:
            messages.append({
                "role": "assistant",
                "content": "".join(current_assistant_parts)
            })
            current_assistant_parts.clear()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            etype = event.get("type")
            
            if etype == "system/message":
                flush_assistant()
                content_arr = event.get("data", {}).get("message", {}).get("content", [])
                text = "".join(c.get("text", "") for c in content_arr if c.get("type") == "text")
                if text:
                    if messages and messages[-1]["role"] == "system":
                        messages[-1]["content"] += "\n" + text
                    else:
                        messages.append({"role": "system", "content": text})
                        
            elif etype == "user/message":
                flush_assistant()
                content_arr = event.get("data", {}).get("content", [])
                text = "".join(c.get("text", "") for c in content_arr if c.get("type") == "text")
                if text:
                    messages.append({"role": "user", "content": text})
                    
            elif etype == "assistant/message":
                msg_data = event.get("data", {}).get("message", {})
                content_arr = msg_data.get("content", [])
                for c in content_arr:
                    ctype = c.get("type")
                    if ctype == "reasoning" and c.get("text"):
                        current_assistant_parts.append(f"<thought>\n{c.get('text')}\n</thought>\n")
                    elif ctype == "text":
                        current_assistant_parts.append(c.get("text", ""))
                    elif ctype == "tool-call":
                        current_assistant_parts.append(f"\n```bash\n{c.get('name')} {c.get('arguments')}\n```\n")
                        
            elif etype == "tool/result":
                flush_assistant()
                msg_data = event.get("data", {}).get("message", {})
                content_arr = msg_data.get("content", [])
                text = "".join(c.get("text", "") for c in content_arr if c.get("type") == "text")
                if text:
                    messages.append({"role": "user", "content": f"Tool Result:\n{text}"})
                    
    flush_assistant()
    return messages


def truncate_messages_forward(messages: list, max_tokens: int, tokenizer) -> list:
    """
    从头往后按顺序装填消息。
    每次加入一条消息后，使用 apply_chat_template 真实渲染并计算 token 数。
    如果超出 max_tokens，则回退，丢弃该条及之后的所有消息。
    """
    if not messages:
        return messages

    kept_msgs = []
    
    for msg in messages:
        # 尝试将当前消息加进去
        kept_msgs.append(msg)
        
        # 真实渲染并计算精准的 Token 数（包含了特殊标记）
        temp_prompt = tokenizer.apply_chat_template(
            kept_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        current_tokens = len(tokenizer.encode(temp_prompt))
        
        # 如果超出了限制，回退这一步，并直接退出
        if current_tokens > max_tokens:
            kept_msgs.pop()
            break

    return kept_msgs


def main():
    args = parse_args()
    
    print(f"加载 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print(f"解析轨迹文件: {args.trace}")
    messages = parse_dsh_trace(args.trace)
    
    if not messages:
        print("错误: 未从轨迹中解析到任何有效消息。")
        return

    original_len = len(messages)

    # 如果指定了长度，进行从头往后的严格截断
    if args.length is not None:
        print(f"执行截断: 目标上限 {args.length} tokens (从头开始保留)")
        messages = truncate_messages_forward(messages, args.length, tokenizer)

    # 使用模型的 chat template 渲染最终结果
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    token_count = len(tokenizer.encode(prompt))
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(prompt)
        
    print(f"转换完成！")
    if args.length is not None:
        print(f"原始消息条数: {original_len} -> 截断后保留: {len(messages)}")
    print(f"最终 Token 数: {token_count}")
    print(f"文件已保存至: {args.output}")


if __name__ == "__main__":
    main()