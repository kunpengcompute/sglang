"""
Run few-shot GSM-8K evaluation.

.. deprecated::
    This module is deprecated. Use ``sglang.test.run_eval`` with
    ``eval_name="gsm8k"`` instead, which routes through the unified
    Chat API evaluation framework with dump_metric support.

Usage:
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
"""

import argparse
import ast
import logging
import os
import re
import time

import numpy as np

from sglang.lang.api import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

# Tokenizer path comes from the MODEL_PATH env var (set by scripts/cpu_kunpeng/env.sh)
model_path = os.environ.get("MODEL_PATH")
tokenizer = AutoTokenizer.from_pretrained(model_path) if model_path else None

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def run_eval(args):
    # Select backend
    set_default_backend(RuntimeEndpoint(f"{args.host}:{args.port}"))

    if args.data_path is None:
        # Read data
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        filename = download_and_cache_file(url)
    else:
        filename = args.data_path

    lines = list(read_jsonl(filename))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]
    logger.info(
        "Prepared %d questions (num_shots=%d, max_new_tokens=%d)",
        len(arguments),
        num_shots,
        args.max_new_tokens,
    )

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen(
            "answer",
            max_tokens=args.max_new_tokens,
            stop=["Question", "Assistant:", "<|separator|>"],
        )

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=args.temperature if hasattr(args, "temperature") else 0,
        num_threads=args.parallel,
        progress_bar=True,
        return_logprob=getattr(args, "return_logprob", None),
        logprob_start_len=getattr(args, "logprob_start_len", None),
    )
    latency = time.perf_counter() - tic

    # Per-request results
    preds = []
    for i in range(len(states)):
        answer_text = states[i]["answer"]
        pred = get_answer_value(answer_text)
        preds.append(pred)

        if pred == INVALID:
            judge = "INVALID (no number extracted)"
        elif pred == labels[i]:
            judge = "RIGHT"
        else:
            judge = "ERROR"

        logger.info("===== Request %d =====", i)
        logger.info("[Question] %s", questions[i])
        logger.info("[Model answer] %s", " ".join(answer_text.split()))
        if tokenizer is not None:
            logger.info("[Output tokens] %d", len(tokenizer.encode(answer_text)))
        logger.info(
            "[Extracted answer] %s | [Expected answer] %s | [Judgment] %s",
            pred,
            labels[i],
            judge,
        )
        logger.info("")

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Summary of all answers
    logger.info("========== Results ==========")
    logger.info("%-5s %-15s %-15s %-8s", "#", "expected", "obtained", "correct")
    for i in range(len(preds)):
        logger.info(
            "%-5d %-15s %-15s %-8s",
            i,
            labels[i],
            preds[i],
            str(preds[i] == labels[i]),
        )

    # Print results
    logger.info("========== Summary ==========")
    logger.info("num_output_tokens: %d", num_output_tokens)
    logger.info("Accuracy: %.3f", acc)
    logger.info("Invalid: %.3f", invalid)
    logger.info("Latency: %.3f s", latency)
    logger.info("Output throughput: %.3f token/s", output_throughput)

    # Dump results
    dump_state_text("tmp_output_gsm8k.txt", states)

    return {
        "accuracy": acc,
        "invalid": invalid,
        "latency": latency,
        "output_throughput": output_throughput,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--parallel", type=int, default=128)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    run_eval(args)
