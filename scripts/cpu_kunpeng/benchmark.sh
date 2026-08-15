#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# benchmark.sh - Launch GSM8K / MMLU benchmarks against the gateway.
#
# Usage:
#   bash benchmark.sh gsm8k [extra python args...]
#   bash benchmark.sh mmlu  [extra python args...]
#   Extra args override the defaults, e.g.:
#     bash benchmark.sh gsm8k --max-new-tokens 128 --num-questions 50
#     bash benchmark.sh mmlu  --nsub 60 --parallel 32
#   (note: use hyphens, e.g. --max-new-tokens, not --max_new_tokens)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Config - edit the defaults here. Env vars override them at runtime:
#   HOST, PORT, PARALLEL
#   (gsm8k) NUM_QUESTIONS, NUM_SHOTS, MAX_NEW_TOKENS, TEMPERATURE, DATA_PATH
#   (mmlu)  NTRAIN, NSUB, BACKEND, RESULT_FILE, RAW_RESULT_FILE, DATA_DIR
# ============================================================================
DEFAULT_PORT=30000                  # (both) gateway port
DEFAULT_PARALLEL=10                 # (both) client concurrency

# gsm8k
DEFAULT_NUM_QUESTIONS=100            # (gsm8k) number of questions
DEFAULT_NUM_SHOTS=5                 # (gsm8k) few-shot count
DEFAULT_MAX_NEW_TOKENS=256          # (gsm8k) max generated tokens
DEFAULT_TEMPERATURE=0               # (gsm8k) sampling temperature

# mmlu
DEFAULT_NTRAIN=2                    # (mmlu) few-shot count
DEFAULT_NSUB=3                     # (mmlu) number of subjects
DEFAULT_BACKEND=srt                 # (mmlu) backend (srt / gpt-*)
DEFAULT_RESULT_FILE=result.jsonl    # (mmlu) result output file
DEFAULT_RAW_RESULT_FILE=            # (mmlu) raw result output file (empty = off)

# Load environment: conda env + SGLANG_PATH / ROUTER_IP / MODEL_PATH / DATASET_PATH
source "$SCRIPT_DIR/env.sh" native
# MODEL_PATH is a plain shell var in env.sh; export it so the python subprocess sees it
export MODEL_PATH="${MODEL_PATH:-}"

SUITE="${1:-}"
shift || true

# Gateway host: $HOST > $ROUTER_IP (from env.sh) > 127.0.0.1
GATEWAY_HOST="${HOST:-${ROUTER_IP:-127.0.0.1}}"
GATEWAY_PORT="${PORT:-$DEFAULT_PORT}"
PARALLEL="${PARALLEL:-$DEFAULT_PARALLEL}"

show_help() {
    cat <<'EOF'
Usage: bash benchmark.sh <suite> [extra python args...]

Available test suites (2):
  gsm8k    Few-shot GSM8K evaluation
           defaults: --num-questions 100 --num-shots 5 --max-new-tokens 256
                     --temperature 0 --parallel 10
  mmlu     MMLU evaluation
           defaults: --ntrain 2 --nsub 3 --parallel 10 --backend srt

Env overrides (runtime): HOST PORT PARALLEL
  gsm8k: NUM_QUESTIONS NUM_SHOTS MAX_NEW_TOKENS TEMPERATURE DATA_PATH
  mmlu:  NTRAIN NSUB BACKEND RESULT_FILE RAW_RESULT_FILE DATA_DIR

Examples:
  bash benchmark.sh gsm8k --max-new-tokens 128 --num-questions 50
  bash benchmark.sh mmlu  --nsub 60 --parallel 32
  bash benchmark.sh mmlu -h    # show the mmlu python script's own args
EOF
}

if [[ -z "$SUITE" ]]; then
    show_help >&2
    exit 1
fi

case "$SUITE" in
    -h|--help|help)
        show_help
        exit 0
        ;;
    gsm8k)
        exec python "$SGLANG_PATH/python/sglang/test/few_shot_gsm8k.py" \
            --num-questions "${NUM_QUESTIONS:-$DEFAULT_NUM_QUESTIONS}" \
            --num-shots "${NUM_SHOTS:-$DEFAULT_NUM_SHOTS}" \
            --max-new-tokens "${MAX_NEW_TOKENS:-$DEFAULT_MAX_NEW_TOKENS}" \
            --temperature "${TEMPERATURE:-$DEFAULT_TEMPERATURE}" \
            --host "http://${GATEWAY_HOST}" \
            --port "$GATEWAY_PORT" \
            --data-path "${DATA_PATH:-$DATASET_PATH/gsm8k/test.jsonl}" \
            --parallel "$PARALLEL" \
            "$@"
        ;;
    mmlu)
        raw_result_args=()
        if [[ -n "${RAW_RESULT_FILE:-$DEFAULT_RAW_RESULT_FILE}" ]]; then
            raw_result_args=(--raw-result-file "${RAW_RESULT_FILE:-$DEFAULT_RAW_RESULT_FILE}")
        fi
        exec python "$SGLANG_PATH/benchmark/mmlu/bench_sglang.py" \
            --parallel "$PARALLEL" \
            --data_dir "${DATA_DIR:-$DATASET_PATH/mmlu}" \
            --ntrain "${NTRAIN:-$DEFAULT_NTRAIN}" \
            --port "$GATEWAY_PORT" \
            --host "http://${GATEWAY_HOST}" \
            --nsub "${NSUB:-$DEFAULT_NSUB}" \
            --backend "${BACKEND:-$DEFAULT_BACKEND}" \
            --result-file "${RESULT_FILE:-$DEFAULT_RESULT_FILE}" \
            "${raw_result_args[@]}" \
            "$@"
        ;;
    *)
        echo "Unknown suite: '$SUITE'" >&2
        show_help >&2
        exit 1
        ;;
esac
