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
DEFAULT_NUM_QUESTIONS=100           # (gsm8k) number of questions
DEFAULT_NUM_SHOTS=5                 # (gsm8k) few-shot count
DEFAULT_MAX_NEW_TOKENS=256          # (gsm8k) max generated tokens
DEFAULT_TEMPERATURE=0               # (gsm8k) sampling temperature

# mmlu
DEFAULT_NTRAIN=2                    # (mmlu) few-shot count
DEFAULT_NSUB=3                      # (mmlu) number of subjects
DEFAULT_BACKEND=srt                 # (mmlu) backend (srt / gpt-*)
DEFAULT_RESULT_FILE=result.jsonl    # (mmlu) result output file
DEFAULT_RAW_RESULT_FILE=            # (mmlu) raw result output file (empty = off)

# aisbench (performance runs via ais_bench in a dedicated conda env)
DEFAULT_AIS_CONDA_ENV=ais_bench     # (aisbench) conda env that has ais_bench installed
DEFAULT_AIS_TYPE=string             # (aisbench) synthetic dataset type (string / tokenid)
DEFAULT_AIS_REQUESTS=1024           # (aisbench) total request count (RequestCount)
DEFAULT_AIS_IN_LEN=128              # (aisbench) input length min (string mode)
DEFAULT_AIS_IN_LEN_MAX=             # (aisbench) input length max (empty = fixed length)
DEFAULT_AIS_OUT_LEN=384             # (aisbench) output length min (string mode)
DEFAULT_AIS_OUT_LEN_MAX=            # (aisbench) output length max (empty = fixed length)
DEFAULT_AIS_MAX_OUT_LEN=384         # (aisbench) model-side max_out_len
DEFAULT_AIS_REQUEST_RATE=32         # (aisbench) request_rate (0 = burst, >0 = req/s)
DEFAULT_AIS_TASKSET_CPUS=266-302    # (aisbench) CPU range for taskset (empty = no binding)

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

Available test suites (3):
  gsm8k    Few-shot GSM8K evaluation
           defaults: --num-questions 100 --num-shots 5 --max-new-tokens 256
                     --temperature 0 --parallel 10
  mmlu     MMLU evaluation
           defaults: --ntrain 2 --nsub 3 --parallel 10 --backend srt
  aisbench Throughput/perf run via ais_bench (in a dedicated conda env).
           Patches the two ais_bench config files in place, switches to the
           ais_bench conda env, then runs "ais_bench --mode perf".
           defaults: string dataset, 1024 requests, in 128 / out 384

Env overrides (runtime): HOST PORT PARALLEL
  gsm8k:    NUM_QUESTIONS NUM_SHOTS MAX_NEW_TOKENS TEMPERATURE DATA_PATH
  mmlu:     NTRAIN NSUB BACKEND RESULT_FILE RAW_RESULT_FILE DATA_DIR
  aisbench: AIS_CONDA_ENV AIS_BASE (default: auto-detected from the conda env)
            AIS_TYPE AIS_REQUESTS AIS_IN_LEN AIS_IN_LEN_MAX
            AIS_OUT_LEN AIS_OUT_LEN_MAX AIS_MAX_OUT_LEN AIS_BATCH
            AIS_TEMPERATURE AIS_IGNORE_EOS(0/1) AIS_TRUST_REMOTE_CODE(0/1)
            AIS_REQUEST_RATE (0=burst, >0 = req/s)
            AIS_REQUEST_SIZE AIS_PREFIX_LEN (tokenid mode)
            AIS_MODEL_CFG AIS_DATASET_CFG AIS_WORK_DIR
            AIS_TASKSET_CPUS (default 266-302, empty = no taskset binding)
  aisbench extra args are forwarded to ais_bench, e.g.:
    bash benchmark.sh aisbench --num-prompts 512
    bash benchmark.sh aisbench --debug

Examples:
  bash benchmark.sh gsm8k --max-new-tokens 128 --num-questions 50
  bash benchmark.sh mmlu  --nsub 60 --parallel 32
  bash benchmark.sh mmlu -h    # show the mmlu python script's own args
  AIS_REQUESTS=512 AIS_IN_LEN=1024 AIS_OUT_LEN=256 bash benchmark.sh aisbench
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
    aisbench)
        AIS_CONDA_ENV="${AIS_CONDA_ENV:-$DEFAULT_AIS_CONDA_ENV}"
        AIS_TYPE="${AIS_TYPE:-$DEFAULT_AIS_TYPE}"
        AIS_REQUESTS="${AIS_REQUESTS:-$DEFAULT_AIS_REQUESTS}"
        AIS_IN_LEN="${AIS_IN_LEN:-$DEFAULT_AIS_IN_LEN}"
        AIS_IN_LEN_MAX="${AIS_IN_LEN_MAX:-$DEFAULT_AIS_IN_LEN_MAX}"
        [[ -z "$AIS_IN_LEN_MAX" ]] && AIS_IN_LEN_MAX="$AIS_IN_LEN"
        AIS_OUT_LEN="${AIS_OUT_LEN:-$DEFAULT_AIS_OUT_LEN}"
        AIS_OUT_LEN_MAX="${AIS_OUT_LEN_MAX:-$DEFAULT_AIS_OUT_LEN_MAX}"
        [[ -z "$AIS_OUT_LEN_MAX" ]] && AIS_OUT_LEN_MAX="$AIS_OUT_LEN"
        AIS_MAX_OUT_LEN="${AIS_MAX_OUT_LEN:-$DEFAULT_AIS_MAX_OUT_LEN}"
        AIS_REQUEST_RATE="${AIS_REQUEST_RATE:-$DEFAULT_AIS_REQUEST_RATE}"
        AIS_BATCH="${AIS_BATCH:-$AIS_REQUESTS}"
        AIS_HOST="${AIS_HOST:-$GATEWAY_HOST}"
        AIS_PORT="${AIS_PORT:-$GATEWAY_PORT}"
        AIS_TASKSET_CPUS="${AIS_TASKSET_CPUS:-$DEFAULT_AIS_TASKSET_CPUS}"
        # Short names passed to ais_bench (resolved by its own config search)
        AIS_MODEL_CFG="${AIS_MODEL_CFG:-vllm_api_stream_chat}"
        AIS_DATASET_CFG="${AIS_DATASET_CFG:-synthetic_gen}"

        # Locate the ais_bench package inside the dedicated conda env
        if [[ -z "${AIS_BASE:-}" ]]; then
            AIS_BASE=$(ls -d "$CONDA_BASE_PATH/envs/$AIS_CONDA_ENV"/lib/python*/site-packages/ais_bench 2>/dev/null | head -n1)
        fi
        if [[ -z "${AIS_BASE:-}" || ! -d "$AIS_BASE" ]]; then
            echo "Error: cannot locate ais_bench under conda env '$AIS_CONDA_ENV' (tried \$AIS_BASE='$AIS_BASE'). Set AIS_BASE manually." >&2
            exit 1
        fi

        AIS_MODEL_CFG_PATH="$AIS_BASE/benchmark/configs/models/vllm_api/$AIS_MODEL_CFG.py"
        AIS_DATASET_CFG_PATH="$AIS_BASE/datasets/synthetic/synthetic_config.py"
        AIS_DATASET_GEN_PATH="$AIS_BASE/benchmark/configs/datasets/synthetic/$AIS_DATASET_CFG.py"
        for f in "$AIS_MODEL_CFG_PATH" "$AIS_DATASET_CFG_PATH" "$AIS_DATASET_GEN_PATH"; do
            if [[ ! -f "$f" ]]; then
                echo "Error: ais_bench config file not found: $f" >&2
                exit 1
            fi
        done

        echo "[aisbench] base:     $AIS_BASE"
        echo "[aisbench] target:   $AIS_HOST:$AIS_PORT"
        echo "[aisbench] dataset:  type=$AIS_TYPE requests=$AIS_REQUESTS in=$AIS_IN_LEN..$AIS_IN_LEN_MAX out=$AIS_OUT_LEN..$AIS_OUT_LEN_MAX"
        echo "[aisbench] model:    batch_size=$AIS_BATCH max_out_len=$AIS_MAX_OUT_LEN request_rate=$AIS_REQUEST_RATE"

        # --- Patch model-side config (rewrites the whole `key=...,` line) ---
        sed -i -E "s/(host_ip=).*/\1\"$AIS_HOST\",/" "$AIS_MODEL_CFG_PATH"
        sed -i -E "s/(host_port=).*/\1$AIS_PORT,/" "$AIS_MODEL_CFG_PATH"
        sed -i -E "s/(batch_size=).*/\1$AIS_BATCH,/" "$AIS_MODEL_CFG_PATH"
        sed -i -E "s/(max_out_len=).*/\1$AIS_MAX_OUT_LEN,/" "$AIS_MODEL_CFG_PATH"
        sed -i -E "s/(request_rate=).*/\1$AIS_REQUEST_RATE,/" "$AIS_MODEL_CFG_PATH"
        if [[ -n "${AIS_TEMPERATURE:-}" ]]; then
            sed -i -E "s/(temperature=).*/\1$AIS_TEMPERATURE,/" "$AIS_MODEL_CFG_PATH"
        fi
        if [[ -n "${AIS_IGNORE_EOS:-}" ]]; then
            if [[ "$AIS_IGNORE_EOS" == "1" ]]; then t=True; else t=False; fi
            sed -i -E "s/(ignore_eos=).*/\1$t,/" "$AIS_MODEL_CFG_PATH"
        fi
        if [[ -n "${AIS_TRUST_REMOTE_CODE:-}" ]]; then
            if [[ "$AIS_TRUST_REMOTE_CODE" == "1" ]]; then t=True; else t=False; fi
            sed -i -E "s/(trust_remote_code=).*/\1$t,/" "$AIS_MODEL_CFG_PATH"
            sed -i -E "s/(\"TrustRemoteCode\":)[^,#]*/\1 $t/" "$AIS_DATASET_CFG_PATH"
        fi

        # --- Patch dataset-side config (value-level, keeps inline comments) ---
        sed -i -E "s/(\"Type\":)[^,]*/\1\"$AIS_TYPE\"/" "$AIS_DATASET_CFG_PATH"
        sed -i -E "s/(\"RequestCount\":)[^,]*/\1 $AIS_REQUESTS/" "$AIS_DATASET_CFG_PATH"
        # Rewrite Input/Output blocks to uniform with the given min/max
        sed -i -E "/\"Input\"/,/\"Output\"/ s/(\"Method\":)[^,]*/\1 \"uniform\"/" "$AIS_DATASET_CFG_PATH"
        sed -i -E "/\"Input\"/,/\"Output\"/ s/(\"Params\":).*/\1 {\"MinValue\": $AIS_IN_LEN, \"MaxValue\": $AIS_IN_LEN_MAX}/" "$AIS_DATASET_CFG_PATH"
        sed -i -E "/\"Output\"/,/\"TokenIdConfig\"/ s/(\"Method\":)[^,]*/\1 \"uniform\"/" "$AIS_DATASET_CFG_PATH"
        sed -i -E "/\"Output\"/,/\"TokenIdConfig\"/ s/(\"Params\":).*/\1 {\"MinValue\": $AIS_OUT_LEN, \"MaxValue\": $AIS_OUT_LEN_MAX}/" "$AIS_DATASET_CFG_PATH"
        if [[ -n "${AIS_REQUEST_SIZE:-}" ]]; then
            sed -i -E "s/(\"RequestSize\":) [0-9]+/\1 $AIS_REQUEST_SIZE/" "$AIS_DATASET_CFG_PATH"
        fi
        if [[ -n "${AIS_PREFIX_LEN:-}" ]]; then
            sed -i -E "s/(\"PrefixLen\":) [0-9]+/\1 $AIS_PREFIX_LEN/" "$AIS_DATASET_CFG_PATH"
        fi

        # --- Switch to the ais_bench conda env and run perf mode ---
        # shellcheck disable=SC1091
        conda activate "$AIS_CONDA_ENV"

        # Build the launcher prefix: LD_PRELOAD (libpthread hook) + optional taskset
        launch_prefix=(env "LD_PRELOAD=$LIBPTHREAD_HOOK_PATH")
        if [[ -n "$AIS_TASKSET_CPUS" ]]; then
            launch_prefix+=(taskset -c "$AIS_TASKSET_CPUS")
        fi

        work_dir_args=()
        if [[ -n "${AIS_WORK_DIR:-}" ]]; then
            work_dir_args=(--work-dir "$AIS_WORK_DIR")
        fi

        echo "[aisbench] launching: LD_PRELOAD=$LIBPTHREAD_HOOK_PATH ${AIS_TASKSET_CPUS:+taskset -c $AIS_TASKSET_CPUS }ais_bench --models $AIS_MODEL_CFG --datasets $AIS_DATASET_CFG --mode perf --debug"
        "${launch_prefix[@]}" \
            ais_bench --models "$AIS_MODEL_CFG" --datasets "$AIS_DATASET_CFG" \
            --mode perf --debug \
            "${work_dir_args[@]}" \
            "$@"
        ;;
    *)
        echo "Unknown suite: '$SUITE'" >&2
        show_help >&2
        exit 1
        ;;
esac
