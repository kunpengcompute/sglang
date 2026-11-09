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

# gpqa / aime26 (accuracy runs via ais_bench, same conda env as aisbench)
DEFAULT_AIS_GPQA_TASK=gpqa_gen_0_shot_cot_chat_prompt   # 198 questions (gpqa_diamond.csv)
DEFAULT_AIS_AIME26_TASK=aime2026_gen_0_shot_chat_prompt # 30 questions (aime2026.jsonl)
DEFAULT_AIS_ACC_BATCH=16            # (accuracy) client concurrency
DEFAULT_AIS_ACC_MAX_OUT_LEN=8192    # (accuracy) model-side max_out_len (upstream uses 131072)
DEFAULT_AIS_ACC_TEMPERATURE=0.0     # (accuracy) sampling temperature
DEFAULT_AIS_ACC_RETRY=0             # (accuracy) per-request retry
DEFAULT_AIS_ACC_MODEL=              # (accuracy) served model name (empty = keep config)
DEFAULT_AIS_ACC_MODEL_PATH=         # (accuracy) tokenizer dir (empty = $MODEL_PATH)
DEFAULT_AIS_ACC_IGNORE_EOS=0        # (accuracy) 0 = stop at EOS (perf uses 1)
DEFAULT_AIS_ACC_NUM_PROMPTS=        # (accuracy) first N questions (empty = all)

# sharegpt (perf run via the official sglang bench_serving, offline)
DEFAULT_SHAREGPT_PORT=30000         # (sharegpt) gateway port (not $DEFAULT_PORT:
                                    #  .user_env_base.sh overrides that to 30002
                                    #  for aisbench's decode-tokenizer target)
DEFAULT_SHAREGPT_BACKEND=sglang     # (sharegpt) bench_serving backend
DEFAULT_SHAREGPT_MODEL=glm-5.3      # (sharegpt) served model name
DEFAULT_SHAREGPT_NUM_PROMPTS=1000   # (sharegpt) number of requests
# (sharegpt) local ShareGPT JSON. Set SHAREGPT_DATASET_PATH in
# runtime/.user_env_base.sh to point at the pre-downloaded file; bench_serving
# downloads from HuggingFace only when this is empty, which offline nodes can't.
DEFAULT_SHAREGPT_DATASET_PATH=/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
# (sharegpt) dataset mode: sharegpt (real conversations) or random (fixed-length
# prompts built from ShareGPT tokens). "random*" is passed through to
# bench_serving, so random-ids (pure integer tokens, no ShareGPT file) works too.
DEFAULT_SHAREGPT_DATASET_NAME=sharegpt
DEFAULT_SHAREGPT_RANDOM_INPUT_LEN=1024
DEFAULT_SHAREGPT_RANDOM_OUTPUT_LEN=1024
# 1 = strictly fixed in/out lengths; 0 = uniform in [1, full_len]
DEFAULT_SHAREGPT_RANDOM_RANGE_RATIO=1
DEFAULT_SHAREGPT_MAX_CONCURRENCY=256
# (sharegpt) client core binding. Default reuses the aisbench client range
# (DEFAULT_AIS_TASKSET_CPUS, tuned in runtime/.user_env_base.sh, e.g.
# 532-568 = NUMA 14) — a NUMA that is idle on the router node. Override with
# SHAREGPT_TASKSET_CPUS; empty = no taskset binding.

# Load environment: conda env + SGLANG_PATH / ROUTER_IP / MODEL_PATH / DATASET_PATH
source "$SCRIPT_DIR/env.sh" native
# MODEL_PATH is a plain shell var in env.sh; export it so the python subprocess sees it
export MODEL_PATH="${MODEL_PATH:-}"

# ais_bench overrides (DEFAULT_AIS_* such as the conda env name) live in their
# own snippet; load it here so the perf and accuracy suites share them.
if [[ -f "$SCRIPT_DIR/runtime/.user_env_aisbench.sh" ]]; then
    source "$SCRIPT_DIR/runtime/.user_env_aisbench.sh"
fi

SUITE="${1:-}"
shift || true

# Gateway host: $HOST > $ROUTER_IP (from env.sh) > 127.0.0.1
GATEWAY_HOST="${HOST:-${ROUTER_IP:-127.0.0.1}}"
GATEWAY_PORT="${PORT:-$DEFAULT_PORT}"
PARALLEL="${PARALLEL:-$DEFAULT_PARALLEL}"

show_help() {
    cat <<'EOF'
Usage: bash benchmark.sh <suite> [extra python args...]

Available test suites (6):
  gsm8k    Few-shot GSM8K evaluation
           defaults: --num-questions 100 --num-shots 5 --max-new-tokens 256
                     --temperature 0 --parallel 10
  mmlu     MMLU evaluation
           defaults: --ntrain 2 --nsub 3 --parallel 10 --backend srt
  gpqa     GPQA accuracy via ais_bench (gpqa_diamond, 198 questions).
           Needs gpqa_diamond.csv under $DATASET_PATH (see runtime/.user_env_aisbench.sh).
           defaults: batch 32, max_out_len 32768, temperature 1.0
  aime26   AIME2026 accuracy via ais_bench (30 questions).
           Needs aime2026.jsonl (fields problem/answer) under $DATASET_PATH.
           defaults: same as gpqa
  sharegpt Perf run via the official sglang bench_serving on the local
           ShareGPT JSON (offline: never downloads from HuggingFace).
           Requires SHAREGPT_DATASET_PATH in runtime/.user_env_base.sh.
           defaults: backend sglang, model glm-5.3, dataset-name sharegpt,
           1000 prompts (SHAREGPT_DATASET_NAME=random switches to fixed-length
           random prompts built from ShareGPT tokens)
  aisbench Throughput/perf run via ais_bench (in a dedicated conda env).
           Patches the two ais_bench config files in place, switches to the
           ais_bench conda env, then runs "ais_bench --mode perf".
           defaults: string dataset, 1024 requests, in 128 / out 384

Env overrides (runtime): HOST PORT PARALLEL
  gsm8k:    NUM_QUESTIONS NUM_SHOTS MAX_NEW_TOKENS TEMPERATURE DATA_PATH
  mmlu:     NTRAIN NSUB BACKEND RESULT_FILE RAW_RESULT_FILE DATA_DIR
  gpqa/aime26: AIS_CONDA_ENV AIS_BASE AIS_HOST AIS_PORT AIS_MODEL_CFG
            AIS_ACC_DATA_DIR (dataset dir; default
            $DATASET_PATH/aisbench/<gpqa|aime2026>)
            AIS_TASKSET_CPUS (default 266-302, empty = no binding)
            AIS_ACC_BATCH AIS_ACC_MAX_OUT_LEN AIS_ACC_TEMPERATURE
            AIS_ACC_RETRY AIS_ACC_IGNORE_EOS (0 = stop at EOS)
            AIS_ACC_MODEL AIS_ACC_MODEL_PATH
            AIS_ACC_NUM_PROMPTS (first N questions; injected as
            reader_cfg.test_range, empty = whole dataset)
            plus AIS_GPQA_TASK / AIS_AIME26_TASK to point at a different
            ais_bench dataset task name

  sharegpt: SHAREGPT_BACKEND SHAREGPT_MODEL SHAREGPT_NUM_PROMPTS
            SHAREGPT_DATASET_NAME (sharegpt / random / random-ids)
            SHAREGPT_RANDOM_INPUT_LEN SHAREGPT_RANDOM_OUTPUT_LEN
            SHAREGPT_RANDOM_RANGE_RATIO (1=fixed, 0=uniform 1..len)
            SHAREGPT_MAX_CONCURRENCY
            SHAREGPT_DATASET_PATH (dataset path default is in this script;
            tokenizer reuses ST_TOKENIZER, i.e. $MODEL_PATH; port defaults to
            the gateway's 30000 and is overridden by PORT)
            SHAREGPT_TASKSET_CPUS (default = aisbench's
            DEFAULT_AIS_TASKSET_CPUS, the router node's idle NUMA;
            empty = no taskset binding)
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
  SHAREGPT_NUM_PROMPTS=2000 bash benchmark.sh sharegpt
  SHAREGPT_DATASET_NAME=random bash benchmark.sh sharegpt \
      --output-file /tmp/collect.jsonl
  AIS_REQUESTS=512 AIS_IN_LEN=1024 AIS_OUT_LEN=256 bash benchmark.sh aisbench
  bash benchmark.sh gpqa                    # GPQA diamond, 198 questions
  bash benchmark.sh aime26                   # AIME2026, 30 questions
  AIS_ACC_MAX_OUT_LEN=131072 bash benchmark.sh gpqa
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
            sed -i -E "s/(temperature[[:space:]]*=[[:space:]]*)[0-9.]+/\1$AIS_TEMPERATURE/" "$AIS_MODEL_CFG_PATH"
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
    gpqa|aime26)
        # Accuracy run: only model + dataset configs are patched; this build
        # has no "--mode accuracy" (default "all" runs infer + eval).
        AIS_CONDA_ENV="${AIS_CONDA_ENV:-$DEFAULT_AIS_CONDA_ENV}"
        if [[ "$SUITE" == "gpqa" ]]; then
            AIS_TASK="${AIS_GPQA_TASK:-$DEFAULT_AIS_GPQA_TASK}"
        else
            AIS_TASK="${AIS_AIME26_TASK:-$DEFAULT_AIS_AIME26_TASK}"
        fi
        AIS_BATCH="${AIS_ACC_BATCH:-$DEFAULT_AIS_ACC_BATCH}"
        AIS_MAX_OUT_LEN="${AIS_ACC_MAX_OUT_LEN:-$DEFAULT_AIS_ACC_MAX_OUT_LEN}"
        AIS_TEMPERATURE="${AIS_ACC_TEMPERATURE:-$DEFAULT_AIS_ACC_TEMPERATURE}"
        AIS_RETRY="${AIS_ACC_RETRY:-$DEFAULT_AIS_ACC_RETRY}"
        AIS_ACC_MODEL="${AIS_ACC_MODEL:-${DEFAULT_AIS_ACC_MODEL:-}}"
        AIS_ACC_MODEL_PATH="${AIS_ACC_MODEL_PATH:-${DEFAULT_AIS_ACC_MODEL_PATH:-$MODEL_PATH}}"
        AIS_ACC_IGNORE_EOS="${AIS_ACC_IGNORE_EOS:-$DEFAULT_AIS_ACC_IGNORE_EOS}"
        # Question cap: this build ignores --num-prompts for accuracy, so it is
        # injected as reader_cfg.test_range below.
        AIS_NUM_PROMPTS="${AIS_ACC_NUM_PROMPTS:-$DEFAULT_AIS_ACC_NUM_PROMPTS}"
        if [[ -n "$AIS_NUM_PROMPTS" && ! "$AIS_NUM_PROMPTS" =~ ^[0-9]+$ ]]; then
            echo "Error: AIS_ACC_NUM_PROMPTS must be an integer, got '$AIS_NUM_PROMPTS'" >&2
            exit 1
        fi
        AIS_HOST="${AIS_HOST:-$GATEWAY_HOST}"
        AIS_PORT="${AIS_PORT:-$GATEWAY_PORT}"
        AIS_MODEL_CFG="${AIS_MODEL_CFG:-vllm_api_stream_chat}"
        AIS_TASKSET_CPUS="${AIS_TASKSET_CPUS:-$DEFAULT_AIS_TASKSET_CPUS}"

        # Locate the ais_bench package inside the dedicated conda env
        if [[ -z "${AIS_BASE:-}" ]]; then
            AIS_BASE=$(ls -d "$CONDA_BASE_PATH/envs/$AIS_CONDA_ENV"/lib/python*/site-packages/ais_bench 2>/dev/null | head -n1)
        fi
        if [[ -z "${AIS_BASE:-}" || ! -d "$AIS_BASE" ]]; then
            echo "Error: cannot locate ais_bench under conda env '$AIS_CONDA_ENV' (tried \$AIS_BASE='$AIS_BASE'). Set AIS_BASE manually." >&2
            exit 1
        fi

        AIS_MODEL_CFG_PATH="$AIS_BASE/benchmark/configs/models/vllm_api/$AIS_MODEL_CFG.py"
        if [[ ! -f "$AIS_MODEL_CFG_PATH" ]]; then
            echo "Error: ais_bench model config not found: $AIS_MODEL_CFG_PATH" >&2
            exit 1
        fi

        # Dataset config dir is "<dataset>_gen..." -> strip from "_gen".
        AIS_CFG_DIR="${AIS_TASK%%_gen*}"
        AIS_DATASET_CFG_PATH="$AIS_BASE/benchmark/configs/datasets/$AIS_CFG_DIR/$AIS_TASK.py"
        if [[ ! -f "$AIS_DATASET_CFG_PATH" ]]; then
            echo "Error: ais_bench dataset config not found: $AIS_DATASET_CFG_PATH" >&2
            exit 1
        fi

        # Dataset dir; `path=` is rewritten to an absolute path below.
        if [[ "$SUITE" == "gpqa" ]]; then
            AIS_ACC_DATA_DIR="${AIS_ACC_DATA_DIR:-$DATASET_PATH/aisbench/gpqa}"
            AIS_DATA_PATH="$AIS_ACC_DATA_DIR"
            AIS_DATA_PROBE="$AIS_ACC_DATA_DIR/gpqa_diamond.csv"
        else
            AIS_ACC_DATA_DIR="${AIS_ACC_DATA_DIR:-$DATASET_PATH/aisbench/aime2026}"
            AIS_DATA_PATH="$AIS_ACC_DATA_DIR/aime2026.jsonl"
            AIS_DATA_PROBE="$AIS_DATA_PATH"
        fi
        if [[ ! -e "$AIS_DATA_PROBE" ]]; then
            echo "Error: dataset file not found: $AIS_DATA_PROBE" >&2
            echo "       Put the data there, or set AIS_ACC_DATA_DIR." >&2
            exit 1
        fi

        echo "[aisbench] base:     $AIS_BASE"
        echo "[aisbench] dataset:  $AIS_TASK ($AIS_DATA_PROBE)"
        echo "[aisbench] target:   $AIS_HOST:$AIS_PORT"
        echo "[aisbench] model:    batch_size=$AIS_BATCH max_out_len=$AIS_MAX_OUT_LEN temperature=$AIS_TEMPERATURE retry=$AIS_RETRY model=${AIS_ACC_MODEL:-<config>} ignore_eos=$AIS_ACC_IGNORE_EOS"

        # Patch model config (perf defaults are wrong for accuracy: ignore_eos
        # keeps generating past the answer, model/path may point elsewhere).
        sed -i -E "s/(host_ip=).*/\1\"$AIS_HOST\",/" "$AIS_MODEL_CFG_PATH"
        sed -i -E "s/(host_port=).*/\1$AIS_PORT,/" "$AIS_MODEL_CFG_PATH"
        sed -i -E "s/(batch_size=).*/\1$AIS_BATCH,/" "$AIS_MODEL_CFG_PATH"
        sed -i -E "s/(max_out_len=).*/\1$AIS_MAX_OUT_LEN,/" "$AIS_MODEL_CFG_PATH"
        sed -i -E "s/(retry=).*/\1$AIS_RETRY,/" "$AIS_MODEL_CFG_PATH"
        sed -i -E "s/(temperature[[:space:]]*=[[:space:]]*)[0-9.]+/\1$AIS_TEMPERATURE/" "$AIS_MODEL_CFG_PATH"
        if ! grep -qE "temperature[[:space:]]*=[[:space:]]*${AIS_TEMPERATURE}([,)]|$)" "$AIS_MODEL_CFG_PATH"; then
            echo "Error: failed to set temperature=$AIS_TEMPERATURE in $AIS_MODEL_CFG_PATH" >&2
            exit 1
        fi
        if [[ "$AIS_ACC_IGNORE_EOS" == "1" ]]; then AIS_EOS=True; else AIS_EOS=False; fi
        sed -i -E "s/ignore_eos[[:space:]]*=[[:space:]]*[^,)]*/ignore_eos=$AIS_EOS/" "$AIS_MODEL_CFG_PATH"
        if ! grep -qE "ignore_eos[[:space:]]*=[[:space:]]*$AIS_EOS" "$AIS_MODEL_CFG_PATH"; then
            echo "Error: failed to set ignore_eos=$AIS_EOS in $AIS_MODEL_CFG_PATH" >&2
            exit 1
        fi
        if [[ -n "$AIS_ACC_MODEL" ]]; then
            sed -i -E "s|^([[:space:]]*model=).*|\1'$AIS_ACC_MODEL',|" "$AIS_MODEL_CFG_PATH"
        fi
        if [[ -n "$AIS_ACC_MODEL_PATH" ]]; then
            sed -i -E "s|^([[:space:]]*path=).*|\1'$AIS_ACC_MODEL_PATH',|" "$AIS_MODEL_CFG_PATH"
        fi

        # Absolute dataset path, then the question cap via reader_cfg.test_range
        # (drop any old value first so runs are idempotent).
        sed -i -E "s|(\\bpath=)['\"][^'\"]*['\"]|\\1'$AIS_DATA_PATH'|" "$AIS_DATASET_CFG_PATH"
        sed -i -E "s/test_range='[^']*',[[:space:]]*//g" "$AIS_DATASET_CFG_PATH"
        if [[ -n "$AIS_NUM_PROMPTS" ]]; then
            sed -i -E "s/(reader_cfg[[:space:]]*=[[:space:]]*dict\()/\1test_range='[:$AIS_NUM_PROMPTS]', /" "$AIS_DATASET_CFG_PATH"
            echo "[aisbench] limit:    first $AIS_NUM_PROMPTS question(s) via reader_cfg.test_range"
        fi

        # shellcheck disable=SC1091
        conda activate "$AIS_CONDA_ENV"

        launch_prefix=()
        if [[ -n "$AIS_TASKSET_CPUS" ]]; then
            launch_prefix=(taskset -c "$AIS_TASKSET_CPUS")
        fi

        # Explicit work dir: results never depend on where this script runs.
        AIS_ACC_WORK_DIR="${AIS_ACC_WORK_DIR:-$PWD/ais_acc_outputs}"
        mkdir -p "$AIS_ACC_WORK_DIR"

        echo "[aisbench] launching: ${AIS_TASKSET_CPUS:+taskset -c $AIS_TASKSET_CPUS }ais_bench --models $AIS_MODEL_CFG --datasets $AIS_TASK --work-dir $AIS_ACC_WORK_DIR --dump-eval-details"
        "${launch_prefix[@]}" \
            ais_bench --models "$AIS_MODEL_CFG" --datasets "$AIS_TASK" \
            --work-dir "$AIS_ACC_WORK_DIR" \
            --dump-eval-details \
            "$@"
        rc=$?

        # Print the newest summary csv so the accuracy is visible at a glance.
        summary=$(ls -t "$AIS_ACC_WORK_DIR"/outputs/default/*/results/*/*/summary*.csv \
                       "$AIS_ACC_WORK_DIR"/*/results/*/*/summary*.csv 2>/dev/null | head -n1)
        if [[ -n "$summary" ]]; then
            echo "[aisbench] result: $summary"
            cat "$summary"
        fi
        exit "$rc"
        ;;
    sharegpt)
        # Offline ShareGPT perf run via the official sglang bench_serving.
        # The host reuses GATEWAY_HOST (HOST/ROUTER_IP); the tokenizer reuses
        # ST_TOKENIZER (= $MODEL_PATH). The port defaults to the gateway's
        # 30000 (PORT overrides). The dataset path comes from
        # SHAREGPT_DATASET_PATH (set it in runtime/.user_env_base.sh); a local
        # file is mandatory — bench_serving only downloads from HuggingFace
        # when the path is empty, which offline nodes cannot do.
        #
        # Core binding (mirrors aisbench's AIS_TASKSET_CPUS): the default
        # reuses the aisbench client range (DEFAULT_AIS_TASKSET_CPUS, tuned in
        # .user_env_base.sh, e.g. 532-568 = NUMA 14) so this client runs on a
        # NUMA that is idle on the router node. Empty = no taskset binding.
        SHAREGPT_TASKSET_CPUS="${SHAREGPT_TASKSET_CPUS:-${AIS_TASKSET_CPUS:-$DEFAULT_AIS_TASKSET_CPUS}}"
        launch_prefix=()
        if [[ -n "$SHAREGPT_TASKSET_CPUS" ]]; then
            launch_prefix=(taskset -c "$SHAREGPT_TASKSET_CPUS")
            echo "taskset: $SHAREGPT_TASKSET_CPUS (router-node cores)" >&2
        fi
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        SHAREGPT_DATASET_NAME="${SHAREGPT_DATASET_NAME:-$DEFAULT_SHAREGPT_DATASET_NAME}"
        random_args=()
        if [[ "$SHAREGPT_DATASET_NAME" == random* ]]; then
            random_args=(
                --random-input-len "${SHAREGPT_RANDOM_INPUT_LEN:-$DEFAULT_SHAREGPT_RANDOM_INPUT_LEN}"
                --random-output-len "${SHAREGPT_RANDOM_OUTPUT_LEN:-$DEFAULT_SHAREGPT_RANDOM_OUTPUT_LEN}"
                --random-range-ratio "${SHAREGPT_RANDOM_RANGE_RATIO:-$DEFAULT_SHAREGPT_RANDOM_RANGE_RATIO}"
                --max-concurrency "${SHAREGPT_MAX_CONCURRENCY:-$DEFAULT_SHAREGPT_MAX_CONCURRENCY}"
            )
        fi
        exec "${launch_prefix[@]}" python3 -m sglang.bench_serving \
            --backend "${SHAREGPT_BACKEND:-$DEFAULT_SHAREGPT_BACKEND}" \
            --host "$GATEWAY_HOST" \
            --port "${PORT:-$DEFAULT_SHAREGPT_PORT}" \
            --model "${SHAREGPT_MODEL:-$DEFAULT_SHAREGPT_MODEL}" \
            --tokenizer "${ST_TOKENIZER:-$MODEL_PATH}" \
            --dataset-name "$SHAREGPT_DATASET_NAME" \
            --dataset-path "${SHAREGPT_DATASET_PATH:-$DEFAULT_SHAREGPT_DATASET_PATH}" \
            --num-prompts "${SHAREGPT_NUM_PROMPTS:-$DEFAULT_SHAREGPT_NUM_PROMPTS}" \
            "${random_args[@]}" \
            "$@"
        ;;
    *)
        echo "Unknown suite: '$SUITE'" >&2
        show_help >&2
        exit 1
        ;;
esac
