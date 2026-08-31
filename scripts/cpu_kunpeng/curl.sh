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

#!/bin/bash

usage() {
  echo "Usage:"
  echo "  $0 [-n NUM] [-r RATE] [-m TOKENS] [-p] [-s] [-f FILE]                # paced batch (RATE req/s, default 32)"
  echo "  $0 -n NUM -r 0 [-m TOKENS] [-p] [-s] [-f FILE]                       # legacy batch: all prompts in one request"
  echo "  $0 -d RANGE [-n NUM] [-c CONC] [-m TOKENS] [-p] [-s] [-v] [-f FILE]  # round-robin benchmark"
  echo "  $0 [-i] [-d RANK] [-m TOKENS] [-h]                                   # interactive chat"
  echo ""
  echo "Modes (mutually exclusive):"
  echo "  (default)   Paced batch — one prompt per request, sent at -r RATE req/s (default 32)."
  echo "  -d RANGE    Round-robin benchmark — triggered by -d with a range (e.g. 0-15, 0,2,5)."
  echo "              Reports: throughput (req/s, tok/s) + per-request latency (P50/P99)."
  echo "  -i          Interactive chat mode — multi-turn streaming conversation."
  echo ""
  echo "Options:"
  echo "  -h          Show this help message"
  echo "  -m TOKENS   Max tokens per request / per turn (default: 10 batch, 128 chat)"
  echo "  -d RANK|RANGE  DP rank (e.g. 5) or range (e.g. 0-15, 0,2,5 → triggers round-robin)"
  echo "  -r RATE     Paced batch send rate in req/s (default: 32; 0 = legacy single array request)"
  echo ""
  echo "  --- batch & round-robin only ---"
  echo "  -p          Enable profiling (start/stop profile via separate curl calls)"
  echo "  -s          Enable streaming mode"
  echo "  -f FILE     Prompt file, one prompt per line (default: prompts/128.txt)"
  echo "  -n NUM      Number of requests to send (default: all lines from file)"
  echo ""
  echo "  --- round-robin & paced batch only ---"
  echo "  -c CONC     Max concurrent requests (default: 256; paced batch defaults to unbounded)"
  echo "  -v          Write per-request latency rows + response bodies to <run-id>_detail.txt"
  exit 0
}

PROFILE=false
MAX_TOKENS=10
MAX_TOKENS_SET=false
NUM_REQUESTS=0
STREAM=false
VERBOSE=false
INTERACTIVE=false
DP_ENABLED=false
DP_RANK=0
CONCURRENCY=256
CONCURRENCY_SET=false
PROMPT_FILE="prompts/128.txt"
ROUND_ROBIN=false
RATE=32
PACED=false

while getopts "d:hipsvn:m:c:r:f:" opt; do
  case $opt in
    h) usage ;;
    i) INTERACTIVE=true ;;
    p) PROFILE=true ;;
    s) STREAM=true ;;
    v) VERBOSE=true ;;
    d) DP_ENABLED=true; DP_RANK=$OPTARG ;;
    n) NUM_REQUESTS=$OPTARG ;;
    c) CONCURRENCY=$OPTARG; CONCURRENCY_SET=true ;;
    r) RATE=$OPTARG ;;
    m) MAX_TOKENS=$OPTARG; MAX_TOKENS_SET=true ;;
    f) PROMPT_FILE=$OPTARG ;;
    *) echo "Invalid option: -$OPTARG" >&2
       exit 1 ;;
  esac
done

shift $((OPTIND - 1))

# =============================================================================
# Common setup
# =============================================================================

if [ "$INTERACTIVE" = false ] && [ ! -f "$PROMPT_FILE" ]; then
  echo "Error: file not found: $PROMPT_FILE" >&2
  exit 1
fi

# Read prompts from file (one per line) — only for non-interactive modes
PROMPTS=()
if [ "$INTERACTIVE" = false ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    PROMPTS+=("$line")
  done < "$PROMPT_FILE"

  if [ ${#PROMPTS[@]} -eq 0 ]; then
    echo "Error: prompt file is empty" >&2
    exit 1
  fi

  # Default NUM_REQUESTS to all lines from file if not specified
  if [ "$NUM_REQUESTS" -eq 0 ]; then
    NUM_REQUESTS=${#PROMPTS[@]}
  fi
fi

# Escape JSON special characters in a string
json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/}"
  s="${s//$'\t'/\\t}"
  echo "$s"
}

# Parse DP ranks spec (e.g. "0-15", "0,2,4", "0-3,7,10-12")
parse_ranks() {
  local spec="$1"
  local ranks=()
  IFS=',' read -ra parts <<< "$spec"
  for part in "${parts[@]}"; do
    part="${part// /}"
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      for ((r=${BASH_REMATCH[1]}; r<=${BASH_REMATCH[2]}; r++)); do
        ranks+=("$r")
      done
    elif [[ "$part" =~ ^[0-9]+$ ]]; then
      ranks+=("$part")
    else
      echo "Error: invalid rank '$part' in '$spec'" >&2
      exit 1
    fi
  done
  echo "${ranks[*]}"
}

IP=$(ifconfig enp26s0f0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
PORT=30000
URL="http://${IP}:${PORT}/v1/completions"

# =============================================================================
# Non-interactive dispatch: round-robin detection + profile start
# =============================================================================

if [ "$INTERACTIVE" = false ]; then
  # Detect round-robin mode: -d value contains '-' or ','
  ROUND_ROBIN=false
  DP_RANKS=()
  NUM_RANKS=0
  if [ "$DP_ENABLED" = true ] && [[ "$DP_RANK" =~ [-,] ]]; then
    ROUND_ROBIN=true
    read -ra DP_RANKS <<< "$(parse_ranks "$DP_RANK")"
    NUM_RANKS=${#DP_RANKS[@]}
    if [ "$NUM_RANKS" -eq 0 ]; then
      echo "Error: no DP ranks parsed from '$DP_RANK'" >&2
      exit 1
    fi
  fi

  # Paced batch mode: without -d, send one prompt per HTTP request at RATE
  # req/s instead of one request carrying all prompts (RATE=0 restores that).
  if [ "$RATE" -gt 0 ] && [ "$NUM_REQUESTS" -gt 1 ]; then
    PACED=true
  fi

  if [ "$PROFILE" = true ]; then
    curl --noproxy "*" http://${IP}:${PORT}/start_profile
  fi
fi

# =============================================================================
# Mode 1: Original batch — single request with array of prompts
# =============================================================================

if [ "$INTERACTIVE" = false ] && [ "$ROUND_ROBIN" = false ] && [ "$PACED" = false ]; then

  # Build prompt JSON array from PROMPTS, cycling if NUM_REQUESTS exceeds list length
  PROMPT_JSON="["
  for ((i=0; i<NUM_REQUESTS; i++)); do
    idx=$((i % ${#PROMPTS[@]}))
    escaped=$(json_escape "${PROMPTS[$idx]}")
    if [ $i -gt 0 ]; then
      PROMPT_JSON+=","
    fi
    PROMPT_JSON+="\"$escaped\""
  done
  PROMPT_JSON+="]"

  if [ "$DP_ENABLED" = true ]; then
    DP_LINE=",\"routed_dp_rank\": $DP_RANK"
  else
    DP_LINE=""
  fi

  BODY="{
      \"model\": \"DeepSeek-R1\",
      \"prompt\": $PROMPT_JSON,
      \"stream\": $STREAM,
      \"max_tokens\": $MAX_TOKENS,
      \"temperature\": 0$DP_LINE"

  if [ "$STREAM" = true ]; then
    BODY+=",\"stream_options\":{\"include_usage\":true}"
  fi

  BODY+="
    }"

  BODY_FILE=$(mktemp)
  printf '%s' "$BODY" > "$BODY_FILE"

  if [ "$STREAM" = true ]; then
      TURN_START=$(date +%s.%N)
      FIRST_TOKEN_TS=""
      CHUNK_COUNT=0
      USAGE_COMP_TOKENS=""
      USAGE_RAW=""

      while read -r line; do
        echo "$line"
        # Count chunks that carry actual content. The completions endpoint
        # puts the generated text in a "text" field; usage-only chunks have
        # empty choices and no "text" field. Counting "text" occurrences
        # avoids miscounting when continuous_usage_stats embeds usage in
        # every chunk.
        TEXT=$(echo "$line" | grep -o '"text":"[^"]*"' | cut -d'"' -f4)
        if [[ ! -z "$TEXT" ]]; then
          if [ -z "$FIRST_TOKEN_TS" ]; then
            FIRST_TOKEN_TS=$(date +%s.%N)
          fi
          CHUNK_COUNT=$((CHUNK_COUNT + 1))
        fi
        # Capture the usage block emitted in the final chunk when
        # stream_options.include_usage is set.
        USAGE_OBJ=$(echo "$line" | grep -o '"usage":{[^}]*}')
        if [[ ! -z "$USAGE_OBJ" ]]; then
          USAGE_RAW="$USAGE_OBJ"
          COMP=$(echo "$USAGE_OBJ" | grep -o '"completion_tokens":[0-9]*' | head -n1 | cut -d':' -f2)
          if [[ ! -z "$COMP" ]]; then
            USAGE_COMP_TOKENS="$COMP"
          fi
        fi
      done < <(curl --noproxy "*" -N -s http://${IP}:${PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d @"$BODY_FILE")
      TURN_END=$(date +%s.%N)

      if [ "$CHUNK_COUNT" -gt 0 ]; then
          TOKEN_COUNT="${USAGE_COMP_TOKENS:-$MAX_TOKENS}"
          TTFT=$(awk -v s="$TURN_START" -v f="$FIRST_TOKEN_TS" 'BEGIN { printf "%.3f", f - s }')
          TOTAL=$(awk -v s="$TURN_START" -v e="$TURN_END" 'BEGIN { printf "%.3f", e - s }')
          TPOT=$(awk -v n="$TOKEN_COUNT" -v tt="$TTFT" -v total="$TOTAL" 'BEGIN { dn=n-1; if (dn>0) printf "%.1f", (total-tt)/dn*1000; else print "0" }')
          RATE=$(awk -v n="$TOKEN_COUNT" -v c="$CHUNK_COUNT" \
              'BEGIN { if (c>0) printf "%.2f", n / c; else print "0" }')
          echo "" >&2
          echo "==================================================" >&2
          echo "TTFT: ${TTFT}s | Total: ${TOTAL}s | TPOT: ${TPOT} ms/tok" >&2
          echo "Output Tokens: $TOKEN_COUNT | Chunks: $CHUNK_COUNT | Accept Rate: $RATE" >&2
          echo "==================================================" >&2
      fi
  else
      time curl --noproxy "*" -s http://${IP}:${PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d @"$BODY_FILE"
  fi

  rm -f "$BODY_FILE"

  if [ "$PROFILE" = true ]; then
    curl --noproxy "*" http://${IP}:${PORT}/stop_profile
  fi

  exit 0
fi

# =============================================================================
# Mode 2: Round-robin benchmark — concurrent requests across DP ranks
# =============================================================================

if [ "$ROUND_ROBIN" = true ] || [ "$PACED" = true ]; then
  RESULT_DIR=$(mktemp -d)
  trap 'rm -rf "$RESULT_DIR"' EXIT

  # Unique run ID for correlating with router logs (router uses X-Request-Id header)
  RUN_ID="curl-$(date +%s)-$$"

  send_request() {
    local idx=$1
    local rank_line=""
    if [ "$ROUND_ROBIN" = true ]; then
      rank_line=",\"routed_dp_rank\": ${DP_RANKS[$((idx % NUM_RANKS))]}"
    elif [ "$DP_ENABLED" = true ]; then
      rank_line=",\"routed_dp_rank\": $DP_RANK"
    fi
    local escaped
    escaped=$(json_escape "${PROMPTS[$((idx % ${#PROMPTS[@]}))]}")

    local body="{
        \"model\": \"DeepSeek-R1\",
        \"prompt\": \"$escaped\",
        \"stream\": $STREAM,
        \"max_tokens\": $MAX_TOKENS,
        \"temperature\": 0$rank_line"

    if [ "$STREAM" = true ]; then
      body+=",\"stream_options\":{\"include_usage\":true}"
    fi

    body+="
      }"

    local body_file
    body_file=$(mktemp)
    printf '%s' "$body" > "$body_file"

    local resp_file="$RESULT_DIR/resp_${idx}"
    local start_ns
    start_ns=$(date +%s%N)

    # Client-generated request ID sent via X-Request-Id header;
    # router uses this ID in its logs for easy correlation.
    local rid="${RUN_ID}-${idx}"
    echo "$rid" > "$RESULT_DIR/rid_${idx}"

    curl --noproxy "*" -s "$URL" \
      -H "Content-Type: application/json" \
      -H "X-Request-Id: $rid" \
      -d @"$body_file" > "$resp_file" 2>/dev/null

    local end_ns
    end_ns=$(date +%s%N)
    echo "$start_ns $end_ns" > "$RESULT_DIR/time_${idx}"

    if [ "$STREAM" = true ]; then
      local usage_raw comp_tokens chunks done_count
      usage_raw=$(grep -o '"usage":{[^}]*}' "$resp_file" 2>/dev/null | tail -n1)
      comp_tokens=$(echo "$usage_raw" | grep -o '"completion_tokens":[0-9]*' | head -n1 | cut -d':' -f2)
      chunks=$(grep -c "^data:" "$resp_file" 2>/dev/null || true)
      chunks="${chunks:-0}"
      done_count=$(grep -c "\[DONE\]" "$resp_file" 2>/dev/null || true)
      done_count="${done_count:-0}"
      # Prefer actual token count from usage; fall back to chunk count
      if [[ ! -z "$comp_tokens" ]]; then
        echo "$comp_tokens" > "$RESULT_DIR/tokens_${idx}"
      else
        echo $((chunks - done_count)) > "$RESULT_DIR/tokens_${idx}"
      fi
    else
      local tokens
      tokens=$(grep -oP '"completion_tokens":\s*\K\d+' "$resp_file" 2>/dev/null | head -1)
      echo "${tokens:-0}" > "$RESULT_DIR/tokens_${idx}"
    fi

    rm -f "$body_file"
  }

  # Paced mode is open-loop: don't throttle on the client unless -c is given.
  if [ "$PACED" = true ] && [ "$CONCURRENCY_SET" = false ]; then
    CONCURRENCY=$NUM_REQUESTS
  fi

  if [ "$ROUND_ROBIN" = true ]; then
    echo "Round-robin: $NUM_REQUESTS requests, concurrency=$CONCURRENCY, $NUM_RANKS DP ranks ($DP_RANK)"
  else
    echo "Paced batch: $NUM_REQUESTS requests, rate=$RATE req/s, concurrency=$CONCURRENCY"
  fi
  echo "  URL: $URL"
  echo "  Max tokens/req: $MAX_TOKENS, Stream: $STREAM"
  echo "  Run ID: $RUN_ID  (grep this in router logs)"
  echo ""

  # FIFO-based concurrency semaphore
  FIFO=$(mktemp -u)
  mkfifo "$FIFO"
  exec 3<>"$FIFO"
  rm "$FIFO"
  for ((i = 0; i < CONCURRENCY; i++)); do
    echo >&3
  done

  # One date call provides both the stats epoch (WALL_START) and the
  # integer-ns pacing base (START_NS).
  START_NS=$(date +%s%N)
  printf -v WALL_START '%d.%09d' $((START_NS / 1000000000)) $((START_NS % 1000000000))

  for ((i = 0; i < NUM_REQUESTS; i++)); do
    read -u 3
    {
      send_request "$i"
      echo >&3
    } &
    if [ "$PACED" = true ] && (( (i + 1) % RATE == 0 )); then
      # Re-anchor to the schedule once per second (every RATE requests).
      # Per-request clock reads cost a fork each and fall behind under
      # load (iter cost > 1/RATE), so pace in 1s batches instead: fire
      # this second's batch, then sleep to the next boundary. If behind
      # (delta <= 0) the next batch fires immediately to catch up.
      delta_ns=$(( START_NS + (i + 1) * 1000000000 / RATE - $(date +%s%N) ))
      if (( delta_ns > 0 )); then
        printf -v delay '%d.%09d' $((delta_ns / 1000000000)) $((delta_ns % 1000000000))
        sleep "$delay"
      fi
    fi
  done

  wait
  WALL_END=$(date +%s.%N)
  exec 3>&-

  if [ "$PROFILE" = true ]; then
    curl --noproxy "*" http://${IP}:${PORT}/stop_profile
  fi

  # --- Compute and print results ---
  WALL_TIME=$(awk -v s="$WALL_START" -v e="$WALL_END" 'BEGIN { printf "%.3f", e - s }')

  TOTAL_TOKENS=0
  FAILED=0
  for ((i = 0; i < NUM_REQUESTS; i++)); do
    t=$(cat "$RESULT_DIR/tokens_${i}" 2>/dev/null)
    if [[ "$t" =~ ^[0-9]+$ ]]; then
      TOTAL_TOKENS=$((TOTAL_TOKENS + t))
      [ "$t" -eq 0 ] && FAILED=$((FAILED + 1))
    else
      FAILED=$((FAILED + 1))
    fi
  done

  REQ_PER_SEC=$(awk -v n="$NUM_REQUESTS" -v t="$WALL_TIME" 'BEGIN { printf "%.2f", n / t }')
  TOKENS_PER_SEC=$(awk -v n="$TOTAL_TOKENS" -v t="$WALL_TIME" 'BEGIN { printf "%.2f", n / t }')

  echo "==================================="
  echo "Throughput Results"
  echo "==================================="
  echo "  Total requests:      $NUM_REQUESTS"
  echo "  Concurrency:         $CONCURRENCY"
  if [ "$ROUND_ROBIN" = true ]; then
    echo "  DP ranks:            $DP_RANK ($NUM_RANKS ranks, round-robin)"
  else
    echo "  Send rate:           $RATE req/s"
  fi
  echo "  Max tokens/req:      $MAX_TOKENS"
  echo "  Stream:              $STREAM"
  echo "  Total wall time:     ${WALL_TIME}s"
  echo "  Total output tokens: $TOTAL_TOKENS"
  echo "  Failed requests:     $FAILED"
  echo "  Requests/sec:        $REQ_PER_SEC"
  echo "  Tokens/sec:          $TOKENS_PER_SEC"
  echo "==================================="

  # --- Per-Request Latency ---
  LAT_FILE="$RESULT_DIR/all_latencies"
  : > "$LAT_FILE"

  # With -v, per-request rows and response bodies go to a file instead of
  # the terminal (row count scales with -n and can be huge).
  DETAIL_FILE=""
  if [ "$VERBOSE" = true ]; then
    DETAIL_FILE="${RUN_ID}_detail.txt"
    {
      echo "Per-Request Latency (relative to test start)"
      printf "%-6s %-6s %-12s %-12s %-10s %-8s %s\n" "Req#" "Rank" "Start(s)" "End(s)" "Latency(s)" "Tokens" "ReqID"
    } > "$DETAIL_FILE"
  fi

  # Fork-free per-request stats: builtin reads + integer-ns arithmetic.
  # The previous awk/cat-per-row version forked ~5 times per request and
  # silently stalled for minutes on a loaded node after the summary.
  for ((i = 0; i < NUM_REQUESTS; i++)); do
    if [ "$ROUND_ROBIN" = true ]; then rank=${DP_RANKS[$((i % NUM_RANKS))]} ; else rank="-"; fi
    if [ -f "$RESULT_DIR/time_${i}" ]; then
      read -r s e < "$RESULT_DIR/time_${i}"
      dur_ns=$((e - s))
      printf -v dur '%d.%03d' $((dur_ns / 1000000000)) $((dur_ns / 1000000 % 1000))
      printf -v rel_start '%d.%03d' $(((s - START_NS) / 1000000000)) $(((s - START_NS) / 1000000 % 1000))
      printf -v rel_end '%d.%03d' $(((e - START_NS) / 1000000000)) $(((e - START_NS) / 1000000 % 1000))
      echo "$dur" >> "$LAT_FILE"
      toks="?"
      rid="N/A"
      read -r toks < "$RESULT_DIR/tokens_${i}" 2>/dev/null || toks="?"
      read -r rid < "$RESULT_DIR/rid_${i}" 2>/dev/null || rid="N/A"
      if [ "$VERBOSE" = true ]; then
        printf "%-6d %-6s %-12s %-12s %-10s %-8s %s\n" "$i" "$rank" "$rel_start" "$rel_end" "$dur" "$toks" "$rid" >> "$DETAIL_FILE"
      fi
    else
      if [ "$VERBOSE" = true ]; then
        printf "%-6d %-6s %-12s %-12s %-10s %-8s %s\n" "$i" "$rank" "N/A" "N/A" "N/A" "N/A" "N/A" >> "$DETAIL_FILE"
      fi
    fi
  done

  if [ -s "$LAT_FILE" ]; then
    awk '{ a[NR]=$1; sum+=$1 } END {
      n=NR; if(n==0) exit;
      for(i=1;i<=n;i++) for(j=i+1;j<=n;j++) if(a[i]>a[j]){t=a[i];a[i]=a[j];a[j]=t}
      p50=a[int((n+1)*0.5)]; p99=a[int((n+1)*0.99)];
      if(p99=="") p99=a[n];
      printf "\n===================================\n";
      printf "Latency Summary\n";
      printf "===================================\n";
      printf "  Min:      %.3fs\n", a[1];
      printf "  Max:      %.3fs\n", a[n];
      printf "  Avg:      %.3fs\n", sum/n;
      printf "  P50:      %.3fs\n", p50;
      printf "  P99:      %.3fs\n", p99;
      printf "===================================\n";
    }' "$LAT_FILE"
  fi

  if [ "$VERBOSE" = true ]; then
    {
      echo ""
      echo "==================================="
      echo "Per-Request Responses"
      echo "==================================="
      for ((i = 0; i < NUM_REQUESTS; i++)); do
        if [ "$ROUND_ROBIN" = true ]; then rank=${DP_RANKS[$((i % NUM_RANKS))]} ; else rank="-"; fi
        echo "----- Request #$i (rank=$rank) -----"
        if [ -s "$RESULT_DIR/resp_${i}" ]; then
          cat "$RESULT_DIR/resp_${i}"
          echo ""
        else
          echo "<no response>"
        fi
      done
      echo "==================================="
    } >> "$DETAIL_FILE"
    echo ""
    echo "Per-request details written to: $DETAIL_FILE"
  fi

  exit 0
fi

# =============================================================================
# Mode 3: Interactive chat — multi-turn streaming conversation
# =============================================================================

if [ "$INTERACTIVE" = true ]; then
  CHAT_URL="http://${IP}:${PORT}/v1/chat/completions"

  # Use a larger default for chat if -m was not explicitly passed
  if [ "$MAX_TOKENS_SET" = false ]; then
    MAX_TOKENS=1024
  fi

  # In interactive mode, -d only supports a single rank. If a range is
  # provided, pick the first rank to avoid emitting invalid JSON.
  if [ "$DP_ENABLED" = true ] && [[ "$DP_RANK" =~ [-,] ]]; then
    FIRST_RANK=$(parse_ranks "$DP_RANK" | tr ' ' '\n' | head -n1)
    echo "Warning: interactive mode uses a single DP rank; using rank $FIRST_RANK (from '$DP_RANK')" >&2
    DP_RANK="$FIRST_RANK"
  fi

  echo "--- AI chat mode (Ctrl+C to exit) ---"
  echo "  URL: $CHAT_URL"
  echo "  Model: deepseek-v3, Max tokens: $MAX_TOKENS, Stream: true"
  if [ "$DP_ENABLED" = true ]; then
    echo "  DP rank: $DP_RANK"
  fi
  echo ""

  HISTORY=()

  trap 'echo ""; exit 0' INT

  while true; do
    read -e -p "User: " USER_INPUT
    if [[ -z "$USER_INPUT" ]]; then continue; fi

    echo -n "AI: "

    # Build messages array from conversation history plus the new user turn
    escaped_input=$(json_escape "$USER_INPUT")
    MESSAGES="["
    first=true
    for msg in "${HISTORY[@]}"; do
      if [ "$first" = true ]; then
        first=false
      else
        MESSAGES+=","
      fi
      MESSAGES+="$msg"
    done
    if [ "$first" = false ]; then
      MESSAGES+=","
    fi
    MESSAGES+="{\"role\":\"user\",\"content\":\"$escaped_input\"}"
    MESSAGES+="]"

    # Record user turn in history
    HISTORY+=("{\"role\":\"user\",\"content\":\"$escaped_input\"}")

    DP_LINE=""
    if [ "$DP_ENABLED" = true ]; then
      DP_LINE=",\"routed_dp_rank\":$DP_RANK"
    fi

    BODY="{"
    BODY+="\"model\":\"DeepSeek-R1\","
    BODY+="\"messages\":$MESSAGES,"
    BODY+="\"stream\":true,"
    BODY+="\"max_tokens\":$MAX_TOKENS,"
    BODY+="\"temperature\":0"
    BODY+="$DP_LINE"
    BODY+=",\"stream_options\":{\"include_usage\":true}"
    BODY+="}"

    BODY_FILE=$(mktemp)
    printf '%s' "$BODY" > "$BODY_FILE"

    # Build curl header args; X-Data-Parallel-Rank has higher priority than
    # the body field and survives proxy/router hops.
    CURL_HEADERS=(-H "Content-Type: application/json")
    if [ "$DP_ENABLED" = true ]; then
      CURL_HEADERS+=(-H "X-Data-Parallel-Rank: $DP_RANK")
    fi

    TURN_START=$(date +%s.%N)
    FIRST_TOKEN_TS=""
    CHUNK_COUNT=0
    TOKEN_COUNT=0
    USAGE_COMP_TOKENS=""
    USAGE_RAW=""
    FULL_RESPONSE=""
    while read -r line; do
      CONTENT=$(echo "$line" | grep -o '"content":"[^"]*"' | cut -d'"' -f4)
      if [[ ! -z "$CONTENT" ]]; then
        if [ -z "$FIRST_TOKEN_TS" ]; then
          FIRST_TOKEN_TS=$(date +%s.%N)
        fi
        printf "%b" "$CONTENT"
        FULL_RESPONSE+="$CONTENT"
        CHUNK_COUNT=$((CHUNK_COUNT + 1))
      fi
      # Capture the usage block emitted in the final chunk when
      # stream_options.include_usage is set.
      USAGE_OBJ=$(echo "$line" | grep -o '"usage":{[^}]*}')
      if [[ ! -z "$USAGE_OBJ" ]]; then
        USAGE_RAW="$USAGE_OBJ"
        COMP=$(echo "$USAGE_OBJ" | grep -o '"completion_tokens":[0-9]*' | head -n1 | cut -d':' -f2)
        if [[ ! -z "$COMP" ]]; then
          USAGE_COMP_TOKENS="$COMP"
        fi
      fi
    done < <(curl --noproxy "*" -N -s -X POST "$CHAT_URL" \
      "${CURL_HEADERS[@]}" \
      -d @"$BODY_FILE")
    TURN_END=$(date +%s.%N)

    # Prefer the real token count from usage when the server provided it
    if [[ ! -z "$USAGE_COMP_TOKENS" ]]; then
      TOKEN_COUNT="$USAGE_COMP_TOKENS"
    else
      TOKEN_COUNT="$CHUNK_COUNT"
    fi

    rm -f "$BODY_FILE"

    # Record assistant reply in history
    if [[ ! -z "$FULL_RESPONSE" ]]; then
      escaped_resp=$(json_escape "$FULL_RESPONSE")
      HISTORY+=("{\"role\":\"assistant\",\"content\":\"$escaped_resp\"}")
    fi

    # --- Timing summary ---
    if [ -n "$FIRST_TOKEN_TS" ]; then
      TTFT=$(awk -v s="$TURN_START" -v f="$FIRST_TOKEN_TS" 'BEGIN { printf "%.3f", f - s }')
      TOTAL=$(awk -v s="$TURN_START" -v e="$TURN_END" 'BEGIN { printf "%.3f", e - s }')
      TPOT=$(awk -v n="$TOKEN_COUNT" -v tt="$TTFT" -v total="$TOTAL" 'BEGIN { dn=n-1; if (dn>0) printf "%.1f", (total-tt)/dn*1000; else print "0" }')
      RATE=$(awk -v n="$TOKEN_COUNT" -v c="$CHUNK_COUNT" 'BEGIN { if (c>0) printf "%.2f", n / c; else print "0" }')
      echo -e "\n=================================================="
      echo "TTFT: ${TTFT}s | Total: ${TOTAL}s | TPOT: ${TPOT} ms/tok"
      echo "Output Tokens: $TOKEN_COUNT | Chunks: $CHUNK_COUNT | Accept Rate: $RATE"
      if [[ ! -z "$USAGE_RAW" ]]; then
        echo "$USAGE_RAW"
      fi
      echo "=================================================="
    else
      echo -e "\n=================================================="
      echo "(no response)"
      echo "=================================================="
    fi
  done

  exit 0
fi
