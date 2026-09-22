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

usage() {
  echo "Usage:"
  echo "  $0 -f FILE [-s] [-m TOKENS] [-d RANK|RANGE] [-n NUM] [-r RATE] [-v] [-F] [-p] [-c CONC]"
  echo "  $0 [-i] [-d RANK] [-m TOKENS] [-h]                                   # interactive chat"
  echo ""
  echo "Benchmark mode (default, non-interactive): one prompt per request,"
  echo "sent at -r RATE req/s. -i starts a multi-turn interactive chat instead."
  echo ""
  echo "Options:"
  echo "  -h          Show this help message"
  echo "  -f FILE     Prompt file, one prompt per line (default: prompts/128.txt)."
  echo "              JSON-string lines (gen_st_prompts.py output) are spliced"
  echo "              verbatim."
  echo "  -s          Enable streaming mode"
  echo "  -m TOKENS   Max tokens per request / per turn (default: 10 benchmark, 1024 chat)"
  echo "  -d RANK|RANGE  DP rank (e.g. 5: every request pinned to it) or ranks"
  echo "              (e.g. 0-15, 0,2,5: requests round-robin across them)."
  echo "              Omit -d for the server default load balancing (round robin)."
  echo "  -n NUM      Number of requests (default: all lines from file; prompts"
  echo "              cycle when NUM exceeds the file size)"
  echo "  -r RATE     Send rate in req/s (default: 32, must be > 0)"
  echo "  -v          Verbose stats (only active when NUM > 1): decode throughput"
  echo "              table; with -s also accept rate. Writes per-request latency"
  echo "              rows + response bodies to <run-id>_detail.txt"
  echo "  -F          Fake-transfer mode: inject bootstrap_host=2.2.2.2 + a unique"
  echo "              bootstrap_room per request; decode skips KV transfer without"
  echo "              --disaggregation-transfer-backend. Port defaults to 30002."
  echo "  -p          Enable profiling (start/stop profile via separate curl calls)"
  echo "  -c CONC     Max concurrent requests (default: unbounded)"
  echo "  -i          Interactive chat mode — multi-turn streaming conversation"
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
PROMPT_FILE_SET=false
ROUND_ROBIN=false
RATE=32
FAKE=false

while getopts "d:hiFpsvn:m:c:r:f:" opt; do
  case $opt in
    h) usage ;;
    i) INTERACTIVE=true ;;
    p) PROFILE=true ;;
    s) STREAM=true ;;
    v) VERBOSE=true ;;
    F) FAKE=true ;;
    d) DP_ENABLED=true; DP_RANK=$OPTARG ;;
    n) NUM_REQUESTS=$OPTARG ;;
    c) CONCURRENCY=$OPTARG; CONCURRENCY_SET=true ;;
    r) RATE=$OPTARG ;;
    m) MAX_TOKENS=$OPTARG; MAX_TOKENS_SET=true ;;
    f) PROMPT_FILE=$OPTARG; PROMPT_FILE_SET=true ;;
    *) echo "Invalid option: -$OPTARG" >&2
       exit 1 ;;
  esac
done

shift $((OPTIND - 1))

# Pacing is always on: -r must be a positive rate (the old unlimited
# single-array-request mode is gone) and -n a non-negative count (0 = all).
if ! [[ "$RATE" =~ ^[0-9]+$ ]] || [ "$RATE" -le 0 ]; then
  echo "Error: -r must be a positive integer (req/s)" >&2
  exit 1
fi
if ! [[ "$NUM_REQUESTS" =~ ^[0-9]+$ ]]; then
  echo "Error: -n must be a non-negative integer" >&2
  exit 1
fi

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

# Target host/port: defaults to the local NIC (in-cluster router/instance).
# Override to hit a standalone decode master directly, e.g. fake-transfer
# decode-throughput tests:
#   CURL_HOST=<decode master IP> ./curl.sh -d 0-15 -n 512 -m 384
# With -F (fake transfer), the port defaults to 30002 — the decode tokenizer
# HTTP server on the route node — so the IP needs no adjustment.
IP=${CURL_HOST:-$(ifconfig enp26s0f0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')}
if [ "$FAKE" = true ]; then
  PORT=${CURL_PORT:-30002}
else
  PORT=${CURL_PORT:-30000}
fi
URL="http://${IP}:${PORT}/v1/completions"

# Fake-transfer injection: per-request unique bootstrap_room (ns timestamp
# base) + the magic fake host recognized by decode._is_fake_transfer.
ROOM_BASE=0
if [ "$FAKE" = true ]; then
  ROOM_BASE=$(date +%s%N)
fi

# =============================================================================
# Non-interactive dispatch: DP rank parsing + profile start
# =============================================================================

if [ "$INTERACTIVE" = false ]; then
  # -d with a range (e.g. 0-15, 0,2,5): requests round-robin across ranks.
  # -d with a single rank: every request pinned to that rank.
  # No -d: no routed_dp_rank — server default load balancing (round robin).
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

  if [ "$PROFILE" = true ]; then
    curl --noproxy "*" http://${IP}:${PORT}/start_profile
  fi
fi

# =============================================================================
# Benchmark mode: paced concurrent requests — one prompt per request,
# routed via -d (single rank / rank range / server default), paced via -r.
# =============================================================================

if [ "$INTERACTIVE" = false ]; then
  RESULT_DIR=$(mktemp -d)
  _cleaned=0
  cleanup() {
    # Idempotent cleanup: kill leftover send_request subshells and the progress
    # monitor first, wait for them to finish writing, then remove the temp
    # dir. This avoids two races:
    #   (a) "Directory not empty" (rm while children still write), and
    #   (b) "No such file or directory" (dir removed while children still write).
    # Both are killed by their RECORDED PIDs, not SIGINT: async subshells
    # ignore SIGINT (Ctrl+C only reaches the main script), so a monitor that
    # is not explicitly killed would outlive the script and keep redrawing
    # the bar on the terminal forever.
    if [ "$_cleaned" != "1" ]; then
      _cleaned=1
      [ -n "$_PROG_PID" ] && kill "$_PROG_PID" 2>/dev/null
      [ "${#_REQ_PIDS[@]}" -gt 0 ] && kill "${_REQ_PIDS[@]}" 2>/dev/null
      wait 2>/dev/null
      rm -rf -- "$RESULT_DIR" 2>/dev/null || true
    fi
  }
  # EXIT trap only cleans up; INT/TERM must actually exit, otherwise the
  # interrupted script resumes and touches RESULT_DIR/all_latencies after the
  # dir was removed ("No such file or directory").
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

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
    # Fake transfer: unique room per request + magic host; the decode side
    # (_is_fake_transfer) then force-selects the FAKE receiver and decodes
    # without any KV transfer.
    local fake_line=""
    if [ "$FAKE" = true ]; then
      fake_line=",\"bootstrap_host\": \"2.2.2.2\",\"bootstrap_room\": $((ROOM_BASE + idx))"
    fi
    local raw="${PROMPTS[$((idx % ${#PROMPTS[@]}))]}"
    local prompt_field
    # JSON string lines (gen_st_prompts.py) splice verbatim — they may carry
    # escaped newlines a plain-line re-escape would corrupt. Heuristic: a
    # line both starting and ending with '"' is taken as a JSON string.
    case "$raw" in
      '"'*'"') prompt_field="$raw" ;;
      *) prompt_field="\"$(json_escape "$raw")\"" ;;
    esac

    local body="{
        \"model\": \"DeepSeek-R1\",
        \"prompt\": $prompt_field,
        \"stream\": $STREAM,
        \"max_tokens\": $MAX_TOKENS,
        \"ignore_eos\": true,
        \"temperature\": 0$rank_line$fake_line"

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

    # -o keeps the response body clean; -w captures curl-INTERNAL clocks so
    # the TPOT decode span (time_total - starttransfer) is sampled inside
    # the curl process, immune to subshell scheduling gaps around fork/exit.
    # Captured in both modes: non-streaming stats use tokens/throughput only.
    if [ "$STREAM" = true ] && [ "$NUM_REQUESTS" -eq 1 ]; then
      # Single streaming request: echo each SSE line live (like the old
      # single-request path) while still capturing the body for parsing.
      # A FIFO splits curl's two outputs: body -> FIFO (read + echoed here),
      # -w clocks -> ttfb file, so the TPOT math stays curl-internal.
      local fifo="$RESULT_DIR/body_${idx}.fifo"
      mkfifo "$fifo"
      curl --noproxy "*" -N -s "$URL" \
        -H "Content-Type: application/json" \
        -H "X-Request-Id: $rid" \
        -o "$fifo" \
        -w '%{time_starttransfer} %{time_total}' \
        -d @"$body_file" > "$RESULT_DIR/ttfb_${idx}" 2>/dev/null &
      local curl_pid=$!
      while IFS= read -r line; do
        printf '%s\n' "$line"
        printf '%s\n' "$line" >> "$resp_file"
      done < "$fifo"
      wait "$curl_pid" 2>/dev/null
      rm -f "$fifo"
    else
      curl --noproxy "*" -s "$URL" \
        -H "Content-Type: application/json" \
        -H "X-Request-Id: $rid" \
        -o "$resp_file" \
        -w '%{time_starttransfer} %{time_total}' \
        -d @"$body_file" > "$RESULT_DIR/ttfb_${idx}" 2>/dev/null
    fi

    local end_ns
    end_ns=$(date +%s%N)
    echo "$start_ns $end_ns" > "$RESULT_DIR/time_${idx}"

    # Usage parsing is mode-agnostic: streaming carries usage in the final
    # chunk (continuous stats embed it in every chunk — take the last),
    # non-streaming in the response body. \s* tolerates spaced JSON.
    local comp_tokens prompt_tokens
    comp_tokens=$(grep -oP '"completion_tokens":\s*\K\d+' "$resp_file" 2>/dev/null | tail -n1)
    prompt_tokens=$(grep -oP '"prompt_tokens":\s*\K\d+' "$resp_file" 2>/dev/null | tail -n1)
    if [ "$STREAM" = true ]; then
      # Content chunks ≈ decode steps: spec decoding emits ALL tokens accepted
      # in one step inside a single chunk, so tokens/chunks is the per-request
      # accept rate. Count "text" keys only — usage-only chunks and
      # "data: [DONE]" carry no text field.
      local chunks
      chunks=$(grep -c '"text"' "$resp_file" 2>/dev/null || true)
      chunks="${chunks:-0}"
      # Prefer actual token count from usage; fall back to chunk count
      echo "${comp_tokens:-$chunks}" > "$RESULT_DIR/tokens_${idx}"
      echo "$chunks" > "$RESULT_DIR/chunks_${idx}"
    else
      echo "${comp_tokens:-0}" > "$RESULT_DIR/tokens_${idx}"
    fi
    # Prompt token count (aisbench InputTokens row)
    [[ ! -z "$prompt_tokens" ]] && echo "$prompt_tokens" > "$RESULT_DIR/ptokens_${idx}"

    rm -f "$body_file"
  }

  # Open-loop pacing: don't throttle on the client unless -c is given.
  if [ "$CONCURRENCY_SET" = false ]; then
    CONCURRENCY=$NUM_REQUESTS
  fi

  # Single request runs silent: only the response + TTFT/TPOT summary.
  if [ "$NUM_REQUESTS" -gt 1 ]; then
    echo "Paced batch: $NUM_REQUESTS requests, rate=$RATE req/s, concurrency=$CONCURRENCY"
    if [ "$ROUND_ROBIN" = true ]; then
      echo "  DP ranks: $DP_RANK ($NUM_RANKS ranks, round-robin)"
    elif [ "$DP_ENABLED" = true ]; then
      echo "  DP rank: $DP_RANK"
    fi
    echo "  URL: $URL"
    echo "  Max tokens/req: $MAX_TOKENS, Stream: $STREAM"
    echo "  Run ID: $RUN_ID  (grep this in router logs)"
    echo ""
  fi

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

  # Live progress bar on stderr: counts completed result files (tokens_* is
  # written last by send_request), refreshed every 0.5s. Doesn't pollute the
  # final stdout stats. Killed after wait.
  # Safety: also bounded by a wall-clock cap so it can NEVER outlive the main
  # `wait` (which would block script exit forever if some requests fail and
  # tokens_* never reaches NUM_REQUESTS).
  _PROG_PID=""
  if [ "$NUM_REQUESTS" -gt 1 ]; then
    (
      _S_W=28
      _PROG_FIRST=1
      # Hard wall-clock cap (1h): if the monitor is ever orphaned without a
      # cleanup kill, it still cannot redraw the bar forever.
      _PROG_DEADLINE=$(( START_NS + 3600000000000 ))
      while :; do
        # Sent = rid_* (written before each curl fires); Done = time_* (written
        # after EVERY request ends, success or fail). Both monotone and every
        # request writes both, so Done always reaches NUM_REQUESTS.
        _PROG_SENT=$(ls "$RESULT_DIR"/rid_* 2>/dev/null | wc -l)
        _PROG_DONE=$(ls "$RESULT_DIR"/time_* 2>/dev/null | wc -l)
        _PROG_PCT=$(( _PROG_DONE * 100 / NUM_REQUESTS ))
        _S_FILL=$(( _PROG_SENT * _S_W / NUM_REQUESTS ))
        _D_FILL=$(( _PROG_DONE * _S_W / NUM_REQUESTS ))
        if [ "$_PROG_FIRST" -eq 1 ]; then
          # First render: print both lines once.
          printf "\r  Sent [%s%s] %d/%d%s\n" \
            "$(printf '%*s' "$_S_FILL" '' | tr ' ' '#')" \
            "$(printf '%*s' $((_S_W - _S_FILL)) '')" \
            "$_PROG_SENT" "$NUM_REQUESTS" $'\033[K' >&2
          printf "  Done [%s%s] %d/%d (%d%%)%s" \
            "$(printf '%*s' "$_D_FILL" '' | tr ' ' '#')" \
            "$(printf '%*s' $((_S_W - _D_FILL)) '')" \
            "$_PROG_DONE" "$NUM_REQUESTS" "$_PROG_PCT" $'\033[K' >&2
          _PROG_FIRST=0
        else
          # Later ticks: cursor is on the Done line (below Sent), so move up
          # ONE line (\033[1A) back to Sent, then redraw both lines.
          printf "\033[1A\r  Sent [%s%s] %d/%d%s\n" \
            "$(printf '%*s' "$_S_FILL" '' | tr ' ' '#')" \
            "$(printf '%*s' $((_S_W - _S_FILL)) '')" \
            "$_PROG_SENT" "$NUM_REQUESTS" $'\033[K' >&2
          printf "  Done [%s%s] %d/%d (%d%%)%s" \
            "$(printf '%*s' "$_D_FILL" '' | tr ' ' '#')" \
            "$(printf '%*s' $((_S_W - _D_FILL)) '')" \
            "$_PROG_DONE" "$NUM_REQUESTS" "$_PROG_PCT" $'\033[K' >&2
        fi
        [ "$_PROG_DONE" -ge "$NUM_REQUESTS" ] && break
        [ "$(date +%s%N)" -ge "$_PROG_DEADLINE" ] && break
        sleep 0.5
      done
      _D_FILL=$(( _PROG_DONE * _S_W / NUM_REQUESTS ))
      printf "\033[1A\r  Sent [%s] %d/%d%s\n" \
        "$(printf '%*s' "$_S_W" '' | tr ' ' '#')" \
        "$_PROG_SENT" "$NUM_REQUESTS" $'\033[K' >&2
      printf "  Done [%s] %d/%d (%d%%)%s\n" \
        "$(printf '%*s' "$_D_FILL" '' | tr ' ' '#')" \
        "$_PROG_DONE" "$NUM_REQUESTS" "$_PROG_PCT" $'\033[K' >&2
    ) &
    _PROG_PID=$!
  fi

  _REQ_PIDS=()
  for ((i = 0; i < NUM_REQUESTS; i++)); do
    read -u 3
    {
      send_request "$i"
      echo >&3
    } &
    _REQ_PIDS+=($!)
    if (( (i + 1) % RATE == 0 )); then
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
  # Stop the progress monitor (it normally exits at 100%, this is a safety net).
  if [ -n "$_PROG_PID" ]; then
    kill "$_PROG_PID" 2>/dev/null
    wait "$_PROG_PID" 2>/dev/null
  fi
  WALL_END=$(date +%s.%N)
  exec 3>&-

  if [ "$PROFILE" = true ]; then
    curl --noproxy "*" http://${IP}:${PORT}/stop_profile
  fi

  # --- Compute and print results ---
  WALL_TIME=$(awk -v s="$WALL_START" -v e="$WALL_END" 'BEGIN { printf "%.3f", e - s }')

  # Single request: echo the response + a TTFT/TPOT summary (the old
  # single-request output format). -v is a no-op here by design.
  if [ "$NUM_REQUESTS" -eq 1 ]; then
    # Streaming already echoed the body live from the FIFO; only
    # non-streaming (buffered to the resp file) prints it here.
    if [ "$STREAM" = true ]; then
      echo ""
    elif [ -s "$RESULT_DIR/resp_0" ]; then
      cat "$RESULT_DIR/resp_0"
      echo ""
    else
      echo "<no response>"
    fi
    ttfb=""; total=""
    read -r ttfb total 2>/dev/null < "$RESULT_DIR/ttfb_0" || true
    toks=""; read -r toks 2>/dev/null < "$RESULT_DIR/tokens_0" || toks="?"
    echo ""
    echo "=================================================="
    if [[ "$ttfb" =~ ^[0-9.]+$ ]] && [[ "$total" =~ ^[0-9.]+$ ]]; then
      if [ "$STREAM" = true ]; then
        ch=0; read -r ch 2>/dev/null < "$RESULT_DIR/chunks_0" || ch=0
        # TPOT from curl-internal clocks: (time_total - starttransfer) /
        # (tokens - 1). Accept rate = (tokens-1)/(chunks-1). Both exclude
        # the prefill's first token/chunk.
        TPOT=$(awk -v t="$ttfb" -v tot="$total" -v n="$toks" \
          'BEGIN { dn=n-1; if (dn>0 && tot>t) printf "%.1f", (tot-t)/dn*1000; else print "N/A" }')
        ACC=$(awk -v t="$toks" -v c="$ch" \
          'BEGIN { if (c>1 && t ~ /^[0-9]+$/) printf "%.2f", (t-1)/(c-1); else print "N/A" }')
        echo "TTFT: ${ttfb}s | Total: ${total}s | TPOT: ${TPOT} ms/tok"
        echo "Output Tokens: $toks | Chunks: $ch | Accept Rate: $ACC"
      else
        echo "Total: ${total}s | Output Tokens: $toks"
      fi
    else
      echo "(no timing data)"
    fi
    echo "=================================================="
    exit 0
  fi

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
  elif [ "$DP_ENABLED" = true ]; then
    echo "  DP rank:             $DP_RANK"
  fi
  echo "  Send rate:           $RATE req/s"
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
      printf "%-6s %-6s %-12s %-12s %-10s %-8s %-8s %s\n" "Req#" "Rank" "Start(s)" "End(s)" "Latency(s)" "Tokens" "Accept" "ReqID"
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
      acc="N/A"
      read -r toks < "$RESULT_DIR/tokens_${i}" 2>/dev/null || toks="?"
      read -r rid < "$RESULT_DIR/rid_${i}" 2>/dev/null || rid="N/A"
      # Per-request accept rate = (tokens-1)/(chunks-1): the first chunk
      # carries the prefill's first token; only decode steps can accept
      # draft tokens.
      if [ -f "$RESULT_DIR/chunks_${i}" ]; then
        read -r ch < "$RESULT_DIR/chunks_${i}"
        acc=$(awk -v t="$toks" -v c="$ch" 'BEGIN {
          if (c > 1 && t ~ /^[0-9]+$/) printf "%.2f", (t-1)/(c-1); else print "N/A" }')
        # Collect for overall summary (only valid rows).
        [[ "$acc" != "N/A" ]] && echo "$toks $ch" >> "$RESULT_DIR/all_accepts"
      fi
      if [ "$VERBOSE" = true ]; then
        printf "%-6d %-6s %-12s %-12s %-10s %-8s %-8s %s\n" "$i" "$rank" "$rel_start" "$rel_end" "$dur" "$toks" "$acc" "$rid" >> "$DETAIL_FILE"
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

  # --- Accept Rate Summary (-v + streaming: needs decode-step counts) ---
  # Rows are "tokens chunks"; both sides exclude the prefill first token/step.
  if [ "$VERBOSE" = true ] && [ "$STREAM" = true ] && [ -s "$RESULT_DIR/all_accepts" ]; then
    awk '
      { t=$1+0; c=$2+0; if (c>1) {
          n++; st+=t-1; sc+=c-1; r=(t-1)/(c-1);
          if (min=="" || r<min) min=r;
          if (max=="" || r>max) max=r;
      } }
      END {
        if (n==0) exit;
        printf "\n===================================\n";
        printf "Accept Rate Summary (tokens/decode step, excl. first)\n";
        printf "===================================\n";
        printf "  Requests:  %d\n", n;
        printf "  Overall:   %.3f\n", st/sc;
        printf "  Min:       %.3f\n", min;
        printf "  Max:       %.3f\n", max;
        printf "===================================\n";
      }' "$RESULT_DIR/all_accepts"
  elif [ "$VERBOSE" = true ] && [ "$STREAM" = false ]; then
    echo ""
    echo "Accept Rate: N/A (requires streaming mode -s)"
  fi
  # --- Per-Request TPOT + aisbench-style decode table (-v only) ---
  # TPOT = (latency - TTFB) / (tokens - 1) — same semantics as ais_bench:
  # decode span (first streamed byte -> last byte) over the output-token
  # gaps; tokens is the usage completion_tokens, not max_tokens.
  # TPOT needs streaming (first-token timing); the other rows — per-request
  # InputTokens / OutputTokens / OutputTokenThroughput (= tokens / E2E
  # latency, ais_bench semantics) — are valid for non-streaming too.
  # Fork-free: builtin reads + integer-ns math, so large -n does not stall
  # on loaded nodes.
  if [ "$VERBOSE" = true ]; then
    AIS_FILE="$RESULT_DIR/ais_rows"
    : > "$AIS_FILE"
    for ((i = 0; i < NUM_REQUESTS; i++)); do
      [ -f "$RESULT_DIR/time_${i}" ] || continue
      [ -f "$RESULT_DIR/ttfb_${i}" ] || continue
      read -r s e < "$RESULT_DIR/time_${i}"
      toks=""; read -r toks < "$RESULT_DIR/tokens_${i}" 2>/dev/null || true
      ttfb=""; total=""; read -r ttfb total < "$RESULT_DIR/ttfb_${i}" 2>/dev/null || true
      # 2>/dev/null BEFORE <file: bash applies redirections left-to-right, so
      # a missing-file error must hit an already-redirected stderr.
      ptok=""; read -r ptok 2>/dev/null < "$RESULT_DIR/ptokens_${i}" || true
      [[ "$toks" =~ ^[0-9]+$ ]] || continue
      [ "$toks" -ge 1 ] || continue
      e2e_ns=$(( e - s ))
      [ "$e2e_ns" -gt 0 ] || continue
      # TPOT (streaming only — non-streaming TTFB ≈ total): decode span from
      # curl-internal clocks (time_total - starttransfer), no subshell
      # scheduling gaps, falling back to the subshell E2E clock for old
      # single-field files. Float seconds -> integer ns without awk forks.
      tpot="-"
      if [ "$STREAM" = true ] && [[ "$ttfb" =~ ^[0-9.]+$ ]] && [ "$toks" -ge 2 ]; then
        if [[ "$ttfb" == *.* ]]; then
          ttfb_s=${ttfb%%.*}; ttfb_f=${ttfb#*.}
          [[ "$ttfb_f" =~ ^[0-9]+$ ]] || ttfb_f=0
          ttfb_f="${ttfb_f}000000000"; ttfb_f="${ttfb_f:0:9}"
        else
          ttfb_s=$ttfb; ttfb_f=0
        fi
        ttfb_ns=$(( 10#${ttfb_s:-0} * 1000000000 + 10#$ttfb_f ))
        if [[ "$total" =~ ^[0-9.]+$ ]]; then
          if [[ "$total" == *.* ]]; then
            tot_s=${total%%.*}; tot_f=${total#*.}
            [[ "$tot_f" =~ ^[0-9]+$ ]] || tot_f=0
            tot_f="${tot_f}000000000"; tot_f="${tot_f:0:9}"
          else
            tot_s=$total; tot_f=0
          fi
          decode_ns=$(( (10#${tot_s:-0} * 1000000000 + 10#$tot_f) - ttfb_ns ))
        else
          decode_ns=$(( e2e_ns - ttfb_ns ))
        fi
        if [ "$decode_ns" -gt 0 ]; then
          per_tok_ns=$(( decode_ns / (toks - 1) ))
          printf -v tpot '%d.%03d' $(( per_tok_ns / 1000000 )) $(( (per_tok_ns / 1000) % 1000 ))
        fi
      fi
      # aisbench per-request throughput: output tokens / E2E latency (tok/s).
      thr_scaled=$(( toks * 10000000000000 / e2e_ns ))
      printf -v thr '%d.%04d' $(( thr_scaled / 10000 )) $(( thr_scaled % 10000 ))
      printf '%s %s %s %s\n' "$tpot" "${ptok:-?}" "$toks" "$thr" >> "$AIS_FILE"
    done
    # aisbench CSV layout: Average/Min/Max/Median/P75/P90/P99/N per metric;
    # N counts the requests that contributed to that row. The TPOT row is
    # emitted only for streaming (first-token timing exists).
    if [ -s "$AIS_FILE" ]; then
      if [ "$STREAM" = true ]; then SHOW_TPOT=1; else SHOW_TPOT=0; fi
      awk -v show_tpot="$SHOW_TPOT" '
        function pidx(n, p,   i) {
          i = int((n + 1) * p / 100)
          return i < 1 ? 1 : (i > n ? n : i)
        }
        function fmtc(v) { return (v == int(v)) ? sprintf("%d", v) : sprintf("%.4f", v) }
        function row(name, a, n, unit,   i, j, t, sum, s) {
          if (n == 0) { printf "%-22s %-7s %s\n", name, "total", "(no data)"; return }
          for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++)
            if (a[i] > a[j]) { t = a[i]; a[i] = a[j]; a[j] = t }
          for (i = 1; i <= n; i++) sum += a[i]
          s = sprintf("%-22s %-7s", name, "total")
          if (unit == "")
            s = s sprintf(" %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-7s", \
              fmtc(sum / n), fmtc(a[1]), fmtc(a[n]), fmtc(a[pidx(n, 50)]), \
              fmtc(a[pidx(n, 75)]), fmtc(a[pidx(n, 90)]), fmtc(a[pidx(n, 99)]), n)
          else
            s = s sprintf(" %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-7d", \
              sprintf("%.4f", sum / n) unit, sprintf("%.4f", a[1]) unit, \
              sprintf("%.4f", a[n]) unit, sprintf("%.4f", a[pidx(n, 50)]) unit, \
              sprintf("%.4f", a[pidx(n, 75)]) unit, sprintf("%.4f", a[pidx(n, 90)]) unit, \
              sprintf("%.4f", a[pidx(n, 99)]) unit, n)
          print s
        }
        {
          if ($1 ~ /^[0-9]+(\.[0-9]+)?$/) T[++nt] = $1 + 0
          if ($2 ~ /^[0-9]+$/) PI[++ni] = $2 + 0
          if ($3 ~ /^[0-9]+$/) PO[++no] = $3 + 0
          if ($4 ~ /^[0-9]+(\.[0-9]+)?$/) TH[++nh] = $4 + 0
        }
        END {
          printf "\n===============================================================\n"
          printf "Decode Throughput (aisbench style)\n"
          printf "===============================================================\n"
          printf "%-22s %-7s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-7s\n", \
            "Metric", "Stage", "Average", "Min", "Max", "Median", "P75", "P90", "P99", "N"
          if (show_tpot) row("TPOT", T, nt, " ms")
          row("InputTokens", PI, ni, "")
          row("OutputTokens", PO, no, "")
          row("OutputTokenThroughput", TH, nh, " token/s")
          printf "===============================================================\n"
        }
      ' "$AIS_FILE"
    fi
  fi

  if [ "$VERBOSE" = true ]; then
    {
      echo ""
      echo "==================================="
      echo "Per-Request Responses"
      echo "==================================="
      for ((i = 0; i < NUM_REQUESTS; i++)); do
        if [ "$ROUND_ROBIN" = true ]; then rank=${DP_RANKS[$((i % NUM_RANKS))]} ; else rank="-"; fi
        # Header carries the per-request accept stats (streaming only).
        acc="N/A"
        if [ -f "$RESULT_DIR/chunks_${i}" ]; then
          read -r ch < "$RESULT_DIR/chunks_${i}"
          read -r tk < "$RESULT_DIR/tokens_${i}" 2>/dev/null || tk="?"
          acc=$(awk -v t="$tk" -v c="$ch" 'BEGIN {
            if (c > 1 && t ~ /^[0-9]+$/) printf "%.2f", (t-1)/(c-1); else print "N/A" }')
          echo "----- Request #$i (rank=$rank, tokens=$tk, steps=$ch, accept=$acc) -----"
        else
          echo "----- Request #$i (rank=$rank) -----"
        fi
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
# Mode 2: Interactive chat — multi-turn streaming conversation
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
      # Same convention as batch mode: exclude the first chunk (prefill token).
      RATE=$(awk -v n="$TOKEN_COUNT" -v c="$CHUNK_COUNT" 'BEGIN { if (c>1) printf "%.2f", (n-1)/(c-1); else print "N/A" }')
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
