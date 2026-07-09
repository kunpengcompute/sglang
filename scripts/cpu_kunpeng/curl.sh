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
  echo "Usage: $0 [-p] [-s] [-d RANK] [-n NUM] [-m TOKENS] [-f FILE] [-h]"
  echo ""
  echo "Options:"
  echo "  -p          Enable profiling (start/stop profile)"
  echo "  -s          Enable streaming mode"
  echo "  -d RANK     Route to specific DP rank"
  echo "  -n NUM      Number of requests / prompt entries (default: all lines from file)"
  echo "  -m TOKENS   Max tokens per request (default: 10)"
  echo "  -f FILE     Prompt file with one prompt per line (default: prompts/5.txt)"
  echo "  -h          Show this help message"
  exit 0
}

PROFILE=false
MAX_TOKENS=10
NUM_REQUESTS=0
STREAM=false
DP_ENABLED=false
DP_RANK=0
PROMPT_FILE="prompts/5.txt"

while getopts "d:hpsn:m:f:" opt; do
  case $opt in
    h) usage ;;
    p) PROFILE=true ;;
    s) STREAM=true ;;
    d) DP_ENABLED=true; DP_RANK=$OPTARG ;;
    n) NUM_REQUESTS=$OPTARG ;;
    m) MAX_TOKENS=$OPTARG ;;
    f) PROMPT_FILE=$OPTARG ;;
    *) echo "Invalid option: -$OPTARG" >&2
       exit 1 ;;
  esac
done

shift $((OPTIND - 1))

if [ ! -f "$PROMPT_FILE" ]; then
  echo "Error: file not found: $PROMPT_FILE" >&2
  exit 1
fi

# Read prompts from file (one per line)
PROMPTS=()
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

IP=$(ifconfig enp26s0f0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
PORT=30000

if [ "$PROFILE" = true ]; then
  curl --noproxy "*" http://${IP}:${PORT}/start_profile
fi

if [ "$DP_ENABLED" = true ]; then
  DP_LINE=",\"routed_dp_rank\": $DP_RANK"
else
  DP_LINE=""
fi

BODY="{
    \"model\": \"deepseek-v2\",
    \"prompt\": $PROMPT_JSON,
    \"stream\": $STREAM,
    \"max_tokens\": $MAX_TOKENS,
    \"temperature\": 0.01$DP_LINE
  }"

time curl --noproxy "*" -s http://${IP}:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d "$BODY"

if [ "$PROFILE" = true ]; then
  curl --noproxy "*" http://${IP}:${PORT}/stop_profile
fi
