#!/bin/bash

PROFILE=false
MAX_TOKENS=10
NUM_REQUESTS=1

# support both: ./curl.sh -n 4  and  ./curl.sh 4
if [[ "$1" != -* ]] && [ -n "$1" ]; then
  NUM_REQUESTS=$1
  shift
fi

while getopts "pn:m:" opt; do
  case $opt in
    p) PROFILE=true ;;
    n) NUM_REQUESTS=$OPTARG ;;
    m) MAX_TOKENS=$OPTARG ;;
  esac
done

IP=$(ifconfig enp26s0f0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
PORT=30000

if [ "$PROFILE" = true ]; then
  curl --noproxy "*" http://${IP}:${PORT}/start_profile
fi

PROMPTS=(
  "Once upon a time"
  "The meaning of life is"
  "In the future, artificial intelligence"
  "Scientists have discovered that the deep-sea hydrothermal vents, often referred to as "black smokers," host a remarkable diversity of extremophilic microorganisms that thrive in complete darkness, high pressure, and temperatures exceeding 350 degrees Celsius. These microbes, which rely on chemosynthesis rather than photosynthesis, are now being studied for their unique enzymes that could revolutionize industrial biotechnology, including"
  "Scientists have discovered that"
  "Deep in the forest, there"
  "The history of mathematics shows"
  "When considering the problem of"
)

echo "Sending $NUM_REQUESTS concurrent requests with max_tokens=$MAX_TOKENS"

send_request() {
  local i=$1
  local prompt_idx=$(( (i - 1) % ${#PROMPTS[@]} ))
  local prompt="${PROMPTS[$prompt_idx]}"
  curl --noproxy "*" -s http://${IP}:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "deepseek-v2",
      "prompt": "'"$prompt"'",
      "stream": false,
      "max_tokens": '"$MAX_TOKENS"',
      "temperature": 0.01
    }' 2>&1
}

if [ "$NUM_REQUESTS" -eq 1 ]; then
  send_request 1
else
  for i in $(seq 1 $NUM_REQUESTS); do
    send_request $i &
  done
  wait
fi

if [ "$PROFILE" = true ]; then
  curl --noproxy "*" http://${IP}:${PORT}/stop_profile
fi
