#!/bin/bash

PROFILE=false
while getopts "p" opt; do
  case $opt in
    p) PROFILE=true ;;
  esac
done

IP=$(ifconfig enp26s0f0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
PORT=30000

if [ "$PROFILE" = true ]; then
  curl --noproxy "*" http://${IP}:${PORT}/start_profile
fi

time curl --noproxy "*" -s http://${IP}:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v2",
    "prompt": [
        "Once upon a time"
    ],
    "stream": true,
    "max_tokens": 10,
    "temperature": 0.01
  }'

if [ "$PROFILE" = true ]; then
  curl --noproxy "*" http://${IP}:${PORT}/stop_profile
fi
