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

PROFILE=false
MAX_TOKENS=10
while getopts "pm:" opt; do
  case $opt in
    p) PROFILE=true ;;
    m) MAX_TOKENS=$OPTARG ;;
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
    "max_tokens": '"$MAX_TOKENS"',
    "temperature": 0.01
  }'

if [ "$PROFILE" = true ]; then
  curl --noproxy "*" http://${IP}:${PORT}/stop_profile
fi
