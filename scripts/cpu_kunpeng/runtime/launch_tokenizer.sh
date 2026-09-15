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
# launch_tokenizer.sh - Launch ONE tokenizer HTTP server on the router node
# (prefill: port 30001 / decode: port 30002), via runtime/server_tokenizer.sh.
# Invoked by: ./launch.sh tokenizer <prefill|decode> [instance]
#   instance (e.g. "128p") selects the per-instance env file
#   (runtime/.user_env_<side>_<instance>.sh) so the tokenizer points at
#   that instance's cluster (master addr, DP size, ...).

if [[ $# -lt 1 || $# -gt 2 || ( "$1" != "prefill" && "$1" != "decode" ) ]]; then
    echo "Usage: $0 <prefill|decode> [instance]" >&2
    exit 1
fi

TOK_SIDE="$1"
TOK_INSTANCE="${2:-}"

# Instance name must be a plain identifier (used inside file names)
if [[ -n "$TOK_INSTANCE" && ! "$TOK_INSTANCE" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Error: invalid instance name '$TOK_INSTANCE'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# env.sh's 2nd arg for tokenizer is "<side>[_<instance>]": only that
# side's role env is loaded (with the instance file if given).
source "$SCRIPT_DIR/env.sh" tokenizer "${TOK_SIDE}${TOK_INSTANCE:+_$TOK_INSTANCE}"

# Note: safe to start in parallel with the prefill/decode servers. The
# http-only process never runs torch.distributed init (dist-init-addr is
# unused), its ZMQ PUSH to the remote scheduler lazily queues until the
# compute side binds, and the bootstrap server (port 9001) only listens
# for inbound registrations.

mkdir -p "$LOG_DIR"
# Refresh "latest" symlink to this run's time dir (e.g. latest -> 214700)
ln -sfn "$LOG_TIME" "$LOG_BASE_DIR/$LOG_DATE/tokenizer/latest"
sh "$SCRIPT_DIR/stop.sh" tokenizer "${TOK_SIDE}${TOK_INSTANCE:+_$TOK_INSTANCE}"

echo "[$(date +%T)] Launching $TOK_SIDE${TOK_INSTANCE:+ ($TOK_INSTANCE)} tokenizer HTTP server on $ROUTER_IP"
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
    "root@$ROUTER_IP" \
    "cd \"$PWD\" && bash runtime/server_tokenizer.sh \"$TOK_SIDE\" \"$LOG_DIR\" \"$TOK_INSTANCE\"" \
    >"$LOG_DIR/ssh_tokenizer_${TOK_SIDE}${TOK_INSTANCE:+_${TOK_INSTANCE}}.log" 2>&1 &

echo "Tokenizer launched. The remote script polls until the port is ready."

if [[ "${SKIP_LOG:-0}" == "1" ]]; then
    exit 0
fi

rank0_log_file="$LOG_DIR/tokenizer_${TOK_SIDE}_http.log"
echo "Log file of rank_0: $rank0_log_file"

while [ ! -f "$rank0_log_file" ]; do
    sleep 1
done
tail -f "$rank0_log_file"
