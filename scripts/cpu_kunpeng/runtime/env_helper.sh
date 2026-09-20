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
# Helper functions for env.sh. This file is sourced by env.sh and should not
# be executed directly.

# ------------------------------------------------------------
# Helper: expand IP range notation "base_ip | ranges"
# Example: "192.168.1. | 1-3,5" -> "192.168.1.1 192.168.1.2 192.168.1.3 192.168.1.5"
# If a second argument (IP_FILE) is given and the file exists,
# read one IP per line instead (empty lines and '#' comments are skipped).
# ------------------------------------------------------------
expand_ip_range() {
    local spec="$1"
    local ip_file="$2"
    local ips=()

    # If an IP file is provided, read one IP per line
    if [[ -n "$ip_file" ]] && [[ -f "$ip_file" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            line="${line//[$'\r' ]/}"  # strip CR and spaces
            [[ -z "$line" ]] && continue
            [[ "$line" == \#* ]] && continue
            ips+=("$line")
        done < "$ip_file"
        echo "${ips[@]}"
        return 0
    fi

    # Collect all IP-base prefixes (e.g. "10.36.182.") in order
    local bases=() temp="$spec"
    while [[ "$temp" =~ ([0-9]+\.[0-9]+\.[0-9]+\.) ]]; do
        bases+=("${BASH_REMATCH[1]}")
        temp="${temp#*"${BASH_REMATCH[1]}"}"
    done

    for ((idx=0; idx<${#bases[@]}; idx++)); do
        local base="${bases[idx]}"

        # Extract substring between this base and the next base (or end of string)
        local sub="${spec#*"$base"}"
        sub="${sub#"${sub%%[![:space:]]*}"}"  # trim leading spaces
        sub="${sub#|}"
        sub="${sub#"${sub%%[![:space:]]*}"}"  # trim leading spaces

        if ((idx+1 < ${#bases[@]})); then
            local next_base="${bases[idx+1]}"
            sub="${sub%%"$next_base"*}"
        fi

        # Trim trailing spaces / commas
        sub="${sub%"${sub##*[![:space:]]}"}"

        IFS=',' read -ra parts <<< "$sub"
        for part in "${parts[@]}"; do
            part="${part// /}"
            [[ -z "$part" ]] && continue
            if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            # range: start-end
                for ((i=${BASH_REMATCH[1]}; i<=${BASH_REMATCH[2]}; i++)); do
                    ips+=("${base}${i}")
                done
            elif [[ "$part" =~ ^[0-9]+$ ]]; then
            # single number
                ips+=("${base}${part}")
            else
                echo "Error: invalid IP range part '$part'" >&2
                return 1
            fi
        done
    done

    echo "${ips[@]}"
}


# ------------------------------------------------------------
# Helper: per-instance router-node resource indexes.
# _instance_indexes <entry> prints "<global_idx> <side_idx>" for the entry
# ("<role>" or "<role>_<instance>") within the comma-separated INSTANCES
# list: global_idx = position among all entries, side_idx = position among
# entries of the same role prefix. Both are -1 when the entry is not
# listed (caller falls back to that side's single-instance defaults).
# ------------------------------------------------------------
_instance_indexes() {
    local entry="$1" role="${1%%_*}"
    local g=0 s=0 gidx=-1 sidx=-1 e r
    IFS=',' read -ra _il <<< "${INSTANCES:-}"
    for e in "${_il[@]}"; do
        e="${e//[[:space:]]/}"
        [[ -z "$e" ]] && continue
        if [[ "$e" == "$entry" ]]; then
            gidx=$g; sidx=$s
            break
        fi
        ((g++))
        r="${e%%_*}"
        [[ "$r" == "$role" ]] && ((s++))
    done
    echo "$gidx $sidx"
}


# ------------------------------------------------------------
# Helpers: resolve variables by role prefix (PREFILL / DECODE / NATIVE)
# ------------------------------------------------------------
_export_node_config() {
    local prefix="$1"
    local _var
    _var="${prefix}_IP_FILE"; local _ip_file="${!_var:-}"
    _var="${prefix}_IP_SPEC"; NODE_IPS=($(expand_ip_range "${!_var}" "$_ip_file"))
    _var="${prefix}_MASTER_ADDR"; export MASTER_ADDR="${!_var}"
    _var="${prefix}_MASTER_PORT"; export MASTER_PORT="${!_var}"
    export WORLD_SIZE=${#NODE_IPS[@]}
    export NODE_IPS_LIST="${NODE_IPS[*]}"
}

_export_pd_vars() {
    local prefix="$1"
    local _var
    _var="${prefix}_TP_SIZE";    export TP_SIZE="${!_var}"
    _var="${prefix}_DP_SIZE";    export DP_SIZE="${!_var}"
    _var="${prefix}_EP_SIZE";    export EP_SIZE="${!_var}"
    _var="${prefix}_PP_SIZE";    export PP_SIZE="${!_var}"
    _var="${prefix}_REDUNDANT_EXPERTS"; export REDUNDANT_EXPERTS="${!_var}"
    _var="${prefix}_INIT_EXPERT_LOCATION"; export INIT_EXPERT_LOCATION="${!_var}"
    _var="${prefix}_EP_DISPATCH_ALGORITHM"; export EP_DISPATCH_ALGORITHM="${!_var}"
    _var="${prefix}_SGLANG_KUNPENG_MOE_SHUFFLE_MODE"; export SGLANG_KUNPENG_MOE_SHUFFLE_MODE="${!_var}"
    _var="MODEL_PATH_${prefix}"; export MODEL_PATH="${!_var}"
    _var="SPECULATIVE_DRAFT_MODEL_PATH_${prefix}"; export SPECULATIVE_DRAFT_MODEL_PATH="${!_var}"
    _var="${prefix}_WEIGTHS_HBW_POOL_SIZE_MB"; export SGLANG_KUNPENG_WEIGTHS_HBW_POOL_SIZE_MB="${!_var}"
    _var="${prefix}_SWAP_KV_IN"; export SGLANG_KUNPENG_SWAP_KV_IN="${!_var}"
    _var="${prefix}_SWAP_KV_OUT"; export SGLANG_KUNPENG_SWAP_KV_OUT="${!_var}"
    _var="${prefix}_SWAP_KV_BLOCKWISE"; export SGLANG_KUNPENG_SWAP_KV_BLOCKWISE="${!_var}"
}

# ------------------------------------------------------------
# Per-role config functions (called via "${ACTION}_config")
# ------------------------------------------------------------
prefill_config() {
    _export_pd_vars "PREFILL"
    _export_node_config "PREFILL"
    export SGLANG_SKIP_HTTP=1
}

decode_config() {
    _export_pd_vars "DECODE"
    _export_node_config "DECODE"
    export SGLANG_SKIP_HTTP=1
}

native_config() {
    _export_node_config "NATIVE"
}

tokenizer_config() {
    export NODE_IPS_LIST="$ROUTER_IP"
    export SGLANG_LAUNCH_HTTP_ONLY=1
}

router_config() {
    export NODE_IPS_LIST="$ROUTER_IP"
}

build_config() {
    :
}
