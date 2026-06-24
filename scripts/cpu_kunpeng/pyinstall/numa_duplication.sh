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

BASE_DIR="./dist"
SOURCE="${BASE_DIR}/sglang_server"
PREFIX="sglang_server_tp"

if [ ! -e "$SOURCE" ]; then
    echo "[numa_duplication] error: source file: $SOURCE not found!"
    exit 1
fi

if [ -e "$SOURCE" ]; then
    mv "$SOURCE" "${BASE_DIR}/${PREFIX}0"
    echo "[numa_duplication] source file rename to: ${PREFIX}0"
else
    echo "[numa_duplication] source file $SOURCE does not exist, skipping rename"
fi

for i in $(seq 1 15); do
    echo "[numa_duplication] copy to: ${PREFIX}${i}"
    cp -r "${BASE_DIR}/${PREFIX}0" "${BASE_DIR}/${PREFIX}${i}" &
done
wait

echo "[numa_duplication] complete!"