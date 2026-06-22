#!/bin/bash

BASE_DIR="./dist"
SOURCE="${BASE_DIR}/sglang_server"
PREFIX="sglang_server_tp"

if [ ! -e "$SOURCE" ]; then
    echo "[numa_duplication] error: source file: $SOURCE not found!"
    exit 1
fi

mv "$SOURCE" "${BASE_DIR}/${PREFIX}0"
echo "[numa_duplication] source file rename to: ${PREFIX}0"

for i in $(seq 1 15); do
    echo "[numa_duplication] copy to: ${PREFIX}${i}"
    cp -r "${BASE_DIR}/${PREFIX}0" "${BASE_DIR}/${PREFIX}${i}" &
done
wait

echo "[numa_duplication] complete!"