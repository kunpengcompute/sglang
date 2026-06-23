#!/bin/bash
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

UPDATE_SGLANG=false
UPDATE_KERNEL=false

case "$1" in
    sglang)
        UPDATE_SGLANG=true
        ;;
    kernel)
        UPDATE_KERNEL=true
        ;;
    ""|all)
        UPDATE_SGLANG=true
        UPDATE_KERNEL=true
        ;;
    *)
        echo "Usage: $0 [sglang|kernel|all] (default: all)"
        exit 1
        ;;
esac

for i in $(seq 0 15); do
    (
        if [ "${UPDATE_SGLANG}" = "true" ]; then
            rm -rf "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sglang"
            cp -rf "$SGLANG_PATH/python/sglang" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sglang"
        fi
        if [ "${UPDATE_KERNEL}" = "true" ]; then
            rm -rf "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sgl_kernel"
            cp -rf "$SITE_PACKAGES/sgl_kernel" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sgl_kernel"
        fi
    ) &
done
wait