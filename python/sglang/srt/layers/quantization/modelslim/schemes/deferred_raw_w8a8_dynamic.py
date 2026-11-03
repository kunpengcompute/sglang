"""Raw original I8/F32 storage for loaded, execution-disabled components.

Inherit allocation/parameter-loader ABI, never the native pack/apply operation.
No model-name or geometry-specific native path is introduced here.
"""

from .cpu920f_w8a8_dynamic import CPU920FModelSlimW8A8Dynamic, validate_int8_payload
from sglang.srt.model_executor.model_capabilities import ModelCapabilityError


class DeferredRawW8A8Dynamic(CPU920FModelSlimW8A8Dynamic):
    def process_weights_after_loading(self, layer):
        if getattr(layer, "_cpu920f_w8a8_packed", False):
            raise RuntimeError("Disabled component was packed")
        if getattr(layer, "_raw_prepack_deferred", False):
            raise RuntimeError("Duplicate deferred completion")
        validate_int8_payload(layer.weight, layer.weight_scale, layer.weight_offset)
        layer._raw_prepack_deferred = True

    def apply_weights(self, layer, x, bias=None):
        raise ModelCapabilityError(
            "INT8 component loaded_not_executable; prepack deferred"
        )
