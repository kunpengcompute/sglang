"""Load the complete Qwen checkpoint through its parameter loader callbacks.

The name/shard map contains model geometry, not data hashes or deployment paths.
All components are loaded; unsupported vision/MTP execution remains gated by the
model capability policy. Packing is still performed by DefaultModelLoader.
"""

import json
import re
from pathlib import Path


class UnifiedCheckpointLoader:
    def __init__(self, model, spec=None):
        from sglang.srt.distributed import (
            get_pp_group,
            get_tensor_model_parallel_world_size,
        )

        self.model = model
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size != 16 or get_pp_group().world_size != 1:
            raise ValueError("Qwen W8A8 checkpoint mapping currently supports TP16/PP1")
        mapping = (
            Path(__file__).parents[1]
            / "hardware_backend"
            / "cpu_kunpeng"
            / "qwen"
            / "weight_mapping.json"
        )
        self.rules = json.loads(mapping.read_text())["rules"]
        self._bound = self._loaded = self._postprocessed = False
        self._postprocess_seen = set()

    def bind(self, model_config, load_config):
        if (
            str(model_config.dtype) != "torch.bfloat16"
            or model_config.quantization != "modelslim"
        ):
            raise ValueError(
                "Qwen W8A8 requires BF16 activations and ModelSlim weights"
            )
        if getattr(load_config, "draft_model_idx", None) is not None:
            raise ValueError("This loader handles the complete target checkpoint")
        self._bound = True

    def load(self, weights):
        import torch
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        if not self._bound or self._loaded:
            raise RuntimeError("Checkpoint loader must be bound and called once")
        params = dict(self.model.named_parameters(remove_duplicate=False))
        loaded_ids, seen = set(), set()
        expected_sources = set()
        for source, rule in self.rules.items():
            pattern = re.escape(rule["target"]).replace(r"\{index\}", r"(\d+)")
            for target in params:
                match = re.fullmatch(pattern, target)
                if match:
                    expected_sources.add(
                        source.format(index=match[1] if match.groups() else "")
                    )
        dtypes = {"I8": torch.int8, "BF16": torch.bfloat16, "F32": torch.float32}
        for name, tensor in weights:
            if name in seen:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            seen.add(name)
            match = re.search(r"\.(?:layers|blocks)\.(\d+)\.", name)
            key = re.sub(r"\.(layers|blocks)\.\d+\.", r".\1.{index}.", name)
            rule = self.rules.get(key)
            if rule is None:
                raise ValueError(f"Unsupported checkpoint tensor: {name}")
            if (
                list(tensor.shape) != rule["source_shape"]
                or tensor.dtype != dtypes[rule["source_dtype"]]
            ):
                raise ValueError(f"Checkpoint dtype/geometry mismatch: {name}")
            target = rule["target"].format(index=match[1] if match else "")
            param = params[target]
            target_dtype = dtypes[rule["dtype"]]
            # Vision quantized biases are F32 in the original checkpoint. Keep
            # their bits instead of copying through the default BF16 allocation.
            if param.dtype != target_dtype:
                if rule["component"] != "vision" or not target.endswith(".bias"):
                    raise ValueError(f"Unexpected target dtype: {target}")
                owner_name, leaf = target.rsplit(".", 1)
                replacement = torch.nn.Parameter(
                    torch.empty_like(param, dtype=target_dtype), requires_grad=False
                )
                replacement.__dict__.update(param.__dict__)
                setattr(self.model.get_submodule(owner_name), leaf, replacement)
                params[target] = param = replacement
            loader = getattr(param, "weight_loader", default_weight_loader)
            shards = rule["shard"]
            if isinstance(shards, list):
                for shard, (start, stop) in zip(shards, rule["chunks"]):
                    loader(param, tensor.narrow(0, start, stop - start), shard)
            elif shards is None:
                loader(param, tensor)
            else:
                loader(param, tensor, shards)
            loaded_ids.add(id(param))
        if seen != expected_sources:
            raise ValueError(
                f"Checkpoint source coverage mismatch: missing={sorted(expected_sources - seen)}, "
                f"unexpected={sorted(seen - expected_sources)}"
            )
        missing = [
            name for name, param in params.items() if id(param) not in loaded_ids
        ]
        if missing:
            raise ValueError(f"Checkpoint did not load model parameters: {missing}")
        self._loaded = True
        return set(params)

    def defer_postprocess(self, module):
        if not self._loaded:
            raise RuntimeError("Packing cannot precede checkpoint loading")
        component = getattr(module, "_runtime_component", "language")
        if component == "language":
            return False
        if self.model.runtime_capabilities.policy_enabled(component):
            return False
        scheme = getattr(module, "scheme", None)
        if scheme is not None:
            from sglang.srt.layers.quantization.modelslim.schemes.deferred_raw_w8a8_dynamic import (
                DeferredRawW8A8Dynamic,
            )

            if not isinstance(scheme, DeferredRawW8A8Dynamic):
                raise ValueError(
                    "Disabled component must retain its raw quantized weights"
                )
            scheme.process_weights_after_loading(module)
        module._postprocess_deferred = True
        return True

    def after_postprocess(self, name, module):
        if name in self._postprocess_seen:
            raise RuntimeError(f"Repeated quantization postprocess: {name}")
        self._postprocess_seen.add(name)

    def finish(self):
        if not self._loaded or self._postprocessed:
            raise RuntimeError("Invalid checkpoint finalization")
        for component in ("language", "vision", "mtp"):
            self.model.runtime_capabilities.loaded_validated(component)
        self._postprocessed = True
