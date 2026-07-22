import torch
import torch.nn as nn

_DEBUG = False


def collect_model_weights(module: nn.Module, prefix: str = "",
                          result: list | None = None,
                          seen: set | None = None,
                          depth: int = 0) -> list[torch.Tensor]:
    if result is None:
        result = []
    if seen is None:
        seen = set()

    indent = "  " * depth
    children = dict(module.named_children())

    for name, param in module.named_parameters(recurse=False):
        key = param.untyped_storage().data_ptr()
        if key and key not in seen:
            seen.add(key)
            result.append(param)
            if _DEBUG:
                print(f"{indent}[PARAM ] {prefix}.{name}  "
                      f"shape={tuple(param.shape)}  dtype={param.dtype}  "
                      f"base=0x{key:x}  device={param.device}")

    for name, buf in module.named_buffers(recurse=False):
        key = buf.untyped_storage().data_ptr()
        if key and key not in seen:
            seen.add(key)
            result.append(buf)
            if _DEBUG:
                print(f"{indent}[BUFFER] {prefix}.{name}  "
                      f"shape={tuple(buf.shape)}  dtype={buf.dtype}  "
                      f"base=0x{key:x}")

    skip = set(module._parameters.keys()) | set(module._buffers.keys())
    extra = []
    for name in dir(module):
        if name.startswith('_') or name in skip:
            continue
        try:
            val = getattr(module, name, None)
        except Exception:
            continue
        if isinstance(val, torch.Tensor):
            extra.append((name, val))
    for name, val in sorted(extra):
        key = val.untyped_storage().data_ptr()
        if key and key not in seen:
            seen.add(key)
            result.append(val)
            if _DEBUG:
                print(f"{indent}[EXTRA ] {prefix}.{name}  "
                      f"shape={tuple(val.shape)}  dtype={val.dtype}  "
                      f"base=0x{key:x}")

    for name, child in children.items():
        child_prefix = f"{prefix}.{name}" if prefix else name
        collect_model_weights(child, child_prefix, result, seen, depth + 1)

    return result
