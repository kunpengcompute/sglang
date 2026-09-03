import torch

from sglang.srt.graph._capture import (
    is_capturing,
    lookup_or_register,
    record_op,
    register_output,
    upgrade_storage_memory_type,
)

_DEBUG = False
_IDLE = False

class idle_forward_mode:
    """Mark the current forward as IDLE to suppress op debug prints."""

    def __enter__(self):
        global _IDLE
        _IDLE = True

    def __exit__(self, *exc):
        global _IDLE
        _IDLE = False


class GraphOp:
    def __init__(self, name, shape_infer, eager_fn=None, shm_fn=None):
        self.name = name
        self.shape_infer = shape_infer
        self.eager_fn = eager_fn
        self.shm_fn = shm_fn

    def __call__(self, *args, **kwargs):
        profile_name = kwargs.pop('profile_name', '')
        if is_capturing():
            if _DEBUG and not _IDLE:
                print(f"[capture] {self.name}", flush=True)
            return self._capture(args, kwargs, profile_name)
        if self.eager_fn:
            if _DEBUG and not _IDLE:
                print(f"[eager] {self.name}", flush=True)
            return self.eager_fn(*args, **kwargs)
        raise RuntimeError("Op not available outside capture graph")

    def _capture(self, args, kwargs, profile_name):
        tensor_args = [a for a in args if isinstance(a, torch.Tensor) or a is None]
        non_tensor_args = [a for a in args if not isinstance(a, torch.Tensor) and a is not None]

        inputs = [lookup_or_register(t, idx=i) for i, t in enumerate(tensor_args)]

        out_infos = self.shape_infer(*args, **kwargs)
        if not isinstance(out_infos, list):
            out_infos = [out_infos]

        output_tensors = []
        for info in out_infos:
            if isinstance(info, tuple):
                shape, dtype = info
            else:
                shape, dtype = info, tensor_args[0].dtype
            if any(s == 0 for s in shape):
                out = torch.empty(1, dtype=dtype)[:0].view(shape)
            else:
                out = torch.empty(shape, dtype=dtype)
            output_tensors.append(out)

        # shm_fn receives the kernel-expanded args:
        # input tensors, then output tensors, then scalar args.
        shm_ids = set()
        if self.shm_fn is not None:
            shm_tensors = self.shm_fn(
                *tensor_args, *output_tensors, *non_tensor_args, **kwargs) or []
            shm_ids = {id(t) for t in shm_tensors if t is not None}

        outputs = []
        for out in output_tensors:
            memory_type = 'shm' if id(out) in shm_ids else 'regular'
            vid, so = register_output(out, memory_type=memory_type)
            outputs.append((vid, so))

        for t in tensor_args:
            if t is not None and id(t) in shm_ids:
                upgrade_storage_memory_type(t)

        record_op(self.name, inputs, outputs, non_tensor_args, profile_name)

        if len(output_tensors) == 1:
            return output_tensors[0]
        return tuple(output_tensors)


class _Ops:
    def __getattr__(self, name):
        op = _registry.get(name)
        if op is None:
            raise AttributeError(f"op '{name}' not registered")
        return op


_registry = {}


def register_op(name, shape_infer, eager_fn=None, shm_fn=None):
    op = GraphOp(name, shape_infer, eager_fn, shm_fn)
    _registry[name] = op


ops = _Ops()
