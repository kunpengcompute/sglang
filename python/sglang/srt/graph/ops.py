import torch

from sglang.srt.graph._capture import (
    is_capturing,
    lookup_or_register,
    register_output,
    record_op,
)

_DEBUG = False


class GraphOp:
    def __init__(self, name, shape_infer, eager_fn=None):
        self.name = name
        self.shape_infer = shape_infer
        self.eager_fn = eager_fn

    def __call__(self, *args, **kwargs):
        profile_name = kwargs.pop('profile_name', '')
        if is_capturing():
            if _DEBUG:
                print(f"[capture] {self.name}", flush=True)
            return self._capture(args, kwargs, profile_name)
        if self.eager_fn:
            if _DEBUG:
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
        outputs = []
        for info in out_infos:
            if isinstance(info, tuple):
                shape, dtype = info
            else:
                shape, dtype = info, tensor_args[0].dtype
            out = torch.empty(shape, dtype=dtype)
            vid, so = register_output(out)
            outputs.append((vid, so))
            output_tensors.append(out)

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


def register_op(name, shape_infer, eager_fn=None):
    op = GraphOp(name, shape_infer, eager_fn)
    _registry[name] = op


ops = _Ops()
