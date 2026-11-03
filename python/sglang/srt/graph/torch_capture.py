"""Opt-in automatic registration of ATen data operations during graph capture.

Native/custom operators must have explicit registered adapters. Data-dependent
scalar extraction and unsupported aliasing fail at capture, never silently
freeze Python values. The common C++ graph still owns planning and replay.
"""

import ctypes
import threading
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten
from sglang.srt.graph._capture import (
    is_capturing,
    lookup_or_register,
    register_output,
    record_op,
)

_libraries = []
_native_shapes = {}
_context = threading.local()
_views = {
    "aten::view",
    "aten::_unsafe_view",
    "aten::alias",
    "aten::detach",
    "aten::t",
    "aten::transpose",
    "aten::permute",
    "aten::slice",
    "aten::select",
    "aten::unsqueeze",
    "aten::squeeze",
    "aten::as_strided",
    "aten::split",
    "aten::split_with_sizes",
    "aten::unbind",
    "aten::expand",
}
_inplace = {
    "aten::copy_",
    "aten::zero_",
    "aten::fill_",
    "aten::index_copy_",
    "aten::index_put_",
    "aten::add_",
    "aten::mul_",
}
_dtypes = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.bool: 11,
    torch.bfloat16: 15,
}


def install_boxed_bridge(path):
    from sgl_kernel import graph_cpp

    library = ctypes.CDLL(str(path))
    _libraries.append(library)
    info = graph_cpp.OpInfo()
    info.dispatch_fn = ctypes.cast(
        library.registered_aten_dispatch, ctypes.c_void_p
    ).value
    graph_cpp.GraphOpRegistry.instance().register_op("aten_boxed", info)
    library.registered_aten_calls.restype = ctypes.c_uint64
    return library


def register_native_shape(operator, infer):
    """Register a pure functional native Torch op's explicit output contract."""
    schema = operator._schema
    if any(a.alias_info is not None for a in (*schema.arguments, *schema.returns)):
        raise ValueError(f"Native boxed adapters require functional schemas: {schema}")
    _native_shapes[(schema.name, schema.overload_name)] = infer


def native_call(operator, *args, **kwargs):
    """Explicit capture boundary for CatchAll kernels that bypass dispatch mode.

    Eager and C++ replay still invoke the original operator. Capture only records
    the registered shape/schema; it must not enter the native kernel body.
    """
    if not is_capturing():
        return operator(*args, **kwargs)
    schema = operator._schema
    if (schema.name, schema.overload_name) not in _native_shapes:
        raise RuntimeError(f"Explicit native graph shape not registered: {schema}")
    mode = getattr(_context, "mode", None) or AutomaticTorchCapture()
    with torch._C._DisableTorchDispatch():
        return mode.__torch_dispatch__(operator, (), args, kwargs)


def contiguous_clone(x, *, memory_format=None):
    """Use the existing planned copy only when clone's output is contiguous."""
    # is_contiguous alone ignores singleton strides; preserve-format clone
    # can retain those strides, whereas a planned contiguous output cannot.
    canonical = True
    expected = 1
    for size, stride in zip(reversed(x.shape), reversed(x.stride())):
        if stride != expected:
            canonical = False
            break
        expected *= max(int(size), 1)
    if memory_format is torch.contiguous_format or (
        memory_format in (None, torch.preserve_format) and canonical
    ):
        from sglang.srt.graph import ops

        return ops.contiguous_kunpeng(x)
    return NotImplemented


class AutomaticTorchCapture(TorchDispatchMode):
    def __init__(self, native_decompositions=None):
        super().__init__()
        self.operations = []
        # Provider-owned decomposition callbacks may use existing GraphOps.
        # Their persistent scratch must already be in the fixed bindings.
        self.native_decompositions = native_decompositions or {}
        self.decomposition_counts = {}

    def __enter__(self):
        self.previous = getattr(_context, "mode", None)
        super().__enter__()
        _context.mode = self
        return self

    def __exit__(self, *exc):
        _context.mode = self.previous
        return super().__exit__(*exc)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        schema = func._schema
        name = schema.name
        key = (name, schema.overload_name)
        decomposition = self.native_decompositions.get(key)
        if decomposition is not None:
            assert is_capturing(), "Native decomposition is capture-only"
            with torch._C._DisableTorchDispatch():
                output = decomposition(*args, **kwargs)
            if output is not NotImplemented:
                self.decomposition_counts[key] = (
                    self.decomposition_counts.get(key, 0) + 1
                )
                self.operations.append("registered_decomposition:" + str(func))
                return output
        native_shape = _native_shapes.get((name, schema.overload_name))
        if not name.startswith("aten::") and native_shape is None:
            raise RuntimeError(f"Native op needs a registered graph adapter: {schema}")
        if name in _views:
            return func(*args, **kwargs)
        if name in (
            "aten::_local_scalar_dense",
            "aten::nonzero",
            "aten::masked_select",
        ):
            raise RuntimeError(
                f"Data-dependent graph operation requires dynamic tensor adapter: {schema}"
            )
        mutable = any(
            a.alias_info is not None and a.alias_info.is_write for a in schema.arguments
        )
        if mutable and name not in _inplace:
            raise RuntimeError(f"Graph mutation contract is not registered: {schema}")
        if not mutable and any(r.alias_info is not None for r in schema.returns):
            raise RuntimeError(f"Graph alias contract is not registered: {schema}")
        normalized = []
        position = 0
        for argument in schema.arguments:
            if argument.name in kwargs:
                value = kwargs[argument.name]
            elif not argument.kwarg_only and position < len(args):
                value = args[position]
                position += 1
            elif argument.has_default_value():
                value = argument.default_value
            else:
                raise RuntimeError(f"Missing graph operator argument {argument.name}")
            # Python may coerce a scalar into a required Tensor for an ATen
            # overload. The boxed ABI does not perform that coercion. Reject
            # during capture, before replay can mutate persistent model state.
            if (
                str(argument.type) == "Tensor"
                and value is not None
                and not isinstance(value, torch.Tensor)
            ):
                raise RuntimeError(
                    f"Graph Tensor argument {argument.name} requires an explicit Tensor; scalar coercion is unsupported: {schema}"
                )
            normalized.append(value)
        tensors = []
        descriptors = []

        def pack(value):
            if isinstance(value, torch.Tensor):
                descriptors.extend(["T", len(tensors)])
                tensors.append(value)
            elif value is None:
                descriptors.append("N")
            elif isinstance(value, bool):
                descriptors.extend(["B", value])
            elif isinstance(value, int):
                descriptors.extend(["I", value])
            elif isinstance(value, float):
                descriptors.extend(["F", value])
            elif isinstance(value, str):
                descriptors.extend(["S", value])
            elif isinstance(value, torch.dtype):
                descriptors.extend(["I", _dtypes[value]])
            elif isinstance(value, torch.device):
                descriptors.extend(["D", str(value)])
            elif isinstance(value, (list, tuple)):
                descriptors.extend(["L", len(value)])
                for element in value:
                    pack(element)
            elif value is torch.strided:
                descriptors.extend(["I", 0])
            elif value in (
                torch.contiguous_format,
                torch.preserve_format,
                torch.channels_last,
                torch.channels_last_3d,
            ):
                descriptors.extend(
                    [
                        "I",
                        (
                            torch.contiguous_format,
                            torch.preserve_format,
                            torch.channels_last,
                            torch.channels_last_3d,
                        ).index(value),
                    ]
                )
            else:
                raise RuntimeError(f"Unsupported graph argument type {type(value)}")

        for value in normalized:
            pack(value)

        def meta(value):
            if isinstance(value, torch.Tensor):
                return torch.empty_strided(
                    value.shape, value.stride(), dtype=value.dtype, device="meta"
                )
            if isinstance(value, list):
                return [meta(x) for x in value]
            if isinstance(value, tuple):
                return tuple(meta(x) for x in value)
            return value

        meta_args = [meta(x) for x in normalized]
        for i, argument in enumerate(schema.arguments):
            if argument.name == "device":
                meta_args[i] = torch.device("meta")
        positional = []
        keywords = {}
        for argument, value in zip(schema.arguments, meta_args):
            if argument.kwarg_only:
                keywords[argument.name] = value
            else:
                positional.append(value)
        inferred = (
            native_shape(*positional, **keywords)
            if native_shape is not None
            else func(*positional, **keywords)
        )
        inputs = [lookup_or_register(t, idx=i) for i, t in enumerate(tensors)]
        outputs = []
        actual = []
        if mutable:
            if (
                not isinstance(inferred, torch.Tensor)
                or not tensors
                or tuple(inferred.shape) != tuple(tensors[0].shape)
            ):
                raise RuntimeError(f"Unsupported graph in-place result: {schema}")
            result = tensors[0]
        else:
            flat, spec = tree_flatten(inferred)
            if not flat or not all(isinstance(t, torch.Tensor) for t in flat):
                raise RuntimeError(
                    f"Graph operation returned non-tensor data: {schema}"
                )
            for t in flat:
                value = torch.empty_strided(
                    t.shape, t.stride(), dtype=t.dtype, device="cpu"
                )
                actual.append(value)
                outputs.append(register_output(value))
            result = tree_unflatten(actual, spec)
        record_op(
            "aten_boxed",
            inputs,
            outputs,
            [name, schema.overload_name, len(tensors), len(outputs), mutable]
            + descriptors,
            str(func),
        )
        self.operations.append(str(func))
        return result
