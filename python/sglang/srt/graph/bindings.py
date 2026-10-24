"""Backend-owned tensor contracts for the common registered-op graph runner.

Providers expose runtime inputs and persistent storages, not model execution.
The normal model forward still supplies the registered operation sequence.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol


def tensor_signature(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        tensor.storage_offset(),
    )


@dataclass(frozen=True)
class StaticTensorBindings:
    """Provider-owned tensors whose allocation and metadata stay immutable.

    Tensor contents may change. Resizing/rebinding these tensors while a group
    or its captured graphs remain alive is outside this explicit contract.
    Runtime state buffers should use ordinary per-forward fixed bindings.
    """

    tensors: Mapping[str, Any]
    storage_bases: frozenset
    signature: tuple

    @classmethod
    def create(cls, tensors):
        tensors = dict(tensors)
        bases = set()
        signature = []
        for name, tensor in tensors.items():
            if tensor is None or tensor.device.type != "cpu":
                raise ValueError(f"Static binding {name} must be a CPU tensor")
            base = tensor.untyped_storage().data_ptr()
            if not base or base in bases:
                raise ValueError(
                    f"Static binding {name} has empty or duplicate storage"
                )
            bases.add(base)
            signature.append((name, tensor_signature(tensor), base))
        return cls(MappingProxyType(tensors), frozenset(bases), tuple(signature))


@dataclass(frozen=True)
class GraphBindings:
    inputs: Mapping[str, Any]
    fixed: Mapping[str, Any]
    contract: tuple
    static: tuple[StaticTensorBindings, ...] = ()

    def all_fixed_tensors(self):
        for group in self.static:
            yield from group.tensors.values()
        yield from self.fixed.values()

    def validate(self):
        if not self.inputs or not self.contract:
            raise ValueError("Graph provider must name inputs and a versioned contract")
        hash(self.contract)
        input_storages = set()
        for name, tensor in self.inputs.items():
            if tensor is None or tensor.device.type != "cpu":
                raise ValueError(f"Graph input {name} must be a CPU tensor")
            base = tensor.untyped_storage().data_ptr()
            if base in input_storages:
                raise ValueError(
                    f"Graph input {name} aliases another dynamic input; "
                    "expose one base input and derive views in forward"
                )
            input_storages.add(base)
        fixed_storages = set()
        fixed_names = set(self.fixed)
        for group in self.static:
            if (
                group.storage_bases & (input_storages | fixed_storages)
                or group.tensors.keys() & fixed_names
            ):
                raise ValueError("Static bindings overlap another binding")
            fixed_storages.update(group.storage_bases)
            fixed_names.update(group.tensors)
        for name, tensor in self.fixed.items():
            if tensor is None or tensor.device.type != "cpu":
                raise ValueError(f"Graph fixed storage {name} must be a CPU tensor")
            base = tensor.untyped_storage().data_ptr()
            if not base or base in input_storages or base in fixed_storages:
                raise ValueError(f"Graph fixed storage {name} overlaps another binding")
            fixed_storages.add(base)
        return self

    def cache_signature(self):
        # Dynamic addresses can change; shapes, strides and view offsets cannot.
        # Persistent storage replacement invalidates graphs retaining old state.
        return (
            self.contract,
            tuple((name, tensor_signature(t)) for name, t in self.inputs.items()),
            tuple(
                (name, tensor_signature(t), t.untyped_storage().data_ptr())
                for name, t in self.fixed.items()
            ),
            tuple(group.signature for group in self.static),
        )


class GraphBindingProvider(Protocol):
    # Optional capture_options may opt into input aliases / adapter dispatch
    # suppression. Providers without it retain capture()'s upstream defaults.
    def supports(self, forward_batch, kwargs: dict) -> bool:
        """Reject unsupported modes before any model/state/collective execution."""
        ...

    def bind(self, model_runner, forward_batch, kwargs: dict) -> GraphBindings:
        """Return fresh per-forward inputs and complete persistent storages."""
        ...


def resolve_graph_provider(model_runner):
    factory = getattr(model_runner.attn_backend, "get_kunpeng_graph_provider", None)
    return factory(model_runner) if factory is not None else None
