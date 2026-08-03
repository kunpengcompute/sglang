import torch

from sgl_kernel import graph_cpp as _C

_captured_storages = None
_captured_views = None
_captured_ops = None
_captured_num_inputs = None
_captured_fixed = None
_none_storage_id = None
_captured_input_map = None


class _CaptureContext:
    def __init__(self, inputs, fixed=None):
        self.inputs = inputs
        self.fixed = fixed or []

    def __enter__(self):
        global _captured_storages, _captured_views, _captured_ops, \
               _captured_num_inputs, _captured_fixed, _none_storage_id, \
               _captured_input_map
        _captured_storages = None
        _captured_views = None
        _captured_ops = None
        _captured_num_inputs = None
        _captured_fixed = None
        _none_storage_id = None
        _captured_input_map = None
        mgr = _C.CaptureManager.instance()
        mgr.begin_capture(self.inputs, self.fixed)
        _captured_input_map = {}
        for i, t in enumerate(self.inputs):
            _captured_input_map[id(t)] = (i, _C.storage_offset(t), i)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _captured_storages, _captured_views, _captured_ops, \
               _captured_num_inputs, _captured_fixed, _captured_input_map
        mgr = _C.CaptureManager.instance()
        if mgr.is_capturing():
            _captured_storages, _captured_views, _captured_ops, \
                _captured_num_inputs = mgr.end_capture()
            n = len(self.inputs)
            _captured_fixed = {n + i: t for i, t in enumerate(self.fixed)}
        _captured_input_map = None
        return False


def capture(inputs, fixed=None):
    return _CaptureContext(inputs, fixed)


def finalize(outputs, external_pool=None, external_shm_pool=None):
    global _captured_storages, _captured_views, _captured_ops, \
           _captured_num_inputs, _captured_fixed, _captured_input_map
    if _captured_storages is None:
        raise RuntimeError("No captured graph to finalize")
    storages = _captured_storages
    views = _captured_views
    ops = _captured_ops
    num_inputs = _captured_num_inputs
    fixed = _captured_fixed or {}
    _captured_storages = None
    _captured_views = None
    _captured_ops = None
    _captured_num_inputs = None
    _captured_fixed = None
    _captured_input_map = None

    output_view_ids = []
    mgr = _C.CaptureManager.instance()
    for t in outputs:
        base = _C.storage_base(t)
        so = _C.storage_offset(t)
        sid = mgr.lookup_storage(base)
        if sid < 0:
            raise RuntimeError(
                f"Output tensor (base {base}) not found in any storage")

        if sid < num_inputs:
            # Output on input storage: exact match with existing view
            t_shape = list(t.shape)
            t_strides = list(t.stride())
            found = None
            for view in views:
                if (view.storage_id == sid and
                    view.storage_offset == so and
                    view.shape == t_shape and
                    view.strides == t_strides and
                    not view.is_return):
                    found = view
                    break
            if found is None:
                raise RuntimeError(
                    f"Output tensor (sid {sid}) has no matching input view")
            output_view_ids.append(found.id)
        else:
            # Output on fixed or intermediate storage: always create new view
            _, view = _C.tensor_to_buf_and_view(t)
            view.storage_id = sid
            view.is_return = True
            view.id = len(views)
            views.append(view)
            output_view_ids.append(view.id)

    gh = _C.Graph(storages, views, ops, output_view_ids, num_inputs, fixed,
                  external_pool, external_shm_pool)
    return gh


def is_capturing():
    return _C.CaptureManager.instance().is_capturing()


def lookup_or_register(tensor, idx=0):
    global _none_storage_id, _captured_input_map
    mgr = _C.CaptureManager.instance()
    if tensor is None:
        if _none_storage_id is None:
            buf = _C.StorageBuf()
            buf.storage_base = 0
            buf.size = 0
            buf.born_op = 0
            _none_storage_id = mgr.register_storage(buf)
        vid = mgr.find_or_register_view(_none_storage_id, 0, tensor)
        return vid, 0, _none_storage_id
    if _captured_input_map is not None:
        cached = _captured_input_map.get(id(tensor))
        if cached is not None:
            return cached
    base = _C.storage_base(tensor)
    assert base != 0, (
        f"non-input tensor has nullptr storage at {idx}-th parameter: "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
    )
    so = _C.storage_offset(tensor)
    sid = mgr.lookup_storage(base)
    if sid < 0:
        print(
            f"[capture] lookup_or_register FAIL at {idx}-th parameter: "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"numel={tensor.numel()}, "
            f"data_ptr=0x{tensor.data_ptr():x}, "
            f"storage_base=0x{base:x}",
            flush=True,
        )
    assert sid >= 0, "non-return-value parameter tensor not registered"

    vid = mgr.find_or_register_view(sid, so, tensor)
    return vid, so, sid


_MEMORY_TYPE_TO_ENUM = {
    'regular': _C.MemoryType.REGULAR,
    'shm': _C.MemoryType.SHM,
}


def register_output(tensor, memory_type='regular'):
    mgr = _C.CaptureManager.instance()
    so = _C.storage_offset(tensor)
    buf, view = _C.tensor_to_buf_and_view(tensor)
    sid = mgr.register_output_storage(buf, _MEMORY_TYPE_TO_ENUM[memory_type])
    view.storage_id = sid
    view.is_return = True
    vid = mgr.register_view(view)
    return vid, so


def upgrade_storage_memory_type(tensor):
    """Upgrade the storage backing ``tensor`` to SHM memory type."""
    mgr = _C.CaptureManager.instance()
    base = _C.storage_base(tensor)
    sid = mgr.lookup_storage(base)
    assert sid >= 0, (
        f"upgrade_storage_memory_type: storage not registered, "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
    )
    mgr.upgrade_storage_memory_type(sid)


def record_op(op_name, inputs, outputs, scalar_args, profile_name=""):
    mgr = _C.CaptureManager.instance()
    op = _C.OpRecord()
    op.op_name = op_name
    op.profile_name = profile_name
    op.input_view_ids = [vid for vid, _, _ in inputs]
    op.output_view_ids = [vid for vid, _ in outputs]
    op.scalar_args = list(scalar_args)
    mgr.record_op(op)
