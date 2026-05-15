import os
from contextlib import contextmanager

import numpy as np
import pytest
import torch
import tvm_ffi

import tilelang
from tvm import runtime
import tilelang.language as T
import tilelang.testing


_ACTIVE_COMPUTE_ENCODER_ENV = "TILELANG_MLX_TVM_FFI_USE_ACTIVE_COMPUTE_ENCODER"


@contextmanager
def _temporary_env(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _make_write_2d_kernel():
    @T.prim_func
    def write_2d(C: T.Tensor((2, 3), T.float32)):
        with T.Kernel(1, threads=1):
            C[0, 0] = T.float32(3.0)
            C[1, 2] = T.float32(7.0)

    return write_2d


def _make_add_2d_kernel():
    @T.prim_func
    def add_2d(A: T.Tensor((2, 3), T.float32), C: T.Tensor((2, 3), T.float32)):
        with T.Kernel(1, threads=1):
            C[0, 0] = A[0, 0] + T.float32(2.0)
            C[1, 2] = A[1, 2] + T.float32(4.0)

    return add_2d


def _make_add_all_2d_kernel():
    @T.prim_func
    def add_all_2d(A: T.Tensor((2, 3), T.float32), C: T.Tensor((2, 3), T.float32)):
        with T.Kernel(1, threads=1):
            C[0, 0] = A[0, 0] + T.float32(1.0)
            C[0, 1] = A[0, 1] + T.float32(2.0)
            C[0, 2] = A[0, 2] + T.float32(3.0)
            C[1, 0] = A[1, 0] + T.float32(4.0)
            C[1, 1] = A[1, 1] + T.float32(5.0)
            C[1, 2] = A[1, 2] + T.float32(6.0)

    return add_all_2d


def _make_parallel_add_1d_kernel():
    @T.prim_func
    def parallel_add_1d(A: T.Tensor((8,), T.float32), C: T.Tensor((8,), T.float32)):
        with T.Kernel(1, threads=8):
            for i in T.Parallel(8):
                C[i] = A[i] + T.float32(1.0)

    return parallel_add_1d


def _make_flattened_1d_input_kernel():
    @T.prim_func
    def flattened_1d_input(A: T.Tensor((8,), T.float32), C: T.Tensor((8,), T.float32)):
        with T.Kernel(1, threads=8):
            for i in T.Parallel(8):
                C[i] = A[i] * T.float32(2.0)

    return flattened_1d_input


def _make_abi_2d_owner_output_kernel():
    @T.prim_func
    def abi_2d_owner_output(C: T.Tensor((1, 8), T.float32)):
        with T.Kernel(1, threads=8):
            for i in T.Parallel(8):
                C[0, i] = T.cast(i, T.float32) + T.float32(10.0)

    return abi_2d_owner_output


def _make_large_parallel_add_1d_kernel(size=65536, threads=256):
    @T.prim_func
    def parallel_add_1d(A: T.Tensor((size,), T.float32), C: T.Tensor((size,), T.float32)):
        with T.Kernel(T.ceildiv(size, threads), threads=threads) as bx:
            for tx in T.Parallel(threads):
                i = bx * threads + tx
                if i < size:
                    C[i] = A[i] + T.float32(1.0)

    return parallel_add_1d


def _capsule_data_ptr(capsule) -> int:
    import ctypes

    class DLDevice(ctypes.Structure):
        _fields_ = [
            ("device_type", ctypes.c_int32),
            ("device_id", ctypes.c_int32),
        ]

    class DLDataType(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
        ]

    class DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", DLDevice),
            ("ndim", ctypes.c_int32),
            ("dtype", DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    class DLManagedTensor(ctypes.Structure):
        _fields_ = [
            ("dl_tensor", DLTensor),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.c_void_p),
        ]

    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    ptr = get_pointer(capsule, b"dltensor")
    tensor = ctypes.cast(ptr, ctypes.POINTER(DLManagedTensor)).contents
    return int(tensor.dl_tensor.data)


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_external_stream_and_compact_strides():
    kernel = tilelang.compile(
        _make_write_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=None,
    )

    if kernel.artifact is not None:
        src = kernel.artifact.host_mod.inspect_source("c")
    else:
        src = kernel.adapter.get_host_source()
    assert "metal.SetExternalCommandBuffer" in src
    assert "metal.GetExternalCommandBuffer" in src
    assert "metal.GetCurrentTVMStream" in src
    assert "torch::mps::get_command_buffer" not in src
    assert "Metal external command buffer runtime hooks are not registered" in src

    dev = runtime.device("metal", 0)
    out = runtime.empty((2, 3), "float32", dev)
    kernel.adapter.executable(out)
    torch.mps.synchronize()
    dev.sync()

    np.testing.assert_allclose(
        out.numpy(),
        np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_result_idx_reuses_caller_owned_tvm_output():
    kernel = tilelang.compile(
        _make_write_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=-1,
    )

    dev = runtime.device("metal", 0)
    out = runtime.empty((2, 3), "float32", dev)
    returned = kernel(out)
    assert returned is out
    torch.mps.synchronize()
    dev.sync()

    np.testing.assert_allclose(
        out.numpy(),
        np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_env_stream_context_path():
    kernel = tilelang.compile(
        _make_write_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=None,
    )

    dev = runtime.device("metal", 0)
    stream = dev.create_raw_stream()
    try:
        out = runtime.empty((2, 3), "float32", dev)
        with tvm_ffi.use_raw_stream(dev, stream):
            kernel.adapter.executable(out)
        dev.sync(stream)
    finally:
        dev.free_raw_stream(stream)

    np.testing.assert_allclose(
        out.numpy(),
        np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_explicit_tvm_stream_path():
    set_external = tilelang.tvm.ffi.get_global_func(
        "metal.SetExternalCommandBuffer", allow_missing=True
    )
    get_external = tilelang.tvm.ffi.get_global_func(
        "metal.GetExternalCommandBuffer", allow_missing=True
    )
    clear_external = tilelang.tvm.ffi.get_global_func(
        "metal.ClearExternalCommandBuffer", allow_missing=True
    )
    get_tvm_stream = tilelang.tvm.ffi.get_global_func(
        "metal.GetCurrentTVMStream", allow_missing=True
    )

    assert set_external is not None
    assert get_external is not None
    assert clear_external is not None
    assert get_tvm_stream is not None

    kernel = tilelang.compile(
        _make_write_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=None,
    )

    dev = runtime.device("metal", 0)
    stream = dev.create_raw_stream()
    try:
        clear_external()
        dev.set_raw_stream(stream)
        out = runtime.empty((2, 3), "float32", dev)
        kernel.adapter.executable(out)
        dev.sync(stream)
    finally:
        dev.set_raw_stream(0)
        dev.free_raw_stream(stream)
        clear_external()

    np.testing.assert_allclose(
        out.numpy(),
        np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_command_buffer_path():
    mx = pytest.importorskip("mlx.core")
    if not hasattr(mx, "metal") or not hasattr(mx.metal, "_current_command_buffer"):
        pytest.skip("MLX build does not expose Metal command buffer interop hook")

    kernel = tilelang.compile(
        _make_write_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=None,
    )

    out = mx.zeros((2, 3), dtype=mx.float32)
    mx.eval(out)
    kernel(out)
    mx.synchronize()

    np.testing.assert_allclose(
        np.array(out),
        np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_materialized_array_command_buffer_path():
    mx = pytest.importorskip("mlx.core")
    if not hasattr(mx, "metal") or not hasattr(mx.metal, "_current_command_buffer"):
        pytest.skip("MLX build does not expose Metal command buffer interop hook")

    kernel = tilelang.compile(
        _make_write_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=None,
    )

    out = mx.zeros((2, 3), dtype=mx.float32) + 1
    mx.eval(out)
    kernel(out)
    mx.synchronize()

    np.testing.assert_allclose(
        np.array(out),
        np.array([[3.0, 1.0, 1.0], [1.0, 1.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_mlx_array_to_tvm_tensor_preserves_metal_buffer():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_interop import mlx_array_to_tvm_tensor

    x = mx.arange(6, dtype=mx.float32).reshape(2, 3)
    mx.eval(x)
    tvm_tensor = mlx_array_to_tvm_tensor(x)

    assert tvm_tensor.__dlpack_device__() == x.__dlpack_device__() == (8, 0)
    assert _capsule_data_ptr(tvm_tensor.__dlpack__()) == _capsule_data_ptr(x.__dlpack__())


@tilelang.testing.requires_metal
def test_tvm_metal_output_exports_to_mlx_without_torch():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_interop import mlx_array_to_tvm_tensor, mlx_metal_output

    out = mlx_metal_output((2, 3), "float32")
    assert isinstance(out, mx.array)
    assert not isinstance(out, torch.Tensor)
    assert out.__dlpack_device__() == (8, 0)
    tvm_view = mlx_array_to_tvm_tensor(out)
    assert _capsule_data_ptr(out.__dlpack__()) == _capsule_data_ptr(tvm_view.__dlpack__())


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_bad_dlpack_device_raises_typed_error():
    from tilelang.contrib.mlx_interop import DLPackDeviceError

    class CPUProducer:
        def __dlpack_device__(self):
            return (1, 0)

        def __dlpack__(self):
            raise AssertionError("__dlpack__ should not be called after device preflight fails")

    kernel = tilelang.compile(
        _make_write_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=None,
    )

    with pytest.raises(DLPackDeviceError, match="kDLCPU:0"):
        kernel(CPUProducer())


@tilelang.testing.requires_metal
def test_raw_cpu_dlpack_capsule_bad_device_raises_typed_error():
    from tilelang.contrib.mlx_interop import DLPackDeviceError, dlpack_to_tvm_tensor

    capsule = torch.zeros((2, 3), dtype=torch.float32).__dlpack__()

    with pytest.raises(DLPackDeviceError, match="kDLCPU:0"):
        dlpack_to_tvm_tensor(capsule, expected_device_type=8)


@tilelang.testing.requires_metal
def test_consumed_dlpack_capsule_raises_ownership_error():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_interop import DLPackOwnershipError, dlpack_to_tvm_tensor

    capsule = mx.zeros((2, 3), dtype=mx.float32).__dlpack__()
    runtime.from_dlpack(capsule)

    with pytest.raises(DLPackOwnershipError):
        dlpack_to_tvm_tensor(capsule)


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_result_idx_reuses_caller_owned_mlx_output():
    mx = pytest.importorskip("mlx.core")
    if not hasattr(mx, "metal") or not hasattr(mx.metal, "_current_command_buffer"):
        pytest.skip("MLX build does not expose Metal command buffer interop hook")

    kernel = tilelang.compile(
        _make_write_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=-1,
    )

    out = mx.zeros((2, 3), dtype=mx.float32) + 1
    mx.eval(out)
    returned = kernel(out)
    assert returned is out
    mx.synchronize()

    np.testing.assert_allclose(
        np.array(out),
        np.array([[3.0, 1.0, 1.0], [1.0, 1.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_owner_output_can_return_graph_alias_without_sync():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_tvm_ffi import (
        debug_state,
        is_available as native_bridge_is_available,
        reset_debug_state,
    )

    if not native_bridge_is_available():
        pytest.skip("native MLX TVM-FFI bridge is not built")
    if not hasattr(mx, "metal") or not hasattr(mx.metal, "_current_command_buffer"):
        pytest.skip("MLX build does not expose Metal command buffer interop hook")

    with _temporary_env(_ACTIVE_COMPUTE_ENCODER_ENV, "0"):
        reset_debug_state()
        kernel = tilelang.compile(
            _make_write_2d_kernel(),
            target="metal",
            execution_backend="tvm_ffi",
            out_idx=-1,
        )

        out = mx.zeros((2, 3), dtype=mx.float32) + 1
        mx.eval(out)
        returned = kernel(out, _tilelang_mlx_async_owner_outputs=True)
        assert isinstance(returned, mx.array)
        assert returned is not out
        mx.eval(returned)

        state = debug_state()
    assert state["use_active_compute_encoder_enabled"] is False
    assert state["direct_device_launches"] >= 1
    assert state["direct_compute_encoder_launches"] == 0
    assert state["command_buffers_checked"] >= 1
    assert _capsule_data_ptr(returned.__dlpack__()) == _capsule_data_ptr(out.__dlpack__())
    np.testing.assert_allclose(
        np.array(out),
        np.array([[3.0, 1.0, 1.0], [1.0, 1.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_compact_result_idx_allocates_mlx_output():
    mx = pytest.importorskip("mlx.core")
    if not hasattr(mx, "metal") or not hasattr(mx.metal, "_current_command_buffer"):
        pytest.skip("MLX build does not expose Metal command buffer interop hook")

    kernel = tilelang.compile(
        _make_add_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=-1,
    )

    source = mx.ones((2, 3), dtype=mx.float32)
    mx.eval(source)
    returned = kernel(source)
    assert isinstance(returned, mx.array)
    assert returned.__dlpack_device__() == (8, 0)
    mx.synchronize()

    np.testing.assert_allclose(
        np.array(returned),
        np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_compile_compact_result_idx_is_graph_safe():
    mx = pytest.importorskip("mlx.core")

    kernel = tilelang.compile(
        _make_add_all_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=-1,
    )

    compiled = mx.compile(lambda source: kernel(source))
    returned = compiled(mx.ones((2, 3), dtype=mx.float32))
    assert isinstance(returned, mx.array)
    mx.eval(returned)

    np.testing.assert_allclose(
        np.array(returned),
        np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_compile_uses_native_graph_primitive():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_tvm_ffi import (
        debug_state,
        is_available as native_bridge_is_available,
        reset_debug_state,
    )

    if not native_bridge_is_available():
        pytest.skip("native MLX TVM-FFI bridge is not built")

    with _temporary_env(_ACTIVE_COMPUTE_ENCODER_ENV, "1"):
        reset_debug_state()

        kernel = tilelang.compile(
            _make_add_all_2d_kernel(),
            target="metal",
            execution_backend="tvm_ffi",
            out_idx=-1,
        )

        compiled = mx.compile(lambda source: kernel(source))
        returned = compiled(mx.ones((2, 3), dtype=mx.float32))
        mx.eval(returned)

        state = debug_state()
    assert state["use_active_compute_encoder_enabled"] is True
    assert state["launches"] >= 1
    assert state["direct_device_launches"] >= 1
    assert state["direct_compute_encoder_launches"] >= 1
    np.testing.assert_allclose(
        np.array(returned),
        np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_native_bridge_uses_tilelang_abi_shape_for_inputs():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_tvm_ffi import is_available as native_bridge_is_available

    if not native_bridge_is_available():
        pytest.skip("native MLX TVM-FFI bridge is not built")

    kernel = tilelang.compile(
        _make_flattened_1d_input_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=-1,
    )

    source = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    returned = kernel(source)
    mx.eval(returned)

    np.testing.assert_allclose(
        np.array(returned),
        np.arange(8, dtype=np.float32) * 2.0,
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_owner_output_returns_owner_shape_with_tilelang_abi_shape():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_tvm_ffi import is_available as native_bridge_is_available

    if not native_bridge_is_available():
        pytest.skip("native MLX TVM-FFI bridge is not built")

    kernel = tilelang.compile(
        _make_abi_2d_owner_output_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=-1,
    )

    out = mx.zeros((8,), dtype=mx.float32)
    mx.eval(out)
    returned = kernel(out, _tilelang_mlx_async_owner_outputs=True)
    assert returned.shape == out.shape
    assert _capsule_data_ptr(returned.__dlpack__()) == _capsule_data_ptr(out.__dlpack__())
    mx.eval(returned)

    np.testing.assert_allclose(
        np.array(out),
        np.arange(8, dtype=np.float32) + 10.0,
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_lazy_producer_and_consumer_are_device_ordered():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_tvm_ffi import is_available as native_bridge_is_available

    if not native_bridge_is_available():
        pytest.skip("native MLX TVM-FFI bridge is not built")

    size = 65536
    kernel = tilelang.compile(
        _make_large_parallel_add_1d_kernel(size=size),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=-1,
    )

    seed = mx.arange(size, dtype=mx.float32)
    source = seed * mx.array(1.5, dtype=mx.float32) - mx.array(7.0, dtype=mx.float32)
    returned = kernel(source)
    consumed = returned * mx.array(2.0, dtype=mx.float32) + source
    mx.eval(consumed)

    expected_source = np.arange(size, dtype=np.float32) * 1.5 - 7.0
    expected = (expected_source + 1.0) * 2.0 + expected_source
    np.testing.assert_allclose(np.array(consumed), expected, rtol=1e-6, atol=1e-6)


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_native_debug_completion_hook_is_nonblocking():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_tvm_ffi import (
        debug_state,
        is_available as native_bridge_is_available,
        reset_debug_state,
    )

    if not native_bridge_is_available():
        pytest.skip("native MLX TVM-FFI bridge is not built")

    previous_debug_completion = os.environ.get("TILELANG_MLX_TVM_FFI_DEBUG_COMPLETION")
    os.environ["TILELANG_MLX_TVM_FFI_DEBUG_COMPLETION"] = "1"
    reset_debug_state()

    try:
        kernel = tilelang.compile(
            _make_add_all_2d_kernel(),
            target="metal",
            execution_backend="tvm_ffi",
            out_idx=-1,
        )

        compiled = mx.compile(lambda source: kernel(source))
        returned = compiled(mx.ones((2, 3), dtype=mx.float32))
        mx.eval(returned)
        state = debug_state()
    finally:
        if previous_debug_completion is None:
            os.environ.pop("TILELANG_MLX_TVM_FFI_DEBUG_COMPLETION", None)
        else:
            os.environ["TILELANG_MLX_TVM_FFI_DEBUG_COMPLETION"] = previous_debug_completion

    assert state["debug_completion_enabled"] is True
    assert state["launches"] >= 1
    assert state["debug_completion_launches"] >= 1
    assert state["direct_compute_encoder_launches"] == 0
    assert state["command_buffers_checked"] >= 1
    assert state["completion_handlers_installed"] >= 1
    assert state["null_command_buffers"] == 0
    assert state["null_input_buffers"] == 0
    assert state["null_output_buffers"] == 0
    assert state["errored_command_buffers"] == 0
    np.testing.assert_allclose(
        np.array(returned),
        np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_graph_same_domain_uses_encode_order_without_device_event():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_tvm_ffi import (
        debug_state,
        is_available as native_bridge_is_available,
        reset_debug_state,
    )

    if not native_bridge_is_available():
        pytest.skip("native MLX TVM-FFI bridge is not built")

    with _temporary_env(_ACTIVE_COMPUTE_ENCODER_ENV, "0"):
        reset_debug_state()
        kernel = tilelang.compile(
            _make_add_all_2d_kernel(),
            target="metal",
            execution_backend="tvm_ffi",
            out_idx=-1,
        )

        compiled = mx.compile(lambda source: kernel(kernel(source)))
        returned = compiled(mx.ones((2, 3), dtype=mx.float32))
        mx.eval(returned)

        state = debug_state()
    assert state["use_active_compute_encoder_enabled"] is False
    assert state["direct_compute_encoder_launches"] == 0
    assert state["command_buffers_checked"] >= 1
    assert state["device_event_waits_encoded"] == 0
    assert state["device_event_signals_encoded"] == 0
    np.testing.assert_allclose(
        np.array(returned),
        np.array([[3.0, 5.0, 7.0], [9.0, 11.0, 13.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_graph_cross_domain_emits_device_event():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_tvm_ffi import (
        debug_state,
        is_available as native_bridge_is_available,
        reset_debug_state,
    )

    if not native_bridge_is_available():
        pytest.skip("native MLX TVM-FFI bridge is not built")

    reset_debug_state()
    kernel = tilelang.compile(
        _make_add_all_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=-1,
    )

    def graph(source):
        mid = kernel(
            source,
            _tilelang_metal_command_buffer_domain="producer-domain",
        )
        return kernel(
            mid,
            _tilelang_metal_command_buffer_domain="consumer-domain",
        )

    compiled = mx.compile(graph)
    returned = compiled(mx.ones((2, 3), dtype=mx.float32))
    mx.eval(returned)

    state = debug_state()
    assert state["direct_compute_encoder_launches"] == 0
    assert state["device_event_waits_encoded"] >= 1
    assert state["device_event_signals_encoded"] >= 1
    np.testing.assert_allclose(
        np.array(returned),
        np.array([[3.0, 5.0, 7.0], [9.0, 11.0, 13.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_graph_kernels_with_same_symbol_do_not_collide():
    mx = pytest.importorskip("mlx.core")
    from tilelang.contrib.mlx_interop import mlx_tilelang_metal_kernel

    msl_template = """
#include <metal_stdlib>
using namespace metal;
kernel void same_symbol(device const float* A [[buffer(0)]],
                        device float* C [[buffer(1)]],
                        uint3 thread_position_in_grid [[thread_position_in_grid]]) {{
    uint i = thread_position_in_grid.x;
    if (i < 6) {{
        C[i] = {expr};
    }}
}}
"""
    add_kernel = mlx_tilelang_metal_kernel(
        msl_template.format(expr="A[i] + 1.0f"),
        input_names=["A"],
        output_names=["C"],
    )
    mul_kernel = mlx_tilelang_metal_kernel(
        msl_template.format(expr="A[i] * 2.0f"),
        input_names=["A"],
        output_names=["C"],
    )
    assert add_kernel is not None
    assert mul_kernel is not None

    source = mx.arange(6, dtype=mx.float32).reshape(2, 3)
    added = add_kernel(
        inputs=[source],
        output_shapes=[source.shape],
        output_dtypes=[source.dtype],
        grid=(6, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]
    doubled = mul_kernel(
        inputs=[source],
        output_shapes=[source.shape],
        output_dtypes=[source.dtype],
        grid=(6, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]
    mx.eval(added, doubled)

    np.testing.assert_allclose(
        np.array(added),
        np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0,
    )
    np.testing.assert_allclose(
        np.array(doubled),
        np.arange(6, dtype=np.float32).reshape(2, 3) * 2.0,
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_cached_host_source_restores_launch_config():
    kernel = tilelang.compile(
        _make_parallel_add_1d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=-1,
    )

    kernel.adapter.device_mod = None

    assert kernel.adapter._metal_launch_config() == ((1, 1, 1), (8, 1, 1))
