import numpy as np
import pytest
import torch
import tvm_ffi

import tilelang
from tvm import runtime
import tilelang.language as T
import tilelang.testing


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
