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


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_external_stream_and_compact_strides():
    kernel = tilelang.compile(
        _make_write_2d_kernel(),
        target="metal",
        execution_backend="tvm_ffi",
        out_idx=None,
    )

    src = kernel.artifact.host_mod.inspect_source("c")
    assert "metal.SetExternalCommandBuffer" in src
    assert "metal.GetExternalCommandBuffer" in src
    assert "metal.GetCurrentTVMStream" in src
    assert "torch::mps::get_command_buffer" in src
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
    kernel(out)
    mx.synchronize()

    np.testing.assert_allclose(
        np.array(out),
        np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 7.0]], dtype=np.float32),
    )


@tilelang.testing.requires_metal
def test_tvm_ffi_metal_mlx_lazy_array_command_buffer_path():
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
    kernel(out)
    mx.synchronize()

    np.testing.assert_allclose(
        np.array(out),
        np.array([[3.0, 1.0, 1.0], [1.0, 1.0, 7.0]], dtype=np.float32),
    )
