import numpy as np
import torch

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
    assert "metal.SetStream" in src
    assert "torch::mps::get_command_buffer" in src
    assert "metal.SetStream runtime hook is not registered" in src

    dev = runtime.device("metal", 0)
    out = runtime.empty((2, 3), "float32", dev)
    kernel.adapter.executable(out)
    torch.mps.synchronize()
    dev.sync()

    np.testing.assert_allclose(
        out.numpy(),
        np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 7.0]], dtype=np.float32),
    )
