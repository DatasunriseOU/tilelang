"""Standalone numeric check that drives Triton -> TileLang end-to-end.

Importing ``triton`` (and therefore ``triton._C.libtriton``) at module
scope conflicts with ``_triton_frontend_cxx`` -- both statically link
their own LLVM. We therefore guard the triton import behind a module-
level skip that fires when the shim is already loaded in this process,
and lazy-import triton inside the test body. Running this file on its
own (no shim loaded yet) behaves exactly like before.
"""
from __future__ import annotations

import pytest

from poc.triton_frontend._test_harness.native_import_guard import (
    triton_import_block_reason,
)


# If the C++ shim / TileLang / TVM side has already been imported in this
# Python process, loading Triton's libtriton too can register duplicate LLVM
# cl::opts and SIGABRT the interpreter. Skip the whole module cleanly in that
# case; operators can rerun this file in a fresh process to exercise it.
_triton_block_reason = triton_import_block_reason()
if _triton_block_reason is not None:
    pytest.skip(
        "test_standalone.py skipped: "
        f"{_triton_block_reason} Re-run "
        "`pytest poc/triton_frontend/tests/test_standalone.py` in a fresh "
        "Python process to exercise it.",
        allow_module_level=True,
    )

import numpy as np  # noqa: E402
import tvm  # noqa: E402
import tvm.testing  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from poc.triton_frontend import from_triton_kernel  # noqa: E402


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@pytest.mark.xfail(reason="Expected locally if LLVM is missing", raises=ValueError)
def test_run():
    N = 1024
    BLOCK = 128
    func = from_triton_kernel(_vector_add_kernel, constexprs={"BLOCK": BLOCK}, target="llvm")
    print("Lowered func!")

    rt_mod = tvm.build(func, target="llvm")
    print("Built func!")

    x_np = np.random.rand(N).astype("float32")
    y_np = np.random.rand(N).astype("float32")
    out_np = np.zeros(N, dtype="float32")

    dev = tvm.cpu()
    x_tvm = tvm.nd.array(x_np, dev)
    y_tvm = tvm.nd.array(y_np, dev)
    out_tvm = tvm.nd.array(out_np, dev)

    rt_mod(x_tvm, y_tvm, out_tvm, N)
    print("Ran func!")
    tvm.testing.assert_allclose(out_tvm.numpy(), x_np + y_np)
    print("Passed!")


if __name__ == "__main__":
    test_run()
