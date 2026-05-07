"""Surface-level test for the ``tl.sync_threads_partial`` builtin.

The full backend codegen (CUDA __syncwarp, HIP no-op, Metal
simdgroup_barrier) lives in ``src/target/codegen_{cuda,hip,metal}.cc`` and
needs a built libtilelang.so to exercise — that build doesn't run on this
test host (no MLIR), so this test only verifies the Python-side surface:

1. the Op is registered with TVM's Op registry under
   ``"tl.sync_threads_partial"``;
2. ``T.sync_threads_partial(mask, n)`` produces a ``tir.Call`` carrying
   that Op with ``num_args == 2``.

Codegen lowering matrix (verified by reading source, not invoked here):
    CUDA  → ``__syncwarp(<mask>);``
    HIP   → ``// no-op (wave-level convergence)``
    Metal → ``simdgroup_barrier(mem_flags::mem_threadgroup);``
"""
from __future__ import annotations

import pytest


def _tvm_or_skip():
    try:
        import tvm  # noqa: F401
        from tvm import tir  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-only
        pytest.skip(f"tvm unavailable: {exc!r}")


def test_sync_threads_partial_op_is_registered():
    _tvm_or_skip()
    from tvm import tir

    op = tir.op.Op.get("tl.sync_threads_partial")
    assert op is not None
    assert op.name == "tl.sync_threads_partial"


def test_sync_threads_partial_python_wrapper_emits_call():
    _tvm_or_skip()
    try:
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - env-only
        pytest.skip(f"tilelang unavailable: {exc!r}")

    call = T.sync_threads_partial(0xFFFFFFFF, 32)
    # tir.Call: op + 2 args + dtype "void"
    assert call is not None
    name = str(call.op) if hasattr(call, "op") else ""
    assert "sync_threads_partial" in name.lower()
    assert len(call.args) == 2
