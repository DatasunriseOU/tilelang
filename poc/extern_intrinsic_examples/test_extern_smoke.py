"""Smoke test for ``tl.extern_intrinsic`` (RFC §6).

Registration must work without CUDA. Actual TIR emission is gated on TVM
being importable; compilation/runtime is not exercised here.
"""

from __future__ import annotations

import importlib.util
import pytest

from tilelang.language import extern_registry
from tilelang.language.extern import Frag, EXTERN_CALL_PREFIX, extern_intrinsic

_HAS_TVM = importlib.util.find_spec("tvm") is not None
_HAS_CUDA = False
try:  # pragma: no cover - environment dependent
    import torch  # type: ignore[import-untyped]
    _HAS_CUDA = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
except ImportError:
    _HAS_CUDA = False


_FUSED_RELU_ADD_CU = r"""
__device__ void fused_relu_add(const float *a, const float *b, float *out) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float v = a[i] + b[i];
        out[i] = v > 0.f ? v : 0.f;
    }
}
"""


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Drop our test entry between tests to avoid cross-test pollution."""
    yield
    try:
        extern_registry._REGISTRY.unregister("fused_relu_add_16")  # type: ignore[attr-defined]
    except KeyError:
        pass


def test_registration_no_cuda_required():
    """Decorator must work even when CUDA / TVM are absent."""
    op = extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32", layout="row_major"),
            Frag("b", (16,), "shared", "float32", layout="row_major"),
            Frag("out", (16,), "shared", "float32", layout="row_major", is_output=True),
        ),
        bodies={"cuda": _FUSED_RELU_ADD_CU},
    )
    entry = extern_registry.lookup("fused_relu_add_16")
    assert entry is not None
    assert entry.has_target("cuda")
    assert not entry.has_target("metal")
    frags = entry.signature()
    assert [f.name for f in frags] == ["a", "b", "out"]
    assert frags[2].is_output is True


@pytest.mark.skipif(not _HAS_TVM, reason="TVM not importable")
def test_emit_returns_tir_call(monkeypatch):
    """Calling the decorated op should yield a TIR call_extern node."""
    from tvm import tir

    op = extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("b", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cuda": _FUSED_RELU_ADD_CU},
    )
    # Build three fake shared buffers and call the emitter.
    a = tir.decl_buffer((16,), "float32", scope="shared", name="a")
    b = tir.decl_buffer((16,), "float32", scope="shared", name="b")
    c = tir.decl_buffer((16,), "float32", scope="shared", name="out")
    node = op(a, b, c)
    assert isinstance(node, tir.Call)
    # Symbol must use the documented prefix so codegen can grep for it.
    assert any(EXTERN_CALL_PREFIX + "fused_relu_add_16" in str(arg) for arg in node.args)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
def test_cuda_device_present_for_real_compile():
    """Placeholder for a future end-to-end compile-and-run test."""
    assert _HAS_CUDA
