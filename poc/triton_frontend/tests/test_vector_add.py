"""End-to-end smoke test: Triton vector_add -> TileLang ``PrimFunc``.

Validates the lowering surface only (no GPU execution): defines a
``@triton.jit`` vector_add kernel, calls
:func:`poc.triton_frontend.from_triton_kernel`, and asserts the result
is a real ``tvm.tir.PrimFunc``. Runtime, codegen, and host bindings are
out of scope.

RFC reference: section 5 op-by-op map; vector_add is item 1 of the
section 5.5 conformance ladder.
"""
from __future__ import annotations

import pytest

from poc.triton_frontend import from_triton_kernel, from_ttir

# Lazily import triton/tvm so test collection still works when neither
# is installed; we skip rather than fail the whole module.
triton = pytest.importorskip("triton")
tvm = pytest.importorskip("tvm")
import triton.language as tl  # noqa: E402  -- needs triton importable


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """Triton vector_add: out = x + y, masked tail."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def test_vector_add_lowers_to_prim_func() -> None:
    """``from_triton_kernel`` returns a ``tvm.tir.PrimFunc``."""
    func = from_triton_kernel(
        _vector_add_kernel,
        constexprs={"BLOCK": 128},
        target="cuda",
    )
    assert isinstance(func, tvm.tir.PrimFunc), (
        f"expected tvm.tir.PrimFunc, got {type(func)!r}"
    )
    assert func.attrs is not None
    assert func.attrs["global_symbol"] == "_vector_add_kernel"


def test_from_ttir_accepts_text() -> None:
    """``from_ttir`` accepts a textual TTIR string and produces a PrimFunc."""
    ttir_text = """\
    module {
      tt.func @vector_add(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>) {
        tt.return
      }
    }
    """
    func = from_ttir(ttir_text, target="cuda", name="vector_add", _allow_text_ttir=True)
    assert isinstance(func, tvm.tir.PrimFunc)
