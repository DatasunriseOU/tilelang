"""Public ``tilelang.compile`` dispatch tests for the Triton TTIR frontend.

Lock the production contract that promoted ``poc.triton_frontend`` into
``tilelang.frontends.triton``:

* The package is importable as ``tilelang.frontends.triton``.
* ``tilelang.compile`` accepts a Triton TTIR string (and an ``mlir.ir.Module``)
  via the registered dispatch hook in :mod:`tilelang.jit`.
* The dispatch is module-identity preserving with the historical
  ``poc.triton_frontend`` location so the two import paths share state.
"""

from __future__ import annotations

import importlib

import pytest


def _capture_vector_add_ttir():
    """Return a freshly-captured vector_add TTIR text or skip the suite.

    The capture path imports Triton; if Triton or the harness aren't
    available, the dispatch contract is still tested via the structural
    checks below.
    """
    try:
        numeric_smoke = importlib.import_module("poc.triton_frontend._test_harness.numeric_smoke")
        va = importlib.import_module("poc.triton_frontend._test_harness.numeric_kernels.vector_add")
    except Exception as exc:  # pragma: no cover -- environment-dependent
        pytest.skip(f"vector_add TTIR capture unavailable: {exc!r}")
    text, err, _opts = numeric_smoke._capture_ttir(va)
    if err is not None:
        pytest.skip(f"_capture_ttir() returned err={err!r}")
    return text


def test_tilelang_frontends_triton_is_importable() -> None:
    """``tilelang.frontends.triton`` must be importable and expose the public surface."""
    tlt = importlib.import_module("tilelang.frontends.triton")
    for attr in ("from_ttir", "from_triton_kernel", "compile_ttir", "is_ttir_input", "OP_TABLE", "WalkerCtx"):
        assert hasattr(tlt, attr), f"tilelang.frontends.triton missing {attr!r}"


def test_tilelang_frontends_triton_shares_state_with_poc() -> None:
    """The production package re-exports the same callables/state as ``poc``."""
    tlt = importlib.import_module("tilelang.frontends.triton")
    poc = importlib.import_module("poc.triton_frontend")
    assert tlt.from_ttir is poc.from_ttir
    assert tlt.from_triton_kernel is poc.from_triton_kernel
    assert tlt.OP_TABLE is poc.OP_TABLE
    assert tlt.WalkerCtx is poc.WalkerCtx


def test_tilelang_top_level_reexports_triton_frontend() -> None:
    """``tilelang.from_ttir`` / ``tilelang.compile_ttir`` should be ergonomic shortcuts."""
    import tilelang

    assert hasattr(tilelang, "from_ttir")
    assert hasattr(tilelang, "compile_ttir")
    assert hasattr(tilelang, "from_triton_kernel")
    from tilelang.frontends.triton import from_ttir as canonical

    assert tilelang.from_ttir is canonical


def test_compile_dispatches_ttir_text_to_triton_frontend() -> None:
    """``tilelang.compile(ttir_text)`` must return a ``JITKernel`` on Metal/MPS or CPU."""
    import tilelang

    text = _capture_vector_add_ttir()
    kernel = tilelang.compile(
        text,
        arg_buffer_shapes=[(1024,), (1024,), (1024,)],
        grid=(8,),
    )
    assert type(kernel).__name__ == "JITKernel"


def test_compile_rejects_unknown_kwargs_for_primfunc_inputs() -> None:
    """When the input is a PrimFunc, unknown kwargs must raise instead of being absorbed."""
    import tilelang
    from tvm import tir

    func = tir.PrimFunc(params=[], body=tir.Evaluate(0))
    func = func.with_attr("global_symbol", "tiny")
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        tilelang.compile(func, arg_buffer_shapes=[(1,)])
