"""Unit tests for the new memory/shape emitters in
:mod:`poc.triton_frontend.op_emitters.memory`.

Same dict-shaped fake-op pattern as ``test_dot_reduce_atomic.py``: each
TTIR op is a plain dict with ``name``, ``operands``, ``results`` and
``attrs`` keys so we don't need MLIR Python bindings. We do still need
``tvm`` importable because the emitters lazy-import it for real TIR
nodes (BufferLoad/Ramp/Broadcast/...).

Coverage:
* ``tt.load``/``tt.store`` round-trip on a 1-D buffer with mask
  -> asserts ``tir.if_then_else`` is in the printed expression.
* ``tt.make_range`` -> ``tir.Ramp``.
* ``tt.broadcast`` (scalar -> tile) -> ``tir.Broadcast``.
* ``tt.load`` on a multi-element tile when the C++ shim is *not*
  available -> asserts the ``# DEGRADED:`` annotation appears in the
  printed PrimFunc text. This is the regression that the
  "never silent-fallback" hard constraint is meant to catch.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

tvm = pytest.importorskip("tvm")

from poc.triton_frontend.op_emitters.memory import (  # noqa: E402
    MEMORY_EMITTERS,
    emit_tt_addptr,
    emit_tt_broadcast,
    emit_tt_load,
    emit_tt_make_range,
    emit_tt_store,
    has_cxx_shim,
)
from poc.triton_frontend.op_mapping import WalkerCtx  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_value(name: str, *, shape: List[int] = (), dtype: str = "float32") -> Dict[str, Any]:
    return {"name": name, "shape": tuple(shape), "dtype": dtype}


def _stringify(node: Any) -> str:
    """Stringify a TIR node / list of stmts for substring assertions."""
    if isinstance(node, list):
        return "\n".join(str(s) for s in node)
    return str(node)


def _force_no_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``has_cxx_shim`` to return False inside ``op_emitters.memory``.

    The C++ shim's availability is not under test here; we want to exercise
    the degraded path deterministically.
    """
    monkeypatch.setattr(
        "poc.triton_frontend.op_emitters.memory.has_cxx_shim",
        lambda: False,
    )


# ---------------------------------------------------------------------------
# tt.make_range -> tir.Ramp
# ---------------------------------------------------------------------------


def test_make_range_emits_ramp() -> None:
    """``tt.make_range(0, 16)`` becomes ``tir.Ramp(0, 1, 16)``."""
    ctx = WalkerCtx()
    out = _fake_value("range_out", shape=[16], dtype="int32")
    op = {
        "name": "tt.make_range",
        "operands": [],
        "results": [out],
        "attrs": {"start": 0, "end": 16},
    }
    ramp = emit_tt_make_range(op, ctx)
    assert isinstance(ramp, tvm.tir.Ramp), f"expected tir.Ramp, got {type(ramp).__name__}"
    assert int(ramp.lanes) == 16
    # ``base`` must be the constant 0; ``stride`` 1.
    assert int(ramp.base) == 0
    assert int(ramp.stride) == 1


def test_make_range_wide_spills_to_for_loop() -> None:
    """A 4096-lane range exceeds the default vector width and spills to a For."""
    ctx = WalkerCtx()
    out = _fake_value("wide_range_out", shape=[4096], dtype="int32")
    op = {
        "name": "tt.make_range",
        "operands": [],
        "results": [out],
        "attrs": {"start": 0, "end": 4096},
    }
    buf = emit_tt_make_range(op, ctx)
    assert isinstance(buf, tvm.tir.Buffer)
    # The For loop should have been emitted into ctx.stmts.
    text = _stringify(ctx.stmts)
    assert "for" in text.lower(), f"expected serial For loop in: {text!r}"


# ---------------------------------------------------------------------------
# tt.broadcast -> tir.Broadcast
# ---------------------------------------------------------------------------


def test_broadcast_scalar_to_tile_emits_broadcast() -> None:
    """``tt.broadcast(scalar) : f32 -> 16xf32`` becomes ``tir.Broadcast``."""
    ctx = WalkerCtx()
    scalar_ssa = _fake_value("s", shape=[], dtype="float32")
    out_ssa = _fake_value("v", shape=[16], dtype="float32")
    # Bind the scalar to a real PrimExpr so emit_tt_broadcast resolves it.
    ctx.bind(scalar_ssa, tvm.tir.const(1.5, "float32"))
    op = {
        "name": "tt.broadcast",
        "operands": [scalar_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    bcast = emit_tt_broadcast(op, ctx)
    assert isinstance(bcast, tvm.tir.Broadcast)
    assert int(bcast.lanes) == 16


# ---------------------------------------------------------------------------
# tt.load / tt.store -- masked round-trip on a 1-D buffer
# ---------------------------------------------------------------------------


def test_load_with_mask_emits_if_then_else(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scalar masked load wraps the BufferLoad in ``tir.if_then_else``."""
    _force_no_shim(monkeypatch)
    ctx = WalkerCtx()
    buf = tvm.tir.decl_buffer([16], "float32", name="A")
    ptr_ssa = _fake_value("ptr", shape=[], dtype="float32")
    mask_ssa = _fake_value("mask", shape=[], dtype="bool")
    other_ssa = _fake_value("other", shape=[], dtype="float32")
    out_ssa = _fake_value("loaded", shape=[], dtype="float32")
    ctx.bind(ptr_ssa, buf)
    ctx.bind(mask_ssa, tvm.tir.const(True, "bool"))
    ctx.bind(other_ssa, tvm.tir.const(0.0, "float32"))
    op = {
        "name": "tt.load",
        "operands": [ptr_ssa, mask_ssa, other_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    expr = emit_tt_load(op, ctx)
    text = str(expr).lower()
    assert "if_then_else" in text, f"expected if_then_else mask guard, got {text!r}"


def test_store_with_mask_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-trip: load masked, store masked into the same buffer shape."""
    _force_no_shim(monkeypatch)
    ctx = WalkerCtx()
    buf = tvm.tir.decl_buffer([16], "float32", name="B")

    ptr_in = _fake_value("ptr_in", shape=[], dtype="float32")
    mask_in = _fake_value("mask", shape=[], dtype="bool")
    other = _fake_value("other", shape=[], dtype="float32")
    loaded = _fake_value("loaded", shape=[], dtype="float32")
    ctx.bind(ptr_in, buf)
    ctx.bind(mask_in, tvm.tir.const(True, "bool"))
    ctx.bind(other, tvm.tir.const(0.0, "float32"))
    emit_tt_load(
        {
            "name": "tt.load",
            "operands": [ptr_in, mask_in, other],
            "results": [loaded],
            "attrs": {},
        },
        ctx,
    )

    ptr_out = _fake_value("ptr_out", shape=[], dtype="float32")
    ctx.bind(ptr_out, buf)
    emit_tt_store(
        {
            "name": "tt.store",
            "operands": [ptr_out, loaded, mask_in],
            "results": [],
            "attrs": {},
        },
        ctx,
    )
    text = _stringify(ctx.stmts)
    assert "IfThenElse" in text or "if " in text.lower(), (
        f"expected guarded BufferStore in stmts, got {text!r}"
    )


# ---------------------------------------------------------------------------
# Multi-element tile load -- degraded path when no shim is available.
# ---------------------------------------------------------------------------


def test_multi_element_load_without_shim_emits_degraded_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the PtrAnalysis shim a tile load must emit ``# DEGRADED:``.

    The maintainer's hard constraint: never silent-fallback. We force the
    shim probe to ``False`` and then verify the printed PrimFunc text
    contains the ``# DEGRADED:`` marker that the AttrStmt pragma_comment
    survives all the way through.
    """
    _force_no_shim(monkeypatch)
    ctx = WalkerCtx()
    ptr_ssa = _fake_value("tile_ptr", shape=[32], dtype="float32")
    out_ssa = _fake_value("tile_out", shape=[32], dtype="float32")
    op = {
        "name": "tt.load",
        "operands": [ptr_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    emit_tt_load(op, ctx)
    text = _stringify(ctx.stmts)
    assert "DEGRADED" in text, (
        f"expected '# DEGRADED:' marker in printed stmts; got:\n{text}"
    )
    # And we should see a serial For driving the per-element scalar loads.
    assert "for" in text.lower(), f"expected per-element For loop; got:\n{text}"


def test_addptr_without_shim_emits_degraded_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pointer-arithmetic test for the no-shim degraded path.

    Without the shim, ``tt.addptr`` cannot fold into a strided layout, so
    we must emit a visible ``# DEGRADED:`` breadcrumb so reviewers know
    the result is a scalar offset add only.
    """
    _force_no_shim(monkeypatch)
    ctx = WalkerCtx()
    ptr_ssa = _fake_value("ptr", shape=[], dtype="float32")
    off_ssa = _fake_value("off", shape=[], dtype="int32")
    out_ssa = _fake_value("ptr2", shape=[], dtype="float32")
    buf = tvm.tir.decl_buffer([1024], "float32", name="P")
    ctx.bind(ptr_ssa, (buf, [tvm.tir.const(0, "int32")]))
    ctx.bind(off_ssa, tvm.tir.const(7, "int32"))
    op = {
        "name": "tt.addptr",
        "operands": [ptr_ssa, off_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    emit_tt_addptr(op, ctx)
    text = _stringify(ctx.stmts)
    assert "DEGRADED" in text, (
        f"expected '# DEGRADED:' breadcrumb for shim-less tt.addptr; got:\n{text}"
    )


# ---------------------------------------------------------------------------
# Dispatch table sanity
# ---------------------------------------------------------------------------


def test_memory_emitters_table_keys() -> None:
    """The exported dict carries every op the maintainer asked for."""
    expected = {
        "tt.load",
        "tt.store",
        "tt.make_range",
        "tt.expand_dims",
        "tt.broadcast",
        "tt.splat",
        "tt.view",
        "tt.reshape",
        "tt.addptr",
        "tts.make_tptr",
    }
    assert expected.issubset(MEMORY_EMITTERS.keys()), (
        f"missing emitters: {expected - MEMORY_EMITTERS.keys()}"
    )


def test_has_cxx_shim_returns_bool() -> None:
    """``has_cxx_shim()`` returns a real bool (not None / dict)."""
    assert isinstance(has_cxx_shim(), bool)
