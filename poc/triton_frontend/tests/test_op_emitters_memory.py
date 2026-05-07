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

from typing import Any, Dict, List, Sequence

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


class _HashableSSA(dict):
    """Hashable dict-shaped SSA fixture.

    The legacy ``_fake_value`` returns a plain ``dict`` which the old
    ``WalkerCtx.bind`` happily accepted (TVM's value_map used to allow
    unhashable keys via a list-of-pairs lookup). ``WalkerCtx.bind`` now
    indexes ``self.value_map[ssa_value]`` directly, so a plain dict keys
    raise ``TypeError: unhashable type: 'dict'`` -- exactly the symptom
    that took the existing test suite offline.

    The fix that's compatible with every emitter helper (``_shape_of``
    / ``_dtype_of`` / ``_ssa_name`` -- all of them check ``isinstance(v,
    dict)``) is a tiny ``dict`` subclass that overrides ``__hash__`` to
    use ``id()``. It is still a real dict so the helpers see the
    ``shape`` / ``dtype`` / ``name`` keys; ``bind`` is happy because
    ``hash(_HashableSSA(...))`` returns an int.
    """

    __slots__ = ()

    def __hash__(self) -> int:  # type: ignore[override]
        return id(self)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        return self is other


def _ssa(name: str, *, shape: Sequence[int] = (), dtype: str = "float32") -> _HashableSSA:
    """Hashable counterpart to ``_fake_value`` for tests that need ``ctx.bind``."""
    return _HashableSSA(name=name, shape=tuple(shape), dtype=dtype)


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


# ---------------------------------------------------------------------------
# Regression: generic-form Triton 3.6 attribute parsing for ``tt.make_range``.
# ---------------------------------------------------------------------------
#
# When the harness rounds the captured TTIR through the C++ shim's
# ``to_generic()`` and re-parses it with jaxlib's ``mlir.ir`` (which only
# supports ``allow_unregistered_dialects``), the ``<{end = 256, start = 0}>``
# block stays as Properties storage. jaxlib does NOT surface property
# values via ``op.attributes`` for unregistered dialects, so the legacy
# helper :func:`poc.triton_frontend.op_mapping._attrs` returns ``{}`` and
# the emitter computed a degenerate ``[0, 0)`` range -> ``ValueError``.
#
# The fix in ``op_emitters/memory.py`` introduces ``_attrs_with_properties``,
# which falls back to parsing the printed ``<{...}>`` block out of
# ``str(op)`` when the dict-shaped accessor comes back empty.


def test_make_range_parses_generic_form_attrs() -> None:
    """Generic-form attrs (``<{start = 0, end = 256}>``) lower to a 256-lane Ramp.

    We cover the dict-shape form here (the synthetic op the bug report
    handed us); the textual ``<{...}>`` parser is exercised separately
    via ``test_make_range_parses_real_mlir_properties`` below.
    """
    ctx = WalkerCtx()
    out = _ssa("range_out", shape=[256], dtype="int32")
    op = {
        "name": "tt.make_range",
        "operands": [],
        "results": [out],
        # Mirrors the dict produced by the walker after extracting the
        # ``<{end = 256 : i32, start = 0 : i32}>`` properties block.
        "attrs": {"start": 0, "end": 256},
    }
    out_node = emit_tt_make_range(op, ctx)
    # The bug surfaced as ``Ramp(0, 1, 0)`` -- a degenerate range that
    # raised ``ValueError: end < start`` from inside the emitter. After
    # the fix, 256 lanes exceeds the vector-width cap (128) so the
    # emitter spills to a serial ``tir.For`` over a 256-element buffer.
    # The structural invariant the regression proves out is "lanes is
    # 256, not 0" -- we check that on whichever shape the emitter chose
    # (Ramp for narrow, Buffer for spilled).
    if isinstance(out_node, tvm.tir.Ramp):
        assert int(out_node.lanes) == 256
        assert int(out_node.base) == 0
        assert int(out_node.stride) == 1
    elif isinstance(out_node, tvm.tir.Buffer):
        assert int(out_node.shape[0]) == 256, (
            f"spilled buffer must have 256 elements; got shape={out_node.shape}"
        )
        # And the For loop driving the spill must have been emitted.
        text = _stringify(ctx.stmts)
        assert "for " in text.lower(), (
            f"expected serial For driving the spill; got {text!r}"
        )
    else:
        raise AssertionError(
            f"expected tir.Ramp or tir.Buffer; got {type(out_node).__name__}"
        )


def test_make_range_parses_real_mlir_properties() -> None:
    """Parse a real generic-form MLIR op whose properties ARE textual-only.

    Uses jaxlib's ``mlir.ir`` to construct an op with the exact
    ``<{end = 256, start = 0}>`` printed shape Triton 3.6 emits. The
    jaxlib bindings expose properties through ``str(op)`` but NOT through
    ``op.attributes`` (the latter is empty for unregistered dialects);
    the emitter's ``_attrs_with_properties`` helper recovers the values
    by parsing the printed assembly.
    """
    ir = pytest.importorskip("jaxlib.mlir.ir")

    text = (
        "module {\n"
        '  "tt.make_range"() <{end = 256 : i32, start = 0 : i32}>'
        " : () -> tensor<256xi32>\n"
        "}"
    )
    mctx = ir.Context()
    mctx.allow_unregistered_dialects = True
    with mctx, ir.Location.unknown(mctx):
        mod = ir.Module.parse(text, mctx)

    target_op = None
    for region in mod.operation.regions:
        for block in region.blocks:
            for child in block.operations:
                if getattr(child, "name", None) == "tt.make_range":
                    target_op = child
                    break

    assert target_op is not None, "did not find tt.make_range in parsed module"
    # Sanity: confirm the legacy attrs accessor is empty -- this is the
    # exact regression precondition the helper now compensates for.
    assert len(list(target_op.attributes)) == 0, (
        "test precondition violated: jaxlib now surfaces properties via "
        ".attributes (would mean the regression cannot reproduce here)"
    )

    ctx = WalkerCtx()
    out_node = emit_tt_make_range(target_op, ctx)
    # 256 lanes spills past the default vector cap (128) -> Buffer
    # backed by a serial For. The regression-preventing invariant is
    # "we got 256 elements out, not 0".
    if isinstance(out_node, tvm.tir.Ramp):
        assert int(out_node.lanes) == 256
        assert int(out_node.base) == 0
    else:
        assert isinstance(out_node, tvm.tir.Buffer)
        assert int(out_node.shape[0]) == 256


def test_load_handles_f32_dtype_string() -> None:
    """``tt.load`` over a ``tensor<256xf32>`` operand must NOT raise ``unknown dtype 'f32'``.

    The MLIR generic-form prints scalar element types using the short
    spelling (``f32``), which TVM's ``tir.decl_buffer`` rejects with
    ``ValueError: unknown dtype 'f32'``. The fix to ``_dtype_of`` in
    ``op_mapping.py`` normalises the alias map (``f32 -> float32``,
    ``i32 -> int32``, ...) so emitters that thread the dtype through
    ``tir.decl_buffer`` keep working with both shapes.
    """
    ctx = WalkerCtx()
    # The ``dtype`` here is the SHORT MLIR spelling -- exactly what
    # ``_dtype_of`` returned before the fix when the result type was
    # printed in generic form.
    ptr_ssa = _ssa("ptr_in", shape=[], dtype="f32")
    out_ssa = _ssa("loaded", shape=[256], dtype="f32")

    # Bind the pointer to a real buffer with the SHORT dtype too -- mimics
    # the state that ``map_tt_func`` would leave us in when it received
    # generic-form ``!tt.ptr<f32>`` block-arg types from the walker.
    buf = tvm.tir.decl_buffer([256], "float32", name="A")
    ctx.bind(ptr_ssa, buf)
    op = {
        "name": "tt.load",
        "operands": [ptr_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    # The regression: this used to raise ``ValueError: unknown dtype 'f32'``
    # from tir.decl_buffer when the tile-load fallback path tried to
    # allocate a fresh buffer with the unnormalised dtype.
    emit_tt_load(op, ctx)


def test_dtype_helper_normalises_short_mlir_spellings() -> None:
    """Guard the dtype alias map against accidental coverage gaps.

    Hard constraint from the maintainer: every short MLIR dtype Triton
    can produce must canonicalise to a TVM dtype string. A genuinely
    unknown dtype must raise ``ValueError`` so we never silently fall
    back to ``float32``.
    """
    from poc.triton_frontend.op_mapping import _normalize_mlir_dtype

    cases = {
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "bf16": "bfloat16",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "index": "int64",
    }
    for short, canonical in cases.items():
        assert _normalize_mlir_dtype(short) == canonical, (
            f"expected {short} -> {canonical}, got {_normalize_mlir_dtype(short)}"
        )

    # Already-canonical TVM spellings round-trip unchanged.
    for tvm_name in ("float32", "int64", "bool", "bfloat16"):
        assert _normalize_mlir_dtype(tvm_name) == tvm_name

    # Unknown dtypes must NOT silently default to float32 (the no-silent-
    # fallback rule).
    with pytest.raises(ValueError, match="unsupported MLIR dtype"):
        _normalize_mlir_dtype("complex_who_knows")
