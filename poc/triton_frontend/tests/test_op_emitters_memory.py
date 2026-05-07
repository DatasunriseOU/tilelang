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
    emit_tt_expand_dims,
    emit_tt_load,
    emit_tt_make_range,
    emit_tt_splat,
    emit_tt_store,
    has_cxx_shim,
)
from poc.triton_frontend.op_mapping import WalkerCtx  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_value(name: str, *, shape: List[int] = (), dtype: str = "float32") -> Dict[str, Any]:
    return {"name": name, "shape": tuple(shape), "dtype": dtype}


# ``_HashableSSA`` previously lived inline; the canonical implementation
# now lives in :mod:`poc.triton_frontend.tests._fixtures` so all
# op-emitter tests share the same hashable-dict surface.
from ._fixtures import FakeSSA as _HashableSSA  # noqa: E402


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
    out = _ssa("range_out", shape=[16], dtype="int32")
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
    out = _ssa("wide_range_out", shape=[4096], dtype="int32")
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


def test_make_range_uses_alloc_fragment_not_decl_buffer() -> None:
    """Wide ``tt.make_range`` spill must produce a *tile-scoped* buffer.

    Regression for "Memory verification failed -- Variable 'range_2' is
    directly accessed by host memory". The fix routes the spill through
    ``op_mapping._alloc_tile_buffer`` so the buffer:

    * is registered in ``ctx.local_buffers`` (NOT ``ctx.buffers``), so
      it doesn't end up in the PrimFunc ``buffer_map`` / become a
      function arg, and
    * carries a non-empty storage scope (``"local"`` by default), so
      downstream lowering treats it as thread-private storage.
    """
    ctx = WalkerCtx()
    out = _ssa("wide_range_out", shape=[4096], dtype="int32")
    op = {
        "name": "tt.make_range",
        "operands": [],
        "results": [out],
        "attrs": {"start": 0, "end": 4096},
    }
    buf = emit_tt_make_range(op, ctx)
    assert isinstance(buf, tvm.tir.Buffer)
    # Tile-scoped allocation contract: the spill buffer is in
    # ``local_buffers`` (so ``_make_prim_func`` wraps it in an
    # ``AllocBuffer`` stmt) and NOT in ``buffers`` (so it doesn't get
    # promoted to a PrimFunc argument).
    assert buf in ctx.local_buffers, (
        "wide tt.make_range must register its spill buffer in "
        "ctx.local_buffers so VerifyMemory sees it as locally allocated"
    )
    assert buf.name not in ctx.buffers, (
        "wide tt.make_range MUST NOT promote its spill buffer to "
        "ctx.buffers (which would make it a PrimFunc argument and trip "
        "VerifyMemory's host-memory access check)"
    )
    # Storage scope check: tile-scoped buffers carry a non-empty scope
    # so the LowerOpaqueBlock / FlattenBuffer passes treat them as
    # thread-private.  The default helper emits ``"local"``.
    scope = getattr(buf, "scope", None)
    if callable(scope):
        scope_str = scope()
    else:
        scope_str = str(scope) if scope is not None else ""
    assert scope_str and scope_str != "global", (
        f"expected non-global storage scope on spill buffer; got {scope_str!r}"
    )


# ---------------------------------------------------------------------------
# tt.broadcast -> tir.Broadcast
# ---------------------------------------------------------------------------


def test_broadcast_scalar_to_tile_emits_broadcast() -> None:
    """``tt.broadcast(scalar) : f32 -> 16xf32`` becomes ``tir.Broadcast``."""
    ctx = WalkerCtx()
    scalar_ssa = _ssa("s", shape=[], dtype="float32")
    out_ssa = _ssa("v", shape=[16], dtype="float32")
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
    ptr_ssa = _ssa("ptr", shape=[], dtype="float32")
    mask_ssa = _ssa("mask", shape=[], dtype="bool")
    other_ssa = _ssa("other", shape=[], dtype="float32")
    out_ssa = _ssa("loaded", shape=[], dtype="float32")
    ctx.bind(ptr_ssa, buf)
    # Use a non-constant mask Var so the constant-folding pass doesn't
    # elide the IfThenElse before the test gets to inspect it.
    mask_var = tvm.tir.Var("m", "bool")
    ctx.bind(mask_ssa, mask_var)
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

    ptr_in = _ssa("ptr_in", shape=[], dtype="float32")
    mask_in = _ssa("mask", shape=[], dtype="bool")
    other = _ssa("other", shape=[], dtype="float32")
    loaded = _ssa("loaded", shape=[], dtype="float32")
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

    ptr_out = _ssa("ptr_out", shape=[], dtype="float32")
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
    ptr_ssa = _ssa("tile_ptr", shape=[32], dtype="float32")
    out_ssa = _ssa("tile_out", shape=[32], dtype="float32")
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
    ptr_ssa = _ssa("ptr", shape=[], dtype="float32")
    off_ssa = _ssa("off", shape=[], dtype="int32")
    out_ssa = _ssa("ptr2", shape=[], dtype="float32")
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
# tt.addptr -> arith.addf pipeline-shape regression (vector_add e2e blocker)
# ---------------------------------------------------------------------------
#
# Triton TTIR for vector_add combines tile-shaped operands at the arith
# layer (``arith.addi(splat<256xi32>, range<256xi32>)`` -- the splat is a
# ``tir.Broadcast`` PrimExpr, the range a ``tir.Buffer`` once it spilled
# past the vector-width cap). Before the C2-D1 fix, mixed shapes blew up
# the scalar-only ``tirx.Add`` with
# ``TypeError: Expected ir.PrimExpr but got tirx.Buffer``. The fix routes
# Buffer / Broadcast / Ramp operands through ``_emit_tile_binop`` in
# ``op_emitters/arith.py``; the regression below pins down the contract:
# (1) ``tt.addptr`` returns a 2-tuple ``(buffer, indices)`` consumed by
#     ``tt.load`` / ``tt.store``, never by an ``arith.*`` op;
# (2) tile-shaped ``arith.*`` consumers compose by emitting per-lane
#     ``tir.For`` loops that produce a fresh result Buffer.


def test_addptr_returns_buffer_indices_tuple_for_load_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tt.addptr`` result is a ``(Buffer, [PrimExpr])`` tuple."""
    _force_no_shim(monkeypatch)
    ctx = WalkerCtx()
    ptr_ssa = _ssa("ptr", shape=[], dtype="float32")
    off_ssa = _ssa("off", shape=[], dtype="int32")
    out_ssa = _ssa("ptr_added", shape=[], dtype="float32")
    buf = tvm.tir.decl_buffer([1024], "float32", name="A")
    ctx.bind(ptr_ssa, (buf, [tvm.tir.const(0, "int32")]))
    ctx.bind(off_ssa, tvm.tir.const(7, "int32"))
    op = {
        "name": "tt.addptr",
        "operands": [ptr_ssa, off_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    result = emit_tt_addptr(op, ctx)
    assert isinstance(result, tuple), (
        f"tt.addptr must return a (buffer, indices) tuple; got {type(result).__name__}"
    )
    assert len(result) == 2, f"tt.addptr tuple must have 2 elements; got {len(result)}"
    addptr_buf, addptr_indices = result
    assert isinstance(addptr_buf, tvm.tir.Buffer), (
        f"tt.addptr[0] must be a tir.Buffer; got {type(addptr_buf).__name__}"
    )
    assert isinstance(addptr_indices, list), (
        f"tt.addptr[1] must be a list of PrimExpr; got {type(addptr_indices).__name__}"
    )


def test_arith_addi_handles_buffer_and_broadcast_tile_operands() -> None:
    """``arith.addi(broadcast, buffer)`` emits a per-lane For into a fresh Buffer.

    This is the regression that took ``vector_add`` offline: ``tt.make_range``
    spills to a Buffer at lane > 128 and ``tt.splat`` produces a Broadcast,
    so ``arith.addi(splat, make_range)`` had two different tile shapes
    feeding ``ctx.tir().Add`` -- which is scalar-only.
    """
    from poc.triton_frontend.op_emitters.arith import _emit_addi  # noqa: WPS433

    ctx = WalkerCtx()
    splat_ssa = _ssa("splat", shape=[256], dtype="int32")
    range_ssa = _ssa("range", shape=[256], dtype="int32")
    out_ssa = _ssa("offsets", shape=[256], dtype="int32")

    splat_val = tvm.tir.Broadcast(tvm.tir.const(7, "int32"), 256)
    range_buf = tvm.tir.decl_buffer([256], "int32", name="R")
    ctx.bind(splat_ssa, splat_val)
    ctx.bind(range_ssa, range_buf)

    op = {
        "name": "arith.addi",
        "operands": [splat_ssa, range_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    result = _emit_addi(op, ctx)
    assert isinstance(result, tvm.tir.Buffer), (
        f"tile-shaped arith.addi must produce a tir.Buffer; got "
        f"{type(result).__name__}"
    )
    assert int(result.shape[0]) == 256
    text = _stringify(ctx.stmts)
    assert "for" in text.lower(), (
        f"expected per-lane For loop in emitted stmts; got:\n{text}"
    )


def test_arith_addf_handles_two_buffer_tile_operands() -> None:
    """``arith.addf(buffer, buffer)`` emits a per-lane For (the load+load+addf path)."""
    from poc.triton_frontend.op_emitters.arith import _emit_addf  # noqa: WPS433

    ctx = WalkerCtx()
    x_ssa = _ssa("x", shape=[256], dtype="float32")
    y_ssa = _ssa("y", shape=[256], dtype="float32")
    out_ssa = _ssa("z", shape=[256], dtype="float32")

    x_buf = tvm.tir.decl_buffer([256], "float32", name="X")
    y_buf = tvm.tir.decl_buffer([256], "float32", name="Y")
    ctx.bind(x_ssa, x_buf)
    ctx.bind(y_ssa, y_buf)

    op = {
        "name": "arith.addf",
        "operands": [x_ssa, y_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    result = _emit_addf(op, ctx)
    assert isinstance(result, tvm.tir.Buffer), (
        f"tile-shaped arith.addf must produce a tir.Buffer; got "
        f"{type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# tt.addptr in-loop pointer increment (matmul scf.for body)
# ---------------------------------------------------------------------------
#
# Matmul's main loop carries ``a_ptrs += BLOCK_K * stride_ak`` -- a tile-shaped
# offset-buffer increment that can't be combined with the previous indices via
# scalar TIR ``+`` (the operand is a Buffer, not a PrimExpr). Before this
# Wave, ``emit_tt_addptr`` blindly evaluated ``new_indices[-1] + off`` and
# raised ``Mismatched type ... Expected ir.PrimExpr but got tirx.Buffer``.
# The fix routes Buffer-shaped offsets through ``_compose_addptr_index``
# which allocates a fresh tile buffer and emits a per-lane ``tir.For`` nest.


def test_addptr_buffer_offset_emits_per_lane_for_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tt.addptr(ptr, buffer_offset)`` allocates a fresh tile and emits a For.

    Mirrors the matmul scf.for body where the offset operand of an in-loop
    ``tt.addptr`` is a per-lane offset Buffer (the ``BLOCK_K * stride_ak``
    tile from a prior ``arith.muli``), not a scalar PrimExpr. The trailing
    index entry of the result must itself be a fresh ``tir.Buffer`` so
    downstream ``tt.load`` consumers see a tile-shaped index, and the
    walker stmts must contain a serial ``tir.For`` that drives the
    elementwise add.
    """
    _force_no_shim(monkeypatch)
    ctx = WalkerCtx()
    ptr_ssa = _ssa("a_ptrs", shape=[32], dtype="float32")
    off_ssa = _ssa("k_step", shape=[32], dtype="int32")
    out_ssa = _ssa("a_ptrs_new", shape=[32], dtype="float32")

    base_buf = tvm.tir.decl_buffer([1024], "float32", name="A")
    # Previous iteration's trailing index entry is a per-lane offset Buffer.
    prev_off = tvm.tir.decl_buffer([32], "int32", name="prev_off")
    step_buf = tvm.tir.decl_buffer([32], "int32", name="k_step_tile")

    ctx.bind(ptr_ssa, (base_buf, [prev_off]))
    ctx.bind(off_ssa, step_buf)

    op = {
        "name": "tt.addptr",
        "operands": [ptr_ssa, off_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    result = emit_tt_addptr(op, ctx)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"tt.addptr must return (buf, indices); got {type(result).__name__}"
    )
    out_buf, out_indices = result
    assert out_buf is base_buf, "tt.addptr must thread the base buffer through"
    assert isinstance(out_indices, list) and len(out_indices) == 1
    assert isinstance(out_indices[0], tvm.tir.Buffer), (
        f"trailing index entry must be a fresh tile Buffer; got "
        f"{type(out_indices[0]).__name__}"
    )
    assert int(out_indices[0].shape[0]) == 32

    text = _stringify(ctx.stmts)
    assert "for" in text.lower(), (
        f"expected per-lane For loop driving the addptr accumulation; got:\n"
        f"{text}"
    )
    # The new tile buffer must be different from either input tile.
    assert out_indices[0] is not prev_off
    assert out_indices[0] is not step_buf


def test_addptr_scalar_offset_keeps_scalar_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scalar PrimExpr offsets must NOT trigger a tile-buffer allocation.

    The fast path is what the original (pre-loop) matmul ``a_ptrs =
    a_ptr + offs_am`` lowering relies on; we must preserve it so we don't
    regress the non-loop call site.
    """
    _force_no_shim(monkeypatch)
    ctx = WalkerCtx()
    ptr_ssa = _ssa("ptr", shape=[], dtype="float32")
    off_ssa = _ssa("off", shape=[], dtype="int32")
    out_ssa = _ssa("ptr_added", shape=[], dtype="float32")
    buf = tvm.tir.decl_buffer([1024], "float32", name="A")
    ctx.bind(ptr_ssa, (buf, [tvm.tir.const(0, "int32")]))
    ctx.bind(off_ssa, tvm.tir.const(7, "int32"))

    pre_count = len(ctx.local_buffers)
    op = {
        "name": "tt.addptr",
        "operands": [ptr_ssa, off_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    result = emit_tt_addptr(op, ctx)
    assert isinstance(result, tuple)
    _, indices = result
    # Trailing index must remain a scalar PrimExpr (not a Buffer).
    assert not isinstance(indices[-1], tvm.tir.Buffer), (
        f"scalar offset must keep scalar fast path; got "
        f"{type(indices[-1]).__name__}"
    )
    assert len(ctx.local_buffers) == pre_count, (
        "scalar offset must NOT allocate a tile buffer"
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


# ---------------------------------------------------------------------------
# Vector-src lowering regressions for expand_dims / broadcast / splat.
#
# Bug: TVM's ``tir.Broadcast(value, lanes)`` REQUIRES a scalar ``value``.
# Calling it with a vector PrimExpr (e.g. ``Broadcast(pid*64, 64) +
# Ramp(0,1,64)`` whose dtype is ``int32x64``) blows up with
# ``Check failed: (value.dtype().is_scalar()) is false``.
# A separate symptom in the same call -- passing the result-shape tuple
# (an ``ffi.Array``) for ``lanes`` instead of an int -- raises
# ``TypeError: ... Expected ir.PrimExpr but got ffi.Array``.
#
# The fix in ``op_emitters/memory.py``:
#  * Vector-src case: emit a serial ``tir.For`` nest into a fresh buffer,
#    indexing the source per-lane via the innermost loop var.
#  * ``ffi.Array`` lanes: ``_coerce_lanes_to_int`` unwraps tile shapes
#    before they reach ``tir.Broadcast``.
# ---------------------------------------------------------------------------


def test_expand_dims_on_vector_primexpr_emits_for_nest() -> None:
    """``tt.expand_dims`` on a vector PrimExpr lowers to a For-nest, not Broadcast.

    Regression: emit_tt_expand_dims used to call ``tir.Broadcast(src, lanes)``
    unconditionally, which blows up on a vector ``src``.
    """
    ctx = WalkerCtx()
    src_ssa = _ssa("offsets", shape=[64], dtype="int32")
    out_ssa = _ssa("offsets_2d", shape=[1, 64], dtype="int32")

    # Build a real vector PrimExpr: Broadcast(pid*64, 64) + Ramp(0, 1, 64).
    pid = tvm.tir.Var("pid", "int32")
    base = tvm.tir.Broadcast(pid * tvm.tir.const(64, "int32"), 64)
    ramp = tvm.tir.Ramp(tvm.tir.const(0, "int32"), tvm.tir.const(1, "int32"), 64)
    vec = base + ramp
    assert "x" in str(vec.dtype), "test precondition: src must be a vector PrimExpr"
    ctx.bind(src_ssa, vec)

    op = {
        "name": "tt.expand_dims",
        "operands": [src_ssa],
        "results": [out_ssa],
        "attrs": {"axis": 0},
    }
    out = emit_tt_expand_dims(op, ctx)
    # Result must be a fresh Buffer of shape [1, 64], NOT a Broadcast PrimExpr.
    assert isinstance(out, tvm.tir.Buffer), (
        f"expected tir.Buffer (For-nest result), got {type(out).__name__}"
    )
    assert tuple(int(s) for s in out.shape) == (1, 64)
    text = _stringify(ctx.stmts)
    assert "for" in text.lower(), f"expected For-nest in stmts; got:\n{text}"


def test_broadcast_vector_to_tile_emits_for_nest() -> None:
    """``tt.broadcast`` from a vector PrimExpr to a tile lowers to a For-nest."""
    ctx = WalkerCtx()
    src_ssa = _ssa("offs", shape=[64], dtype="int32")
    out_ssa = _ssa("offs_tile", shape=[16, 64], dtype="int32")

    # Vector PrimExpr source.
    base = tvm.tir.Broadcast(tvm.tir.const(7, "int32"), 64)
    ramp = tvm.tir.Ramp(tvm.tir.const(0, "int32"), tvm.tir.const(1, "int32"), 64)
    vec = base + ramp
    ctx.bind(src_ssa, vec)

    op = {
        "name": "tt.broadcast",
        "operands": [src_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    out = emit_tt_broadcast(op, ctx)
    assert isinstance(out, tvm.tir.Buffer), (
        f"expected tir.Buffer (For-nest result), got {type(out).__name__}"
    )
    assert tuple(int(s) for s in out.shape) == (16, 64)
    text = _stringify(ctx.stmts)
    assert "for" in text.lower(), f"expected For-nest in stmts; got:\n{text}"


def test_tt_load_with_buffer_other_emits_per_lane_buffer_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tile load whose ``other`` resolved to a Buffer (post Wave E2 lowering)
    must emit ``BufferLoad(other_buf, [lane])`` inside the IfThenElse arg #2,
    NOT a bare buffer reference (which TVM rejects with
    ``Mismatched type on argument #N: Expected ir.PrimExpr but got
    tirx.Buffer``).

    Setup mirrors ``vector_add`` post-E2: the ``other`` operand resolves to
    a freshly declared buffer (the lowered ``arith.constant dense<0.0>``)
    and the ``mask`` likewise resolves to a Buffer-shaped boolean tile.
    """
    _force_no_shim(monkeypatch)
    ctx = WalkerCtx()

    src_buf = tvm.tir.decl_buffer([256], "float32", name="A")
    mask_buf = tvm.tir.decl_buffer([256], "bool", name="M")
    other_buf = tvm.tir.decl_buffer([256], "float32", name="O")

    ptr_ssa = _ssa("ptr", shape=[256], dtype="float32")
    mask_ssa = _ssa("mask", shape=[256], dtype="bool")
    other_ssa = _ssa("other", shape=[256], dtype="float32")
    out_ssa = _ssa("loaded", shape=[256], dtype="float32")

    ctx.bind(ptr_ssa, src_buf)
    ctx.bind(mask_ssa, mask_buf)
    ctx.bind(other_ssa, other_buf)

    op = {
        "name": "tt.load",
        "operands": [ptr_ssa, mask_ssa, other_ssa],
        "results": [out_ssa],
        "attrs": {},
    }

    # The crucial assertion is that the call does NOT raise the TVM
    # ``Mismatched type ... Expected ir.PrimExpr but got tirx.Buffer``
    # error -- which it would if we fed bare Buffers into ``if_then_else``.
    result = emit_tt_load(op, ctx)
    assert isinstance(result, tvm.tir.Buffer), (
        "tile-shaped tt.load must produce a tir.Buffer; got "
        f"{type(result).__name__}"
    )

    text = _stringify(ctx.stmts)
    # The IfThenElse / if_then_else expression must reference the other
    # buffer via a per-lane BufferLoad. We assert the buffer name appears
    # AND the if_then_else marker is present, so a reviewer can see the
    # buffer is accessed lane-wise rather than passed through bare.
    assert "if_then_else" in text.lower(), (
        f"expected if_then_else mask guard in stmts; got:\n{text}"
    )
    # The other buffer name appearing in the printed text means it was
    # consumed as a BufferLoad (a bare Buffer reference would not survive
    # the IfThenElse type-check, so the absence of an exception above
    # already proves the per-lane path; the substring assertion locks in
    # the readable contract.)
    assert "O[" in text, (
        f"expected per-lane BufferLoad on the 'other' buffer (e.g. 'O[i0]') "
        f"in printed stmts; got:\n{text}"
    )


def test_load_2d_tile_emits_2d_for_nest() -> None:
    """Rank-tracking regression for matmul's 2D tile load.

    matmul builds 2D pointer tiles via ``offs_m[:, None] * stride_am +
    offs_k[None, :]`` and then issues a ``tt.load`` with a rank-2 result
    type. Before this fix the no-PtrState branch in ``emit_tt_load``
    declared a 2D buffer for the placeholder source but indexed it with a
    single ``[0]`` index, which TVM rejects with
    ``buffer->shape.size() == indices.size() (2 vs. 1) : Buffer ... is
    2-dimensional, cannot be indexed with the 1-dimensional indices``.

    The contract this test pins:

    1. The emitted IR has a *nested* ``tir.For`` (rank == result rank,
       i.e. two ``for`` keywords for a 2D tile).
    2. The innermost ``BufferStore`` references *both* loop variables, so
       the tile buffer is filled rank-N rather than smashed to lane 0.
    3. The call must not raise the rank-mismatch ``InternalError``.
    """
    ctx = WalkerCtx()
    ptr_ssa = _ssa("ptr2d", shape=[16, 16], dtype="float32")
    out_ssa = _ssa("tile2d", shape=[16, 16], dtype="float32")
    op = {
        "name": "tt.load",
        "operands": [ptr_ssa],
        "results": [out_ssa],
        "attrs": {},
    }

    result = emit_tt_load(op, ctx)

    # The result must be a rank-2 tile buffer (not a scalar BufferLoad).
    assert isinstance(result, tvm.tir.Buffer), (
        f"expected tir.Buffer for 2D tile load; got {type(result).__name__}"
    )
    assert len(result.shape) == 2, (
        f"expected rank-2 result buffer; got shape={list(result.shape)}"
    )

    text = _stringify(ctx.stmts)
    # The TVM TIR printer collapses adjacent ``tir.For`` nodes into a
    # single ``T.grid(...)`` call, so a rank-2 nest may print as one
    # ``for i0, i1 in T.grid(16, 16)``. We assert on the rank-N axis
    # extents instead: a 2D tile must show both extents.
    assert "16, 16" in text or text.lower().count("for ") >= 2, (
        f"expected rank-2 loop nest (T.grid(16, 16) or two nested for "
        f"keywords) for a 2D tile; got:\n{text}"
    )
    # The fix must not have silently demoted the load to a single
    # ``BufferLoad(src, [0])``. We check that two loop variables drive
    # the rank-2 BufferStore index.
    assert "i0" in text and "i1" in text, (
        "expected both loop variables (i0, i1) to drive the rank-2 "
        f"BufferStore index; got:\n{text}"
    )


def test_store_2d_tile_emits_2d_for_nest() -> None:
    """Symmetric rank-tracking regression for ``tt.store`` on a 2D tile.

    Same rationale as :func:`test_load_2d_tile_emits_2d_for_nest`: matmul
    writes its 2D accumulator back through ``tt.store`` and would
    otherwise hit the ``buffer->shape.size() == indices.size()`` check on
    the destination buffer.
    """
    ctx = WalkerCtx()
    ptr_ssa = _ssa("ptr_out_2d", shape=[16, 16], dtype="float32")
    val_ssa = _ssa("val_2d", shape=[16, 16], dtype="float32")
    val_buf = tvm.tir.decl_buffer([16, 16], "float32", name="V")
    ctx.bind(val_ssa, val_buf)
    op = {
        "name": "tt.store",
        "operands": [ptr_ssa, val_ssa],
        "results": [],
        "attrs": {},
    }
    emit_tt_store(op, ctx)
    text = _stringify(ctx.stmts)
    # See ``test_load_2d_tile_emits_2d_for_nest`` for why a single
    # ``T.grid(16, 16)`` is also acceptable: the TIR printer collapses
    # adjacent ``tir.For`` nodes into a grid form.
    assert "16, 16" in text or text.lower().count("for ") >= 2, (
        f"expected rank-2 loop nest for a 2D tile store; got:\n{text}"
    )
    assert "i0" in text and "i1" in text, (
        "expected both loop variables (i0, i1) in the rank-2 BufferStore "
        f"index list; got:\n{text}"
    )


def test_emit_tile_load_from_input_buffer_2d_offset_tile() -> None:
    """Rank-tracking regression: ``_emit_tile_load_from_input_buffer`` must
    not index a rank-N offset tile buffer with a single 1-D index.

    matmul resolves ``a_ptrs`` (a ``tensor<64x64xi32>`` flat-address tile)
    via PtrAnalysis to ``(src_buf, [tile_2d_offsets])`` -- a single
    rank-2 offset buffer, not one buffer per axis. Before this fix the
    helper took the per-axis fallback branch and emitted
    ``BufferLoad(tile_2d_offsets, [lv0])`` -- a 1-D index list against a
    2-D buffer -- which TVM rejects with
    ``buffer->shape.size() == indices.size() (2 vs. 1) : Buffer ... is
    2-dimensional, cannot be indexed with the 1-dimensional indices``.

    Contract: when ``offset_indices`` is ``[buf]`` and ``buf.shape``
    matches the surrounding loop-var rank, the emitter indexes the
    offset buffer with the FULL loop-var nest and produces a single
    flat-address load against a 1D-redecl'd source buffer.
    """
    from poc.triton_frontend import (
        _emit_tile_load_from_input_buffer,
        _redecl_input_buffer,  # noqa: F401  -- ensure side-effect path imports
    )

    ctx = WalkerCtx()
    # Pre-populate ctx.buffers so _redecl_input_buffer can rebind the
    # source buffer in-place (mirrors how _materialize_func_args sets up
    # PrimFunc parameter slots).
    src_buf = tvm.tir.decl_buffer([64, 64], "float32", name="src")
    ctx.buffers["src_key"] = src_buf
    # The 2D offset tile that addptr produced (flat-address layout).
    offset_buf = tvm.tir.decl_buffer([64, 64], "int32", name="tile_offsets_2d")

    out_ssa = _ssa("tile_load", shape=[64, 64], dtype="float32")
    op = {
        "name": "tt.load",
        "operands": [_ssa("ptr", shape=[64, 64], dtype="float32")],
        "results": [out_ssa],
        "attrs": {},
    }

    # Must not raise the rank-mismatch InternalError.
    result = _emit_tile_load_from_input_buffer(
        op, ctx, src_buf, [offset_buf], (64, 64), "float32",
        mask_ssa=None, other_ssa=None,
    )
    assert isinstance(result, tvm.tir.Buffer), (
        f"expected rank-N tile Buffer; got {type(result).__name__}"
    )
    assert len(result.shape) == 2, (
        f"expected rank-2 result tile; got shape={list(result.shape)}"
    )

    text = _stringify(ctx.stmts)
    # Both loop variables must drive the offset buffer's index list (not
    # just a single ``[i0]`` against a 2D buffer).
    assert "i0" in text and "i1" in text, (
        "expected both loop variables (i0, i1) to index the rank-2 offset "
        f"tile; got:\n{text}"
    )
    # The original bug printed ``tile_offsets_2d[i0]`` (1-D index on a 2-D
    # buffer). With the fix the emitter indexes with the FULL loop nest.
    assert "tile_offsets_2d[i0]" not in text or "tile_offsets_2d[i0, i1]" in text, (
        "offset tile must be indexed with the full rank-N loop-var list, "
        f"not a single 1-D index; got:\n{text}"
    )


def test_splat_pointer_buffer_passthrough() -> None:
    """``tt.splat`` of a pointer-backed Buffer must propagate the buffer unchanged.

    The downstream ``tt.addptr`` needs to see the underlying base buffer so
    it can pair it with the per-lane offset tile. Wrapping the buffer in
    ``tir.Broadcast`` would defeat that and also crash (Broadcast rejects
    non-PrimExpr operands).
    """
    ctx = WalkerCtx()
    ptr_ssa = _ssa("ptr", shape=[], dtype="float32")
    out_ssa = _ssa("ptrs", shape=[256], dtype="float32")
    buf = tvm.tir.decl_buffer([1024], "float32", name="P")
    ctx.bind(ptr_ssa, buf)
    op = {
        "name": "tt.splat",
        "operands": [ptr_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    out = emit_tt_splat(op, ctx)
    assert out is buf, (
        "pointer-typed splat must passthrough the source buffer unchanged "
        f"(got {type(out).__name__})"
    )


# ---------------------------------------------------------------------------
# Regression: matmul LOWER_FAIL (Cannot store value with 4096, expected 1)
# ---------------------------------------------------------------------------


def test_addptr_compose_handles_buffer_plus_vector_primexpr_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tt.addptr`` inside ``scf.for`` with a Buffer prev-tile and a
    *vector* PrimExpr offset (e.g. ``Broadcast(scalar, prod(out_shape))``)
    must scalarise the vector per-lane via the flat lane index.

    Regression for: ``InternalError: Cannot store value with 4096, expected
    value with 1`` -- ``_compose_addptr_index._lane`` previously returned
    the full ``Broadcast(int32x4096)`` operand verbatim, so the
    ``tir.BufferStore(out_buf, _lane(prev) + _lane(off), [i0, i1])``
    inside the surrounding ``T.grid(64, 64)`` saw a 4096-lane RHS against
    a single-lane storage slot.
    """
    _force_no_shim(monkeypatch)
    ctx = WalkerCtx()

    ptr_ssa = _ssa("a_ptrs", shape=[64, 64], dtype="float32")
    off_ssa = _ssa("k_step", shape=[64, 64], dtype="int32")
    out_ssa = _ssa("a_ptrs_new", shape=[64, 64], dtype="float32")

    base_buf = tvm.tir.decl_buffer([1024 * 1024], "float32", name="A")
    # Previous iteration's trailing index entry is a 2-D offset Buffer
    # (matmul's a_ptrs in-loop accumulator).
    prev_off = tvm.tir.decl_buffer([64, 64], "int32", name="prev_off")
    # The new offset is a vector PrimExpr -- exactly what tt.splat +
    # arith.muli produces upstream of an in-loop tt.addptr (a single
    # ``BLOCK_K * stride_ak`` constant broadcast across all 4096 lanes).
    vec_off = tvm.tir.Broadcast(tvm.tir.const(7, "int32"), 64 * 64)

    ctx.bind(ptr_ssa, (base_buf, [prev_off]))
    ctx.bind(off_ssa, vec_off)

    op = {
        "name": "tt.addptr",
        "operands": [ptr_ssa, off_ssa],
        "results": [out_ssa],
        "attrs": {},
    }
    # MUST NOT raise.
    result = emit_tt_addptr(op, ctx)
    assert isinstance(result, tuple) and len(result) == 2
    out_buf, out_indices = result
    assert out_buf is base_buf
    assert isinstance(out_indices[0], tvm.tir.Buffer)
    # Resulting accumulator buffer must inherit the 2-D shape (one slot
    # per lane), not collapse to scalar or to the vector lane count.
    assert tuple(int(d) for d in out_indices[0].shape) == (64, 64)


def test_tt_load_scalarizes_buffer_typed_index_per_lane() -> None:
    """``emit_tt_load`` scalar path must scalarise a Buffer-typed index.

    Post Wave L2 the in-loop ``tt.addptr`` may leave the trailing index
    of its ``(buf, indices)`` descriptor as a ``tir.Buffer``. The scalar
    ``tt.load`` path has no enclosing per-lane ``tir.For`` to index that
    Buffer with, so it must conservatively read lane 0 instead of feeding
    the bare Buffer into ``tir.BufferLoad`` (which would expand to
    ``buf.shape[0]`` lanes against a single storage slot and trip the
    ``index_lanes * buffer_lanes == value_dtype_lanes`` check).
    """
    ctx = WalkerCtx()
    # Scalar result -- forces the scalar BufferLoad path.
    ptr_ssa = _ssa("ptr", shape=[], dtype="float32")
    result_ssa = _ssa("loaded", shape=[], dtype="float32")
    base_buf = tvm.tir.decl_buffer([1024], "float32", name="A")
    offset_tile = tvm.tir.decl_buffer([32], "int32", name="off_tile")
    ctx.bind(ptr_ssa, (base_buf, [offset_tile]))
    op = {
        "name": "tt.load",
        "operands": [ptr_ssa],
        "results": [result_ssa],
        "attrs": {},
    }
    # MUST NOT raise. The bare Buffer-typed index would otherwise trip
    # ``index_lanes * buffer_lanes == value_dtype_lanes``.
    out = emit_tt_load(op, ctx)
    text = _stringify(out)
    # The index must read off the offset buffer rather than embed it
    # raw -- look for a BufferLoad against the offset buffer's name.
    assert "off_tile" in text, (
        f"expected off_tile to appear via BufferLoad-as-index; got:\n{text}"
    )


def test_tt_store_scalarizes_buffer_typed_index_per_lane() -> None:
    """Symmetric to :func:`test_tt_load_scalarizes_buffer_typed_index_per_lane`."""
    ctx = WalkerCtx()
    ptr_ssa = _ssa("ptr", shape=[], dtype="float32")
    val_ssa = _ssa("v", shape=[], dtype="float32")
    base_buf = tvm.tir.decl_buffer([1024], "float32", name="C")
    offset_tile = tvm.tir.decl_buffer([32], "int32", name="off_tile_s")
    ctx.bind(ptr_ssa, (base_buf, [offset_tile]))
    ctx.bind(val_ssa, tvm.tir.const(1.0, "float32"))
    op = {
        "name": "tt.store",
        "operands": [ptr_ssa, val_ssa],
        "results": [],
        "attrs": {},
    }
    # MUST NOT raise.
    emit_tt_store(op, ctx)
    text = _stringify(ctx.stmts)
    assert "off_tile_s" in text, (
        f"expected off_tile_s BufferLoad inside the BufferStore index; got:\n{text}"
    )


# ---------------------------------------------------------------------------
# Regression: per-row store-ptr bound must thread the row-base offset.
#
# Bug: ``softmax`` / ``layer_norm`` emitted ``tl.store(out + pid*stride +
# col_offsets, ...)`` lowered to a BufferStore on a buffer whose declared
# shape was the **tile extent** (e.g. ``(128,)``). TileLang's
# ``LegalizeSafeMemoryAccess`` then synthesised a runtime guard
# ``if (pid * stride + i < 128)`` against the global store, which silently
# dropped every row beyond ``pid == 0`` -- rows 1..3 came out as zeros and
# the kernel reported ``NUMERIC_DIVERGE`` with row 0 correct.
#
# Fix: ``_redecl_input_buffer`` now consumes the offset_indices and, when
# the index references a program_id Var, derives a buffer extent that
# upper-bounds ``(gridDim - 1) * stride + tile_extent``. The per-row
# guard then becomes ``idx < (gridDim - 1) * stride + tile_extent``,
# which the analyzer can prove for the actual runtime values.
# ---------------------------------------------------------------------------


def test_tt_store_threads_row_base_into_flat_index() -> None:
    """``_redecl_input_buffer`` must derive a pid-aware flat extent.

    For a softmax-shaped pattern (``pid * stride + col_offsets``) the
    redecl'd buffer shape must NOT be the per-row tile extent (128) --
    it must symbolically encompass every reachable index across the
    full launch grid (4 rows -> at least 4 * 128 = 512 elements).
    """
    from poc.triton_frontend import (  # noqa: WPS433
        _flat_extent_for_indices,
        _redecl_input_buffer,
    )

    ctx = WalkerCtx()

    # Simulate the kernel-level state: input buffer ``arg1`` registered
    # in ctx.buffers and one program_id Var with ``gridDim_0`` extent.
    arg1 = tvm.tir.decl_buffer([1], "float32", name="arg1")
    ctx.buffers["arg1"] = arg1
    pid = tvm.tir.Var("pid0", "int32")
    grid_dim_0 = tvm.tir.Var("gridDim_0", "int32")
    ctx.program_id_vars.append((pid, 0, grid_dim_0))

    # Build the offset expression as the wrapper sees it. The trailing
    # index entry, after addptr fold, is ``Broadcast(pid * stride, 128)
    # + Ramp(0, 1, 128)``.
    stride = tvm.tir.Var("arg3", "int32")
    base = tvm.tir.Broadcast(pid * stride, 128)
    ramp = tvm.tir.Ramp(tvm.tir.const(0, "int32"), tvm.tir.const(1, "int32"), 128)
    offset_indices = [base + ramp]
    tile_shape = [128]

    # Direct probe of the helper -- the shape must reference both
    # ``gridDim_0`` and the stride Var.
    extent_shape = _flat_extent_for_indices(ctx, offset_indices, tile_shape)
    assert len(extent_shape) == 1
    extent_text = str(extent_shape[0])
    assert "gridDim_0" in extent_text, (
        f"derived extent must reference gridDim_0; got {extent_text!r}"
    )
    assert "arg3" in extent_text, (
        f"derived extent must reference the stride Var; got {extent_text!r}"
    )

    # End-to-end: redecl returns a Buffer whose shape is NOT the bare
    # tile extent. This is the load-bearing property -- the
    # downstream LegalizeSafeMemoryAccess upper-bound check uses
    # ``buffer->shape`` directly.
    new_buf = _redecl_input_buffer(
        ctx, arg1, tile_shape, "float32",
        offset_indices=offset_indices,
    )
    assert isinstance(new_buf, tvm.tir.Buffer)
    assert len(new_buf.shape) == 1
    new_shape_text = str(new_buf.shape[0])
    # The bug-state shape was a literal ``128``; the fix produces a
    # symbolic expression mentioning gridDim_0 and the stride Var.
    assert new_shape_text != "128", (
        "redecl returned the per-row tile extent; row-base was lost"
    )
    assert "gridDim_0" in new_shape_text and "arg3" in new_shape_text, (
        f"redecl shape must thread row base + stride; got {new_shape_text!r}"
    )


def test_tt_store_redecl_falls_back_to_tile_shape_without_pid() -> None:
    """Without ``program_id_vars``, redecl keeps the original tile shape.

    Vector_add-class kernels that do not reference a program_id (or a
    single-launch grid where the whole buffer fits in one tile) must
    not get the symbolic-extent rewrite -- otherwise the analyzer
    might generate weaker guards or break the "no-regression" property
    for the pre-fix passing kernel.
    """
    from poc.triton_frontend import (  # noqa: WPS433
        _flat_extent_for_indices,
    )

    ctx = WalkerCtx()
    # Empty program_id_vars -- mimics a pre-launch-context test fixture.
    assert ctx.program_id_vars == []

    # A pure ramp index, no pid involvement.
    ramp = tvm.tir.Ramp(tvm.tir.const(0, "int32"), tvm.tir.const(1, "int32"), 256)
    offset_indices = [ramp]
    extent_shape = _flat_extent_for_indices(ctx, offset_indices, [256])
    # Falls back to the original tile shape.
    assert extent_shape == [256], (
        f"expected fallback to tile shape [256]; got {extent_shape!r}"
    )
