"""Memory / shape op emitters for the Triton-IR -> TileLang TIR frontend.

This module is a *new* sibling of :mod:`poc.triton_frontend.op_mapping` that
parks the next batch of emitters in their own file so multiple agents can
work on op_mapping.py without merge-conflicting on shared lines (one of
the hard constraints called out by the maintainer of the dispatch table).

What lives here
---------------
``MEMORY_EMITTERS`` is a flat ``dict[str, Callable]`` keyed by canonical
TTIR op names. The walker (see :mod:`poc.triton_frontend.op_mapping`)
treats this dict as an additive overlay: any name present here takes
precedence over the legacy stub in op_mapping. Stubs in op_mapping are
**not** deleted -- per the ``feedback_no_silent_delete`` rule we leave
the existing scaffolding in place with a TODO so a reviewer can compare.

Op coverage (this file)
-----------------------
* ``tt.load``            -- BufferLoad with optional ``mask``/``other``;
                            multi-element tile path defers to ``tts.load``
                            when PtrAnalysis is available, otherwise emits
                            a per-element ``tir.For`` with a ``# DEGRADED:``
                            annotation that survives PrimFunc pretty-print.
* ``tt.store``           -- symmetric BufferStore with optional ``mask``;
                            same degraded path as load.
* ``tt.make_range``      -- ``tir.Ramp(start, 1, lanes)``; spill to a
                            ``tir.For`` over a small buffer when ``lanes``
                            exceeds the target vector width.
* ``tt.expand_dims``     -- shape rebind; for vector inputs we wrap in
                            ``tir.Broadcast``; for buffer views we emit a
                            buffer alias.
* ``tt.broadcast``       -- ``tir.Broadcast`` for scalar->vector; for
                            shape-broadcast (vector->tile) we emit a
                            ``tir.For`` rebuild.
* ``tt.splat``           -- ``tir.Broadcast(scalar, lanes)``.
* ``tt.view`` / ``tt.reshape`` -- buffer alias with new shape (no data
                                  movement; a TIR-level rebind).
* ``tt.addptr``          -- pointer arithmetic; calls into PtrAnalysis when
                            the C++ shim is built, else emits a scalar
                            offset add on the underlying var.
* ``tts.make_tptr``      -- TritonStructured opaque structured-pointer
                            constructor; lowers to a TileLang
                            fragment-buffer view.

Hard constraints (from the maintainer)
--------------------------------------
* **Never silent-fallback.** When PtrAnalysis is unavailable for a
  multi-element tile load/store, we emit a visible ``# DEGRADED:`` comment
  in the printed PrimFunc text via ``tir.AttrStmt(..., "pragma_comment", ...)``.
* **Stay inside ``poc/triton_frontend/``.** No edits to TileLang core or
  the vendored TVM tree.
* **Don't delete stubs.** The legacy stubs in op_mapping are flagged with
  TODOs but stay live so callers that depend on the old call shape keep
  working until the walker is rewired through ``MEMORY_EMITTERS``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Reuse the public surface of op_mapping (WalkerCtx, helpers) so this file
# stays a thin overlay rather than a parallel implementation.
from ..op_mapping import (
    WalkerCtx,
    _alloc_tile_buffer,
    _attrs,
    _attrs_with_properties_shared,
    _dtype_of,
    _operands,
    _parse_generic_properties_shared,
    _ptrstate_buffer,
    _ptrstate_is_tile,
    _ptrstate_offsets_or_zero,
    _ptrstate_sizes_int,
    _results,
    _shape_of,
)


# ---------------------------------------------------------------------------
# Generic-form properties parser
# ---------------------------------------------------------------------------
#
# The real implementation now lives in :mod:`poc.triton_frontend.op_mapping`
# as ``_parse_generic_properties_shared`` / ``_attrs_with_properties_shared``
# so other emitter modules (``op_emitters/arith.py`` for cmpi/cmpf) can
# reuse the same Triton 3.6 ``<{...}>`` parser without duplicating the
# regex. We keep these module-private aliases for backwards compatibility
# with code that imports the unprefixed names directly.

_parse_generic_properties = _parse_generic_properties_shared
_attrs_with_properties = _attrs_with_properties_shared

__all__ = ["MEMORY_EMITTERS", "has_cxx_shim"]


# Target vector width above which ``tt.make_range`` spills to a ``tir.For``
# over a small buffer instead of materialising a single Ramp lane. This
# matches the conservative SM_70 / RDNA / Apple7 shared cap; LayoutInference
# can re-vectorize when the actual target supports more.
_DEFAULT_VECTOR_WIDTH = 128


# ---------------------------------------------------------------------------
# Shim probe -- the maintainer asked for a `has_cxx_shim` re-export under
# this name. Delegates to the existing :func:`shim_available` helper.
# ---------------------------------------------------------------------------


def has_cxx_shim() -> bool:
    """Return True iff the PtrAnalysis C++ shim is importable.

    The shim is built out of ``poc/triton_frontend/_cxx`` and exposes
    ``mlir::tts::PtrAnalysis::rewriteOp`` so we can resolve multi-element
    pointer arithmetic into a ``tts.load``/``tts.store`` pair. When the
    shim is missing we fall back to per-element scalar BufferLoad and
    annotate the emitted IR with ``# DEGRADED:`` (see
    :func:`_emit_degraded_tile_load`).
    """
    # Import lazily so that test-time monkeypatching of ``shim_available``
    # also captures callers of ``has_cxx_shim`` without an extra hook.
    from ..ptr_analysis import shim_available

    return bool(shim_available())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolved_or_none(ctx: WalkerCtx, ssa_value: Any) -> Any:
    """Return the value bound to ``ssa_value`` if any, else ``None``.

    ``WalkerCtx.get`` raises on missing keys; the memory emitters need a
    "look but don't break" probe for the optional ``mask``/``other`` and
    the speculative PtrState inspection on ``ptr``.
    """
    if ssa_value is None:
        return None
    try:
        return ctx.value_map.get(ssa_value)
    except TypeError:
        # ssa_value is unhashable (e.g. a dict-shaped fake op from tests, or
        # a raw operand record from the regex walker). value_map keys are
        # always hashable SSA names; an unhashable key is by definition
        # not in the map, so return None.
        return None


def _resolve_lane_operand(
    ctx: WalkerCtx,
    value: Any,
    loop_vars: Sequence[Any],
    *,
    role: str,
) -> Any:
    """Coerce an ``IfThenElse`` mask/other operand to a scalar PrimExpr lane.

    After Wave E2 lowered ``arith.constant dense<0.0>`` (and the broadcasted
    boolean mask comparator) into a ``tir.decl_buffer`` with a per-lane store
    nest, both the ``mask`` and the ``other`` operands of ``tt.load`` /
    ``tt.store`` may resolve to a ``tir.Buffer`` rather than a scalar
    PrimExpr. ``tir.IfThenElse``/``tir.if_then_else`` reject Buffer args
    outright (``Mismatched type on argument #N: Expected ir.PrimExpr but got
    tirx.Buffer``), so we materialise a per-lane ``BufferLoad`` indexed by
    the surrounding loop vars instead.

    Parameters
    ----------
    ctx
        Walker context (provides ``tir`` / ``tvm`` modules).
    value
        Resolved operand: scalar ``PrimExpr`` (passes through), ``tir.Buffer``
        (rewritten to ``BufferLoad(buf, loop_vars)``), or ``None`` (returned
        unchanged for the no-operand sentinel).
    loop_vars
        The enclosing per-lane ``tir.Var`` indices. Must have length matching
        the buffer rank when ``value`` is a Buffer; we trim/pad with the
        trailing axes to match the buffer's rank, mirroring the broadcast
        convention used by ``_emit_tile_binop``.
    role
        ``"mask"`` or ``"other"`` -- only used to give a precise EmitError
        message when the operand has an unexpected type.

    Raises
    ------
    EmitError
        If ``value`` is neither a Buffer nor a PrimExpr (and not None).
    """
    if value is None:
        return None
    tvm_mod = ctx.tvm()
    tir = ctx.tir()
    if isinstance(value, tvm_mod.tir.Buffer):
        rank = len(value.shape)
        if rank == 0:
            return tir.BufferLoad(value, [tir.const(0, "int32")])
        # Match the trailing ``rank`` axes of ``loop_vars`` so a rank-1
        # mask/other indexed by the innermost lane works as expected; if
        # the caller has fewer loop vars than the buffer rank we zero-pad
        # the leading axes (the conservative scalar-broadcast read).
        lv = list(loop_vars)
        if len(lv) >= rank:
            indices = lv[-rank:]
        else:
            indices = [tir.const(0, "int32")] * (rank - len(lv)) + lv
        return tir.BufferLoad(value, indices)
    # Scalar PrimExpr passes through. We accept anything that quacks like a
    # PrimExpr (has a ``dtype`` attr) -- the actual type-check happens inside
    # ``tir.IfThenElse`` / ``tir.if_then_else`` for free.
    if hasattr(value, "dtype") or isinstance(value, (int, float, bool)):
        return value
    from ..op_mapping import EmitError  # local import: avoid circular at import time
    raise EmitError(
        f"unsupported tt.load/tt.store {role!r} operand type: "
        f"{type(value).__name__}; expected tir.PrimExpr or tir.Buffer"
    )


def _ssa_name(value: Any) -> Optional[str]:
    """Best-effort printed SSA name for a TTIR value (e.g. ``"%2"``).

    Mirrors the lookup pattern used by ``mlir_walker._block_arg_name``: try
    the standard ``get_name``/``name`` accessors, then fall back to ``str()``
    and take the first whitespace-delimited token (which is the SSA name in
    the printed form ``"%2 = tt.load ..."``).
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("name") or value.get("ssa") or None
    for attr in ("get_name", "name"):
        getter = getattr(value, attr, None)
        if callable(getter):
            try:
                out = str(getter())
                if out:
                    return out
            except Exception:
                pass
        elif isinstance(getter, str) and getter:
            return getter
    try:
        s = str(value).strip()
        if s:
            head = s.split()[0]
            return head if head.startswith("%") else None
    except Exception:
        pass
    return None


def _lookup_ptr_state(ctx: WalkerCtx, op: Any, ptr_ssa: Any) -> Any:
    """Find a :class:`PtrState` for the load/store at ``op``.

    Search order:

    1. ``ctx.ptr_states[result_ssa_name]`` -- the most precise match
       (``tts.load``/``tts.store`` ops post-rewrite carry the result name
       that PtrAnalysis emitted).
    2. ``ctx.ptr_states[ptr_ssa_name]`` -- pre-rewrite case where the load
       still references the original pointer.
    3. ``None``.
    """
    states_map = getattr(ctx, "ptr_states", None) or {}
    if not states_map:
        return None
    results = _results(op)
    if results:
        rname = _ssa_name(results[0])
        if rname and rname in states_map:
            return states_map[rname]
    pname = _ssa_name(ptr_ssa)
    if pname and pname in states_map:
        return states_map[pname]
    return None


def _wrap_pragma_comment(ctx: WalkerCtx, body: Any, comment: str) -> Any:
    """Wrap ``body`` in an ``AttrStmt`` whose pretty-print includes ``comment``.

    TVM's TIR printer renders ``AttrStmt(node, "pragma_comment", value)``
    as ``attr [node] "pragma_comment" = "<value>"`` -- visible in the
    PrimFunc text *and* in the lowered C/CUDA output (the codegen lifts
    pragma_comment annotations into ``// <value>`` comments). That is
    exactly the visibility the constraint asks for: a ``# DEGRADED:``
    marker that survives all the way to the printed kernel.
    """
    tir = ctx.tir()
    tvm_mod = ctx.tvm()
    stmt = body
    if not hasattr(stmt, "checked_type") and not isinstance(stmt, tvm_mod.tir.Stmt):
        # Wrap PrimExprs in Evaluate so AttrStmt has a Stmt body.
        stmt = tir.Evaluate(stmt) if hasattr(tir, "Evaluate") else stmt
    # The first arg of AttrStmt is the "node" the pragma is attached to;
    # 0 is a sentinel TVM accepts when the pragma is body-scoped.
    attr_value = tir.StringImm(f"# DEGRADED: {comment}") if hasattr(tir, "StringImm") else f"# DEGRADED: {comment}"
    return tir.AttrStmt(tvm_mod.tir.IntImm("int32", 0), "pragma_comment", attr_value, stmt)


def _emit_degraded_tile_load(
    op: Any,
    ctx: WalkerCtx,
    buf: Any,
    base_indices: Sequence[Any],
    out_shape: Sequence[int],
    out_dtype: str,
    mask_ssa: Any,
    other_ssa: Any,
) -> Any:
    """Per-element ``tir.For`` fallback when no PtrAnalysis shim is present.

    Emits a loop nest that issues one ``BufferLoad`` per lane into a fresh
    buffer-backed tile. The whole nest is wrapped in an ``AttrStmt`` with a
    ``# DEGRADED:`` ``pragma_comment`` so the regression -- per-element scalar
    loads when we *should* be emitting a tile copy -- shows up in the PrimFunc
    pretty-print and downstream codegen. Without this annotation the
    silent-fallback failure mode is what bites users in production.
    """
    tir = ctx.tir()
    result_value = _results(op)[0] if _results(op) else None
    out_buf_name = ctx.fresh("tile_load")
    # Tile-scoped allocation -- see ``_alloc_tile_buffer`` for why this
    # bypasses ``ctx.buffers`` (otherwise ``VerifyMemory`` flags the
    # per-lane BufferStore as host-memory access).
    tile_buf = _alloc_tile_buffer(ctx, list(out_shape) or [1], out_dtype, out_buf_name)

    # Build a nested ``For`` over the tile shape. We collapse to a single
    # 1-D loop when the tile is rank-1 to keep the output legible; higher
    # ranks materialise as nested ``For`` whose innermost body is one
    # BufferLoad guarded by ``if_then_else(mask, ..., other)``.
    loop_vars: List[Any] = []
    body_indices: List[Any] = list(base_indices) if base_indices else []
    for axis, extent in enumerate(out_shape or [1]):
        loop_vars.append(tir.Var(ctx.fresh(f"i{axis}"), "int32"))

    # Build the source index by adding loop vars to the base offsets,
    # extending base_indices if it's shorter than the tile rank.
    src_indices: List[Any] = []
    for axis, lv in enumerate(loop_vars):
        base = body_indices[axis] if axis < len(body_indices) else tir.const(0, "int32")
        src_indices.append(base + lv)

    load_expr: Any = tir.BufferLoad(buf, src_indices)
    if mask_ssa is not None:
        try:
            mask_expr = ctx.get(mask_ssa)
        except KeyError:
            mask_expr = None
        if mask_expr is not None:
            if other_ssa is not None:
                try:
                    other_expr = ctx.get(other_ssa)
                except KeyError:
                    other_expr = tir.const(0, out_dtype)
            else:
                other_expr = tir.const(0, out_dtype)
            # Per-lane materialisation: when the mask/other resolve to a
            # ``tir.Buffer`` (post Wave E2 broadcast lowering) we must pull
            # one scalar lane out via ``BufferLoad(..., loop_vars)`` because
            # ``tir.if_then_else`` only accepts scalar PrimExpr operands.
            mask_lane = _resolve_lane_operand(ctx, mask_expr, loop_vars, role="mask")
            other_lane = _resolve_lane_operand(ctx, other_expr, loop_vars, role="other")
            load_expr = tir.if_then_else(mask_lane, load_expr, other_lane)

    body = tir.BufferStore(tile_buf, load_expr, list(loop_vars) or [tir.const(0, "int32")])
    # Wrap loops innermost-out.
    for axis in range(len(loop_vars) - 1, -1, -1):
        extent = out_shape[axis] if axis < len(out_shape) else 1
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body,
        )

    annotated = _wrap_pragma_comment(
        ctx,
        body,
        "tt.load multi-element tile, PtrAnalysis shim unavailable -> "
        "per-element BufferLoad. Build poc/triton_frontend/_cxx to recover.",
    )
    ctx.emit(annotated)
    if result_value is not None:
        ctx.bind(result_value, tile_buf)
    return tile_buf


def _emit_degraded_tile_store(
    op: Any,
    ctx: WalkerCtx,
    buf: Any,
    base_indices: Sequence[Any],
    val_expr: Any,
    val_shape: Sequence[int],
    mask_ssa: Any,
) -> Any:
    """Per-element fallback for ``tt.store`` when no shim is present."""
    tir = ctx.tir()
    if not val_shape:
        # Scalar store -- fall back to the simple BufferStore path.
        idx = list(base_indices) or [tir.const(0, "int32")]
        store = tir.BufferStore(buf, val_expr, idx)
        if mask_ssa is not None:
            try:
                mask_expr = ctx.get(mask_ssa)
                # Scalar store with no enclosing loop -> pass an empty
                # loop-var list; ``_resolve_lane_operand`` will index a
                # rank-0 mask buffer as ``BufferLoad(buf, [0])``.
                mask_lane = _resolve_lane_operand(ctx, mask_expr, [], role="mask")
                store = tir.IfThenElse(mask_lane, store, None)
            except KeyError:
                pass
        ctx.emit(store)
        return store

    loop_vars = [tir.Var(ctx.fresh(f"i{axis}"), "int32") for axis in range(len(val_shape))]

    dst_indices: List[Any] = []
    for axis, lv in enumerate(loop_vars):
        base = (
            base_indices[axis]
            if axis < len(base_indices)
            else tir.const(0, "int32")
        )
        dst_indices.append(base + lv)

    # The value being stored is a buffer; index it with the loop vars to
    # get a per-lane scalar.
    if hasattr(val_expr, "shape"):
        rhs = tir.BufferLoad(val_expr, list(loop_vars))
    else:
        rhs = val_expr  # scalar PrimExpr broadcast

    store: Any = tir.BufferStore(buf, rhs, dst_indices)
    if mask_ssa is not None:
        try:
            mask_expr = ctx.get(mask_ssa)
            # Per-lane: if the mask is a Buffer, ``BufferLoad`` it at the
            # current loop-var index instead of feeding the bare Buffer
            # into IfThenElse arg #0 (which TVM rejects).
            mask_lane = _resolve_lane_operand(ctx, mask_expr, loop_vars, role="mask")
            store = tir.IfThenElse(mask_lane, store, None)
        except KeyError:
            pass
    body: Any = store
    for axis in range(len(loop_vars) - 1, -1, -1):
        extent = val_shape[axis]
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body,
        )

    annotated = _wrap_pragma_comment(
        ctx,
        body,
        "tt.store multi-element tile, PtrAnalysis shim unavailable -> "
        "per-element BufferStore. Build poc/triton_frontend/_cxx to recover.",
    )
    ctx.emit(annotated)
    return annotated


def _emit_tile_copy_tir(
    op: Any,
    ctx: WalkerCtx,
    src_buf: Any,
    base_indices: Sequence[Any],
    out_shape: Sequence[int],
    out_dtype: str,
    mask_ssa: Any,
    other_ssa: Any,
) -> Any:
    """Non-DEGRADED tile copy fallback: rolled ``tir.For`` over ``BufferLoad``.

    Used when PtrAnalysis has surfaced a tile descriptor (so we know the load
    is a real tile, not a silent fallback) but the TileLang ``T`` builder
    scope is not active (e.g., unit-test environment that calls the emitter
    directly without ``T.prim_func``). The emitted IR has the same shape as
    ``_emit_load_copy``'s ImportError fallback -- a per-lane BufferLoad/Store
    nest -- but **without** the ``# DEGRADED:`` AttrStmt, because we DO have
    PtrState; it is the *builder scope* that's absent, not the analysis.

    This is the contractually-correct artifact for the
    ``test_tile_load_with_seeded_state_skips_degraded_marker`` invariant
    documented in ``tests/test_pipeline_with_ptr_analysis.py``.
    """
    tir = ctx.tir()
    result_value = _results(op)[0] if _results(op) else None
    out_buf_name = ctx.fresh("tile_load")
    # Tile-scoped allocation; see ``_alloc_tile_buffer`` docstring.
    tile_buf = _alloc_tile_buffer(ctx, list(out_shape) or [1], out_dtype, out_buf_name)

    loop_vars: List[Any] = []
    body_indices: List[Any] = list(base_indices) if base_indices else []
    for axis, _extent in enumerate(out_shape or [1]):
        loop_vars.append(tir.Var(ctx.fresh(f"i{axis}"), "int32"))

    src_indices: List[Any] = []
    for axis, lv in enumerate(loop_vars):
        base = body_indices[axis] if axis < len(body_indices) else tir.const(0, "int32")
        src_indices.append(base + lv)

    load_expr: Any = tir.BufferLoad(src_buf, src_indices)
    if mask_ssa is not None:
        try:
            mask_expr = ctx.get(mask_ssa)
        except KeyError:
            mask_expr = None
        if mask_expr is not None:
            if other_ssa is not None:
                try:
                    other_expr = ctx.get(other_ssa)
                except KeyError:
                    other_expr = tir.const(0, out_dtype)
            else:
                other_expr = tir.const(0, out_dtype)
            mask_lane = _resolve_lane_operand(ctx, mask_expr, loop_vars, role="mask")
            other_lane = _resolve_lane_operand(ctx, other_expr, loop_vars, role="other")
            load_expr = tir.if_then_else(mask_lane, load_expr, other_lane)

    body = tir.BufferStore(tile_buf, load_expr, list(loop_vars) or [tir.const(0, "int32")])
    for axis in range(len(loop_vars) - 1, -1, -1):
        extent = out_shape[axis] if axis < len(out_shape) else 1
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body,
        )

    # Emit the loop nest WITHOUT the ``# DEGRADED:`` AttrStmt -- the
    # PtrAnalysis pre-pass succeeded, so the breadcrumb does not apply.
    ctx.emit(body)
    if result_value is not None:
        ctx.bind(result_value, tile_buf)
    return tile_buf


def _emit_tile_store_tir(
    op: Any,
    ctx: WalkerCtx,
    dst_buf: Any,
    base_indices: Sequence[Any],
    val_expr: Any,
    val_shape: Sequence[int],
    mask_ssa: Any,
) -> Any:
    """Non-DEGRADED tile store fallback: rolled ``tir.For`` over ``BufferStore``.

    Symmetric to :func:`_emit_tile_copy_tir`. Used when the destination is a
    multi-rank tile buffer but no PtrAnalysis tile descriptor or
    ``T``-builder scope is available, so we must emit a rank-N ``tir.For``
    nest by hand. Without this helper, an emitter that synthesises a
    rank-N ``decl_buffer`` based on the result/value shape would index it
    with a single ``[0]`` index and trip the TVM ``Buffer is N-dimensional,
    cannot be indexed with the 1-dimensional indices`` check.
    """
    tir = ctx.tir()
    # Handle the scalar / empty-shape case the same way ``BufferStore`` does
    # for rank-1 (or rank-0) buffers: the caller is responsible for passing
    # a sensible scalar value.
    if not val_shape:
        idx = list(base_indices) or [tir.const(0, "int32")]
        store: Any = tir.BufferStore(dst_buf, val_expr, idx)
        if mask_ssa is not None:
            try:
                mask_expr = ctx.get(mask_ssa)
                mask_lane = _resolve_lane_operand(ctx, mask_expr, [], role="mask")
                store = tir.IfThenElse(mask_lane, store, None)
            except KeyError:
                pass
        ctx.emit(store)
        return store

    loop_vars = [tir.Var(ctx.fresh(f"i{axis}"), "int32") for axis in range(len(val_shape))]

    dst_indices: List[Any] = []
    for axis, lv in enumerate(loop_vars):
        base = (
            base_indices[axis]
            if axis < len(base_indices)
            else tir.const(0, "int32")
        )
        dst_indices.append(base + lv)

    # The value being stored may be a buffer (per-lane BufferLoad) or a
    # PrimExpr that we broadcast to every lane.
    if hasattr(val_expr, "shape"):
        rhs = tir.BufferLoad(val_expr, list(loop_vars))
    else:
        rhs = val_expr  # scalar PrimExpr broadcast

    store = tir.BufferStore(dst_buf, rhs, dst_indices)
    if mask_ssa is not None:
        try:
            mask_expr = ctx.get(mask_ssa)
            mask_lane = _resolve_lane_operand(ctx, mask_expr, loop_vars, role="mask")
            store = tir.IfThenElse(mask_lane, store, None)
        except KeyError:
            pass

    body: Any = store
    for axis in range(len(loop_vars) - 1, -1, -1):
        extent = val_shape[axis]
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body,
        )

    # No ``# DEGRADED:`` AttrStmt: the buffer rank is honoured rank-N, this
    # is just a non-T.copy fallback (e.g., shim present but no PtrState).
    ctx.emit(body)
    return body


def _is_tile_shape(shape: Sequence[int]) -> bool:
    """Return True iff ``shape`` describes more than one element."""
    if not shape:
        return False
    total = 1
    for s in shape:
        try:
            total *= int(s)
        except (TypeError, ValueError):
            return True  # symbolic dims are tile-sized by definition
    return total > 1


# ---------------------------------------------------------------------------
# tt.load / tt.store
# ---------------------------------------------------------------------------


def emit_tt_load(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.load(ptr, mask, other)`` with the new memory-emitter rules.

    Behaviour matrix
    ----------------
    +------------------------------+-----------------------------------+
    | PtrState describes a tile?   | What we emit                      |
    +==============================+===================================+
    | yes, shim available          | ``tts.load``-style ``T.copy``    |
    | yes, shim missing            | per-element ``tir.For`` +         |
    |                              | ``# DEGRADED:`` AttrStmt          |
    | no (scalar)                  | ``tir.BufferLoad`` + optional     |
    |                              | ``tir.if_then_else`` mask guard   |
    +------------------------------+-----------------------------------+
    """
    tir = ctx.tir()
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.load: missing pointer operand")
    ptr_ssa = operands[0]
    mask_ssa = operands[1] if len(operands) >= 2 else None
    other_ssa = operands[2] if len(operands) >= 3 else None

    resolved = _resolved_or_none(ctx, ptr_ssa)
    result_value = _results(op)[0] if _results(op) else None
    out_shape = list(_shape_of(result_value)) if result_value is not None else []
    out_dtype = _dtype_of(result_value) if result_value is not None else "float32"

    # Fallback: ``_shape_of`` only inspects MLIR ``RankedTensorType`` and
    # the dict-shaped fake; bare Python objects that expose ``.shape`` (e.g.
    # the ``_FakeValue`` test fixture) slip through and present as scalar.
    # That silent rank-0 path is exactly what masked the no-shim ``# DEGRADED:``
    # invariant: a 32-element tile fake was being routed into the scalar
    # BufferLoad branch with no breadcrumb. Mirror the dict probe on
    # attribute access -- and on the ptr operand as a secondary source --
    # so any value that *says* it's a tile is treated as one here.
    if not out_shape:
        for candidate in (result_value, ptr_ssa):
            cand_shape = getattr(candidate, "shape", None)
            if cand_shape:
                out_shape = [int(s) if not isinstance(s, str) else s for s in cand_shape]
                break
    if out_dtype in {"", "handle", "float32"} and result_value is not None:
        # Honour an explicit ``.dtype`` when ``_dtype_of`` defaulted us back
        # to float32 with no MLIR type info. (Cheap; doesn't affect the dict
        # path because dicts are caught by ``_dtype_of`` directly.) Pipe
        # through the alias map so a raw MLIR ``f32`` doesn't reach
        # ``tir.decl_buffer`` with the short spelling -- TVM rejects it
        # outright with ``ValueError: unknown dtype 'f32'``.
        attr_dtype = getattr(result_value, "dtype", None)
        if isinstance(attr_dtype, str) and attr_dtype:
            from ..op_mapping import _normalize_mlir_dtype  # local import
            out_dtype = _normalize_mlir_dtype(attr_dtype)

    # Pre-pass-seeded PtrState lookup (run_ptr_analysis_pre_pass populated
    # ``ctx.ptr_states`` keyed by SSA name). When found, we synthesize the
    # legacy tagged-dict shape that ``_emit_load_copy`` already understands
    # so the existing T.copy code path runs and we DON'T emit ``# DEGRADED:``.
    if (resolved is None or not (isinstance(resolved, dict) and "_ptrstate" in resolved)):
        state = _lookup_ptr_state(ctx, op, ptr_ssa)
        if state is not None:
            resolved = {
                "_ptrstate": state,
                "source": state.source,
                "offsets": list(state.offsets),
                "sizes": list(state.sizes),
                "strides": list(state.strides),
            }

    # Tile path (PtrState has explicit sizes > 1).
    if isinstance(resolved, dict) and "_ptrstate" in resolved and _ptrstate_is_tile(resolved):
        tile_shape = _ptrstate_sizes_int(resolved) or list(out_shape) or [1]
        tile_offsets = _ptrstate_offsets_or_zero(resolved)
        tile_buf = _ptrstate_buffer(ctx, resolved, out_dtype)
        if has_cxx_shim():
            # Defer to the legacy ``_emit_load_copy`` path which already
            # knows how to talk to ``T.copy`` once PtrAnalysis is in.
            from ..op_mapping import _emit_load_copy

            try:
                return _emit_load_copy(op, ctx, resolved, mask_ssa, other_ssa)
            except ValueError as exc:
                # ``_emit_load_copy`` calls into ``tilelang.language``
                # builders (``T.alloc_fragment`` / ``T.copy``) which require
                # an active ``T.prim_func`` builder scope. When the emitter
                # is invoked outside that scope (unit tests, dict-walker
                # plumbing) TVM raises ``ValueError: No builder in current
                # scope``. PtrAnalysis still gave us a real tile descriptor,
                # so we fall back to a rolled ``tir.For`` over BufferLoad
                # *without* the ``# DEGRADED:`` AttrStmt -- the breadcrumb
                # is reserved for the no-PtrState path. Re-raise anything
                # that is NOT the builder-scope error so genuine emitter
                # bugs still surface.
                if "No builder in current scope" not in str(exc):
                    raise
                return _emit_tile_copy_tir(
                    op, ctx, tile_buf, tile_offsets, tile_shape, out_dtype,
                    mask_ssa, other_ssa,
                )
        # No shim: degrade visibly.
        return _emit_degraded_tile_load(
            op, ctx, tile_buf, tile_offsets, tile_shape, out_dtype,
            mask_ssa, other_ssa,
        )

    # Tile path inferred from the *result* type even without PtrState. This
    # matters when the dict-shaped fakes don't carry _ptrstate but the user
    # has annotated the result as a multi-element tile.
    if _is_tile_shape(out_shape):
        # No PtrState resolution -- synthesise a placeholder buffer for the
        # underlying global memory. We seed it the same way ``map_tt_load``
        # does in op_mapping.py so existing tests keep passing.
        buf_name = (
            getattr(ptr_ssa, "name", None)
            or (ptr_ssa.get("name") if isinstance(ptr_ssa, dict) else None)
            or ctx.fresh("buf")
        )
        # Tile-scoped placeholder for the no-PtrState path; see
        # ``op_mapping._alloc_tile_buffer`` for why this bypasses
        # ``ctx.buffers`` (Memory verification would otherwise flag
        # every per-lane BufferLoad below).
        if buf_name not in ctx.buffers:
            src_buf = _alloc_tile_buffer(ctx, list(out_shape), out_dtype, buf_name)
        else:
            src_buf = ctx.buffers[buf_name]
        # Rank-N safety: ``src_buf`` was declared with the result tile
        # shape, so indexing it with a single ``[0]`` would trip TVM's
        # "Buffer is N-dimensional, cannot be indexed with the
        # 1-dimensional indices" assertion (matmul's 2D pointer-arithmetic
        # tiles hit this). Always emit a rank-N ``tir.For`` nest -- even
        # with the shim available -- so the BufferLoad index count
        # matches ``src_buf.shape``. PtrAnalysis-driven shim acceleration
        # only fires earlier in this function (the ``_ptrstate_is_tile``
        # branch); here we have no PtrState, so the rolled-loop path is
        # the contractually-correct lowering.
        if has_cxx_shim():
            base_indices = [tir.const(0, "int32") for _ in out_shape]
            return _emit_tile_copy_tir(
                op, ctx, src_buf, base_indices, out_shape, out_dtype,
                mask_ssa, other_ssa,
            )
        return _emit_degraded_tile_load(
            op, ctx, src_buf, [0], out_shape, out_dtype, mask_ssa, other_ssa,
        )

    # Scalar path -- a plain BufferLoad with optional mask/other guard.
    if isinstance(resolved, tuple) and len(resolved) == 2:
        buf, indices = resolved
    elif resolved is not None and not isinstance(resolved, dict):
        buf, indices = resolved, [0]
    else:
        # Seed a placeholder buffer to match the legacy fallback shape.
        # Tile-scoped (see ``_alloc_tile_buffer``) so VerifyMemory skips it.
        buf_name = (
            getattr(ptr_ssa, "name", None)
            or (ptr_ssa.get("name") if isinstance(ptr_ssa, dict) else None)
            or ctx.fresh("buf")
        )
        if buf_name not in ctx.buffers:
            buf = _alloc_tile_buffer(
                ctx, out_shape or [1024], out_dtype, buf_name
            )
        else:
            buf = ctx.buffers[buf_name]
        indices = [0]

    load_expr: Any = tir.BufferLoad(buf, list(indices))
    if mask_ssa is not None:
        mask_expr = ctx.get(mask_ssa)
        if other_ssa is not None:
            other_expr = ctx.get(other_ssa)
        else:
            other_expr = tir.const(0, out_dtype)
        # Scalar load path -- if the mask/other are Buffers (post Wave E2)
        # we have no enclosing loop var, so ``_resolve_lane_operand`` falls
        # back to a rank-0 BufferLoad at index 0.
        mask_lane = _resolve_lane_operand(ctx, mask_expr, [], role="mask")
        other_lane = _resolve_lane_operand(ctx, other_expr, [], role="other")
        load_expr = tir.if_then_else(mask_lane, load_expr, other_lane)

    if result_value is not None:
        ctx.bind(result_value, load_expr)
    return load_expr


def emit_tt_store(op: Any, ctx: WalkerCtx) -> Any:
    """Symmetric to :func:`emit_tt_load`. Honors mask via ``IfThenElse``."""
    tir = ctx.tir()
    operands = _operands(op)
    if len(operands) < 2:
        raise ValueError("tt.store: missing pointer or value operand")
    ptr_ssa, val_ssa = operands[0], operands[1]
    mask_ssa = operands[2] if len(operands) >= 3 else None

    resolved = _resolved_or_none(ctx, ptr_ssa)
    val_expr = ctx.get(val_ssa)
    val_shape = list(_shape_of(val_ssa))

    # Pre-pass-seeded PtrState lookup (see emit_tt_load). Promotes the no-shim
    # ``# DEGRADED:`` path to the real T.copy when run_ptr_analysis_pre_pass
    # has surfaced a tile descriptor for the destination pointer.
    if (resolved is None or not (isinstance(resolved, dict) and "_ptrstate" in resolved)):
        state = _lookup_ptr_state(ctx, op, ptr_ssa)
        if state is not None:
            resolved = {
                "_ptrstate": state,
                "source": state.source,
                "offsets": list(state.offsets),
                "sizes": list(state.sizes),
                "strides": list(state.strides),
            }

    # Tile path via PtrState.
    if isinstance(resolved, dict) and "_ptrstate" in resolved and _ptrstate_is_tile(resolved):
        tile_shape = _ptrstate_sizes_int(resolved) or list(val_shape) or [1]
        tile_offsets = _ptrstate_offsets_or_zero(resolved)
        dtype = _dtype_of(val_ssa) or "float32"
        tile_buf = _ptrstate_buffer(ctx, resolved, dtype)
        if has_cxx_shim():
            from ..op_mapping import _emit_store_copy
            return _emit_store_copy(op, ctx, resolved, val_expr, mask_ssa)
        return _emit_degraded_tile_store(
            op, ctx, tile_buf, tile_offsets, val_expr, tile_shape, mask_ssa,
        )

    # Tile path inferred from the value's shape.
    if _is_tile_shape(val_shape):
        dtype = _dtype_of(val_ssa) or "float32"
        buf_name = (
            getattr(ptr_ssa, "name", None)
            or (ptr_ssa.get("name") if isinstance(ptr_ssa, dict) else None)
            or ctx.fresh("buf")
        )
        # Tile-scoped placeholder; see ``op_mapping._alloc_tile_buffer``.
        # Production with a shim hits ``_emit_store_copy`` above; this is
        # the no-shim / unit-test fallback.
        if buf_name not in ctx.buffers:
            dst_buf = _alloc_tile_buffer(ctx, list(val_shape), dtype, buf_name)
        else:
            dst_buf = ctx.buffers[buf_name]
        if has_cxx_shim():
            # Detect Buffer-shaped mask early: the single-statement path
            # below has no enclosing loop to index a per-lane mask, so we
            # must fall through to the per-lane ``_emit_degraded_tile_store``
            # which DOES emit the surrounding tir.For nest. (E2 lowers the
            # bool tile mask to a tir.decl_buffer, so this is the
            # vector_add hot path.) The store is still a real tile-shaped
            # write -- we just lose the T.copy fast path until the shim
            # learns to consume Buffer masks directly.
            mask_is_buffer = False
            if mask_ssa is not None:
                try:
                    mask_probe = ctx.get(mask_ssa)
                    mask_is_buffer = isinstance(
                        mask_probe, ctx.tvm().tir.Buffer
                    )
                except KeyError:
                    pass
            if mask_is_buffer:
                return _emit_degraded_tile_store(
                    op, ctx, dst_buf, [0], val_expr, val_shape, mask_ssa,
                )
            # Rank-N safety: ``dst_buf`` was declared with the value's tile
            # shape (matmul writes a 2D output tile), so a single ``[0]``
            # index against a 2D buffer would trip TVM's
            # ``buffer->shape.size() == indices.size()`` check. Always
            # emit a rank-N ``tir.For`` nest. Without PtrState the shim
            # cannot fold this into ``T.copy`` anyway, so the rolled
            # loop is the contractually-correct lowering.
            base_indices = [tir.const(0, "int32") for _ in val_shape]
            return _emit_tile_store_tir(
                op, ctx, dst_buf, base_indices, val_expr, val_shape, mask_ssa,
            )
        return _emit_degraded_tile_store(
            op, ctx, dst_buf, [0], val_expr, val_shape, mask_ssa,
        )

    # Scalar path.
    if isinstance(resolved, tuple) and len(resolved) == 2:
        buf, indices = resolved
    elif resolved is not None and not isinstance(resolved, dict):
        buf, indices = resolved, [0]
    else:
        dtype = _dtype_of(val_ssa) or "float32"
        buf_name = (
            getattr(ptr_ssa, "name", None)
            or (ptr_ssa.get("name") if isinstance(ptr_ssa, dict) else None)
            or ctx.fresh("buf")
        )
        # Tile-scoped placeholder; see ``op_mapping._alloc_tile_buffer``.
        if buf_name not in ctx.buffers:
            buf = _alloc_tile_buffer(ctx, val_shape or [1024], dtype, buf_name)
        else:
            buf = ctx.buffers[buf_name]
        indices = [0]

    store_stmt: Any = tir.BufferStore(buf, val_expr, list(indices))
    if mask_ssa is not None:
        mask_expr = ctx.get(mask_ssa)
        # Scalar store path: lane-resolve so a Buffer mask doesn't crash
        # IfThenElse arg #0.
        mask_lane = _resolve_lane_operand(ctx, mask_expr, [], role="mask")
        store_stmt = tir.IfThenElse(mask_lane, store_stmt, None)
    ctx.emit(store_stmt)
    return store_stmt


# ---------------------------------------------------------------------------
# tt.make_range
# ---------------------------------------------------------------------------


def emit_tt_make_range(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.make_range(start, end)`` to ``tir.Ramp`` (or a For spill).

    Spill threshold: when ``end - start > _DEFAULT_VECTOR_WIDTH`` we
    materialise the range into a small buffer with a serial ``tir.For`` so
    the downstream Codegen doesn't fight a too-wide vector type. The
    default is 128, the conservative cap that fits SM_70 / RDNA / Apple7.
    LayoutInference can re-vectorise once the actual target reports more.
    """
    tir = ctx.tir()
    # Use the property-aware helper so generic-form Triton 3.6 ops
    # (``<{end = 256 : i32, start = 0 : i32}>``) parse correctly. The
    # legacy ``_attrs`` returns ``{}`` for those because jaxlib's
    # ``op.attributes`` does NOT surface inherent attributes stored in
    # Properties when the dialect is unregistered.
    attrs = _attrs_with_properties(op)
    start = int(attrs.get("start", 0))
    end = int(attrs.get("end", 0))
    lanes = end - start
    if lanes <= 0:
        raise ValueError(
            f"tt.make_range: invalid range [{start}, {end}); end must be > start"
        )

    if lanes <= _DEFAULT_VECTOR_WIDTH:
        ramp = tir.Ramp(tir.const(start, "int32"), tir.const(1, "int32"), lanes)
        if _results(op):
            ctx.bind(_results(op)[0], ramp)
        return ramp

    # Wide range -- spill to a tile-scoped buffer with a serial init loop.
    buf_name = ctx.fresh("range")
    buf = _alloc_tile_buffer(ctx, [lanes], "int32", buf_name)
    iv = tir.Var(ctx.fresh("ri"), "int32")
    body = tir.BufferStore(buf, tir.const(start, "int32") + iv, [iv])
    loop = tir.For(
        iv,
        tir.const(0, "int32"),
        tir.const(int(lanes), "int32"),
        tir.ForKind.SERIAL,
        body,
    )
    ctx.emit(loop)
    if _results(op):
        ctx.bind(_results(op)[0], buf)
    return buf


# ---------------------------------------------------------------------------
# tt.expand_dims / tt.broadcast / tt.splat
# ---------------------------------------------------------------------------


def _is_buffer(ctx: WalkerCtx, val: Any) -> bool:
    """Type-test: is ``val`` a TVM tir.Buffer?"""
    try:
        return isinstance(val, ctx.tvm().tir.Buffer)
    except Exception:
        return False


def _vector_lanes(value: Any) -> int:
    """Return per-lane count for a vector PrimExpr (dtype ``Tx<N>``), else 1.

    A vector PrimExpr is what ``tir.Broadcast(scalar, N)``,
    ``tir.Ramp(start, 1, N)``, and the result of a vector-vector binop print
    as ``int32xN`` / ``float32xN`` etc. TVM's ``tir.Broadcast`` REQUIRES a
    SCALAR ``value`` operand and rejects vector ones with
    ``Check failed: (value.dtype().is_scalar()) is false``.

    We detect vector-ness by parsing the ``"x"`` suffix of ``value.dtype``.
    Non-PrimExpr inputs (Buffer, dict, None) get ``1`` so callers don't
    need to special-case them.
    """
    dt = getattr(value, "dtype", None)
    if dt is None:
        return 1
    s = str(dt)
    if "x" not in s:
        return 1
    try:
        return int(s.rsplit("x", 1)[-1])
    except ValueError:
        return 1


def _vector_scalar_dtype(value: Any) -> str:
    """Return the per-lane scalar dtype of a vector PrimExpr (``int32xN`` -> ``int32``)."""
    dt = getattr(value, "dtype", None)
    s = str(dt) if dt is not None else "float32"
    if "x" in s:
        s = s.rsplit("x", 1)[0]
    return s


def _read_vector_lane(ctx: WalkerCtx, value: Any, lane_idx: Any) -> Any:
    """Per-lane read from a vector PrimExpr or Buffer at ``lane_idx``.

    Mirrors ``op_emitters.arith._read_lane`` but is local so this module
    has no cross-emitter import dependency. Recognises:

    * ``tir.Buffer`` -> ``BufferLoad(buf, [lane_idx])`` (rank-1 view).
    * ``tir.Broadcast`` -> the splat scalar (constant per lane).
    * ``tir.Ramp`` -> ``base + stride * lane_idx``.
    * Vector elementwise binops (``Add``/``Sub``/``Mul``/``Div``/``Mod``/
      ``Min``/``Max``/``FloorDiv``/``FloorMod``) -> recurse on ``a`` and
      ``b`` and re-apply the op per-lane. This covers the softmax /
      vector_add case where ``tt.make_range`` + ``tt.splat`` produces
      ``Broadcast(pid*N, N) + Ramp(0,1,N)`` -- a generic ``Add`` PrimExpr
      with vector dtype.
    * Vector unary nodes (``Cast``) -> recurse on ``value.value``.
    * Anything else -> raise ``EmitError`` so a missing branch surfaces
      loudly instead of silently degrading.
    """
    from ..op_mapping import EmitError  # local import to avoid circulars at module import

    tir = ctx.tir()
    tvm_mod = ctx.tvm()
    if isinstance(value, tvm_mod.tir.Buffer):
        rank = len(value.shape)
        if rank == 0:
            return tir.BufferLoad(value, [tir.const(0, "int32")])
        return tir.BufferLoad(value, [lane_idx])
    bcast_cls = getattr(tir, "Broadcast", None)
    if bcast_cls is not None and isinstance(value, bcast_cls):
        return value.value
    ramp_cls = getattr(tir, "Ramp", None)
    if ramp_cls is not None and isinstance(value, ramp_cls):
        return value.base + value.stride * lane_idx

    # Scalar PrimExpr: dtype has no ``"x"`` lane suffix; broadcast.
    dt = getattr(value, "dtype", None)
    if dt is not None and "x" not in str(dt):
        return value

    # Vector elementwise binop / unary: recurse on subterms. We use Python
    # operators so the result re-typechecks under TVM's PrimExpr rules.
    binop_pyops = {
        "Add": lambda a, b: a + b,
        "Sub": lambda a, b: a - b,
        "Mul": lambda a, b: a * b,
        "Div": lambda a, b: a / b,
        "Mod": lambda a, b: a % b,
        "FloorDiv": lambda a, b: a // b,
        "FloorMod": lambda a, b: a % b,
    }
    for cls_name, pyop in binop_pyops.items():
        cls = getattr(tir, cls_name, None)
        if cls is not None and isinstance(value, cls):
            la = _read_vector_lane(ctx, value.a, lane_idx)
            lb = _read_vector_lane(ctx, value.b, lane_idx)
            return pyop(la, lb)
    # Min/Max via tir constructors.
    for cls_name, fn_name in (("Min", "min"), ("Max", "max")):
        cls = getattr(tir, cls_name, None)
        if cls is not None and isinstance(value, cls):
            la = _read_vector_lane(ctx, value.a, lane_idx)
            lb = _read_vector_lane(ctx, value.b, lane_idx)
            fn = getattr(tir, fn_name, None)
            if fn is not None:
                return fn(la, lb)
            # Fallback: select.
            return tir.Select(la < lb, la, lb) if fn_name == "min" else tir.Select(la > lb, la, lb)
    cast_cls = getattr(tir, "Cast", None)
    if cast_cls is not None and isinstance(value, cast_cls):
        scalar_dt = _vector_scalar_dtype(value)
        return tir.Cast(scalar_dt, _read_vector_lane(ctx, value.value, lane_idx))

    raise EmitError(
        f"vector PrimExpr of type {type(value).__name__} (dtype "
        f"{getattr(value, 'dtype', '?')!s}) is not lane-indexable; "
        "extend _read_vector_lane to cover this node"
    )


def _coerce_lanes_to_int(lanes: Any) -> int:
    """Unwrap ``lanes`` to a Python int.

    Tile-shape values reach the emitters as MLIR ``ffi.Array`` shape tuples
    (when ``out_shape`` is forwarded directly instead of ``int(out_shape[-1])``).
    TVM's ``tir.Broadcast(value, lanes)`` rejects ``ffi.Array`` with
    ``Expected ir.PrimExpr but got ffi.Array``. Coerce to int -- raise if
    we cannot, so we never silently pass an opaque type through.
    """
    if isinstance(lanes, int):
        return lanes
    if isinstance(lanes, (tuple, list)):
        # Tile shape was passed in by mistake; take the trailing dim.
        if not lanes:
            return 1
        return int(lanes[-1])
    # ffi.Array / other shape-like objects.
    try:
        return int(lanes)
    except (TypeError, ValueError):
        # Best-effort: iterate, take the last entry.
        try:
            seq = list(lanes)
            if seq:
                return int(seq[-1])
        except TypeError:
            pass
    raise ValueError(f"cannot coerce lanes={lanes!r} (type {type(lanes).__name__}) to int")


def _materialise_vector_into_buffer(
    ctx: WalkerCtx, src: Any, out_shape: Sequence[int], dtype: str
) -> Any:
    """Lower a vector PrimExpr / Buffer source into a fresh tile Buffer.

    Emits a serial ``tir.For`` nest over ``out_shape`` whose innermost body
    is ``BufferStore(dst, _read_vector_lane(src, lane_idx), [outer..., lane_idx])``.
    Returns the freshly allocated ``tir.Buffer``. The store nest is appended
    to ``ctx.stmts`` via ``ctx.emit``.

    The convention matches the broadcast semantics used by
    ``_emit_tile_binop`` in ``op_emitters/arith.py``: outer (rank-promoted)
    axes drive the splat, the innermost axis indexes the source vector
    lane-for-lane.
    """
    tir = ctx.tir()
    out_shape_list = [int(s) for s in out_shape]
    if not out_shape_list:
        raise ValueError("_materialise_vector_into_buffer: out_shape is empty")
    dst = _alloc_tile_buffer(ctx, out_shape_list, dtype, ctx.fresh("vec"))

    loop_vars = [tir.Var(ctx.fresh(f"v{axis}"), "int32") for axis in range(len(out_shape_list))]
    lane_idx = loop_vars[-1]
    # Per-lane scalar read from the vector source.
    rhs = _read_vector_lane(ctx, src, lane_idx)
    body: Any = tir.BufferStore(dst, rhs, list(loop_vars))
    for axis in range(len(loop_vars) - 1, -1, -1):
        extent = out_shape_list[axis]
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body,
        )
    ctx.emit(body)
    return dst


def emit_tt_expand_dims(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.expand_dims`` to a Broadcast / buffer alias / For-nest.

    Behaviour matrix
    ----------------
    * Buffer input            -> ``tir.decl_buffer`` alias with the new
                                  shape (no data movement; rebind only).
    * Vector PrimExpr input   -> ``tir.For`` nest over the result shape,
                                  reading per-lane from the source vector
                                  (Broadcast/Ramp/Buffer). This is the
                                  only correct lowering -- ``tir.Broadcast``
                                  REQUIRES a scalar ``value`` operand and
                                  blows up with ``Check failed:
                                  (value.dtype().is_scalar()) is false``
                                  on a vector source.
    * Scalar PrimExpr input   -> ``tir.Broadcast(scalar, lanes)``.
    """
    tir = ctx.tir()
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.expand_dims: missing source operand")
    src = ctx.get(operands[0])
    result_value = _results(op)[0] if _results(op) else None
    out_shape = list(_shape_of(result_value)) if result_value is not None else []

    if _is_buffer(ctx, src):
        # Buffer alias with the new shape.
        new_buf = tir.decl_buffer(
            out_shape or list(getattr(src, "shape", [1])),
            getattr(src, "dtype", "float32"),
            name=ctx.fresh("expanded"),
            data=getattr(src, "data", None),
        )
        if result_value is not None:
            ctx.bind(result_value, new_buf)
        return new_buf

    # Pointer-descriptor passthrough. ``tt.expand_dims`` on a pointer tile
    # is a logical-only rebind; the downstream ``tt.addptr`` / ``tt.load``
    # consume the descriptor.
    if isinstance(src, tuple) and len(src) == 2:
        if result_value is not None:
            ctx.bind(result_value, src)
        return src
    if isinstance(src, dict) and "_ptrstate" in src:
        if result_value is not None:
            ctx.bind(result_value, src)
        return src

    src_vec_lanes = _vector_lanes(src)

    # Vector PrimExpr path: emit a For nest over out_shape, NOT tir.Broadcast
    # (which only accepts scalar ``value``). The innermost lane axis pulls
    # per-lane scalars out of the source vector via ``_read_vector_lane``.
    if src_vec_lanes > 1 and out_shape:
        dtype = _dtype_of(result_value) if result_value is not None else _vector_scalar_dtype(src)
        dst = _materialise_vector_into_buffer(ctx, src, out_shape, dtype)
        if result_value is not None:
            ctx.bind(result_value, dst)
        return dst

    # Scalar PrimExpr path: tir.Broadcast(scalar, lanes) is correct.
    lanes = 1
    for s in out_shape:
        try:
            lanes *= int(s)
        except (TypeError, ValueError):
            lanes = 0
            break
    if lanes > 1 and hasattr(tir, "Broadcast"):
        # Coerce in case lanes ended up as an ffi.Array shape (defensive).
        out = tir.Broadcast(src, _coerce_lanes_to_int(lanes))
    else:
        out = src
    if result_value is not None:
        ctx.bind(result_value, out)
    return out


def emit_tt_broadcast(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.broadcast`` to ``tir.Broadcast`` or a ``tir.For`` rebuild.

    Behaviour matrix
    ----------------
    * Scalar PrimExpr -> tile : ``tir.Broadcast(value, lanes)``.
    * Vector PrimExpr / Buffer -> tile (rank promotion) : emit a serial
      ``tir.For`` nest into a fresh buffer. The innermost loop indexes the
      source per-lane (Broadcast.value, Ramp.base+stride*idx, or
      BufferLoad). NEVER ``tir.Broadcast(vector_src, ...)`` -- TVM's
      Broadcast node rejects vector ``value`` outright.
    """
    tir = ctx.tir()
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.broadcast: missing source operand")
    src_ssa = operands[0]
    src = ctx.get(src_ssa)
    src_shape = list(_shape_of(src_ssa))
    result_value = _results(op)[0] if _results(op) else None
    out_shape = list(_shape_of(result_value)) if result_value is not None else []

    # Pointer-descriptor passthrough. ``tt.broadcast`` on a pointer tile
    # is logical-only; downstream load/store consumes the descriptor.
    if isinstance(src, tuple) and len(src) == 2:
        if result_value is not None:
            ctx.bind(result_value, src)
        return src
    if isinstance(src, dict) and "_ptrstate" in src:
        if result_value is not None:
            ctx.bind(result_value, src)
        return src

    out_lanes = 1
    for s in out_shape:
        try:
            out_lanes *= int(s)
        except (TypeError, ValueError):
            out_lanes = 0
            break

    src_lanes = 1
    for s in src_shape:
        try:
            src_lanes *= int(s)
        except (TypeError, ValueError):
            src_lanes = 0
            break

    # Refine src_lanes: a vector PrimExpr (e.g. ``int32x64``) carries lane
    # info in its dtype suffix even when ``_shape_of`` returned ``()``
    # because the SSA value had no MLIR type info attached.
    src_vec_lanes = _vector_lanes(src)
    if src_vec_lanes > 1 and src_lanes <= 1:
        src_lanes = src_vec_lanes

    # Scalar -> tile : Broadcast. Only emit ``tir.Broadcast`` when the
    # source is a *scalar* PrimExpr; vectors must take the For-nest path.
    src_is_buffer = _is_buffer(ctx, src)
    if (
        src_lanes == 1
        and not src_is_buffer
        and src_vec_lanes == 1
        and out_lanes > 1
        and hasattr(tir, "Broadcast")
    ):
        out = tir.Broadcast(src, _coerce_lanes_to_int(out_lanes))
        if result_value is not None:
            ctx.bind(result_value, out)
        return out

    # Vector / Buffer -> tile : emit a tir.For that materialises the
    # broadcast into a fresh buffer.
    dtype = _dtype_of(result_value) if result_value is not None else _dtype_of(src_ssa)
    # Normalise a possibly-short MLIR dtype (e.g. ``f32``) so decl_buffer
    # accepts it.
    if dtype in ("", "handle"):
        dtype = _vector_scalar_dtype(src)
    if out_shape and (out_lanes > src_lanes or src_is_buffer or src_vec_lanes > 1):
        out_buf = _alloc_tile_buffer(ctx, out_shape, dtype, ctx.fresh("bcast"))
        loop_vars = [tir.Var(ctx.fresh(f"b{i}"), "int32") for i in range(len(out_shape))]
        # Index the source with the trailing dims that map into the source
        # shape (the broadcast convention is to match trailing dims).
        if src_is_buffer:
            tail = loop_vars[-len(src_shape):] if src_shape else []
            rhs = tir.BufferLoad(src, tail or [tir.const(0, "int32")])
        elif src_vec_lanes > 1:
            # Vector PrimExpr: index the lane via the innermost loop var.
            rhs = _read_vector_lane(ctx, src, loop_vars[-1])
        else:
            # Scalar PrimExpr broadcast across all lanes.
            rhs = src
        body: Any = tir.BufferStore(out_buf, rhs, list(loop_vars))
        for axis in range(len(loop_vars) - 1, -1, -1):
            extent = out_shape[axis]
            body = tir.For(
                loop_vars[axis],
                tir.const(0, "int32"),
                tir.const(int(extent), "int32"),
                tir.ForKind.SERIAL,
                body,
            )
        ctx.emit(body)
        if result_value is not None:
            ctx.bind(result_value, out_buf)
        return out_buf

    # No-op: same shape, just rebind.
    if result_value is not None:
        ctx.bind(result_value, src)
    return src


def emit_tt_splat(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.splat`` (scalar -> tile) to ``tir.Broadcast(scalar, lanes)``.

    Pointer-typed splats (``tt.splat %ptr : !tt.ptr<f32> -> tensor<Nx!tt.ptr<f32>>``)
    cannot go through ``tir.Broadcast`` -- TVM's Broadcast node rejects
    non-PrimExpr operands and pointer-tile expansion is handled lazily by
    the downstream ``tt.addptr`` / ``tt.load`` pair (which materialises the
    per-lane address from the base buffer + the per-lane offset). For those
    we propagate the source buffer through unchanged so ``tt.addptr`` sees
    the underlying base and can wire it into a ``(buffer, indices)`` tuple.
    """
    tir = ctx.tir()
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.splat: missing source operand")
    src = ctx.get(operands[0])
    result_value = _results(op)[0] if _results(op) else None
    out_shape = list(_shape_of(result_value)) if result_value is not None else []
    lanes = 1
    for s in out_shape:
        try:
            lanes *= int(s)
        except (TypeError, ValueError):
            lanes = 0
            break
    # Pointer-typed splat: pass the underlying buffer through. ``tt.addptr``
    # downstream picks up the buffer via ``_resolved_or_none`` and pairs it
    # with the per-lane offset tile.
    tvm_mod = ctx.tvm()
    if isinstance(src, tvm_mod.tir.Buffer):
        out = src
    elif isinstance(src, tuple) and len(src) == 2:
        # ``(buffer, indices)`` descriptor from a prior ``tt.addptr`` --
        # passthrough so the downstream load/store consumes the tile-shaped
        # pointer descriptor unchanged. Wrapping in ``tir.Broadcast`` would
        # crash (tuple is not a PrimExpr) and is semantically wrong: the
        # splat is widening a scalar pointer to a per-lane pointer tile;
        # the underlying address arithmetic is already encoded in the
        # tuple's ``indices`` list.
        out = src
    elif isinstance(src, dict) and "_ptrstate" in src:
        # PtrState descriptor -- same passthrough rationale as the tuple.
        out = src
    else:
        src_vec_lanes = _vector_lanes(src)
        if src_vec_lanes > 1 and lanes > 1 and out_shape:
            # Defensive: ``tt.splat`` is contractually scalar->tile, but if
            # the producer accidentally bound a vector PrimExpr to the
            # source SSA we cannot pass it to ``tir.Broadcast`` (which
            # rejects vector ``value``). Lower via the same For-nest path
            # used by ``emit_tt_expand_dims`` / ``emit_tt_broadcast``.
            dtype = _dtype_of(result_value) if result_value is not None else _vector_scalar_dtype(src)
            out = _materialise_vector_into_buffer(ctx, src, out_shape, dtype)
        elif lanes > 1 and hasattr(tir, "Broadcast"):
            out = tir.Broadcast(src, _coerce_lanes_to_int(lanes))
        else:
            out = src
    if result_value is not None:
        ctx.bind(result_value, out)
    return out


# ---------------------------------------------------------------------------
# tt.view / tt.reshape -- buffer alias rebind
# ---------------------------------------------------------------------------


def emit_tt_reshape(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.reshape`` / ``tt.view`` to a buffer alias with new shape.

    No data movement: we either rebind the SSA to a new ``tir.decl_buffer``
    that shares the source's underlying ``data`` handle (when the source
    is a Buffer), or simply rebind the PrimExpr (when the source is a
    PrimExpr -- TIR's lazy indexing carries shape implicitly).
    """
    tir = ctx.tir()
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.reshape: missing source operand")
    src = ctx.get(operands[0])
    result_value = _results(op)[0] if _results(op) else None
    out_shape = list(_shape_of(result_value)) if result_value is not None else []

    if _is_buffer(ctx, src) and out_shape:
        new_buf = tir.decl_buffer(
            out_shape,
            getattr(src, "dtype", "float32"),
            name=ctx.fresh("view"),
            data=getattr(src, "data", None),
        )
        if result_value is not None:
            ctx.bind(result_value, new_buf)
        return new_buf

    if result_value is not None:
        ctx.bind(result_value, src)
    return src


# ---------------------------------------------------------------------------
# tt.addptr -- pointer arithmetic
# ---------------------------------------------------------------------------


def _compose_addptr_index(ctx: WalkerCtx, prev: Any, off: Any) -> Any:
    """Return a per-lane sum ``prev + off`` for an in-loop ``tt.addptr``.

    ``prev`` is the trailing entry of the previous-iteration index list and
    may be a scalar ``tir.PrimExpr`` or a ``tir.Buffer`` (a tile of
    per-lane offsets). ``off`` is the new addptr offset operand and may
    likewise be either a PrimExpr or a Buffer.

    * **PrimExpr + PrimExpr** -- delegate to TIR's scalar add (the existing
      fast path).
    * **Buffer + PrimExpr** (or symmetric) -- broadcast the scalar across
      every lane of the buffer; allocate a fresh tile buffer and emit a
      per-lane ``tir.For`` nest that stores ``buf[i,...] + scalar``.
    * **Buffer + Buffer** -- per-lane elementwise add into a fresh tile
      buffer of the broadcast shape (rank picked from the larger-rank
      operand, padded with leading 1s for the smaller).

    Always returns either a ``tir.PrimExpr`` (scalar fast path) or a fresh
    ``tir.Buffer`` allocated via ``_alloc_tile_buffer``. The caller stores
    this as the trailing entry of ``new_indices`` so downstream
    ``tt.load`` / ``tt.store`` consumers see the same shape contract they
    already handled.
    """
    tvm_mod = ctx.tvm()
    tir = ctx.tir()
    Buffer = tvm_mod.tir.Buffer

    prev_is_buf = isinstance(prev, Buffer)
    off_is_buf = isinstance(off, Buffer)

    # Scalar fast path: the existing TIR ``+`` operator handles this.
    if not prev_is_buf and not off_is_buf:
        return prev + off

    # Pick the result tile shape: whichever operand is a Buffer wins; if
    # both, pick the higher-rank shape (matmul's case where the prev
    # offsets are themselves a 2-D tile).
    if prev_is_buf and off_is_buf:
        if len(prev.shape) >= len(off.shape):
            out_shape = list(prev.shape)
        else:
            out_shape = list(off.shape)
        out_dtype = str(prev.dtype)
    elif prev_is_buf:
        out_shape = list(prev.shape)
        out_dtype = str(prev.dtype)
    else:
        out_shape = list(off.shape)
        out_dtype = str(off.dtype)

    out_buf = _alloc_tile_buffer(ctx, out_shape, out_dtype, ctx.fresh("addptr_acc"))
    loop_vars = [tir.Var(ctx.fresh(f"a{axis}"), "int32") for axis in range(len(out_shape))]

    def _lane(operand: Any) -> Any:
        if isinstance(operand, Buffer):
            rank = len(operand.shape)
            if rank == 0:
                return tir.BufferLoad(operand, [tir.const(0, "int32")])
            # Broadcast: align the trailing ``rank`` axes; pad the leading
            # axes with zero so a rank-1 tile composed with a rank-2 tile
            # behaves the same way ``_emit_tile_binop`` does.
            if len(loop_vars) >= rank:
                idx = list(loop_vars[-rank:])
            else:
                idx = [tir.const(0, "int32")] * (rank - len(loop_vars)) + list(loop_vars)
            return tir.BufferLoad(operand, idx)
        # Scalar PrimExpr broadcasts across every lane.
        return operand

    body: Any = tir.BufferStore(out_buf, _lane(prev) + _lane(off), list(loop_vars))
    for axis in range(len(loop_vars) - 1, -1, -1):
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(out_shape[axis]), "int32"),
            tir.ForKind.SERIAL,
            body,
        )
    ctx.emit(body)
    return out_buf


def emit_tt_addptr(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.addptr(ptr, offset)`` to a pointer descriptor.

    Return-type contract
    --------------------
    The result is **never** a bare ``tir.PrimExpr`` -- pointer arithmetic
    is a memory-side concept. Possible result shapes:

    * Tagged ``dict`` ``{"_ptrstate": PtrState, "offsets", "sizes",
      "strides", "source"}`` when PtrAnalysis (the C++ shim) folded the
      op into a structured pointer. Consumed by ``tt.load`` / ``tt.store``.
    * Plain ``(buffer_or_None, indices_list)`` 2-tuple in the degraded
      path. ``buffer`` is a ``tir.Buffer`` (or ``None`` when the base
      pointer wasn't bound yet) and ``indices_list`` is a list of
      ``tir.PrimExpr`` **or** ``tir.Buffer`` index entries (the latter
      when the offset is a per-lane tile, e.g. matmul's
      ``a_ptrs += BLOCK_K * stride_ak`` inside ``scf.for``).

    In-loop tile increments
    -----------------------
    For an scf.for loop body containing ``%a_ptrs_new = tt.addptr %a_ptrs,
    %k_step``, the ``%k_step`` operand resolves to a ``tir.Buffer`` (the
    per-lane offset tile), not a scalar PrimExpr. The previous iteration's
    indices may also already carry a Buffer in the trailing entry. We
    delegate the per-lane add to :func:`_compose_addptr_index`, which
    allocates a fresh tile buffer and emits a serial ``tir.For`` nest --
    the same shape contract that ``_emit_tile_binop`` uses in
    ``op_emitters/arith.py``.

    Downstream consumers must extract the buffer + index pair (``tt.load`` /
    ``tt.store`` already do this). Pure ``arith.*`` ops MUST NOT consume an
    addptr result -- if they do, ``_emit_tile_binop`` in
    ``op_emitters/arith.py`` raises an ``EmitError`` with the offending op.

    With the C++ shim available we delegate to PtrAnalysis (it does the
    full strided-layout fold). Without the shim we degrade to a scalar
    offset add on the underlying var/value.
    """
    tvm_mod = ctx.tvm()
    tir = ctx.tir()
    operands = _operands(op)
    if len(operands) < 2:
        raise ValueError("tt.addptr: expected (ptr, offset) operands")
    ptr_ssa, off_ssa = operands[0], operands[1]
    base = _resolved_or_none(ctx, ptr_ssa)
    off = ctx.get(off_ssa)

    result_value = _results(op)[0] if _results(op) else None

    if has_cxx_shim():
        # Shim is built -- the PtrAnalysis pass should already have folded
        # this op away in the rewritten module. If we still see it here we
        # respect whatever PtrState the walker has stashed and just
        # propagate it forward; concretely that means re-binding the ptr
        # SSA's resolved descriptor to the result SSA.
        if isinstance(base, dict) and "_ptrstate" in base:
            new_offsets = list(base.get("offsets") or [])
            if new_offsets:
                # The trailing offset slot may be a scalar PrimExpr, an
                # int (untouched here), or a Buffer when an earlier
                # iteration already produced a tile. Compose-or-pass.
                if isinstance(new_offsets[-1], int):
                    new_offsets[-1] = new_offsets[-1] + 0
                else:
                    new_offsets[-1] = _compose_addptr_index(ctx, new_offsets[-1], off)
            else:
                new_offsets = [off]
            new_state = dict(base)
            new_state["offsets"] = new_offsets
            if result_value is not None:
                ctx.bind(result_value, new_state)
            return new_state
        if isinstance(base, tuple) and len(base) == 2:
            buf, indices = base
            new_indices = list(indices) or [tir.const(0, "int32")]
            new_indices[-1] = _compose_addptr_index(ctx, new_indices[-1], off)
            value = (buf, new_indices)
            if result_value is not None:
                ctx.bind(result_value, value)
            return value

    # Degraded path: scalar offset add on the underlying var/value.
    # Wrap the result in a pragma_comment AttrStmt so reviewers can see
    # the missing PtrAnalysis fold the same way as for tt.load.
    if isinstance(base, tuple) and len(base) == 2:
        buf, indices = base
        new_indices = list(indices) or [tir.const(0, "int32")]
        new_indices[-1] = _compose_addptr_index(ctx, new_indices[-1], off)
        value = (buf, new_indices)
    elif base is None:
        # No prior binding: synthesise a flat scalar with the offset.
        value = (None, [off])
    else:
        # Treat the prior value as a buffer/var; emit a fresh tuple.
        value = (base, [off])

    # Note via Evaluate-wrapped AttrStmt so the printed PrimFunc shows the
    # degradation. We don't have a usable Stmt to attach to (the addptr is
    # an expression-level op), so emit a side-effect-free AttrStmt that
    # carries the comment as a hoisted breadcrumb.
    if hasattr(tir, "Evaluate") and hasattr(tir, "AttrStmt"):
        breadcrumb = _wrap_pragma_comment(
            ctx,
            tir.Evaluate(tir.const(0, "int32")),
            "tt.addptr without PtrAnalysis shim -> scalar offset add only",
        )
        ctx.emit(breadcrumb)

    if result_value is not None:
        ctx.bind(result_value, value)
    return value


# ---------------------------------------------------------------------------
# tts.make_tptr -- TritonStructured opaque tensor-pointer
# ---------------------------------------------------------------------------


def emit_tts_make_tptr(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tts.make_tptr`` to a TileLang fragment-buffer view.

    The TritonStructured dialect's ``make_tptr`` packages a base pointer
    plus shape/strides/offsets into an opaque structured pointer. At the
    TIR level we materialise it as a fresh fragment buffer alias whose
    shape/dtype come from the op's result type. ptr_analysis (when the
    shim is present) consumes this binding directly via PtrState; without
    the shim we still produce a buffer so downstream emitters see a value.
    """
    tir = ctx.tir()
    result_value = _results(op)[0] if _results(op) else None
    out_shape = list(_shape_of(result_value)) if result_value is not None else [1]
    out_dtype = _dtype_of(result_value) if result_value is not None else "float32"

    try:
        import tilelang.language as T  # type: ignore
        frag = T.alloc_fragment(out_shape, out_dtype)
    except ImportError:  # pragma: no cover -- TileLang absent
        # Fragment fallback: tile-scoped buffer (see ``_alloc_tile_buffer``)
        # so VerifyMemory's host-memory check skips it.
        frag = _alloc_tile_buffer(ctx, out_shape, out_dtype, ctx.fresh("tptr"))

    if result_value is not None:
        ctx.bind(result_value, frag)
    return frag


# ---------------------------------------------------------------------------
# Dispatch table -- imported by the walker; overlays op_mapping.OP_TABLE.
# ---------------------------------------------------------------------------

MEMORY_EMITTERS: Dict[str, Callable[..., Any]] = {
    "tt.load": emit_tt_load,
    "tt.store": emit_tt_store,
    "tt.make_range": emit_tt_make_range,
    "tt.expand_dims": emit_tt_expand_dims,
    "tt.broadcast": emit_tt_broadcast,
    "tt.splat": emit_tt_splat,
    "tt.view": emit_tt_reshape,
    "tt.reshape": emit_tt_reshape,
    "tt.addptr": emit_tt_addptr,
    "tts.make_tptr": emit_tts_make_tptr,
}
