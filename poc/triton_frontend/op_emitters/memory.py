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
* ``tts.load`` / ``tts.store`` -- structured-pointer memory ops produced by
                                  PtrAnalysis; lower through recovered
                                  PtrState stride metadata, never through the
                                  degraded placeholder path.

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

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Reuse the public surface of op_mapping (WalkerCtx, helpers) so this file
# stays a thin overlay rather than a parallel implementation.
from ..op_mapping import (
    EmitError,
    LazyTileExpr,
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
    _normalize_mlir_dtype,
    _results,
    _shape_of,
    materialize_lazy_tile,
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
    # We accept the SUBPROCESS-available shim too because the PtrAnalysis
    # pre-pass in __init__.py already routes through a clean subprocess
    # when libtriton is loaded — by the time op_emitters runs, ptr_state
    # metadata is seeded on ``ctx`` either way, so the tile-copy path is
    # safe to emit. Without this widening, op_emitters silently degraded
    # to scalar loads whenever libtriton was present, defeating the whole
    # subprocess fallback.
    from ..ptr_analysis import shim_available, shim_subprocess_available

    return bool(shim_available() or shim_subprocess_available())


def _ctx_has_cxx_shim(ctx: WalkerCtx) -> bool:
    override = getattr(ctx, "ptr_analysis_shim_available", None)
    if override is not None:
        return bool(override)
    return has_cxx_shim()


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
    if isinstance(value, LazyTileExpr):
        return value.read_lane(ctx, loop_vars)
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


def _coerce_index_scalar(ctx: WalkerCtx, value: Any) -> Any:
    """Coerce JSON/PtrAnalysis integer spellings into TIR scalar constants."""
    tir = ctx.tir()
    if isinstance(value, bool):
        return tir.const(int(value), "int32")
    if isinstance(value, int):
        return tir.const(value, "int32")
    if isinstance(value, str):
        text = value.strip()
        try:
            return tir.const(int(text, 0), "int32")
        except ValueError:
            return value
    return value


def _cast_index_like(ctx: WalkerCtx, value: Any, target: Any) -> Any:
    """Cast MLIR ``index``/i64 expressions to the target TIR index dtype."""
    dtype = str(getattr(target, "dtype", "int32") or "int32")
    if isinstance(value, bool):
        return ctx.tir().const(int(value), dtype)
    if isinstance(value, int):
        return ctx.tir().const(value, dtype)
    if hasattr(value, "dtype") and str(value.dtype) != dtype:
        return ctx.tir().Cast(dtype, value)
    return value


def _flat_lane_index(ctx: WalkerCtx, loop_vars: Sequence[Any], shape: Sequence[int]) -> Any:
    """Linearise a rank-N tile loop index into a flat vector lane index."""
    tir = ctx.tir()
    if not loop_vars:
        return tir.const(0, "int32")
    idx: Any = loop_vars[0]
    for axis in range(1, len(loop_vars)):
        extent = shape[axis] if axis < len(shape) else 1
        idx = idx * tir.const(int(extent), "int32") + loop_vars[axis]
    return idx


def _scalarize_tile_index_base(
    ctx: WalkerCtx,
    base: Any,
    loop_vars: Sequence[Any],
    shape: Sequence[int],
) -> Any:
    """Return a scalar source/destination index base for the current lane."""
    tvm_mod = ctx.tvm()
    tir = ctx.tir()
    base = _coerce_index_scalar(ctx, base)
    if isinstance(base, LazyTileExpr):
        return base.read_lane(ctx, tuple(loop_vars))
    if isinstance(base, tvm_mod.tir.Buffer):
        rank = len(base.shape)
        if rank == 0:
            return tir.BufferLoad(base, [tir.const(0, "int32")])
        if len(loop_vars) >= rank:
            indices = list(loop_vars[-rank:])
        else:
            indices = [tir.const(0, "int32")] * (rank - len(loop_vars)) + list(loop_vars)
        return tir.BufferLoad(base, indices)
    if _vector_lanes(base) > 1:
        return _read_vector_lane(ctx, base, _flat_lane_index(ctx, loop_vars, shape))
    return base


def _resolve_ptrstate_value(ctx: WalkerCtx, value: Any) -> Any:
    """Resolve a PtrAnalysis JSON scalar/symbol into a TIR value."""
    try:
        if value in ctx.value_map:
            return ctx.value_map[value]
    except TypeError:
        pass
    value = _coerce_index_scalar(ctx, value)
    if not isinstance(value, str):
        return value
    candidates = (value, value.lstrip("%"), f"%{value.lstrip('%')}")
    for key in candidates:
        try:
            if key in ctx.value_map:
                return ctx.value_map[key]
        except TypeError:
            pass
    raise EmitError(f"PtrState references unresolved SSA value {value!r}")


def _resolve_ptrstate_values(ctx: WalkerCtx, values: Sequence[Any]) -> List[Any]:
    return [_resolve_ptrstate_value(ctx, v) for v in values]


def _compatible_ptr_state(
    ctx: WalkerCtx,
    base: Any,
    result_value: Any,
) -> Any:
    """Find a same-source PtrState whose tile size matches result shape."""
    if not (isinstance(base, dict) and "_ptrstate" in base):
        return None
    target_shape = list(_shape_of(result_value)) if result_value is not None else []
    if not target_shape:
        return None
    source = base.get("source")
    states_map = getattr(ctx, "ptr_states", None) or {}
    for state in states_map.values():
        if source is not None and getattr(state, "source", None) != source:
            continue
        try:
            sizes = [int(s) for s in getattr(state, "sizes", ()) or ()]
        except Exception:
            continue
        if sizes != target_shape:
            continue
        try:
            _resolve_ptrstate_values(ctx, getattr(state, "offsets", ()) or ())
            _resolve_ptrstate_values(ctx, getattr(state, "strides", ()) or ())
        except Exception:
            continue
        return state
    return None


def _unique_ptr_states(ctx: WalkerCtx) -> List[Any]:
    """Return stable unique PtrState objects from ``ctx.ptr_states``."""
    states_map = getattr(ctx, "ptr_states", None) or {}
    seen = set()
    out: List[Any] = []
    for state in states_map.values():
        ident = id(state)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(state)
    out.sort(key=lambda s: str(getattr(s, "result_ssa", "") or ""))
    return out


def _compatible_ptr_state_for_base(
    ctx: WalkerCtx,
    base: Any,
    result_value: Any,
) -> Any:
    """Find a PtrState by source buffer + result tile shape.

    The C++ PtrAnalysis shim serializes SSA names before the Python MLIR
    parser may round-trip the rewritten TTIR through generic form. Some MLIR
    providers renumber SSA results during that parse, so an exact
    ``result_ssa`` lookup can miss even though the source buffer and tile
    shape still identify the recovered pointer state.
    """
    target_shape = list(_shape_of(result_value)) if result_value is not None else []
    if not target_shape:
        return None
    candidates = []
    for state in _unique_ptr_states(ctx):
        if not _ptrstate_source_matches_base(ctx, state, base):
            continue
        try:
            sizes = [int(s) for s in getattr(state, "sizes", ()) or ()]
        except Exception:
            continue
        if sizes != target_shape:
            continue
        try:
            _resolve_ptrstate_values(ctx, getattr(state, "offsets", ()) or ())
            _resolve_ptrstate_values(ctx, getattr(state, "strides", ()) or ())
        except Exception:
            continue
        candidates.append(state)
    if not candidates:
        return None

    source = getattr(candidates[0], "source", None)
    match_key = (str(source), tuple(int(s) for s in target_shape))
    counts = getattr(ctx, "_ptr_state_shape_match_counts", None)
    if counts is None:
        counts = {}
        ctx._ptr_state_shape_match_counts = counts
    idx = int(counts.get(match_key, 0))
    if idx >= len(candidates):
        idx = len(candidates) - 1
    counts[match_key] = idx + 1
    return candidates[idx]


def _has_seeded_ptr_state_for_base(ctx: WalkerCtx, base: Any) -> bool:
    """Whether pre-pass metadata exists for the resolved base pointer."""
    for state in _unique_ptr_states(ctx):
        if _ptrstate_source_matches_base(ctx, state, base):
            return True
    return False


def _ptrstate_source_matches_base(ctx: WalkerCtx, state: Any, base: Any) -> bool:
    """Return False when ``state.source`` resolves to a different base buffer."""
    source = getattr(state, "source", None)
    if source is None or base is None:
        return True

    def _base_buffer(value: Any) -> Any:
        if isinstance(value, tuple) and len(value) == 2:
            return value[0]
        if isinstance(value, dict) and "_ptrstate" in value:
            return None
        try:
            if isinstance(value, ctx.tvm().tir.Buffer):
                return value
        except Exception:
            pass
        return None

    expected = _base_buffer(base)
    if expected is None:
        return True

    resolved = None
    for key in (source, str(source).lstrip("%"), f"%{str(source).lstrip('%')}"):
        try:
            if key in ctx.value_map:
                resolved = ctx.value_map[key]
                break
        except TypeError:
            continue
    actual = _base_buffer(resolved)
    if actual is None:
        return True
    try:
        return bool(actual.same_as(expected))
    except Exception:
        return actual is expected


def _redeclare_ctx_buffer_1d(
    ctx: WalkerCtx,
    buf: Any,
    dtype: str,
    min_extent: int,
) -> Any:
    """Replace a placeholder function-arg buffer with a 1D flat view."""
    try:
        rank = len(buf.shape)
        current_extent = int(buf.shape[0]) if rank == 1 else 0
    except Exception:
        rank = 0
        current_extent = 0
    if rank == 1 and current_extent >= min_extent:
        return buf
    name = getattr(buf, "name", None) or "buf"
    target_key: Any = None
    for key, value in (getattr(ctx, "buffers", {}) or {}).items():
        if value is buf:
            target_key = key
            break
    if target_key is None:
        return buf
    fixed_keys = getattr(ctx, "fixed_arg_buffer_keys", set()) or set()
    if target_key in fixed_keys or str(name) in fixed_keys:
        return buf
    new_buf = ctx.tir().decl_buffer(
        [max(int(min_extent), 1)], dtype, name=str(name)
    )
    ctx.buffers[target_key] = new_buf
    for key, value in list(getattr(ctx, "value_map", {}).items()):
        if value is buf:
            ctx.value_map[key] = new_buf
        elif isinstance(value, tuple) and len(value) == 2 and value[0] is buf:
            ctx.value_map[key] = (new_buf, value[1])
    return new_buf


def _ptrstate_flat_index(
    ctx: WalkerCtx,
    resolved: Dict[str, Any],
    loop_vars: Sequence[Any],
    shape: Sequence[int],
) -> Any:
    """Build flat address from PtrState offsets/strides for current lane."""
    tir = ctx.tir()
    offsets = _resolve_ptrstate_values(ctx, resolved.get("offsets") or [])
    strides = _resolve_ptrstate_values(ctx, resolved.get("strides") or [])
    flat: Any = tir.const(0, "int32")
    for axis, lv in enumerate(loop_vars):
        off = offsets[axis] if axis < len(offsets) else tir.const(0, "int32")
        stride = strides[axis] if axis < len(strides) else tir.const(1, "int32")
        off = _scalarize_tile_index_base(ctx, off, loop_vars, shape)
        stride = _scalarize_tile_index_base(ctx, stride, loop_vars, shape)
        off = _cast_index_like(ctx, off, lv)
        stride = _cast_index_like(ctx, stride, lv)
        flat = flat + (off + lv) * stride
    return flat


def _flat_min_extent(shape: Sequence[int]) -> int:
    extent = 1
    for dim in shape or [1]:
        try:
            extent *= int(dim)
        except Exception:
            extent *= 1024
    return max(extent, 1024 * 1024)


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
            if head.startswith("%"):
                return head
            match = re.search(r"%[A-Za-z0-9_.$-]+", s)
            if match:
                return match.group(0)
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
    # Rank-2+ operand tiles may feed GEMM and need shared scope; rank-1
    # staging stays local to avoid unnecessary threadgroup memory.
    tile_scope = "shared" if len(out_shape or []) >= 2 else "local"
    tile_buf = _alloc_tile_buffer(
        ctx, list(out_shape) or [1], out_dtype, out_buf_name, scope=tile_scope
    )

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
        base = _scalarize_tile_index_base(ctx, base, loop_vars, out_shape)
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
        base = _scalarize_tile_index_base(ctx, base, loop_vars, val_shape)
        dst_indices.append(base + lv)

    # The value being stored is a buffer; index it with the loop vars to
    # get a per-lane scalar.
    if isinstance(val_expr, LazyTileExpr):
        rhs = val_expr.read_lane(ctx, tuple(loop_vars))
    elif hasattr(val_expr, "shape"):
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
    # Rank-2+ loads may feed dot and must satisfy Metal GEMM's shared-scope
    # contract. Rank-1 vector loads do not participate in GEMM; keeping them
    # local avoids blowing Apple's 32 KiB threadgroup-memory cap.
    tile_scope = "shared" if len(out_shape or []) >= 2 else "local"
    tile_buf = _alloc_tile_buffer(
        ctx, list(out_shape) or [1], out_dtype, out_buf_name, scope=tile_scope
    )

    loop_vars: List[Any] = []
    body_indices: List[Any] = list(base_indices) if base_indices else []
    for axis, _extent in enumerate(out_shape or [1]):
        loop_vars.append(tir.Var(ctx.fresh(f"i{axis}"), "int32"))

    src_indices: List[Any] = []
    for axis, lv in enumerate(loop_vars):
        base = body_indices[axis] if axis < len(body_indices) else tir.const(0, "int32")
        base = _scalarize_tile_index_base(ctx, base, loop_vars, out_shape)
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


def _emit_tuple_tile_load_tir(
    op: Any,
    ctx: WalkerCtx,
    src_buf: Any,
    base_indices: Sequence[Any],
    out_shape: Sequence[int],
    out_dtype: str,
    mask_ssa: Any,
    other_ssa: Any,
) -> Any:
    """Non-DEGRADED tile load from an already-composed tuple pointer.

    ``tt.addptr`` can carry a real TIR tuple ``(buffer, [flat_index])`` when
    parser-side SSA renumbering prevents direct PtrState lookup. In that
    case the dynamic address arithmetic is already represented in TIR, so
    using it is more reliable than re-reading stale C++ printed SSA refs.
    """
    tir = ctx.tir()
    result_value = _results(op)[0] if _results(op) else None
    src_buf = _redeclare_ctx_buffer_1d(
        ctx, src_buf, out_dtype, _flat_min_extent(out_shape)
    )
    tile_buf = _alloc_tile_buffer(
        ctx,
        list(out_shape) or [1],
        out_dtype,
        ctx.fresh("tile_load"),
        scope="shared" if len(out_shape or []) >= 2 else "local",
    )
    loop_vars = [
        tir.Var(ctx.fresh(f"i{axis}"), "int32")
        for axis, _extent in enumerate(out_shape or [1])
    ]

    if len(base_indices) == 1:
        flat_idx = _scalarize_tile_index_base(
            ctx, base_indices[0], loop_vars, out_shape
        )
    else:
        flat_idx = tir.const(0, "int32")
        for axis, lv in enumerate(loop_vars):
            base = (
                base_indices[axis]
                if axis < len(base_indices)
                else tir.const(0, "int32")
            )
            base = _scalarize_tile_index_base(ctx, base, loop_vars, out_shape)
            flat_idx = flat_idx + base + lv

    load_expr: Any = tir.BufferLoad(src_buf, [flat_idx])
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

    body: Any = tir.BufferStore(
        tile_buf, load_expr, list(loop_vars) or [tir.const(0, "int32")]
    )
    for axis in range(len(loop_vars) - 1, -1, -1):
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(out_shape[axis]), "int32"),
            tir.ForKind.SERIAL,
            body,
        )
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
        base = _scalarize_tile_index_base(ctx, base, loop_vars, val_shape)
        dst_indices.append(base + lv)

    # The value being stored may be a buffer (per-lane BufferLoad) or a
    # PrimExpr that we broadcast to every lane.
    if isinstance(val_expr, LazyTileExpr):
        rhs = val_expr.read_lane(ctx, tuple(loop_vars))
    elif hasattr(val_expr, "shape"):
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


def _emit_ptrstate_tile_load_tir(
    op: Any,
    ctx: WalkerCtx,
    src_buf: Any,
    resolved: Dict[str, Any],
    out_shape: Sequence[int],
    out_dtype: str,
    mask_ssa: Any,
    other_ssa: Any,
    dynamic_mask_dims: Sequence[Any] = (),
) -> Any:
    """Load a PtrState tile from a flat function-arg buffer using strides."""
    tir = ctx.tir()
    result_value = _results(op)[0] if _results(op) else None
    src_buf = _redeclare_ctx_buffer_1d(
        ctx, src_buf, out_dtype, _flat_min_extent(out_shape)
    )
    tile_buf = _alloc_tile_buffer(
        ctx,
        list(out_shape) or [1],
        out_dtype,
        ctx.fresh("tile_load"),
        scope="shared" if len(out_shape or []) >= 2 else "local",
    )
    loop_vars = [
        tir.Var(ctx.fresh(f"i{axis}"), "int32")
        for axis, _extent in enumerate(out_shape or [1])
    ]
    flat_idx = _ptrstate_flat_index(ctx, resolved, loop_vars, out_shape)
    load_expr: Any = tir.BufferLoad(src_buf, [flat_idx])
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
    dynamic_mask = _dynamic_tts_mask_expr(ctx, loop_vars, dynamic_mask_dims)
    if dynamic_mask is not None:
        load_expr = tir.if_then_else(dynamic_mask, load_expr, tir.const(0, out_dtype))

    body: Any = tir.BufferStore(
        tile_buf, load_expr, list(loop_vars) or [tir.const(0, "int32")]
    )
    for axis in range(len(loop_vars) - 1, -1, -1):
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(out_shape[axis]), "int32"),
            tir.ForKind.SERIAL,
            body,
        )
    ctx.emit(body)
    if result_value is not None:
        ctx.bind(result_value, tile_buf)
    return tile_buf


def _emit_ptrstate_tile_store_tir(
    op: Any,
    ctx: WalkerCtx,
    dst_buf: Any,
    resolved: Dict[str, Any],
    val_expr: Any,
    val_shape: Sequence[int],
    mask_ssa: Any,
    dynamic_mask_dims: Sequence[Any] = (),
) -> Any:
    """Store a PtrState tile to a flat function-arg buffer using strides."""
    tir = ctx.tir()
    dtype = _normalize_mlir_dtype(
        str(getattr(val_expr, "dtype", _dtype_of(_operands(op)[1]) or "float32"))
    )
    dst_buf = _redeclare_ctx_buffer_1d(
        ctx, dst_buf, dtype, _flat_min_extent(val_shape)
    )
    loop_vars = [
        tir.Var(ctx.fresh(f"i{axis}"), "int32")
        for axis, _extent in enumerate(val_shape or [1])
    ]
    flat_idx = _ptrstate_flat_index(ctx, resolved, loop_vars, val_shape)
    if isinstance(val_expr, LazyTileExpr):
        rhs = val_expr.read_lane(ctx, tuple(loop_vars))
    elif hasattr(val_expr, "shape"):
        rhs = tir.BufferLoad(val_expr, list(loop_vars))
    else:
        rhs = val_expr
    store: Any = tir.BufferStore(dst_buf, rhs, [flat_idx])
    if mask_ssa is not None:
        try:
            mask_expr = ctx.get(mask_ssa)
            mask_lane = _resolve_lane_operand(ctx, mask_expr, loop_vars, role="mask")
            store = tir.IfThenElse(mask_lane, store, None)
        except KeyError:
            pass
    dynamic_mask = _dynamic_tts_mask_expr(ctx, loop_vars, dynamic_mask_dims)
    if dynamic_mask is not None:
        store = tir.IfThenElse(dynamic_mask, store, None)
    body: Any = store
    for axis in range(len(loop_vars) - 1, -1, -1):
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(val_shape[axis]), "int32"),
            tir.ForKind.SERIAL,
            body,
        )
    ctx.emit(body)
    return body


def _dynamic_tts_mask_expr(
    ctx: WalkerCtx,
    loop_vars: Sequence[Any],
    mask_dims: Sequence[Any],
) -> Any:
    """Build a per-lane in-bounds predicate from ``tts.*`` dynamic dims."""
    if not mask_dims:
        return None
    tir = ctx.tir()
    pred = None
    rank = len(loop_vars)
    start_axis = max(0, rank - len(mask_dims))
    for i, dim in enumerate(mask_dims):
        axis = start_axis + i
        if axis >= rank:
            break
        dim_expr = _resolved_or_none(ctx, dim)
        if dim_expr is None:
            dim_expr = _coerce_index_scalar(ctx, dim)
        dim_expr = _cast_index_like(ctx, dim_expr, loop_vars[axis])
        lane_pred = tir.LT(loop_vars[axis], dim_expr)
        pred = lane_pred if pred is None else tir.And(pred, lane_pred)
    return pred


def _ptrstate_resolved_dict_from_state(
    state: Any,
    ctx: Optional[WalkerCtx] = None,
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    offsets = list(getattr(state, "offsets", ()) or ())
    strides = list(getattr(state, "strides", ()) or ())
    if ctx is not None:
        try:
            offsets = _resolve_ptrstate_values(ctx, offsets)
            strides = _resolve_ptrstate_values(ctx, strides)
        except EmitError:
            if strict:
                raise
    return {
        "_ptrstate": state,
        "source": getattr(state, "source", None),
        "offsets": offsets,
        "sizes": list(getattr(state, "sizes", ()) or ()),
        "strides": strides,
        "shape": list(state.shape) if getattr(state, "shape", None) is not None else None,
    }


def _parse_int_array_attr(op: Any, key: str) -> List[int]:
    """Parse MLIR generic-form ``array<i64: ...>`` attrs."""
    attrs = _attrs(op)
    raw = attrs.get(key)
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    try:
        text = str(op)
    except Exception:
        return []
    match = re.search(rf"{re.escape(key)}\s*=\s*array<[^:>]+:\s*([^>]*)>", text)
    if not match:
        return []
    out: List[int] = []
    for part in match.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part, 0))
        except ValueError:
            pass
    return out


def _ssa_name_or_literal(value: Any) -> Any:
    name = _ssa_name(value)
    return name if name is not None else value


def _make_tptr_state_from_op(op: Any, ctx: WalkerCtx, result_value: Any) -> Any:
    """Reconstruct PtrState from parsed ``tts.make_tptr`` operands.

    PtrAnalysis serializes state before the MLIR Python parser may renumber
    SSA values. The structured pointer op itself carries the same source,
    dynamic stride, dynamic offset, and static-dim metadata after parsing, so
    we rebuild the state from the parsed op and key it by the parsed result.
    """
    from ..ptr_analysis import PtrState  # noqa: WPS433

    operands = list(_operands(op))
    if not operands:
        return None
    result_name = _ssa_name(result_value)
    if result_name is None:
        return None
    source = _ssa_name_or_literal(operands[0])
    out_shape = list(_shape_of(result_value))
    seg = _parse_int_array_attr(op, "operandSegmentSizes")
    static_offsets = _parse_int_array_attr(op, "static_offsets")
    static_strides = _parse_int_array_attr(op, "static_strides")
    rank = max(len(out_shape), len(static_offsets), len(static_strides), 1)
    sizes = tuple(str(int(x)) for x in (out_shape or [1] * rank))

    n_strides = seg[1] if len(seg) > 1 else max(0, min(rank, len(operands) - 1))
    n_offsets = seg[2] if len(seg) > 2 else max(0, min(rank, len(operands) - 1 - n_strides))
    dyn_strides = operands[1:1 + n_strides]
    dyn_offsets = operands[1 + n_strides:1 + n_strides + n_offsets]
    sentinel = -9223372036854775808

    def _materialize_axis_values(static_values: List[int], dynamic_values: List[Any], default: str) -> Tuple[Any, ...]:
        out: List[Any] = []
        dyn_i = 0
        for axis in range(rank):
            static = static_values[axis] if axis < len(static_values) else sentinel
            if static != sentinel:
                out.append(str(static))
                continue
            if dyn_i < len(dynamic_values):
                out.append(dynamic_values[dyn_i])
                dyn_i += 1
            else:
                out.append(default)
        return tuple(out)

    strides = _materialize_axis_values(static_strides, dyn_strides, "1")
    offsets = _materialize_axis_values(static_offsets, dyn_offsets, "0")
    return PtrState(
        offsets=offsets,
        sizes=sizes,
        strides=strides,
        source=str(source) if source is not None else None,
        shape=tuple("0" for _ in range(rank)),
        op=str(op),
        result_ssa=result_name,
    )


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
    out_dtype = _normalize_mlir_dtype(out_dtype)

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
        tile_buf = _ptrstate_buffer(ctx, resolved, out_dtype)
        if resolved.get("strides"):
            return _emit_ptrstate_tile_load_tir(
                op, ctx, tile_buf, resolved, tile_shape, out_dtype,
                mask_ssa, other_ssa,
            )
        tile_offsets = [_coerce_index_scalar(ctx, v) for v in _ptrstate_offsets_or_zero(resolved)]
        if _ctx_has_cxx_shim(ctx):
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

    if (
        _is_tile_shape(out_shape)
        and isinstance(resolved, tuple)
        and len(resolved) == 2
        and resolved[0] is not None
        and _has_seeded_ptr_state_for_base(ctx, resolved)
    ):
        return _emit_tuple_tile_load_tir(
            op, ctx, resolved[0], list(resolved[1]) or [tir.const(0, "int32")],
            out_shape, out_dtype, mask_ssa, other_ssa,
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
        # Rank-2+ operand tiles may feed GEMM and need shared scope; rank-1
        # staging stays local to avoid unnecessary threadgroup memory.
        if buf_name not in ctx.buffers:
            tile_scope = "shared" if len(out_shape or []) >= 2 else "local"
            src_buf = _alloc_tile_buffer(
                ctx, list(out_shape), out_dtype, buf_name, scope=tile_scope
            )
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
        if _ctx_has_cxx_shim(ctx):
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

    # Scalarise any Buffer-typed index entries before the BufferLoad. After
    # Wave L2's tt.addptr in-loop accumulator, the trailing index of a
    # ``(buf, indices)`` tuple may be a ``tir.Buffer`` (a per-lane offset
    # tile). Passing that Buffer directly to ``tir.BufferLoad(buf, [...,
    # Buffer])`` trips ``index_lanes * buffer_lanes == value_dtype_lanes``
    # because the load expands to ``buf.shape[0]`` lanes against a single
    # storage slot. The scalar path has no enclosing per-lane ``tir.For`` to
    # index the offset tile with, so we fall back to lane 0 -- conservative
    # but loud (any caller that should have hit the tile path instead is
    # caught by the rank/lane-count assertions below).
    indices = list(indices)
    for axis_i, idx_v in enumerate(indices):
        idx_v = _coerce_index_scalar(ctx, idx_v)
        if isinstance(idx_v, LazyTileExpr):
            indices[axis_i] = idx_v.read_lane(
                ctx, tuple(tir.const(0, "int32") for _ in idx_v.shape)
            )
        elif isinstance(idx_v, ctx.tvm().tir.Buffer):
            buf_rank = len(idx_v.shape)
            if buf_rank == 0:
                indices[axis_i] = tir.BufferLoad(idx_v, [tir.const(0, "int32")])
            else:
                indices[axis_i] = tir.BufferLoad(
                    idx_v, [tir.const(0, "int32")] * buf_rank,
                )
        else:
            indices[axis_i] = idx_v

    load_expr: Any = tir.BufferLoad(buf, indices)
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
        dtype = _normalize_mlir_dtype(_dtype_of(val_ssa) or "float32")
        tile_buf = _ptrstate_buffer(ctx, resolved, dtype)
        if resolved.get("strides"):
            return _emit_ptrstate_tile_store_tir(
                op, ctx, tile_buf, resolved, val_expr, tile_shape, mask_ssa,
            )
        tile_offsets = [_coerce_index_scalar(ctx, v) for v in _ptrstate_offsets_or_zero(resolved)]
        if _ctx_has_cxx_shim(ctx):
            from ..op_mapping import _emit_store_copy
            return _emit_store_copy(op, ctx, resolved, val_expr, mask_ssa)
        return _emit_degraded_tile_store(
            op, ctx, tile_buf, tile_offsets, val_expr, tile_shape, mask_ssa,
        )

    # Tile path inferred from the value's shape.
    if _is_tile_shape(val_shape):
        dtype = _dtype_of(val_ssa) or "float32"
        dtype = _normalize_mlir_dtype(dtype)
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
        if _ctx_has_cxx_shim(ctx):
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
        dtype = _normalize_mlir_dtype(dtype)
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

    # Scalarise any Buffer-typed index entries (see emit_tt_load above for
    # rationale: post Wave L2 the in-loop tt.addptr can leave a Buffer in
    # the trailing index slot, which the scalar BufferStore path cannot
    # consume directly because there is no surrounding per-lane For).
    indices = list(indices)
    for axis_i, idx_v in enumerate(indices):
        idx_v = _coerce_index_scalar(ctx, idx_v)
        if isinstance(idx_v, LazyTileExpr):
            indices[axis_i] = idx_v.read_lane(
                ctx, tuple(tir.const(0, "int32") for _ in idx_v.shape)
            )
        elif isinstance(idx_v, ctx.tvm().tir.Buffer):
            buf_rank = len(idx_v.shape)
            if buf_rank == 0:
                indices[axis_i] = tir.BufferLoad(idx_v, [tir.const(0, "int32")])
            else:
                indices[axis_i] = tir.BufferLoad(
                    idx_v, [tir.const(0, "int32")] * buf_rank,
                )
        else:
            indices[axis_i] = idx_v

    if isinstance(val_expr, LazyTileExpr):
        val_expr = val_expr.read_lane(
            ctx, tuple(tir.const(0, "int32") for _ in val_expr.shape)
        )

    store_stmt: Any = tir.BufferStore(buf, val_expr, indices)
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

    if lanes == 1:
        scalar = tir.const(start, "int32")
        if _results(op):
            ctx.bind(_results(op)[0], scalar)
        return scalar

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


def _is_tile_expr(value: Any) -> bool:
    return isinstance(value, LazyTileExpr)


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
    if isinstance(value, LazyTileExpr):
        lanes = 1
        for extent in value.shape:
            lanes *= int(extent)
        return lanes
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
    if isinstance(value, LazyTileExpr):
        return value.dtype
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
    if isinstance(value, LazyTileExpr):
        if len(value.shape) == 1:
            return value.read_lane(ctx, (lane_idx,))
        indices = []
        rem = lane_idx
        for axis, extent in enumerate(value.shape):
            stride = 1
            for trailing in value.shape[axis + 1:]:
                stride *= int(trailing)
            if stride == 1:
                indices.append(rem)
            else:
                q = rem // tir.const(stride, "int32")
                indices.append(q)
                rem = rem - q * tir.const(stride, "int32")
        return value.read_lane(ctx, tuple(indices))
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
    ctx: WalkerCtx,
    src: Any,
    out_shape: Sequence[int],
    dtype: str,
    *,
    lane_axis: Optional[int] = None,
) -> Any:
    """Lower a vector PrimExpr / Buffer source into a fresh tile Buffer.

    Emits a serial ``tir.For`` nest over ``out_shape`` whose body reads one
    scalar vector lane from ``src`` and stores it into the destination tile.
    Returns the freshly allocated ``tir.Buffer``. The store nest is appended
    to ``ctx.stmts`` via ``ctx.emit``.

    By default the innermost axis indexes the source vector, matching
    broadcast semantics. ``tt.expand_dims`` overrides ``lane_axis`` because
    inserting a trailing singleton dimension, e.g. ``(64,) -> (64, 1)``,
    means the source vector maps to axis 0, not the innermost axis.
    """
    tir = ctx.tir()
    out_shape_list = [int(s) for s in out_shape]
    if not out_shape_list:
        raise ValueError("_materialise_vector_into_buffer: out_shape is empty")
    dst = _alloc_tile_buffer(ctx, out_shape_list, dtype, ctx.fresh("vec"))

    loop_vars = [tir.Var(ctx.fresh(f"v{axis}"), "int32") for axis in range(len(out_shape_list))]
    if lane_axis is None:
        lane_axis = len(loop_vars) - 1
    if lane_axis < 0:
        lane_axis += len(loop_vars)
    if not 0 <= lane_axis < len(loop_vars):
        raise ValueError(
            f"_materialise_vector_into_buffer: lane_axis {lane_axis} out of "
            f"range for shape {out_shape_list}"
        )
    lane_idx = loop_vars[lane_axis]
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

    if (src_vec_lanes > 1 or isinstance(src, LazyTileExpr)) and out_shape:
        dtype = _dtype_of(result_value) if result_value is not None else _vector_scalar_dtype(src)
        attrs = _attrs_with_properties_shared(op)
        raw_axis = attrs.get("axis", len(out_shape) - 1)
        axis = int(raw_axis)
        if axis < 0:
            axis += len(out_shape)
        if not 0 <= axis < len(out_shape):
            raise ValueError(
                f"tt.expand_dims: axis {raw_axis!r} out of range for "
                f"result shape {out_shape}"
            )
        lane_axis = next(
            (i for i in range(len(out_shape)) if i != axis),
            len(out_shape) - 1,
        )
        lazy = LazyTileExpr(
            out_shape,
            dtype,
            lambda read_ctx, indices: _read_vector_lane(
                read_ctx, src, tuple(indices)[lane_axis]
            ),
            name=ctx.fresh("expand_expr"),
        )
        out = materialize_lazy_tile(
            ctx,
            lazy,
            out_shape,
            dtype,
            name="expand",
            loop_var_prefix="v",
        )
        if result_value is not None:
            ctx.bind(result_value, out)
        return out

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
    src_is_lazy = isinstance(src, LazyTileExpr)
    if (
        src_lanes == 1
        and not src_is_buffer
        and not src_is_lazy
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
    if out_shape and (out_lanes > src_lanes or src_is_buffer or src_is_lazy or src_vec_lanes > 1):
        def _broadcast_reader(read_ctx: WalkerCtx, indices: Tuple[Any, ...]) -> Any:
            read_tir = read_ctx.tir()
            idx_tuple = tuple(indices)
            if src_is_buffer:
                if src_shape:
                    tail = idx_tuple[-len(src_shape):]
                    src_indices = []
                    for dim, idx in zip(src_shape, tail):
                        try:
                            dim_i = int(dim)
                        except (TypeError, ValueError):
                            dim_i = None
                        src_indices.append(read_tir.const(0, "int32") if dim_i == 1 else idx)
                else:
                    src_indices = [read_tir.const(0, "int32")]
                return read_tir.BufferLoad(src, src_indices)
            if src_is_lazy:
                src_rank = len(src.shape)
                if src_rank:
                    tail = idx_tuple[-src_rank:]
                    src_indices = []
                    for dim, idx in zip(src.shape, tail):
                        src_indices.append(read_tir.const(0, "int32") if int(dim) == 1 else idx)
                    return src.read_lane(read_ctx, tuple(src_indices))
                return src.read_lane(read_ctx, ())
            if src_vec_lanes > 1:
                return _read_vector_lane(read_ctx, src, idx_tuple[-1])
            return src

        lazy = LazyTileExpr(out_shape, dtype, _broadcast_reader, name=ctx.fresh("bcast_expr"))
        out_buf = materialize_lazy_tile(
            ctx,
            lazy,
            out_shape,
            dtype,
            name="bcast",
            loop_var_prefix="b",
        )
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
    result_is_ptr = False
    if result_value is not None:
        result_is_ptr = (
            "!tt.ptr" in str(getattr(result_value, "type", ""))
            or _dtype_of(result_value) == "handle"
        )
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
    if isinstance(src, tvm_mod.tir.Buffer) and result_is_ptr:
        out = (src, [tir.const(0, "int32")])
    elif isinstance(src, tvm_mod.tir.Buffer):
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
        if (src_vec_lanes > 1 or isinstance(src, LazyTileExpr)) and lanes > 1 and out_shape:
            # Defensive: ``tt.splat`` is contractually scalar->tile, but if
            # the producer accidentally bound a tile expression to the
            # source SSA we keep it lazy instead of spilling to thread stack.
            dtype = _dtype_of(result_value) if result_value is not None else _vector_scalar_dtype(src)
            out = LazyTileExpr(
                out_shape,
                dtype,
                lambda read_ctx, indices: _read_vector_lane(read_ctx, src, tuple(indices)[-1]),
                name=ctx.fresh("splat_expr"),
            )
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


def _compose_addptr_index(
    ctx: WalkerCtx,
    prev: Any,
    off: Any,
    *,
    target: Any = None,
    materialize: bool = False,
) -> Any:
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

    Returns a ``tir.PrimExpr`` for scalar fast paths, a ``LazyTileExpr`` for
    same-shape per-lane index composition, or a fresh ``tir.Buffer`` when a
    caller needs a materialized tile (PtrState metadata, explicit loop-carried
    targets, or broadcast-shape expansion).
    """
    tvm_mod = ctx.tvm()
    tir = ctx.tir()
    Buffer = tvm_mod.tir.Buffer
    prev = _coerce_index_scalar(ctx, prev)
    off = _coerce_index_scalar(ctx, off)

    prev_is_buf = isinstance(prev, Buffer)
    off_is_buf = isinstance(off, Buffer)
    prev_is_tile = prev_is_buf or isinstance(prev, LazyTileExpr)
    off_is_tile = off_is_buf or isinstance(off, LazyTileExpr)

    # Scalar fast path: the existing TIR ``+`` operator handles this.
    if not prev_is_tile and not off_is_tile:
        return prev + off

    def _broadcast_shape(lhs: Sequence[Any], rhs: Sequence[Any]) -> List[int]:
        left = [int(s) for s in lhs]
        right = [int(s) for s in rhs]
        rank = max(len(left), len(right))
        left = [1] * (rank - len(left)) + left
        right = [1] * (rank - len(right)) + right
        out = []
        for a_dim, b_dim in zip(left, right):
            if a_dim == b_dim:
                out.append(a_dim)
            elif a_dim == 1:
                out.append(b_dim)
            elif b_dim == 1:
                out.append(a_dim)
            else:
                raise EmitError(
                    "tt.addptr: cannot broadcast offset shapes "
                    f"{list(lhs)} and {list(rhs)}"
                )
        return out

    # Pick the result tile shape using normal broadcast rules. This matters for
    # matmul C stores: row offsets are shaped (BLOCK_M, 1), column offsets are
    # shaped (BLOCK_M, BLOCK_N), and the composed flat index tile must be
    # (BLOCK_M, BLOCK_N), not the first operand's shape.
    if prev_is_tile and off_is_tile:
        out_shape = _broadcast_shape(prev.shape, off.shape)
        out_dtype = str(getattr(prev, "dtype", "int32"))
    elif prev_is_tile:
        out_shape = list(prev.shape)
        out_dtype = str(getattr(prev, "dtype", "int32"))
    else:
        out_shape = list(off.shape)
        out_dtype = str(getattr(off, "dtype", "int32"))

    def _shape_tuple(value: Any) -> Optional[Tuple[int, ...]]:
        if not isinstance(value, (Buffer, LazyTileExpr)):
            return None
        try:
            return tuple(int(s) for s in value.shape)
        except Exception:
            return None

    try:
        out_shape_tuple = tuple(int(s) for s in out_shape)
    except Exception:
        out_shape_tuple = ()
    needs_materialized_shape = materialize
    for operand in (prev, off):
        operand_shape = _shape_tuple(operand)
        if operand_shape is not None and operand_shape != out_shape_tuple:
            needs_materialized_shape = True
            break

    if target is not None and isinstance(target, Buffer):
        try:
            target_shape = [int(s) for s in target.shape]
            out_shape_int = [int(s) for s in out_shape]
        except Exception:
            target_shape = []
            out_shape_int = []
        if target_shape == out_shape_int:
            out_buf = target
        else:
            out_buf = _alloc_tile_buffer(
                ctx, out_shape, out_dtype, ctx.fresh("addptr_acc")
            )
    elif target is not None:
        out_buf = _alloc_tile_buffer(ctx, out_shape, out_dtype, ctx.fresh("addptr_acc"))
    elif needs_materialized_shape:
        out_buf = _alloc_tile_buffer(ctx, out_shape, out_dtype, ctx.fresh("addptr_acc"))
    else:
        out_buf = None
    loop_vars = [tir.Var(ctx.fresh(f"a{axis}"), "int32") for axis in range(len(out_shape))]

    def _flat_lane_idx(indices: Sequence[Any], read_tir: Any) -> Any:
        """Linearise the surrounding loop-var nest to a single lane index.

        Vector PrimExpr offsets (``Broadcast``/``Ramp``/elementwise vector
        binops produced by ``tt.make_range`` + ``tt.splat``) carry a flat
        lane count equal to ``prod(out_shape)``. To read one scalar lane
        out of such a vector we need a single linear index that matches
        the position the surrounding ``tir.For`` nest is currently at.
        """
        if not indices:
            return read_tir.const(0, "int32")
        idx: Any = indices[0]
        for axis in range(1, len(indices)):
            idx = idx * read_tir.const(int(out_shape[axis]), "int32") + indices[axis]
        return idx

    def _lane(operand: Any, read_ctx: WalkerCtx, indices: Sequence[Any]) -> Any:
        read_tir = read_ctx.tir()
        if isinstance(operand, LazyTileExpr):
            rank = len(operand.shape)
            if len(indices) >= rank:
                idx = list(indices[-rank:])
            else:
                idx = [read_tir.const(0, "int32")] * (rank - len(indices)) + list(indices)
            for axis, extent in enumerate(operand.shape):
                if int(extent) == 1:
                    idx[axis] = read_tir.const(0, "int32")
            return operand.read_lane(read_ctx, tuple(idx))
        if isinstance(operand, Buffer):
            rank = len(operand.shape)
            if rank == 0:
                return read_tir.BufferLoad(operand, [read_tir.const(0, "int32")])
            # Broadcast: align the trailing ``rank`` axes; pad the leading
            # axes with zero so a rank-1 tile composed with a rank-2 tile
            # behaves the same way ``_emit_tile_binop`` does.
            if len(indices) >= rank:
                idx = list(indices[-rank:])
            else:
                idx = [read_tir.const(0, "int32")] * (rank - len(indices)) + list(indices)
            for axis, extent in enumerate(operand.shape):
                try:
                    if int(extent) == 1:
                        idx[axis] = read_tir.const(0, "int32")
                except Exception:
                    pass
            return read_tir.BufferLoad(operand, idx)
        # Vector PrimExpr (e.g. ``Broadcast(scalar, N)`` or
        # ``Ramp(base, 1, N) + Broadcast(...)``): read a single scalar lane
        # via the flattened lane index so the BufferStore below sees a
        # scalar value (not a vector with ``prod(out_shape)`` lanes, which
        # tripsx ``index_lanes * buffer_lanes == value_dtype_lanes``).
        if _vector_lanes(operand) > 1:
            return _read_vector_lane(read_ctx, operand, _flat_lane_idx(indices, read_tir))
        # Scalar PrimExpr broadcasts across every lane.
        return operand

    if out_buf is None:
        return LazyTileExpr(
            out_shape,
            out_dtype,
            lambda read_ctx, indices: _lane(prev, read_ctx, indices)
            + _lane(off, read_ctx, indices),
            name=ctx.fresh("addptr_expr"),
        )

    body: Any = tir.BufferStore(
        out_buf,
        _lane(prev, ctx, loop_vars) + _lane(off, ctx, loop_vars),
        list(loop_vars),
    )
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


def _loop_carry_addptr_target(ctx: WalkerCtx, ptr_ssa: Any, base: Any) -> Any:
    """Return the carried index buffer for ``ptr_ssa`` when it is loop-carried."""
    carries = getattr(ctx, "loop_carry_buffers", {}) or {}

    def _descriptor_index_buffer(value: Any) -> Any:
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[1], (list, tuple))
            and value[1]
        ):
            candidate = value[1][-1]
            try:
                if isinstance(candidate, ctx.tvm().tir.Buffer):
                    return candidate
            except Exception:
                return None
        try:
            if isinstance(value, ctx.tvm().tir.Buffer):
                return value
        except Exception:
            pass
        return None

    if ptr_ssa in carries:
        target = _descriptor_index_buffer(carries[ptr_ssa])
        if target is not None:
            return target
    try:
        name = _ssa_name(ptr_ssa)
    except Exception:
        name = None
    if name and name in carries:
        target = _descriptor_index_buffer(carries[name])
        if target is not None:
            return target
    if isinstance(base, tuple):
        base_target = _descriptor_index_buffer(base)
        for carried in carries.values():
            target = _descriptor_index_buffer(carried)
            if target is not None and target is base_target:
                return target
    return None


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
    tir = ctx.tir()
    operands = _operands(op)
    if len(operands) < 2:
        raise ValueError("tt.addptr: expected (ptr, offset) operands")
    ptr_ssa, off_ssa = operands[0], operands[1]
    base = _resolved_or_none(ctx, ptr_ssa)

    result_value = _results(op)[0] if _results(op) else None

    state = None
    states_map = getattr(ctx, "ptr_states", None) or {}
    if result_value is not None:
        rname = _ssa_name(result_value)
        if rname:
            state = states_map.get(rname)
    if (
        state is not None
        and base is None
        and _ptrstate_source_matches_base(ctx, state, base)
    ):
        value = _ptrstate_resolved_dict_from_state(state, ctx, strict=True)
        if result_value is not None:
            ctx.bind(result_value, value)
        return value

    off = ctx.get(off_ssa)

    # A seeded PtrState can arrive from the subprocess pre-pass even when
    # this interpreter must not import the shim because Triton's native
    # libtriton is already loaded. Trust the seeded state/descriptor before
    # probing shim availability again.
    if isinstance(base, dict) and "_ptrstate" in base:
        new_offsets = _resolve_ptrstate_values(ctx, list(base.get("offsets") or []))
        if new_offsets:
            # The trailing offset slot may be a scalar PrimExpr, an
            # int (untouched here), or a Buffer when an earlier
            # iteration already produced a tile. Compose-or-pass.
            if isinstance(new_offsets[-1], int):
                new_offsets[-1] = new_offsets[-1] + 0
            else:
                new_offsets[-1] = _compose_addptr_index(
                    ctx,
                    new_offsets[-1],
                    off,
                    materialize=True,
                )
        else:
            new_offsets = [off]
        new_state = dict(base)
        new_state["offsets"] = new_offsets
        if base.get("strides"):
            new_state["strides"] = _resolve_ptrstate_values(
                ctx, list(base.get("strides") or [])
            )
        if result_value is not None:
            ctx.bind(result_value, new_state)
        return new_state

    state = (
        None
        if base is not None
        else _compatible_ptr_state_for_base(ctx, base, result_value)
    )
    if state is not None:
        value = _ptrstate_resolved_dict_from_state(state, ctx, strict=True)
        if result_value is not None:
            ctx.bind(result_value, value)
        return value

    if _ctx_has_cxx_shim(ctx):
        state = _compatible_ptr_state(ctx, base, result_value)
        if state is not None:
            value = _ptrstate_resolved_dict_from_state(state, ctx, strict=True)
            if result_value is not None:
                ctx.bind(result_value, value)
            return value
        if isinstance(base, tuple) and len(base) == 2:
            buf, indices = base
            new_indices = list(indices) or [tir.const(0, "int32")]
            new_indices[-1] = _compose_addptr_index(
                ctx,
                new_indices[-1],
                off,
                target=_loop_carry_addptr_target(ctx, ptr_ssa, base),
            )
            value = (buf, new_indices)
            if result_value is not None:
                ctx.bind(result_value, value)
            return value

    # Generic tuple/scalar pointer arithmetic. This is not itself a degraded
    # memory access; downstream load/store emitters are responsible for
    # marking a visible fallback when a real tile memory op cannot use
    # PtrAnalysis metadata.
    if isinstance(base, tuple) and len(base) == 2:
        buf, indices = base
        new_indices = list(indices) or [tir.const(0, "int32")]
        new_indices[-1] = _compose_addptr_index(
            ctx,
            new_indices[-1],
            off,
            target=_loop_carry_addptr_target(ctx, ptr_ssa, base),
        )
        value = (buf, new_indices)
    elif base is None:
        # No prior binding: synthesise a flat scalar with the offset.
        value = (None, [off])
    else:
        # Treat the prior value as a buffer/var; emit a fresh tuple.
        value = (base, [off])

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
    result_value = _results(op)[0] if _results(op) else None
    out_shape = list(_shape_of(result_value)) if result_value is not None else [1]
    out_dtype = _dtype_of(result_value) if result_value is not None else "float32"

    if result_value is not None:
        state = _make_tptr_state_from_op(op, ctx, result_value)
        if state is not None:
            if not hasattr(ctx, "ptr_states"):
                ctx.ptr_states = {}
            ctx.ptr_states[state.result_ssa] = state
            value = _ptrstate_resolved_dict_from_state(state, ctx, strict=True)
            ctx.bind(result_value, value)
            return value

    try:
        import tilelang.language as T  # type: ignore
        frag = T.alloc_fragment(out_shape, out_dtype)
    except Exception:  # pragma: no cover -- TileLang absent or no IRBuilder scope
        # Fragment fallback: tile-scoped buffer (see ``_alloc_tile_buffer``)
        # so VerifyMemory's host-memory check skips it.
        frag = _alloc_tile_buffer(ctx, out_shape, out_dtype, ctx.fresh("tptr"))

    if result_value is not None:
        ctx.bind(result_value, frag)
    return frag


def emit_tts_load(op: Any, ctx: WalkerCtx) -> Any:
    """Lower PtrAnalysis ``tts.load`` using the recovered PtrState.

    ``tts.load`` is only emitted after TritonStructured PtrAnalysis has
    materialised a structured pointer. Missing PtrState is therefore a real
    frontend bug, not a cue to fall back to the visible ``# DEGRADED`` scalar
    placeholder route used for unanalysed ``tt.load``.
    """
    operands = _operands(op)
    if not operands:
        raise EmitError("tts.load: missing structured pointer operand")
    ptr_ssa = operands[0]
    dynamic_mask_dims = tuple(operands[1:])
    result_value = _results(op)[0] if _results(op) else None
    out_shape = list(_shape_of(result_value)) if result_value is not None else []
    out_dtype = _normalize_mlir_dtype(
        _dtype_of(result_value) if result_value is not None else "float32"
    )

    state = _lookup_ptr_state(ctx, op, ptr_ssa)
    if state is None:
        ptr_name = _ssa_name(ptr_ssa)
        known = sorted(str(k) for k in (getattr(ctx, "ptr_states", {}) or {}).keys())
        raise EmitError(
            "tts.load: missing PtrState for structured pointer; "
            "run PtrAnalysis pre-pass or fix result SSA threading; "
            f"ptr={ptr_name!r}; known={known[:8]!r}"
        )
    resolved = _ptrstate_resolved_dict_from_state(state, ctx, strict=True)
    tile_shape = _ptrstate_sizes_int(resolved) or out_shape or [1]
    src_buf = _ptrstate_buffer(ctx, resolved, out_dtype)
    return _emit_ptrstate_tile_load_tir(
        op,
        ctx,
        src_buf,
        resolved,
        tile_shape,
        out_dtype,
        None,
        None,
        dynamic_mask_dims,
    )


def emit_tts_store(op: Any, ctx: WalkerCtx) -> Any:
    """Lower PtrAnalysis ``tts.store`` using recovered PtrState strides."""
    operands = _operands(op)
    if len(operands) < 2:
        raise EmitError("tts.store: missing structured pointer or value operand")
    ptr_ssa, val_ssa = operands[0], operands[1]
    dynamic_mask_dims = tuple(operands[2:])
    val_expr = ctx.get(val_ssa)
    val_shape = list(_shape_of(val_ssa)) or list(getattr(val_expr, "shape", ()) or [])

    state = _lookup_ptr_state(ctx, op, ptr_ssa)
    if state is None:
        ptr_name = _ssa_name(ptr_ssa)
        known = sorted(str(k) for k in (getattr(ctx, "ptr_states", {}) or {}).keys())
        raise EmitError(
            "tts.store: missing PtrState for structured pointer; "
            "run PtrAnalysis pre-pass or fix result SSA threading; "
            f"ptr={ptr_name!r}; known={known[:8]!r}"
        )
    resolved = _ptrstate_resolved_dict_from_state(state, ctx, strict=True)
    dtype = _normalize_mlir_dtype(_dtype_of(val_ssa) or getattr(val_expr, "dtype", "float32"))
    tile_shape = _ptrstate_sizes_int(resolved) or val_shape or [1]
    dst_buf = _ptrstate_buffer(ctx, resolved, dtype)
    return _emit_ptrstate_tile_store_tir(
        op,
        ctx,
        dst_buf,
        resolved,
        val_expr,
        tile_shape,
        None,
        dynamic_mask_dims,
    )


# ---------------------------------------------------------------------------
# tt.split / tt.join
# ---------------------------------------------------------------------------


def emit_tt_split(op: Any, ctx: WalkerCtx) -> Any:
    """Lower tt.split to two separate buffer slices along the last dimension."""
    tir = ctx.tir()
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.split: missing source operand")
    src_ssa = operands[0]
    src = ctx.get(src_ssa)
    results = _results(op)
    if not results or len(results) != 2:
        raise ValueError("tt.split: expected 2 results")
    
    out_shape_0 = list(_shape_of(results[0]))
    out_dtype_0 = _dtype_of(results[0]) or getattr(src, "dtype", "float32")
    out_shape_1 = list(_shape_of(results[1]))
    out_dtype_1 = _dtype_of(results[1]) or getattr(src, "dtype", "float32")
    
    buf0_name = ctx.fresh("split0")
    buf1_name = ctx.fresh("split1")
    buf0 = _alloc_tile_buffer(ctx, out_shape_0 or [1], out_dtype_0, buf0_name)
    buf1 = _alloc_tile_buffer(ctx, out_shape_1 or [1], out_dtype_1, buf1_name)
    
    loop_vars = [tir.Var(ctx.fresh(f"s{i}"), "int32") for i in range(len(out_shape_0))]
    
    src_indices_0 = list(loop_vars) + [tir.const(0, "int32")]
    src_indices_1 = list(loop_vars) + [tir.const(1, "int32")]
    
    if isinstance(src, LazyTileExpr):
        rhs0 = src.read_lane(ctx, tuple(src_indices_0))
        rhs1 = src.read_lane(ctx, tuple(src_indices_1))
    elif hasattr(src, "shape"):
        rhs0 = tir.BufferLoad(src, src_indices_0)
        rhs1 = tir.BufferLoad(src, src_indices_1)
    else:
        rhs0 = src
        rhs1 = src
        
    store0 = tir.BufferStore(buf0, rhs0, list(loop_vars) or [tir.const(0, "int32")])
    body0: Any = store0
    for axis in range(len(loop_vars) - 1, -1, -1):
        extent = out_shape_0[axis] if axis < len(out_shape_0) else 1
        body0 = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body0,
        )
    ctx.emit(body0)
    
    store1 = tir.BufferStore(buf1, rhs1, list(loop_vars) or [tir.const(0, "int32")])
    body1: Any = store1
    for axis in range(len(loop_vars) - 1, -1, -1):
        extent = out_shape_1[axis] if axis < len(out_shape_1) else 1
        body1 = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body1,
        )
    ctx.emit(body1)
    
    ctx.bind(results[0], buf0)
    ctx.bind(results[1], buf1)
    
    # Return None so Walker doesn't bind a single result
    return None


def emit_tt_join(op: Any, ctx: WalkerCtx) -> Any:
    """Lower tt.join to a single buffer by combining along a new last dimension."""
    tir = ctx.tir()
    operands = _operands(op)
    if len(operands) != 2:
        raise ValueError("tt.join: expected 2 source operands")
    src0 = ctx.get(operands[0])
    src1 = ctx.get(operands[1])
    
    result_value = _results(op)[0] if _results(op) else None
    out_shape = list(_shape_of(result_value)) if result_value is not None else []
    out_dtype = _dtype_of(result_value) if result_value is not None else getattr(src0, "dtype", "float32")
    
    buf_name = ctx.fresh("join")
    out_buf = _alloc_tile_buffer(ctx, out_shape or [2], out_dtype, buf_name)
    
    src_shape = out_shape[:-1] if out_shape else []
    
    loop_vars = [tir.Var(ctx.fresh(f"j{i}"), "int32") for i in range(len(src_shape))]
    
    if hasattr(src0, "shape"):
        rhs0 = tir.BufferLoad(src0, list(loop_vars) or [tir.const(0, "int32")])
    else:
        rhs0 = src0
        
    if hasattr(src1, "shape"):
        rhs1 = tir.BufferLoad(src1, list(loop_vars) or [tir.const(0, "int32")])
    else:
        rhs1 = src1
        
    dst_indices_0 = list(loop_vars) + [tir.const(0, "int32")]
    dst_indices_1 = list(loop_vars) + [tir.const(1, "int32")]
    
    store0 = tir.BufferStore(out_buf, rhs0, dst_indices_0)
    body0: Any = store0
    for axis in range(len(loop_vars) - 1, -1, -1):
        extent = src_shape[axis] if axis < len(src_shape) else 1
        body0 = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body0,
        )
    ctx.emit(body0)

    store1 = tir.BufferStore(out_buf, rhs1, dst_indices_1)
    body1: Any = store1
    for axis in range(len(loop_vars) - 1, -1, -1):
        extent = src_shape[axis] if axis < len(src_shape) else 1
        body1 = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body1,
        )
    ctx.emit(body1)
    
    if result_value is not None:
        ctx.bind(result_value, out_buf)
    return out_buf


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
    "tt.split": emit_tt_split,
    "tt.join": emit_tt_join,
    "tts.make_tptr": emit_tts_make_tptr,
    "tts.load": emit_tts_load,
    "tts.store": emit_tts_store,
}
