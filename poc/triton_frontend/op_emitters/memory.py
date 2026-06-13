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
    should_fold_addressing,
    _result_ssa_name,
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
    # Import lazily so tests and isolated harnesses can exercise shim probing
    # without importing the native extension at module import time.
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
        return ctx.get(ssa_value)
    except (KeyError, TypeError):
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
    """Resolve a PtrAnalysis JSON scalar/symbol into a TIR value.

    The loop-carried pointer advance is handled in :func:`_ptrstate_flat_index`
    via the resolved dict's ``_carry_flat`` field, not in the per-axis offset,
    so this resolver stays a single scalar/symbol path.
    """
    try:
        if value in ctx.value_map:
            resolved = ctx.value_map[value]
            if isinstance(resolved, tuple) and len(resolved) == 2:
                # If it resolved to a (buffer, indices) pointer tuple, the actual
                # numeric offset is the trailing index.
                return resolved[1][-1]
            return resolved
    except TypeError:
        pass
    value = _coerce_index_scalar(ctx, value)
    if not isinstance(value, str):
        return value
    candidates = (value, value.lstrip("%"), f"%{value.lstrip('%')}")
    for key in candidates:
        try:
            if key in ctx.value_map:
                resolved = ctx.value_map[key]
                if isinstance(resolved, tuple) and len(resolved) == 2:
                    return resolved[1][-1]
                return resolved
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
    *,
    grow_fixed: bool = False,
) -> Any:
    """Replace a placeholder function-arg buffer with a flat SYMBOLIC view.

    A strided per-block load/store addresses a region whose true flat
    extent is the real DLTensor element count -- which is a *runtime*
    quantity (it depends on seqlen / grid / strides). Declaring the flat
    arg buffer with a baked numeric extent (the old ``max(numel, 1<<20)``
    floor) MONOMORPHIZES the kernel: the compiled PrimFunc only fits one
    launch shape, and a larger launch (the §P1 grid (1,64,112), 29.36M-elem
    ``dout``) addresses past the baked ``1048576`` extent and SEGFAULTS at
    launch (declared extent << real). RULE #1: a buffer sized below the real
    tensor is a truncation/OOB bug, not a contract.

    We therefore declare the flat buffer with a FRESH SYMBOLIC int64 extent
    ``Var`` (one per arg key, cached on ``ctx.flat_arg_extent_vars``).
    Because the buffer lives in the PrimFunc ``buffer_map``, MakePackedAPI
    binds that extent Var from the passed DLTensor's real element count at
    launch -- so the SAME compiled kernel runs at ANY grid/seqlen. The
    per-block base + tile mask in the flat index / store guard already bound
    every reachable element to the in-tensor region (mirroring the native
    ``tl.store`` row/col mask); the whole-buffer extent only has to MATCH the
    real allocation, which a symbolic-bound Var does exactly.

    ``grow_fixed`` -- a caller-seeded ("fixed") function-arg buffer is only
    authoritative when it already carries a real (non-placeholder) extent we
    must not override. We keep honoring such a seed (early-return) unless a
    strided per-block write needs the symbolic span; ``min_extent`` is no
    longer used to size the buffer (kept for signature/back-compat) -- the
    symbolic Var supersedes any numeric floor.
    """
    name = getattr(buf, "name", None) or "buf"
    target_key: Any = None
    for key, value in (getattr(ctx, "buffers", {}) or {}).items():
        if value is buf:
            target_key = key
            break
    if target_key is None:
        return buf

    # Already symbolic for this key -> idempotent; reuse the cached buffer so
    # every load/store of this arg resolves to ONE extent Var (bound once).
    extent_cache = getattr(ctx, "flat_arg_extent_vars", None)
    if extent_cache is None:
        extent_cache = {}
        ctx.flat_arg_extent_vars = extent_cache
    cached_buf = extent_cache.get(target_key)
    if cached_buf is not None and ctx.buffers.get(target_key) is cached_buf:
        return cached_buf

    fixed_keys = getattr(ctx, "fixed_arg_buffer_keys", set()) or set()
    is_fixed = target_key in fixed_keys or str(name) in fixed_keys
    if is_fixed and not grow_fixed:
        return buf

    tir = ctx.tir()
    # Fresh symbolic flat extent bound by MakePackedAPI from the real tensor.
    try:
        extent_var = tir.Var(ctx.fresh(str(name) + "_numel"), "int64")
    except Exception:
        # Never fall back to a baked under-counted constant (RULE #1): if a
        # symbol cannot be minted, surface the failure rather than silently
        # monomorphizing the kernel to a too-small extent.
        raise RuntimeError(
            "could not mint symbolic flat extent Var for buffer %r; refusing "
            "to bake a monomorphized numeric extent (would segfault at a "
            "larger launch)" % (str(name),)
        )
    # Reuse the ORIGINAL placeholder's backing data Var so the param handle
    # (bound by MakePackedAPI) and this flat view resolve to ONE Var -- minting
    # a fresh data Var would leave a free variable MakePackedAPI rejects.
    data_var = getattr(buf, "data", None)
    if data_var is not None:
        new_buf = tir.decl_buffer(
            [extent_var], dtype, name=str(name), data=data_var,
        )
    else:
        new_buf = tir.decl_buffer([extent_var], dtype, name=str(name))
    ctx.buffers[target_key] = new_buf
    extent_cache[target_key] = new_buf
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
        # The PtrAnalysis shim recovers per-axis ``offsets`` that are ALREADY
        # flat element offsets from the buffer base -- i.e. the FULL pointer
        # arithmetic ``base + pid_b*stride_b + pid_c*stride_chunk + ... +
        # blk_origin*stride_axis`` collapsed into one symbol per axis (each
        # term already carries its own stride). The per-lane tile coordinate
        # ``lv`` (0..tile_extent) is the ONLY part that still needs the axis
        # stride applied. The previous ``(off + lv) * stride`` re-multiplied
        # the already-flat ``off`` by ``stride[axis]`` -- a stride^2
        # double-count that placed every program block (and even the in-tile
        # rows) at the wrong flat address. Correct decomposition:
        #     flat += off[axis]              (already flat, add once)
        #     flat += lv * stride[axis]      (tile coordinate -> flat)
        # RULE #1: one correct flat address, not a stride-squared placement.
        flat = flat + off + lv * stride
    # FORWARDPTR: loop-carried pointer advance. A scf.for block-arg forwarded
    # by ``map_scf_for`` advances its base pointer by ``advance`` elements each
    # trip; after ``trips = (induction - lb) / step`` trips the total flat
    # shift is ``trips * advance``. Adding it here makes the strided in-loop
    # load slide exactly like the native multi-K-trip kernel -- no carry_index
    # scalar gather, and no per-axis stride-name matching required.
    carry = resolved.get("_carry_flat") if isinstance(resolved, dict) else None
    if carry:
        advance = _carry_resolve(ctx, carry.get("advance"))
        induction = _carry_resolve(ctx, carry.get("induction"))
        lb = _carry_resolve(ctx, carry.get("lb"))
        step = _carry_resolve(ctx, carry.get("step"))
        advance = _cast_index_like(ctx, advance, flat)
        induction = _cast_index_like(ctx, induction, flat)
        lb = _cast_index_like(ctx, lb, flat)
        step = _cast_index_like(ctx, step, flat)
        trips = tir.FloorDiv(induction - lb, step)
        flat = flat + trips * advance
    return flat


def _carry_resolve(ctx: WalkerCtx, ref: Any) -> Any:
    """Resolve a carry-flat field (PrimExpr / SSA name / int) to a scalar.

    ``advance`` is already a resolved PrimExpr (from forwarding time);
    ``induction`` / ``lb`` / ``step`` are SSA names bound by ``_emit_region``
    (the loop bounds + induction var) or synthetic constant names. RULE #1:
    an unresolved name raises.
    """
    tir = ctx.tir()
    if ref is None:
        return tir.const(0, "int32")
    if isinstance(ref, int):
        return tir.const(ref, "int32")
    if not isinstance(ref, str):
        return ref  # already a PrimExpr
    for key in (ref, ref.lstrip("%"), f"%{ref.lstrip('%')}"):
        try:
            if key in ctx.value_map:
                val = ctx.value_map[key]
                if isinstance(val, tuple) and len(val) == 2:
                    return val[1][-1]
                return val
        except TypeError:
            pass
    raise EmitError(
        f"FORWARDPTR carry-advance references unresolved field {ref!r}"
    )


def _flat_min_extent(shape: Sequence[int]) -> int:
    extent = 1
    for dim in shape or [1]:
        try:
            extent *= int(dim)
        except Exception:
            extent *= 1024
    return max(extent, 1024 * 1024)


def _tile_numel(shape: Sequence[int]) -> int:
    """Numeric element count of a (store) tile shape; 1 for empty."""
    numel = 1
    for dim in shape or [1]:
        try:
            numel *= max(int(dim), 1)
        except Exception:
            return 0
    return numel


def _grid_scaled_store_extent(ctx: WalkerCtx, val_shape: Sequence[int]) -> int:
    """Lower bound on the flat extent a strided per-block store addresses.

    A ``tt.store`` whose destination pointer carries a per-program base
    offset (``pid_b*stride_b + pid_c*stride_c + pid_h*stride_h + ...``)
    writes a distinct tile for every point in the launch grid. The output
    buffer must therefore span ALL grid blocks, not a single tile. The
    exact extent depends on the (runtime) destination strides, but a sound
    lower bound is ``grid_product * tile_numel`` -- the dense-packing floor
    that every contiguous grid layout meets or exceeds. We use it to refuse
    a caller seed that is smaller than even this floor (the truncation bug
    the prior single-tile ``(4096,)`` seed exhibited).

    Returns 0 when the grid is unknown (no scaling claim can be made).
    """
    grid = getattr(ctx, "launch_grid", None)
    if not grid:
        return 0
    grid_prod = 1
    for ext in grid:
        try:
            grid_prod *= max(int(ext), 1)
        except Exception:
            return 0
    tile_numel = _tile_numel(val_shape)
    if tile_numel <= 0:
        return 0
    return grid_prod * tile_numel


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


def _ptrstate_block_base_and_strides(
    ctx: WalkerCtx,
    resolved: Dict[str, Any],
    out_shape: Sequence[int],
) -> Tuple[Any, List[Any]]:
    """Split the PtrState flat address into (per-block base, per-axis strides).

    :func:`_ptrstate_flat_index` collapses the whole address into ONE scalar
    per lane::

        flat = sum_axis(off[axis] + lv[axis]*stride[axis]) + trips*advance

    For a *real* 2D ``T.copy`` CopyNode we must instead keep the 2D structure
    explicit -- the global source has to be a 2D ``tir.Buffer`` whose
    ``strides=[stride0, stride1]`` preserve axis contiguity (so the lowering
    sees a contiguous innermost axis and can pick TMA/UTMALDG). The part of
    the address that does NOT depend on the per-lane tile coords ``lv`` is the
    block ``elem_offset``::

        elem_offset = sum_axis(off[axis]) + trips*advance       (constant/block)
        strides     = [stride[axis] for axis in tile]           (per-axis)

    RULE #1: the per-axis stride must resolve to a real PrimExpr (the symbolic
    arg stride). A missing stride is a bug, not a default-to-1 -- we raise
    rather than silently fabricate contiguity that the tensor does not have.
    """
    tir = ctx.tir()
    offsets = _resolve_ptrstate_values(ctx, resolved.get("offsets") or [])
    strides_in = _resolve_ptrstate_values(ctx, resolved.get("strides") or [])
    rank = len(out_shape or [])
    # Per-block base: only the loop-independent terms (off[axis]) summed,
    # scalarized exactly like _ptrstate_flat_index does (any axis-indexed
    # off is reduced to its block origin -- off carries no lv dependence).
    base: Any = tir.const(0, "int64")
    for axis in range(rank):
        off = offsets[axis] if axis < len(offsets) else tir.const(0, "int32")
        off = _scalarize_tile_index_base(ctx, off, [], out_shape)
        off = _cast_index_like(ctx, off, base)
        base = base + off
    # Loop-carried pointer advance (multi-K-trip slide) -- same math as the
    # 1D path: trips * advance, added to the block base.
    carry = resolved.get("_carry_flat") if isinstance(resolved, dict) else None
    if carry:
        advance = _carry_resolve(ctx, carry.get("advance"))
        induction = _carry_resolve(ctx, carry.get("induction"))
        lb = _carry_resolve(ctx, carry.get("lb"))
        step = _carry_resolve(ctx, carry.get("step"))
        advance = _cast_index_like(ctx, advance, base)
        induction = _cast_index_like(ctx, induction, base)
        lb = _cast_index_like(ctx, lb, base)
        step = _cast_index_like(ctx, step, base)
        trips = tir.FloorDiv(induction - lb, step)
        base = base + trips * advance
    # Per-axis strides (preserve the 2D contiguity). RULE #1: raise on a
    # missing stride rather than default to 1 (that would claim contiguity
    # the source layout does not have and silently mis-address / mis-TMA).
    strides: List[Any] = []
    for axis in range(rank):
        if axis >= len(strides_in):
            raise EmitError(
                "CopyNode conversion: PtrState has %d strides but tile rank is "
                "%d; refusing to default the missing axis stride to 1 (would "
                "fabricate contiguity). resolved.strides=%r"
                % (len(strides_in), rank, resolved.get("strides"))
            )
        stride = _scalarize_tile_index_base(ctx, strides_in[axis], [], out_shape)
        stride = _cast_index_like(ctx, stride, base)
        strides.append(stride)
    return base, strides


def _decompose_monotone_mask_extents(
    ctx: WalkerCtx,
    mask_lane: Any,
    loop_vars: Sequence[Any],
    out_shape: Sequence[int],
) -> Optional[List[Any]]:
    """Recover per-axis in-bounds extents from an ``And``-of-``LT`` tile mask.

    BOUNDSHOIST (mask_ssa form). The native ``tl.load``/``tl.store`` row/col
    mask lowers to ``mask_lane = AND_a ( lhs_a(i_a) < bound_a )`` where each
    leaf depends on EXACTLY ONE loop var ``i_a`` and ``lhs_a`` is affine in
    that var with unit coefficient (``offs_a = block_base_a + i_a``, the
    monotone iota). For such a leaf the in-bounds lane set on axis ``a`` is the
    contiguous prefix ``i_a < (bound_a - block_base_a)``, i.e. the per-axis
    extent ``bound_a - lhs_a|_{i_a=0}``.

    We split the ``And`` into leaves, attribute each to its single loop var,
    and compute the extent by substituting ``i_a := 0`` into ``lhs_a`` (TVM
    ``Substitute``) and verifying with the arithmetic analyzer that the leaf is
    exactly ``LT(i_a + c, bound)`` (unit-coeff, single-var) so the prefix is
    truly contiguous. RULE #1: return ``None`` -- so the caller keeps the
    correct per-element predicate -- unless EVERY constrained axis yields a
    verified unit-coeff monotone bound and every axis ends up covered. Never
    guess a bound: a wrong extent would drop the guard on real OOB lanes.
    """
    tir = ctx.tir()
    rank = len(loop_vars)
    if rank == 0:
        return None
    try:
        from tvm.tir import stmt_functor as _sf  # noqa: WPS433
        substitute = _sf.substitute
    except Exception:
        return None
    import tvm as _tvm  # noqa: WPS433

    # Flatten the conjunction into LT leaves. The boolean AND of the row/col
    # mask lowers either to a ``tir.And`` node OR to a ``tirx.bitwise_and``
    # Call (the bool tile comparator path); recurse through both. Bail on any
    # other node (Or, Not, a comparison we don't model) so the caller keeps
    # the per-element predicate (RULE #1).
    leaves: List[Any] = []
    stack = [mask_lane]
    while stack:
        node = stack.pop()
        if isinstance(node, tir.And):
            stack.append(node.a)
            stack.append(node.b)
        elif (
            isinstance(node, tir.Call)
            and getattr(node.op, "name", None) == "tirx.bitwise_and"
            and len(node.args) == 2
        ):
            stack.append(node.args[0])
            stack.append(node.args[1])
        elif isinstance(node, tir.LT):
            leaves.append(node)
        else:
            return None
    if not leaves:
        return None

    var_of = {v: i for i, v in enumerate(loop_vars)}
    extents: List[Any] = [None] * rank
    analyzer = _tvm.arith.Analyzer()
    affine_src = getattr(ctx, "affine_tile_source", None) or {}
    from tvm.tir import stmt_functor as _sf2  # noqa: WPS433

    def _lhs_as_affine(lhs: Any) -> Optional[Any]:
        """Return an analyzer-transparent affine form of ``lhs``, or None.

        If ``lhs`` is a bare ``BufferLoad`` of a buffer materialized from a
        recorded affine iota source, substitute the source's per-lane
        ``read_lane`` expression (``base + i``). A plain PrimExpr passes
        through unchanged. RULE #1: when the buffer is NOT a recorded affine
        source we return ``lhs`` as-is -- the unit-coeff check below then
        fails and the caller keeps the predicate (never a guessed bound).
        """
        if isinstance(lhs, tir.BufferLoad):
            src_expr = affine_src.get(getattr(lhs.buffer, "data", None))
            if src_expr is not None:
                try:
                    indices = list(lhs.indices)
                    src_rank = len(getattr(src_expr, "shape", ()) or ())
                    # An ``expand_dims`` alias indexes the SAME backing buffer
                    # with extra broadcast (size-1) axes the rank-<src_rank>
                    # source does not carry. Drop the size-1 axes of the alias
                    # so the surviving indices map onto the source's lane axes.
                    if src_rank and len(indices) > src_rank:
                        buf_shape = list(getattr(lhs.buffer, "shape", []) or [])
                        if len(buf_shape) == len(indices):
                            kept = [
                                idx
                                for idx, dim in zip(indices, buf_shape)
                                if not (
                                    isinstance(dim, tir.IntImm)
                                    and int(dim.value) == 1
                                )
                            ]
                            if len(kept) == src_rank:
                                indices = kept
                    if src_rank and len(indices) != src_rank:
                        # Could not reconcile rank -> do NOT guess; keep the
                        # opaque BufferLoad so the unit-coeff check fails and
                        # the caller retains the predicate (RULE #1).
                        return lhs
                    return src_expr.read_lane(ctx, tuple(indices))
                except Exception:
                    return None
            return lhs
        return lhs

    for leaf in leaves:
        lhs_raw, rhs = leaf.a, leaf.b
        lhs = _lhs_as_affine(lhs_raw)
        if lhs is None:
            return None
        # Which loop vars does lhs depend on? Must be exactly one.
        used_vars = set()
        _sf2.post_order_visit(
            lhs, lambda o: used_vars.add(o) if isinstance(o, tir.Var) else None
        )
        present = [var_of[v] for v in used_vars if v in var_of]
        if len(present) != 1:
            # lhs spans zero or multiple loop vars -> not a clean per-axis
            # monotone bound. Keep the predicate (RULE #1).
            return None
        axis = present[0]
        var = loop_vars[axis]
        # Verify lhs is affine unit-coefficient in var: lhs(var) - lhs(var=0)
        # must simplify to exactly `var` (a strictly-monotone unit-step iota,
        # so the in-bounds lane set is the contiguous prefix [0, rhs - lhs0)).
        lhs0 = substitute(lhs, {var: tir.const(0, var.dtype)})
        delta = analyzer.simplify(lhs - lhs0)
        if not (isinstance(delta, tir.Var) and delta.same_as(var)):
            return None
        # In-bounds extent on this axis: rhs - lhs0 (where lhs = var + lhs0).
        ext = analyzer.simplify(
            _cast_index_like(ctx, rhs, var) - _cast_index_like(ctx, lhs0, var)
        )
        if extents[axis] is not None:
            # Two leaves on the same axis -> take the tighter (min) bound.
            ext = tir.Min(extents[axis], ext)
        extents[axis] = ext
    # Any axis with no leaf is fully in-bounds (extent == tile dim).
    for axis in range(rank):
        if extents[axis] is None:
            extents[axis] = tir.const(int(out_shape[axis]), "int32")
    return extents


def _emit_oob_zero_partition(
    ctx: WalkerCtx,
    tile_buf: Any,
    loop_vars: Sequence[Any],
    out_shape: Sequence[int],
    extents: Sequence[Any],
    other_lane: Any,
) -> bool:
    """Emit the masked-OOB zero-fill as axis-partitioned UNGUARDED loops.

    BOUNDSHOIST (the corrected, evidence-based dominant move). The previous
    epilogue wrapped the WHOLE ``BLK_m x BLK_k`` tile in a per-element
    predicate ``if (i0>=ext0 | i1>=ext1) tile[idx]=0`` -- which native Triton
    never does. For a fully-aligned tile (``ext_axis == BLK_axis``, the §P1
    hd=ds=64 case and every tile where the dim divides the block) the
    predicate is ALWAYS FALSE yet was evaluated ``BLK_m*BLK_k`` times per
    thread per K-iter -- the dominant contributor to the 9.41B
    ``op_integer_pred_on`` saturating L1TEX at 87% SOL.

    We instead PARTITION the OOB set ``{ANY axis i_a >= ext_a}`` into disjoint
    per-axis regions and emit each as a plain clamped ``tir.For`` nest with NO
    inner predicate -- the loop BOUNDS encode the partition:

      region[a] = { i_a in [clamp(ext_a), BLK_a)  (the OOB suffix of axis a)
                    AND i_b in [0, clamp(ext_b))  for every earlier axis b<a
                    AND i_c in [0, BLK_c)         for every later axis c>a }

    Each OOB cell has a UNIQUE smallest-OOB axis, so the regions are disjoint
    and cover the OOB set EXACTLY (a cell fully in-bounds lands in no region).
    For a full tile ``clamp(ext_a) == BLK_a`` -> the axis-``a`` suffix
    ``[BLK_a, BLK_a)`` is EMPTY -> TVM emits no loop body and ptxas drops it:
    ZERO predicate, ZERO iterations. For a genuine partial trailing tile only
    the real OOB cells iterate, UNGUARDED. RULE #1: writes ``other`` (0) to
    EXACTLY the OOB lanes -- bit-identical to the predicated epilogue, never a
    maskless in-bounds overwrite and never an OOB write past the tile.
    """
    tir = ctx.tir()
    rank = len(loop_vars)
    if rank == 0 or len(extents) != rank:
        return False
    zero = tir.const(0, "int32")

    def _clamp(ext: Any, axis: int) -> Any:
        # clamp(ext, 0, BLK_axis) -- a serial For with start>stop is a no-op,
        # but we clamp so the suffix start never exceeds the tile extent (a
        # dim_axis - block_base that overshoots BLK on a non-trailing tile) and
        # the prefix stop is a valid [0, BLK] bound.
        blk = tir.const(int(out_shape[axis]), "int32")
        ext_i = _cast_index_like(ctx, ext, zero)
        return tir.Max(zero, tir.Min(ext_i, blk))

    clamped = [_clamp(extents[a], a) for a in range(rank)]
    blk = [tir.const(int(out_shape[a]), "int32") for a in range(rank)]
    emitted_any = False
    for a in range(rank):
        # Build the per-axis [min, stop) range for partition region[a].
        # axis a: OOB suffix [clamped_a, BLK_a); earlier axes: in-bounds prefix
        # [0, clamped_b); later axes: full [0, BLK_c). ``tir.For`` takes
        # (loop_var, min, EXTENT, kind, body) where extent = stop - min and the
        # loop var iterates [min, min+extent); a non-positive extent makes the
        # body unreachable (the full-tile no-op).
        #
        # Mint FRESH per-region loop vars: each region constrains a given axis
        # to a DIFFERENT const-int range ([clamped, BLK) in its own region,
        # [0, clamped) as an earlier axis elsewhere). Reusing one Var across
        # regions trips the analyzer's single-const-bound-per-Var invariant.
        region_vars = [
            tir.Var(ctx.fresh(f"oobz{a}_{axis}"), "int32")
            for axis in range(rank)
        ]
        # Index the tile with the per-region loop vars (same flat indexing as
        # the predicated epilogue; any swizzle layout is applied by the later
        # TVM layout pass, not here).
        store = tir.BufferStore(tile_buf, other_lane, list(region_vars))
        body: Any = store
        for axis in range(rank - 1, -1, -1):
            if axis == a:
                lo, stop = clamped[axis], blk[axis]
            elif axis < a:
                lo, stop = zero, clamped[axis]
            else:
                lo, stop = zero, blk[axis]
            extent = stop if (lo is zero) else tir.Sub(stop, lo)
            body = tir.For(
                region_vars[axis],
                lo,
                extent,
                tir.ForKind.SERIAL,
                body,
            )
        ctx.emit(body)
        emitted_any = True
    return emitted_any


def _emit_ptrstate_tile_load_copynode(
    op: Any,
    ctx: WalkerCtx,
    src_buf: Any,
    resolved: Dict[str, Any],
    out_shape: Sequence[int],
    out_dtype: str,
    mask_ssa: Any,
    other_ssa: Any,
    dynamic_mask_dims: Sequence[Any],
    tile_buf: Any,
    loop_vars: Sequence[Any],
) -> bool:
    """Emit a REAL 2D-strided ``T.copy`` CopyNode for the K-loop producer.

    Iteration-4 (CopyNode conversion). The 1D-flattened predicated ``tir.For``
    (``shared[i,j] = if_then_else(mask, global[flat_idx], other)``) is NOT
    recognized as a proven copy by ``pipeline_planning.cc`` (the global read is
    hidden inside an ``if_then_else`` condition-guarded subtree, and there is
    no ``tl.tileop.copy`` CallNode for ``ParseOperator`` to match). So
    ``num_stages`` alone yields no async stage.

    Here we instead:
      (a) declare the global source as a 2D ``tir.Buffer`` aliasing the flat
          arg's backing ``data`` Var, with ``strides=[stride0, stride1]`` (the
          symbolic per-axis arg strides) and ``elem_offset`` = the per-block
          flat base -- so the 2D axis-contiguity (innermost stride) survives;
      (b) emit ``T.copy(src2d, shared_tile)`` as a real ``tl.tileop.copy``
          CallNode (recognized as a CopyNode by ParseOperator/PipelinePlanning);
      (c) SPLIT the bounds mask OFF the copy into a separate masked epilogue
          (zero the out-of-bounds shared lanes AFTER the bulk copy) instead of
          guarding the producer with ``if_then_else``.

    Returns True on success (CopyNode emitted). RULE #1: this is gated to the
    routed-triton async path; on any structural mismatch it RAISES (no silent
    fallback to a predicated-For dressed up as a coalesced copy).
    """
    tir = ctx.tir()
    try:
        import tilelang.language as T  # type: ignore
    except Exception as exc:  # pragma: no cover - TileLang must be importable here
        raise EmitError(
            "CopyNode conversion requires tilelang.language (T.copy) to build a "
            "real CopyNode; import failed: %r" % (exc,)
        )
    base, strides = _ptrstate_block_base_and_strides(ctx, resolved, out_shape)
    data_var = getattr(src_buf, "data", None)
    if data_var is None:
        raise EmitError(
            "CopyNode conversion: flat source buffer %r has no backing data Var "
            "to alias into a 2D strided view" % (getattr(src_buf, "name", "?"),)
        )

    # ---- TMA-eligibility gate (RULE #1) ---------------------------------
    # Once this CopyNode is scheduled as a pipeline producer, the CUDA copy
    # lowering classifies a global->shared LOAD as a TMA bulk load and the
    # bulk path HARD-REQUIRES a statically-provable contiguous innermost
    # global stride (``ICHECK(is_one(desc.global_stride[0]))`` in
    # src/backend/cuda/op/copy.cc). ``desc.global_stride`` is the REVERSED
    # buffer strides, so the gate is on the LAST element of ``strides`` (the
    # innermost tile axis). We must NOT emit a CopyNode that the core will
    # then FATAL on -- and we must NOT silently fabricate contiguity. So:
    #   * if the innermost stride is provably 1 -> TMA-eligible, emit;
    #   * elif the route has GROUND-TRUTH-verified the innermost is contiguous
    #     (``ctx.routed_contiguous_innermost``, set only when the caller knows
    #     the real tensor's innermost stride == 1) -> substitute a LITERAL 1
    #     for that axis so the static TMA descriptor is provable (genuine
    #     UTMALDG, not a guess);
    #   * else -> RAISE the exact reason (non-contiguous / opaque-symbolic
    #     innermost; TMA needs the innermost axis contiguous). No predicated-
    #     For dressed up as a coalesced copy.
    if not strides:
        raise EmitError(
            "CopyNode conversion: empty strides for tile %r (rank %d)"
            % (getattr(src_buf, "name", "?"), len(out_shape or []))
        )
    inner = strides[-1]
    try:
        analyzer = ctx.tvm().arith.Analyzer()
        inner_is_one = bool(analyzer.can_prove(inner == tir.const(1, inner.dtype)))
    except Exception:
        inner_is_one = False
    # ITERATION 6 (C-tile executed TMA). Per-SOURCE ground-truth contiguity:
    # the route supplies the set of producer-load source pointers (TTIR
    # ``%argN``) whose INNERMOST tile axis is PROVABLY CONTIGUOUS (global
    # stride == 1) on the real tensor (``ctx.routed_contiguous_innermost_sources``;
    # set in __init__ ONLY for the dstates C (%arg1) tile, never for dout). For
    # such a source the de-monomorphized launch passes the innermost stride as
    # an opaque symbolic arg so the analyzer cannot prove ``== 1`` -- we GROUND
    # it to a literal IntImm(1) so ``copy.cc:988 ICHECK(is_one(...))`` passes
    # and the CopyNode lowers to a REAL TMA (UTMALDG) load. RULE #1: gated to
    # the verified-contiguous source ONLY; a non-contiguous innermost is NEVER
    # grounded.
    ground_innermost = False
    _ground_srcs = getattr(ctx, "routed_contiguous_innermost_sources", None)
    if _ground_srcs:
        _src_name = resolved.get("source") if isinstance(resolved, dict) else None
        if _src_name is not None and str(_src_name) in _ground_srcs:
            ground_innermost = True
    # ASYNCIMPL (generic non-innermost coalesce): the route may verify that a
    # source has a CONTIGUOUS (global stride == 1) tile axis that is NOT the
    # innermost one -- e.g. the dstates dout tile ``[hd, k]`` has ``hd`` (axis 0)
    # contiguous while the innermost ``k`` (seq) axis has stride nheads*headdim.
    # ``routed_contiguous_tile_axis`` (set in __init__ from the real row-major
    # layout) maps the source SSA -> that contiguous axis. A SIMT cp.async copy
    # (the disable_tma path we ALWAYS take on GB10/consumer-Blackwell) does NOT
    # need the contiguous axis to be innermost: ``CopyNode::MakeSIMTLoop`` builds
    # a fully PARALLEL (thread-distributed) loop nest over ALL axes and
    # LayoutInference + InjectPTXAsyncCopy vectorize/coalesce the cp.async over
    # whichever axis has stride 1. So we ground the route-verified contiguous
    # axis to a literal 1 (it genuinely IS 1 on the real tensor; the
    # de-monomorphized launch only PASSES it as an opaque symbolic arg) and emit
    # the CopyNode -> the dout load becomes a COOPERATIVE cp.async load instead
    # of a serial per-lane predicated fill. RULE #1: only a route-VERIFIED
    # contiguous axis is grounded; a genuinely non-contiguous axis is never
    # pinned. The dst shared tile + GEMM operand keep their [hd, k] logical
    # layout (no transpose) -- only the cp.async vectorization axis changes.
    _contig_axis_map = getattr(ctx, "routed_contiguous_tile_axis", None)
    _contig_axis = None
    if _contig_axis_map:
        _src_name2 = resolved.get("source") if isinstance(resolved, dict) else None
        if _src_name2 is not None:
            _contig_axis = _contig_axis_map.get(str(_src_name2))
    if not inner_is_one:
        if ground_innermost:
            # Ground-truth contiguity supplied by the route: pin the innermost
            # stride to a literal 1 so the SIMT cp.async vectorizes over it.
            # This is the axis the route verified contiguous on the real tensor
            # -- grounded, not fabricated.
            strides = list(strides[:-1]) + [tir.const(1, "int64")]
            inner_is_one = True
        elif _contig_axis is not None and 0 <= int(_contig_axis) < len(strides):
            # Route-verified contiguous NON-innermost axis: ground THAT axis to
            # a literal 1 so the cp.async coalesces over it. The CopyNode then
            # lowers via MakeSIMTLoop (parallel, thread-distributed) -> genuine
            # cooperative cp.async over the contiguous axis (dout [hd,k]: hd).
            _ax = int(_contig_axis)
            strides = [
                (tir.const(1, "int64") if i == _ax else s)
                for i, s in enumerate(strides)
            ]
        else:
            # NOT cp.async-eligible: no route-verified contiguous axis at all.
            # Keep this load on the existing (non-pipelined) predicated-For path
            # rather than emit a CopyNode with no coalescible axis. RULE #1: no
            # coalesced-copy claim for a fully non-contiguous load. The exact
            # reason is surfaced via the return value so it is visible end-to-end.
            return False
    src_name = (getattr(src_buf, "name", None) or "src") + "_2d"
    # 2D (rank-N) strided view over the SAME global allocation. The strides
    # carry the symbolic arg strides; the per-block base must reach the LOWERED
    # address. Aliasing src_buf.data keeps MakePackedAPI's bound param handle --
    # no new free Var. scope is global (inherit from the flat arg buffer).
    # Scope: the flat arg buffer is a global function param; mirror it so the
    # 2D view aliases global memory (the copy is global->shared).
    try:
        src_scope = src_buf.scope()
    except Exception:
        src_scope = "global"
    # ITERATION 9 (cp.async elem_offset DROP fix). MEASURED root cause of the
    # §P1 cp.async MAXDIFF 1.28e3: a ``decl_buffer`` whose ``elem_offset`` is a
    # SYMBOLIC per-block base and whose ``data`` ALIASES another buffer's Var
    # has its ``elem_offset`` DROPPED by the CopyNode SIMT lowering
    # (CopyNode::MakeSIMTLoop -> BufferLoad -> LowerParallelLoop/tvm_access_ptr):
    # the lowered cp.async source address is ``threadIdx*..+i*..*stride`` with NO
    # per-block base term, so every CTA reads block 0's slice -> wrong GEMM
    # operand (verified by dumping the lowered TIR: ``arg1_1.data`` offset omits
    # the ``pid*arg16 + ...`` base).
    #
    # Folding ``base`` into the innermost REGION MIN does flow through
    # ``MakeIndices``/``OffsetOf`` into the address, BUT it ALSO makes
    # ``CopyNode::MakePredicate`` emit a bounds check ``base + iv < shape`` that
    # is almost always false (base >> shape) -> the copy becomes a
    # ``cp_async_gs_conditional`` that skips nearly every element (MEASURED:
    # output nonzero collapses to 57344/29360128). So the region-min channel is
    # WRONG for a flat per-block base.
    #
    # The correct carrier is ``elem_offset`` -- which is structurally an offset,
    # NOT a coordinate, so it does NOT feed MakePredicate -- combined with the
    # core fix (src/op/copy.cc MakeSIMTLoop) that re-attaches ``src->elem_offset``
    # to the BufferLoad's flattened address so the SIMT-loop lowering stops
    # dropping it. RULE #1: exact per-block base, no fabricated contiguity, no
    # spurious OOB predicate.
    src2d = tir.decl_buffer(
        list(out_shape),
        out_dtype,
        name=ctx.fresh(src_name),
        data=data_var,
        strides=list(strides),
        elem_offset=base,
        scope=src_scope or "global",
    )
    src2d_region = src2d
    # Real CopyNode: T.copy(src2d -> shared_tile). The source carries the
    # per-block base in elem_offset; copy_op encodes tl.region.
    #
    # TMA-eligibility (RULE #1, measured on sm_121): the CUDA bulk-copy
    # (UTMALDG) lowering REQUIRES a statically-provable contiguous innermost
    # GLOBAL stride -- ``ICHECK(is_one(desc.global_stride[0]))`` in
    # src/backend/cuda/op/copy.cc. Our per-axis strides are SYMBOLIC runtime
    # args (the de-monomorphized kernel passes ``stride_*`` as params), so the
    # analyzer can neither prove ``== 1`` (-> the bulk lowering ICHECK FATALs)
    # nor, for the dout tile, is the innermost axis even contiguous at runtime
    # (dout tile is ``[hd, k]`` -> innermost = K-reduction axis, stride
    # ``nheads*headdim`` = 7168, genuinely != 1; only C's ``[k, ds]`` tile has
    # a runtime-contiguous innermost, but still opaque-symbolic so unprovable).
    # We therefore set ``disable_tma`` so the CopyNode lowers via the SIMT /
    # cp.async coalesced path instead of FATAL-ing on the unprovable TMA
    # descriptor. It is STILL a real CopyNode -> PipelinePlanning schedules it
    # as an async producer; we do NOT fabricate a TMA claim the layout can't
    # honor. If a future monomorphized launch surfaces a literal-1 innermost
    # stride, dropping ``disable_tma`` lets C take UTMALDG.
    #
    # ITERATION 7 (SIDESTEP TMA -- SIMT COALESCED): the sm_90 TMA template
    # (cp.async.bulk.tensor.2d) does NOT execute on sm_121 (Blackwell GB10):
    # iter-6 got C to lower to a real UTMALDG and the kernel LAUNCHED, but the
    # TMA FAULTED at runtime (compute-sanitizer: illegal instruction at
    # tl::tma_load copy_sm90.h:96). So TMA is a dead end on this target. We now
    # route the C AND dout K-loop global->shared loads as SIMT cooperative
    # VECTORIZED COALESCED loads (plain LDG.E.128, thread-distributed over the
    # contiguous axis) -- no tensorMap, no cp.async.bulk.tensor, no sm_90 TMA
    # template. This avoids the sm_121 TMA fault entirely.
    #
    # The lever: set ``disable_tma=True`` on the CopyNode so Copy::Lower
    # (src/backend/cuda/op/copy.cc) takes the SIMT branch (vectorized LDG) via
    # SelectCopyInstForLowering instead of LowerBulk (TMA). This annotation now
    # actually reaches the CopyNode because the per-Call annotations channel was
    # restored (3rdparty/tvm tirx::Call 5-arg ctor + call_intrin; previously the
    # kwarg was silently dropped by ``del annotations`` in tvm/tirx/op.py).
    #
    # ``ground_innermost`` (the route-verified contiguous innermost stride for
    # the dstates C tile) is STILL pinned to a literal 1 above: that contiguity
    # is what lets the SIMT vectorizer prove a 128-bit coalesced LDG.E.128 over
    # the ds axis (stride 1). It is no longer used to enable TMA. For the dout
    # tile and any un-grounded source the same disable_tma=True applies. RULE #1:
    # one clear SIMT path; no fabricated TMA claim, no faulting bulk copy.
    #
    # ITERATION 8 (sm_121f ArchFix re-test, env-gated): the ArchFix commit makes
    # CC 12.x compile to the sm_121f FAMILY arch, under which cp.async.bulk.tensor
    # TMA is validly enabled. When ``TL_TMA_ROUTE=1`` AND the route grounded this
    # source's innermost stride to a literal 1 (the provably-contiguous dstates C
    # tile, ``ground_innermost``), we DROP ``disable_tma`` and let the CopyNode
    # lower to a real UTMALDG TMA load -- the iter-6 behavior -- to MEASURE whether
    # the C-tile TMA now EXECUTES (no copy_sm90.h:96 fault) under sm_121f and
    # whether the coalesced TMA load drops ms. Default (env unset) keeps the
    # committed SIMT route. RULE #1: env-gated experiment, no silent change.
    import os as _os
    _tma_route = _os.environ.get("TL_TMA_ROUTE") == "1"
    # ARCH GATE (measured): cp.async.bulk.tensor TMA works on Hopper (sm_90) but
    # FAULTS on GB10 / consumer-Blackwell sm_120/sm_121 (cudaErrorIllegalInstruction
    # at copy_sm90.h:96, confirmed under sm_121a/sm_121f and a CTA non-cluster
    # barrier). Native Triton uses cp.async/LDGSTS on GB10 (measured: 75 LDGSTS,
    # 0 UTMALDG). So: keep TMA on HOPPER (it works there); force cp.async
    # (disable_tma) on sm_120/sm_121 and by default everywhere else.
    _is_hopper = False
    try:
        from tvm.target import Target as _Tgt
        from tilelang.contrib.nvcc import (
            get_target_compute_version as _gtcv,
            parse_compute_version as _pcv,
        )
        _cur = _Tgt.current(allow_none=True)
        if _cur is not None:
            _mj, _mn = _pcv(_gtcv(_cur))
            _is_hopper = (_mj == 9)
    except Exception:
        _is_hopper = False
    if _os.environ.get("TL_TMA_DEBUG") == "1":
        _sn = resolved.get("source") if isinstance(resolved, dict) else None
        import sys as _sys
        print("TMA_DEBUG src=%r tma_route=%s ground_innermost=%s inner_is_one=%s is_hopper=%s ground_srcs=%r" % (
            _sn, _tma_route, ground_innermost, inner_is_one, _is_hopper,
            getattr(ctx, "routed_contiguous_innermost_sources", None)), file=_sys.stderr, flush=True)
    if _is_hopper and _tma_route and ground_innermost:
        # Hopper: real cp.async.bulk.tensor TMA load (works on sm_90).
        copy_call = T.copy(src2d_region, tile_buf)
    else:
        # GB10 / consumer-Blackwell + default: cp.async / LDGSTS coalesced async
        # copy (the path native Triton uses on GB10), NOT the faulting bulk TMA.
        #
        # ASYNCIMPL lever (c): ``disable_tma`` alone routes the CopyNode through
        # SelectCopyInstForLowering -> SelectSyncLikeInst -> kNormal == PLAIN
        # synchronous LDG (measured: 66-96 LDG, 0 LDGSTS in SASS). This is the
        # CORRECT, parity-clean default (§P1 MAXDIFF 4.88e-04): a synchronous LDG
        # needs no async wait-barrier so the masked-OOB epilogue + GEMM read valid
        # data unconditionally.
        #
        # The ``is_async_copy`` annotation (GetIsAsyncCopy, copy.cc:95) makes
        # ``facts.explicit_cp_async`` true so SelectCopyInstForLowering picks
        # CopyInst::kCPAsync (copy_analysis.cc:617-619) -> Copy::LowerCPAsync emits
        # GENUINE cp.async/LDGSTS (cp.async.cg.shared.global, like native Triton).
        #
        # RACE-FREENESS (RULE #1 -- correct by construction, MEASURED bit-correct):
        # Copy::LowerCPAsync (src/backend/cuda/op/copy.cc:527) closes the async
        # group OUT-OF-LINE for an ``is_async_copy`` copy that is NOT pipeline-
        # managed: it emits ``cp.async`` + ``cp.async.commit_group`` +
        # ``cp.async.wait_group<0>`` + a CTA ``__syncthreads`` so every lane has
        # LANDED before the GEMM consumer (or the masked-OOB epilogue) reads the
        # shared tile. There is NO race: the wait_group<0> drains ALL outstanding
        # async copies and the barrier publishes them CTA-wide. MEASURED §P1
        # MAXDIFF = 4.882812e-04 (2^-11, the bit-correct target) with this exact
        # path (cp_async>0 in the generated .cu, HMMA intact). The stale earlier
        # note about a "FAST-but-RACY 1.28e+03" path predated the copy.cc:527
        # self-contained commit/wait and the num_stages drop below; it no longer
        # applies. We therefore make ``is_async_copy`` the COMMITTED DEFAULT (not
        # env-gated) so the routed global->shared K-loop load lands on the genuine
        # cp.async hardware path like native. ``TL_NO_ASYNC_COPY=1`` reverts to a
        # plain synchronous LDG for A/B measurement only.
        #
        # IMPORTANT: because this copy carries its OWN commit/wait, the enclosing
        # K-loop MUST NOT also be stamped with ``num_stages`` (that would route it
        # through the software pipeline whose overlapping-write check FATALs on the
        # masked-OOB epilogue, a legitimate second writer to the SAME shared tile).
        # We record the async-explicit emission on the ctx so the K-loop emitter
        # (control.py ``_maybe_pipeline``) skips the num_stages stamp. RULE #1: one
        # clear async path; the wait is always present, never a silent half-state.
        import os as _os_async
        if _os_async.environ.get("TL_NO_ASYNC_COPY") == "1":
            copy_call = T.copy(src2d_region, tile_buf, disable_tma=True)
        elif _os_async.environ.get("TL_PIPELINE_CP_ASYNC") == "1":
            # PIPELINENS4 (TRUE multi-stage software pipeline): emit a PLAIN
            # global->shared CopyNode with NO ``is_async_copy`` annotation. Because
            # GetIsAsyncCopy is false, CheckPipelineManagedCPAsyncCopy
            # (inject_pipeline.cc:83) returns TRUE -> InjectSoftwarePipeline takes
            # OWNERSHIP of this copy: it multi-versions (double/quad-buffers) the
            # shared tile across ``num_stages`` and schedules the cp.async
            # commit_group + wait_group<N> (N = num_stages-2 in-flight) so stage
            # k+1 loads OVERLAP stage k MMA -- exactly like native ns4. We do NOT
            # set ``routed_explicit_cp_async`` (the copy is NOT self-contained);
            # ``_maybe_pipeline`` (control.py) STAMPS ``num_stages`` on the K-loop
            # so PipelinePlanning schedules the pipeline. Race-freeness is
            # guaranteed by the pipeline-inserted wait_group<N> ordering (the
            # GEMM consumer waits for its producer stage). RULE #1: the
            # masked-OOB epilogue is a SECOND writer to the SAME private per-stage
            # shared tile in the SAME iteration -- not a cross-stage overlap;
            # pipeline_planning.cc treats it as part of the producer stage (see
            # the AnalyzeCopyLastUse same-buffer-consumer carve-out).
            copy_call = T.copy(src2d_region, tile_buf, disable_tma=True)
        else:
            copy_call = T.copy(
                src2d_region,
                tile_buf,
                disable_tma=True,
                annotations={"is_async_copy": 1},
            )
            # Signal the K-loop emitter to NOT stamp num_stages (the copy is
            # self-contained async; pipeline planning would FATAL on the
            # masked-OOB second writer). Generic + backend-agnostic flag.
            try:
                setattr(ctx, "routed_explicit_cp_async", True)
            except Exception:
                pass
    ctx.emit(tir.Evaluate(copy_call))

    # ---- mask SPLIT OFF the copy: masked epilogue -----------------------
    # The bounds mask is NOT inside the producer (that would re-hide the read
    # and defeat CopyNode recognition). After the bulk copy populates the
    # shared tile, overwrite the out-of-bounds lanes with ``other`` (default
    # 0) in a separate predicated For. In-bounds lanes keep the copied value.
    pred = None
    pred_from_mask_ssa = False
    other_lane: Any = tir.const(0, out_dtype)
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
            # Epilogue writes ``other`` where the lane is OUT of bounds:
            # predicate is the NEGATION of the in-bounds mask.
            pred = tir.Not(mask_lane)
            pred_from_mask_ssa = True
    else:
        mask_lane = None
    dyn = _dynamic_tts_mask_expr(ctx, loop_vars, dynamic_mask_dims)
    if dyn is not None:
        dyn_oob = tir.Not(dyn)
        pred = dyn_oob if pred is None else tir.Or(pred, dyn_oob)
    if pred is not None:
        import os as _os_epi
        _idx = list(loop_vars) or [tir.const(0, "int32")]
        # BOUNDSHOIST: the masked-OOB epilogue zeroes the lanes OUTSIDE the
        # native row/col mask. That in-bounds predicate is an ``AND`` of
        # per-axis monotone bounds ``offs_axis < dim_axis`` (the ``mask_ssa``
        # ``tl.load`` mask) and/or the clamped ``dynamic_mask_dims`` -- BOTH a
        # contiguous per-axis prefix. We DECOMPOSE the combined in-bounds
        # predicate into one extent per axis and emit the zero-fill as
        # axis-partitioned UNGUARDED loops over the OOB suffix, instead of the
        # per-element predicate rebuild native Triton never does. For a fully-
        # aligned tile (the §P1 hd=ds=64 case) every partition loop has an
        # EMPTY range -> zero predicate, zero iterations (collapses the
        # dominant 9.41B op_integer_pred_on saturating L1TEX at 87% SOL). The
        # pipelined path keeps the read-modify-write consumer form (it must
        # READ tile_buf for the async wait ordering), so partitioning is gated
        # to the non-pipelined default. RULE #1: writes 0 to EXACTLY the OOB
        # lanes, bit-identical to the predicate; falls back to the predicated
        # loop whenever the per-axis extents are not provably recoverable
        # (never a maskless overwrite, never an OOB write).
        if (
            _os_epi.environ.get("TL_PIPELINE_CP_ASYNC") != "1"
            and _os_epi.environ.get("TL_NO_BOUNDS_HOIST") != "1"
        ):
            # Combined in-bounds predicate = mask_lane AND dyn (whichever
            # present). Decompose it into per-axis contiguous-prefix extents.
            in_bounds = None
            if mask_lane is not None:
                in_bounds = mask_lane
            if dyn is not None:
                in_bounds = dyn if in_bounds is None else tir.And(in_bounds, dyn)
            extents = None
            if in_bounds is not None:
                extents = _decompose_monotone_mask_extents(
                    ctx, in_bounds, loop_vars, out_shape
                )
            if _os_epi.environ.get("TL_BOUNDSHOIST_DEBUG") == "1":
                import sys as _sys
                print("BOUNDSHOIST epilogue rank=%d mask_ssa=%s dyn=%s "
                      "partitioned=%s" % (
                          len(loop_vars), mask_lane is not None,
                          dyn is not None, extents is not None),
                      file=_sys.stderr, flush=True)
            if extents is not None and _emit_oob_zero_partition(
                ctx, tile_buf, loop_vars, out_shape, extents, other_lane,
            ):
                return True
        if _os_epi.environ.get("TL_PIPELINE_CP_ASYNC") == "1":
            # PIPELINENS4 race-free masked fixup: emit the OOB-zeroing as an
            # UNCONDITIONAL read-modify-write -- tile_buf[idx] = select(oob,
            # other, tile_buf[idx]). Because this READS tile_buf (the cp.async
            # producer's output), InjectSoftwarePipeline classifies it as a
            # CONSUMER of the async group and inserts cp.async.wait_group<N>
            # BEFORE it. The masked fixup therefore runs only AFTER its stage's
            # async load has LANDED -> no WAW race with the in-flight cp.async
            # (RULE #1: race-free by the pipeline's wait ordering, not a hope).
            # In-bounds lanes copy the loaded value back (a no-op); OOB lanes get
            # other -- identical result to the predicated store, but now a
            # genuine read-after-async-load consumer.
            cur = tir.BufferLoad(tile_buf, list(_idx))
            store = tir.BufferStore(
                tile_buf,
                tir.Select(pred, other_lane, cur),
                list(_idx),
            )
            epi: Any = store
        else:
            store = tir.BufferStore(tile_buf, other_lane, list(_idx))
            epi = tir.IfThenElse(pred, store, None)
        for axis in range(len(loop_vars) - 1, -1, -1):
            epi = tir.For(
                loop_vars[axis],
                tir.const(0, "int32"),
                tir.const(int(out_shape[axis]), "int32"),
                tir.ForKind.SERIAL,
                epi,
            )
        ctx.emit(epi)
    return True


# Op-name prefixes that consume a tile ELEMENTWISE (the value flows through
# unchanged in layout: a per-lane arithmetic / math / cast / compare). A rank>=2
# load consumed ONLY by these is re-staged into a downstream gemm-operand or
# store tile, so its own shared staging is redundant (see the A-staging collapse
# in ``_emit_ptrstate_tile_load_tir``).
_ELEMENTWISE_CONSUMER_PREFIXES = (
    "arith.",
    "math.",
    "tt.mulhiui",
    "tt.fp_to_fp",
    "tt.bitcast",
    "tt.int_to_ptr",
    "tt.precise_",
)
# Consumers for which the load tile MUST stay shared (they read it cooperatively
# in a layout-sensitive way, or directly off a function buffer): a gemm operand,
# a store source, an atomic, a transpose/reduce/broadcast view, another load's
# pointer base, etc. Fail-closed: ANY consumer not provably elementwise keeps
# the load in shared (RULE #1 -- never silently demote a layout-sensitive tile).


def _rank2_load_feeds_only_elementwise(ctx: WalkerCtx, result_value: Any) -> bool:
    """True iff EVERY consumer of ``result_value`` is a per-lane elementwise op.

    Used to demote a redundant rank>=2 load tile from ``shared`` to ``local``:
    when the load feeds only elementwise ops (e.g. ``dout`` scaled by
    ``exp(dA)``), the result is re-staged into the swizzled gemm-operand shared
    tile anyway, so the raw load needs no shared buffer of its own. Fail-closed:
    returns False (keep shared) when the use-graph is unavailable, the result is
    unnamed, the consumer set is empty, or ANY consumer is not an elementwise
    op -- never silently demote a tile a ``tt.dot`` / store / transpose reads.
    """
    ssa_users = getattr(ctx, "ssa_users", None)
    if not ssa_users:
        return False
    name = None
    try:
        getter = getattr(result_value, "get_name", None)
        name = getter() if callable(getter) else getattr(result_value, "name", None)
    except Exception:
        name = None
    if not name:
        return False
    consumers = ssa_users.get(str(name)) or ssa_users.get(str(name).lstrip("%"))
    if not consumers:
        return False
    for c in consumers:
        cs = str(c)
        if not any(cs.startswith(p) for p in _ELEMENTWISE_CONSUMER_PREFIXES):
            return False
    return True


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
    # A-STAGING COLLAPSE (RULE #1 -- correctness + shared budget, generic +
    # backend-agnostic): a rank>=2 tile load is staged to SHARED so that a
    # downstream ``tt.dot`` can read it cooperatively. But when this load feeds
    # an ELEMENTWISE op (e.g. the ``dout`` that is scaled by ``exp(dA)`` to
    # build the GEMM A operand in the SSD/dstates chunk kernels) -- and NOT a
    # ``tt.dot`` / ``tt.store`` directly -- its shared tile is REDUNDANT: the
    # elementwise result is re-staged into the swizzled gemm-operand shared tile
    # anyway (``_stage_operand_to_shared``), so the raw load lives in shared
    # only to be read once by the mul. That is the duplicate A buffer
    # (``tile_load`` raw + ``dot_a`` swizzled = 64KB) vs native's single 32KB
    # #shared operand. Stage such loads in ``local`` instead: the elementwise op
    # reads the per-thread tile (exactly as the existing ``tile_binop`` result
    # is already a per-thread local tile on this MVP path) and the SINGLE
    # swizzled gemm-operand shared tile is the only A-sized shared buffer. A
    # load that DOES feed a ``tt.dot`` directly (the B operand) keeps its shared
    # tile -- the gemm reads it directly, native single-#shared. Gated to the
    # routed prologue-opt path; every other path is byte-identical.
    load_scope = "shared" if len(out_shape or []) >= 2 else "local"
    # ASYNCIMPL: if this rank>=2 load can become a COOPERATIVE cp.async CopyNode
    # (the routed async path AND the route verifies a contiguous tile axis to
    # coalesce over -- e.g. the dstates dout tile ``[hd, k]`` with hd contiguous),
    # KEEP it in ``shared`` so it lands on the genuine cp.async hardware path
    # (cp.async.cg.shared.global, thread-distributed) instead of being demoted to
    # a ``local`` per-lane SERIAL predicated fill. Native Triton stages dout via
    # cp.async into shared; the per-lane scalar fill is the measured 3042x
    # bottleneck. The downstream ``dout*exp`` then reads the SHARED tile
    # cooperatively and stages into the swizzled GEMM operand. We accept ONE
    # extra small cooperative shared->shared stage (dout fits ~8KB, well under the
    # 101376 B cap) in exchange for replacing the serial fill with cp.async.
    # RULE #1: only taken when the load is genuinely cp.async-eligible (a
    # contiguous axis exists); otherwise the demotion below still applies.
    _async_eligible = False
    if (
        load_scope == "shared"
        and getattr(ctx, "routed_triton_async_loads", False)
    ):
        _cmap = getattr(ctx, "routed_contiguous_tile_axis", None)
        _gset = getattr(ctx, "routed_contiguous_innermost_sources", None)
        _sname = resolved.get("source") if isinstance(resolved, dict) else None
        if _sname is not None:
            if _cmap and str(_sname) in _cmap:
                _async_eligible = True
            if _gset and str(_sname) in _gset:
                _async_eligible = True
    if (
        load_scope == "shared"
        and getattr(ctx, "routed_triton_prologue_opt", False)
        and _rank2_load_feeds_only_elementwise(ctx, result_value)
        and not _async_eligible
    ):
        load_scope = "local"
    tile_buf = _alloc_tile_buffer(
        ctx,
        list(out_shape) or [1],
        out_dtype,
        ctx.fresh("tile_load"),
        scope=load_scope,
    )
    # BANKSWIZZLE (RULE #1 -- generic shared-bank-conflict elimination, CUDA-gated):
    # a rank>=2 SHARED cp.async-staged load tile that feeds ONLY elementwise ops
    # (the dout tile scaled by exp(dA) into the register-A MMA fragment) keeps a
    # PLAIN row-major shared layout: LayoutInference assigns the GEMM ldmatrix
    # swizzle only to operands handed DIRECTLY to T.gemm (the B/C tile), NOT to a
    # raw load consumed via a register-A fragment fill. The register-A fill then
    # reads this flat shared buffer at MMA-A logical positions, so per-lane LDS
    # reads collide on banks (ncu: ~14.7M shared-LD bank conflicts on tile_load
    # dout). RECORD the buffer so the post-walk FRAMEFIX SBlock pins it to
    # make_swizzled_layout (the SAME XOR swizzle T.gemm already gives the B
    # operand): both the cp.async store AND the register-A read then hit distinct
    # banks. Gated EXACTLY to the cp.async register-A source tile (shared +
    # async-eligible + feeds-only-elementwise); a B operand fed straight to
    # T.gemm already gets the swizzle and is NOT recorded here. Fail-closed: only
    # when the use-graph proves the elementwise-only feed.
    if (
        _ctx_is_cuda_target(ctx)
        and load_scope == "shared"
        and len(out_shape or []) == 2
        and _async_eligible
        and _rank2_load_feeds_only_elementwise(ctx, result_value)
    ):
        sw = getattr(ctx, "swizzle_shared_loads", None)
        if sw is None:
            sw = []
            try:
                ctx.swizzle_shared_loads = sw
            except Exception:
                sw = None
        if sw is not None:
            sw.append(tile_buf)
    loop_vars = [
        tir.Var(ctx.fresh(f"i{axis}"), "int32")
        for axis, _extent in enumerate(out_shape or [1])
    ]
    # ITERATION 4 (CopyNode conversion): on the routed-triton async path, a
    # rank>=2 global->shared producer (the K-loop dout/C tile) is emitted as a
    # REAL 2D-strided ``T.copy`` CopyNode instead of the 1D-flattened
    # predicated ``tir.For``. PipelinePlanning then recognizes it as a proven
    # copy and InjectSoftwarePipeline schedules it as an async stage ->
    # UTMALDG (TMA) on sm_121. The bounds mask is SPLIT OFF into a separate
    # masked epilogue (not inside the producer). Gated tightly; the GEMM
    # cooperative path is untouched. RULE #1: convert or RAISE -- no silent
    # predicated-For dressed up as a coalesced copy.
    if (
        getattr(ctx, "routed_triton_async_loads", False)
        and len(out_shape or []) >= 2
        and load_scope in ("shared", "shared.dyn")
    ):
        converted = _emit_ptrstate_tile_load_copynode(
            op,
            ctx,
            src_buf,
            resolved,
            out_shape,
            out_dtype,
            mask_ssa,
            other_ssa,
            dynamic_mask_dims,
            tile_buf,
            loop_vars,
        )
        if converted:
            # Real CopyNode emitted -> count it so ``map_scf_for`` stamps
            # ``num_stages`` on the enclosing K-loop (the producer is now a
            # proven copy -> InjectSoftwarePipeline schedules it async ->
            # UTMALDG on sm_121).
            shared_copies = getattr(ctx, "_gmem_shared_copies", None)
            if isinstance(shared_copies, list):
                shared_copies.append(1)
            if result_value is not None:
                ctx.bind(result_value, tile_buf)
            return tile_buf
        # NOT TMA-eligible (innermost stride not provably 1). Fall through to
        # the existing predicated-For load path WITHOUT counting it as a
        # shared async copy -- so the K-loop is NOT mis-annotated as pipelined
        # for a load that cannot become a coalesced async copy. RULE #1: no
        # silent coalesced claim.

    flat_idx = _ptrstate_flat_index(ctx, resolved, loop_vars, out_shape)
    load_expr: Any = tir.BufferLoad(src_buf, [flat_idx])
    mask_lane = None
    other_lane = tir.const(0, out_dtype)
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
    dynamic_mask = _dynamic_tts_mask_expr(ctx, loop_vars, dynamic_mask_dims)

    # DALOADBOUNDSHOIST (the two-birds load-side predicate hoist, same family as
    # the cp.async tile-load OOB-zero partition). For a rank-1 (vector) masked
    # load whose native in-bounds predicate is an AND of per-axis MONOTONE
    # bounds ``offs_a < dim_a`` (the dstates dA decay load: ``offs_k <
    # chunk_size - i_132*32``), the per-element ``if_then_else(mask, load,
    # other)`` forces the CUDA codegen to ALSO wrap the symbolic-extent flat
    # ``BufferLoad`` in its own ``(0 <= flat) && (flat < numel)`` safe-index
    # guard -- so every lane recomputes the full int64 flat base THREE times
    # (lower bound, upper bound, and the index) under a predicate. We instead
    # PARTITION the load into (1) an UNGUARDED ``T.For`` over the in-bounds
    # contiguous prefix ``[0, clamp(ext_a))`` that reads ``load_expr`` directly
    # (no per-element mask, so the codegen drops the numel safe-index guard --
    # the index is now a small affine of bounded loop vars over the dense
    # prefix) and (2) the OOB-zero suffix via the existing
    # ``_emit_oob_zero_partition``. For the §P1 full tile (clamp == BLK) the
    # suffix is EMPTY and the prefix is the whole vector -- bit-identical to the
    # masked load, fewer integer-pred ops, fewer live predicate/index registers.
    # Gated to the non-pipelined routed path (the pipelined consumer must keep
    # the read-modify-write form); falls back to the predicated load whenever
    # the per-axis extents are not provably recoverable (RULE #1: never a
    # maskless OOB read, never a dropped guard on a real OOB lane).
    _hoisted = False
    import os as _os_da
    if (
        _os_da.environ.get("TL_PIPELINE_CP_ASYNC") != "1"
        and _os_da.environ.get("TL_NO_BOUNDS_HOIST") != "1"
        and _os_da.environ.get("TL_NO_DA_BOUNDS_HOIST") != "1"
        and (mask_lane is not None or dynamic_mask is not None)
    ):
        in_bounds = None
        if mask_lane is not None:
            in_bounds = mask_lane
        if dynamic_mask is not None:
            in_bounds = dynamic_mask if in_bounds is None else tir.And(in_bounds, dynamic_mask)
        extents = None
        if in_bounds is not None:
            extents = _decompose_monotone_mask_extents(
                ctx, in_bounds, loop_vars, out_shape
            )
        if _os_da.environ.get("TL_BOUNDSHOIST_DEBUG") == "1":
            import sys as _sys
            print("DALOADBOUNDSHOIST rank=%d mask=%s dyn=%s partitioned=%s" % (
                len(loop_vars), mask_lane is not None,
                dynamic_mask is not None, extents is not None),
                file=_sys.stderr, flush=True)
        if extents is not None and len(extents) == len(loop_vars):
            zero = tir.const(0, "int32")
            clamped = [
                tir.Max(
                    zero,
                    tir.Min(
                        _cast_index_like(ctx, extents[a], zero),
                        tir.const(int(out_shape[a]), "int32"),
                    ),
                )
                for a in range(len(loop_vars))
            ]
            # (1) UNGUARDED in-bounds prefix load (no per-element mask -> codegen
            # drops the numel safe-index guard for the dense prefix).
            prefix: Any = tir.BufferStore(
                tile_buf, load_expr, list(loop_vars) or [tir.const(0, "int32")]
            )
            for axis in range(len(loop_vars) - 1, -1, -1):
                prefix = tir.For(
                    loop_vars[axis], zero, clamped[axis],
                    tir.ForKind.SERIAL, prefix,
                )
            ctx.emit(prefix)
            # (2) OOB-zero suffix (partitioned, unguarded) -- bit-identical to
            # the masked-load ``other`` on the out-of-bounds lanes.
            _emit_oob_zero_partition(
                ctx, tile_buf, loop_vars, out_shape, clamped, other_lane,
            )
            _hoisted = True
    if _hoisted:
        if result_value is not None:
            ctx.bind(result_value, tile_buf)
        return tile_buf

    if mask_lane is not None:
        load_expr = tir.if_then_else(mask_lane, load_expr, other_lane)
    if dynamic_mask is not None:
        load_expr = tir.if_then_else(dynamic_mask, load_expr, tir.const(0, out_dtype))

    body: Any = tir.BufferStore(
        tile_buf, load_expr, list(loop_vars) or [tir.const(0, "int32")]
    )
    # ITERATION 5 (DoutTranspose / coalesced strided load). For a rank-2 tile
    # whose CONTIGUOUS global axis (stride==1 on the real tensor) is NOT the
    # innermost tile axis (e.g. the dstates ``dout`` tile ``[hd, k]``: the
    # innermost loop axis ``k`` has global stride ``nheads*headdim`` = 7168
    # while the OUTER axis ``hd`` is the contiguous stride-1 axis), the default
    # innermost-last loop nest reads global memory along the strided axis ->
    # scalar/non-coalesced LDG. Re-ORDER the LOOP NEST so the contiguous axis is
    # iterated innermost; the per-lane global address then advances by 1 element
    # across the innermost iteration -> the downstream coalescing/vectorization
    # pass can emit a coalesced (vectorized) LDG instead of a strided scalar one.
    #
    # PARITY: this changes ONLY the iteration ORDER of a full-tile
    # materialization -- every (i,j) lane stores the SAME value into the SAME
    # logical ``tile_buf[i, j]`` slot. The downstream GEMM consumes the
    # identical logical ``[hd, k]`` tile, so ``dstates`` stays BIT-correct (no
    # GEMM operand-orientation change needed; the transpose is purely in the
    # global-load traversal order, not in the logical tile layout). RULE #1:
    # gated to a GROUND-TRUTH route hint (``routed_contiguous_tile_axis``: the
    # route supplies, per source tensor, the tile axis that is contiguous on the
    # REAL tensor) -- we never reorder on fabricated/assumed contiguity.
    contig_axis = None
    hint = getattr(ctx, "routed_contiguous_tile_axis", None)
    if isinstance(hint, dict) and len(loop_vars) >= 2:
        src_name = resolved.get("source") if isinstance(resolved, dict) else None
        if src_name is not None:
            try:
                contig_axis = hint.get(str(src_name))
            except Exception:
                contig_axis = None
    loop_order = list(range(len(loop_vars)))
    if (
        contig_axis is not None
        and 0 <= int(contig_axis) < len(loop_vars)
        and int(contig_axis) != len(loop_vars) - 1
    ):
        ca = int(contig_axis)
        loop_order = [a for a in loop_order if a != ca] + [ca]
    for axis in reversed(loop_order):
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(out_shape[axis]), "int32"),
            tir.ForKind.SERIAL,
            body,
        )
    ctx.emit(body)
    # NOTE (iteration 4): reaching here on the routed async path means the
    # CopyNode conversion returned False (the tile is NOT TMA-eligible -- its
    # innermost global stride is not provably 1). We deliberately do NOT count
    # this predicated-For load on ``_gmem_shared_copies``: a manual predicated
    # ``tir.For`` is NOT a proven copy (pipeline_planning.cc), so annotating the
    # K-loop with ``num_stages`` for it would only stamp an INERT pipeline (the
    # iteration-3 finding). RULE #1: do not claim a coalesced/async pipeline for
    # a load that cannot become a real CopyNode. Eligible tiles take the
    # CopyNode early-return above and ARE counted there.
    if result_value is not None:
        ctx.bind(result_value, tile_buf)
    return tile_buf


def _frag_store_scope(val_expr: Any) -> Optional[str]:
    """Return the memory scope string of ``val_expr`` if it is a Buffer."""
    scope_fn = getattr(val_expr, "scope", None)
    if scope_fn is None:
        return None
    try:
        return scope_fn() if callable(scope_fn) else scope_fn
    except Exception:
        return None


def _emit_ptrstate_fragment_store_copynode(
    op: Any,
    ctx: WalkerCtx,
    dst_buf: Any,
    resolved: Dict[str, Any],
    frag_buf: Any,
    val_shape: Sequence[int],
    dtype: str,
    mask_ssa: Any,
    dynamic_mask_dims: Sequence[Any],
) -> Any:
    """Emit a mask-aware DIRECT ``local.fragment`` -> global ``T.copy`` store.

    REGCARRYCOLLAPSE epilogue (the §P1 dstates store). When the loop-carried
    MMA-C accumulator is bound DIRECTLY to the post-loop ``tt.store`` (no shared
    ``carry_logical`` staging tile), the value is a swizzled ``local.fragment``.
    A per-lane serial scalar store reading the fragment at LOGICAL ``[i, j]``
    indices would impose a flat layout that CONFLICTS with the tensor-core MMA
    store layout ("Get different layout for carry_tile"). Native Triton instead
    emits a single layout-aware fragment->global masked store (``tt.store(ptrs,
    acc, mask)``). We mirror that with TileLang's own layout-aware ``T.copy``:

      1. Build a 2D-strided global VIEW of the flat function-arg buffer
         (``data`` aliases the arg's backing Var, ``strides`` = the per-axis arg
         strides, ``elem_offset`` = the per-block base) -- the SAME construction
         the K-loop LOAD CopyNode uses (``_emit_ptrstate_tile_load_copynode``).
      2. ``T.copy(frag_region -> view2d_region)``: the CopyNode iterates over the
         fragment (highest scope-level) and ``InferLayout`` propagates the
         fragment's registered ``make_mma_store_layout`` to the global store
         loop, so the per-warp register tile is written DIRECTLY to global at
         the correct ``[i, j]`` positions -- the MxN fp32 shared stage and its
         re-traffic are eliminated.

    Mask-awareness (RULE #1 -- correct for general non-mult-of-64 shapes). The
    native store mask ``(offs_m < hdim) & (offs_n < dstate)`` is, per axis, the
    bound ``offs_axis < dim_axis`` with ``offs_axis = block_base + arange(BLK)``.
    Since ``offs`` is monotone from ``block_base``, the in-bounds lane set is the
    CONTIGUOUS prefix ``[0, dim_axis - block_base)`` -- exactly the clamped
    extent ``min(BLK, dim_axis - block_base)`` already carried per axis by the
    ``tts.store`` ``dynamic_mask_dims`` (verified: each dim resolves to
    ``min(max(min(BLK+base, dim), base) - base, BLK)``). We clamp the copy
    region to that per-axis extent, so the copy writes EXACTLY the in-bounds
    lanes -- provably identical to the masked store, never an out-of-bounds /
    full-tile write. For §P1 (hdim=dstate=64=BLK, single tile) the extent folds
    to a constant 64 (mask always-true); for a partial trailing tile it clamps.
    RULE #1: if the per-axis clamped extent cannot be recovered (no
    ``dynamic_mask_dims`` and no resolvable ``mask_ssa`` bound), we RAISE rather
    than emit a maskless full-tile write that is wrong for general shapes.
    """
    tir = ctx.tir()
    if len(val_shape) != 2:
        raise EmitError(
            "direct fragment->global store: expected rank-2 MMA-C tile, got "
            "shape %r" % (list(val_shape),)
        )
    # Grow the flat arg buffer to the grid-scaled floor (same contract as the
    # serial store path), then alias its backing data Var into a 2D view.
    grid_floor = _grid_scaled_store_extent(ctx, val_shape)
    min_extent = max(_flat_min_extent(val_shape), grid_floor)
    dst_buf = _redeclare_ctx_buffer_1d(ctx, dst_buf, dtype, min_extent, grow_fixed=True)
    data_var = getattr(dst_buf, "data", None)
    if data_var is None:
        raise EmitError(
            "direct fragment->global store: flat dst buffer %r has no backing "
            "data Var to alias into a 2D strided view"
            % (getattr(dst_buf, "name", "?"),)
        )
    base, strides = _ptrstate_block_base_and_strides(ctx, resolved, val_shape)
    try:
        dst_scope = dst_buf.scope()
    except Exception:
        dst_scope = "global"
    view2d = tir.decl_buffer(
        list(val_shape),
        dtype,
        name=ctx.fresh((getattr(dst_buf, "name", None) or "out") + "_2d"),
        data=data_var,
        strides=list(strides),
        elem_offset=base,
        scope=dst_scope or "global",
    )
    # Per-axis clamped in-bounds extents (== the native store mask). Prefer the
    # ``tts.store`` dynamic mask dims (each already the clamped extent); fall
    # back to a resolvable ``mask_ssa`` only if it yields per-axis bounds. RULE
    # #1: no recoverable bound -> RAISE (never a maskless full-tile store).
    extents: List[Any] = []
    if dynamic_mask_dims:
        rank = len(val_shape)
        start_axis = max(0, rank - len(dynamic_mask_dims))
        # Full extent for any leading axis the mask does not constrain.
        for axis in range(start_axis):
            extents.append(tir.const(int(val_shape[axis]), "int32"))
        for i, dim in enumerate(dynamic_mask_dims):
            axis = start_axis + i
            if axis >= rank:
                break
            dim_expr = _resolved_or_none(ctx, dim)
            if dim_expr is None:
                dim_expr = _coerce_index_scalar(ctx, dim)
            # Each dynamic dim is the clamped in-bounds extent for this axis.
            ext = _cast_index_like(ctx, dim_expr, tir.const(0, "int32"))
            extents.append(ext)
    if len(extents) != len(val_shape):
        raise EmitError(
            "direct fragment->global store: could not recover a per-axis "
            "in-bounds extent for every tile axis (got %d of %d); refusing to "
            "emit a maskless full-tile store that is wrong for non-multiple-of-"
            "block shapes. dynamic_mask_dims=%r mask_ssa=%r"
            % (len(extents), len(val_shape), dynamic_mask_dims, mask_ssa)
        )
    import tilelang.language as T  # type: ignore

    tvm_mod = ctx.tvm()
    zero = tir.const(0, "int32")
    src_ranges = [
        tvm_mod.ir.Range.from_min_extent(zero, ext) for ext in extents
    ]
    dst_ranges = [
        tvm_mod.ir.Range.from_min_extent(zero, ext) for ext in extents
    ]
    src_region = tvm_mod.tir.BufferRegion(frag_buf, src_ranges)
    dst_region = tvm_mod.tir.BufferRegion(view2d, dst_ranges)
    copy_handle = T.copy(src_region, dst_region)
    if isinstance(copy_handle, tir.PrimExpr):
        stmt = tir.Evaluate(copy_handle)
        ctx.emit(stmt)
        return stmt
    if copy_handle is not None:
        ctx.emit(copy_handle)
    return copy_handle


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
    """Store a PtrState tile to a flat function-arg buffer using strides.

    Output-buffer grid-scaling (RULE #1): this is a strided per-block write
    (the destination pointer carries a per-program base offset), so the
    target buffer must span the WHOLE launch grid, not a single tile. We
    grow the function-arg buffer to at least the grid-scaled floor
    ``grid_product * tile_numel`` -- and we do so even when the caller
    seeded a smaller ("fixed") shape, because a seed below that floor is a
    provable truncation that would drop every block past the first tile.
    The per-block base + tile mask already live in the flat index / store
    guard below (mirroring the native ``tl.store`` mask
    ``(offs_m<hdim)&(offs_n<dstate)``); there is NO whole-buffer clamp.
    """
    tir = ctx.tir()
    dtype = _normalize_mlir_dtype(
        str(getattr(val_expr, "dtype", _dtype_of(_operands(op)[1]) or "float32"))
    )
    # REGCARRYCOLLAPSE DIRECT FRAGMENT->GLOBAL EPILOGUE. When the post-loop
    # ``tt.store`` value is bound DIRECTLY to the loop-carried MMA-C accumulator
    # (a ``local.fragment``; control.py drops the shared ``carry_logical`` stage
    # for the unfolded-recurrence slot when TL_FRAG_GLOBAL_EPILOGUE != 0), emit a
    # single layout-aware, mask-clamped ``T.copy(fragment -> global_2d_view)``
    # instead of a per-lane serial scalar store. The serial scalar store would
    # read the fragment at LOGICAL [i, j] and impose a flat layout conflicting
    # with the MMA store layout; the CopyNode propagates the fragment's
    # make_mma_store_layout and writes registers DIRECTLY to global, eliminating
    # the 16 KB shared stage + its re-traffic. RULE #1: the helper RAISES if it
    # cannot recover the per-axis in-bounds (mask) extent -- never a maskless
    # full-tile store that is wrong for non-multiple-of-block shapes.
    if _frag_store_scope(val_expr) in {"local.fragment", "metal.simdgroup"}:
        return _emit_ptrstate_fragment_store_copynode(
            op, ctx, dst_buf, resolved, val_expr, list(val_shape), dtype,
            mask_ssa, dynamic_mask_dims,
        )
    grid_floor = _grid_scaled_store_extent(ctx, val_shape)
    min_extent = max(_flat_min_extent(val_shape), grid_floor)
    # A strided per-block store that addresses beyond a too-small caller
    # seed must grow the buffer (grow_fixed=True). When the seed already
    # covers the grid extent (the contract case) this is a no-op because
    # current_extent >= min_extent short-circuits inside the helper.
    dst_buf = _redeclare_ctx_buffer_1d(
        ctx, dst_buf, dtype, min_extent, grow_fixed=True,
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
    if resolved is None or (
        isinstance(resolved, dict) and "_ptrstate" not in resolved
    ):
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
    if resolved is None or (
        isinstance(resolved, dict) and "_ptrstate" not in resolved
    ):
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
    elif isinstance(val_expr, ctx.tvm().tir.Buffer):
        val_expr = _resolve_lane_operand(ctx, val_expr, [], role="value")

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
        # FULL TRANSFORM 1: fold-eligible expand_dims (consumed only as
        # addressing/mask) stays lazy -- folds into the load/store loop body,
        # so no [N] expand_* array is materialized.
        # REGSCALECOLLAPSE/IndexFold-lazy companion: keep the decay-scale
        # expand_dims (``exp(dA) -> expand_dims -> broadcast -> dout*scale ->
        # dot``) LAZY so the exp folds JUST-IN-TIME into the MMA-A fragment fill
        # (register-resident, per-lane) instead of materialising the §P1
        # ``expand_159[32]`` that pins all K columns of exp(dA) live in regs.
        if should_fold_addressing(ctx, op) or _expand_dims_feeds_register_a_scale(ctx, op):
            if result_value is not None:
                ctx.bind(result_value, lazy)
            return lazy
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


def _ctx_is_cuda_target(ctx: WalkerCtx) -> bool:
    """True iff the codegen target is explicitly CUDA / NVIDIA.

    Mirrors ``arith._ctx_is_cuda_target`` / ``reduction._is_cuda_target`` --
    prefers ``ctx.target`` (set by ``from_ttir`` before tilelang lowering passes
    establish ``Target.current()``), then the ambient target, then False.
    """
    ctx_target = getattr(ctx, "target", None) if ctx is not None else None
    if ctx_target:
        t = str(ctx_target).lower()
        return "cuda" in t or "nvidia" in t or t.startswith("sm_") or "nvptx" in t
    try:
        import tvm  # noqa: WPS433

        target = tvm.target.Target.current(allow_none=True)
    except Exception:
        return False
    if target is None:
        return False
    kind = str(getattr(getattr(target, "kind", None), "name", "") or "").lower()
    tstr = str(target).lower()
    return "cuda" in kind or "nvidia" in tstr or "cuda" in tstr


def _broadcast_feeds_register_a_scale(ctx: WalkerCtx, op: Any) -> bool:
    """True iff this ``tt.broadcast`` result feeds ONLY an element-wise float
    multiply/add (the tensor-core decay-scale, e.g. Mamba ``dout * exp(dA)``).

    REGSCALECOLLAPSE companion (RULE #1: scoped + fail-closed). When True the
    broadcast is kept LAZY -- its per-lane value folds directly into the
    consuming binop (which ``arith._feeds_tensorcore_dot_as_register_a`` itself
    keeps lazy and materialises into the MMA-A ``local.fragment``). That removes
    the materialised ``bcast_*`` staging tile that ptxas otherwise places in the
    stack frame (the §P1 dstates ``bcast_156[2048]`` == the entire 8448 B stack
    frame / 17.6 GB local-memory traffic). The scaled value then lives in the A
    fragment REGISTERS at MMA time -- exactly native's ``ldmatrix -> *scale in
    registers -> mma`` -- with no per-K local round-trip.

    Gated to:
      * CUDA / NVIDIA target (tensor-core MMA; gemm_rs reads A from registers).
        On Metal / SIMT the analogous scale is applied by the backend's own
        emitter, so this fold is a no-op there (the broadcast materialises as
        before) -- backend-neutral by construction.
      * The broadcast's SSA consumers are EXACTLY a set of element-wise float
        arithmetic ops (``arith.mulf`` / ``arith.addf`` / ``arith.subf``) -- the
        decay-scale pattern. Any other / unknown consumer keeps the materialised
        path (fail-closed: a broadcast read by a non-arith consumer, e.g. a
        store value or a reduction, is NOT folded). RULE #1: never over-broaden.

    The fold is BIT-EXACT regardless: a broadcast is pure replication, so the
    lazy reader reproduces the identical per-lane value. The gate only decides
    whether keeping it lazy is *beneficial* (register-resident scale) vs neutral.
    """
    if not _ctx_is_cuda_target(ctx):
        return False
    ssa_users = getattr(ctx, "ssa_users", None)
    if not ssa_users:
        return False
    name = _result_ssa_name(op)
    if name is None:
        return False
    users = (
        ssa_users.get(name)
        or ssa_users.get(name.lstrip("%"))
        or ssa_users.get(f"%{name.lstrip('%')}")
    )
    if not users:
        return False
    return set(users).issubset({"arith.mulf", "arith.addf", "arith.subf"})


def _op_name_for_gate(op: Any) -> str:
    """Best-effort MLIR op-name (``dialect.op``) for a consumer op object.

    Mirrors the ``_op_name`` resolution the module prepass uses, so the gate
    classifies consumer ops the same way ``ssa_users`` keys are built.
    """
    for attr in ("name", "OPERATION_NAME"):
        v = getattr(op, attr, None)
        if isinstance(v, str) and v:
            return v
    operation = getattr(op, "operation", None)
    if operation is not None:
        v = getattr(operation, "name", None)
        if isinstance(v, str) and v:
            return v
    return ""


def _expand_dims_feeds_register_a_scale(ctx: WalkerCtx, op: Any) -> bool:
    """True iff this ``tt.expand_dims`` result feeds ONLY the tensor-core
    decay-scale chain (e.g. Mamba ``exp(dA) -> expand_dims -> broadcast ->
    dout*scale -> tt.dot``).

    REGSCALECOLLAPSE / IndexFold-lazy companion (RULE #1: scoped + fail-closed).
    The ``tt.expand_dims`` sits BETWEEN the ``math.exp`` (already lazy) and the
    ``tt.broadcast`` (kept lazy by ``_broadcast_feeds_register_a_scale``). With
    no gate here ``emit_tt_expand_dims`` MATERIALISES the lazy exp into a
    register tile -- the §P1 ``expand_159[32]`` holding all K columns of
    ``exp(dA)`` LIVE across the whole K-step, the register-pressure ceiling that
    pins occupancy at 2 blocks/SM. Keeping it lazy lets the ``exp`` fold all the
    way into the MMA-A fragment fill, recomputed JUST-IN-TIME per lane in
    registers -- NO ``expand_159`` array, NO wide live range, and (RULE #1) NO
    local round-trip (a register recompute, not a spill).

    BIT-EXACT regardless: ``expand_dims`` is a pure shape rebind, so the lazy
    reader reproduces the identical per-lane value. The gate only decides
    whether keeping it lazy is *beneficial*.

    Gated to (fail-closed, never over-broaden):
      * CUDA / NVIDIA target (gemm_rs reads A from registers; on Metal/SIMT the
        scale is applied by the backend emitter, so this is a no-op there --
        the expand_dims materialises as before, backend-neutral).
      * Every consumer is EITHER a direct element-wise float arith op
        (``arith.mulf`` / ``addf`` / ``subf``) OR a ``tt.broadcast`` that ITSELF
        passes ``_broadcast_feeds_register_a_scale`` (transitive, one hop, on
        the real consumer op object via ``ctx.ssa_user_ops``). Any other /
        unknown / unresolved consumer keeps the materialised path.
    """
    if not _ctx_is_cuda_target(ctx):
        return False
    name = _result_ssa_name(op)
    if name is None:
        return False
    user_names = (
        ctx.ssa_users.get(name)
        or ctx.ssa_users.get(name.lstrip("%"))
        or ctx.ssa_users.get(f"%{name.lstrip('%')}")
    ) if getattr(ctx, "ssa_users", None) else None
    if not user_names:
        return False
    arith_scale = {"arith.mulf", "arith.addf", "arith.subf"}
    # Fast accept: every consumer is a direct element-wise arith scale op.
    if set(user_names).issubset(arith_scale):
        return True
    # Otherwise every consumer must be a ``tt.broadcast`` that itself feeds the
    # scale. Resolve the real consumer op objects (fail-closed if unavailable).
    if not set(user_names).issubset(arith_scale | {"tt.broadcast"}):
        return False
    user_ops = (
        ctx.ssa_user_ops.get(name)
        or ctx.ssa_user_ops.get(name.lstrip("%"))
        or ctx.ssa_user_ops.get(f"%{name.lstrip('%')}")
    ) if getattr(ctx, "ssa_user_ops", None) else None
    if not user_ops:
        return False
    for user_op in user_ops:
        op_name = _op_name_for_gate(user_op)
        if op_name in arith_scale:
            continue
        if op_name == "tt.broadcast":
            if not _broadcast_feeds_register_a_scale(ctx, user_op):
                return False
            continue
        return False
    return True


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
        # FULL TRANSFORM 1: fold-eligible broadcasts (consumed only as
        # addressing/mask) stay lazy -- the per-lane broadcast read folds into
        # the load/store loop body, so no [2048]/[4096] bcast_* array is
        # materialized (no local spill).
        # REGSCALECOLLAPSE companion: keep the decay-scale broadcast LAZY so it
        # folds into the MMA-A fragment fill (register-resident scale, native
        # mirror) instead of materialising a ``bcast_*[M*K]`` stack tile.
        if should_fold_addressing(ctx, op) or _broadcast_feeds_register_a_scale(ctx, op):
            if result_value is not None:
                ctx.bind(result_value, lazy)
            return lazy
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
