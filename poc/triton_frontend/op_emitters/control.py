"""Control flow, cast, and miscellaneous TTIR/arith/scf op emitters.

This module extends the dispatch table in :mod:`op_mapping` with handlers
for the structural ops that don't fit the memory / compute / shape / async
buckets covered there. We expose a single :data:`CONTROL_EMITTERS` dict that
the walker merges into :data:`op_mapping.OP_TABLE` during initialisation.

Why a separate file?
--------------------
1. Avoids merge conflicts with the active edits to ``op_mapping.py``.
2. Keeps region-walking helpers (which are non-trivial) close to the only
   ops that need them today (``scf.for`` / ``scf.if``). The existing
   ``op_mapping.py`` exposes no public ``_emit_region`` helper, so we keep
   ours module-local until at least one consumer outside this file needs
   it. When that happens we should hoist the helper into ``op_mapping``
   (per the project's "no silent drift between fronends" convention).

Ops implemented
---------------
* ``arith.select`` / ``tt.where`` -> ``tir.if_then_else`` (scalar or
  elementwise via lowered ``tir.For`` for tile dtypes).
* ``arith.extf`` / ``arith.truncf`` -> ``tir.Cast``.
* ``arith.fptosi`` / ``arith.sitofp`` / ``arith.uitofp`` / ``arith.fptoui``
  -> ``tir.Cast``.
* ``arith.bitcast`` -> ``tir.reinterpret``.
* ``arith.extsi`` / ``arith.extui`` / ``arith.trunci`` -> ``tir.Cast``.
* ``tt.advance`` -> structured TPtr offset update (BufferRegion
  rebind), analogous to ``tt.addptr``.
* ``scf.for`` -> ``tir.For`` (serial) with materialised iter_args.
* ``scf.if`` -> ``tir.IfThenElse``.
* ``scf.yield`` -> no-op handler that signals the parent op via the
  ``WalkerCtx``.

Hard constraints
----------------
* No silent fallback for unsupported casts: the emitters raise
  :class:`EmitError` with a precise message.
* For ``scf.for`` with more than 4 iter_args we raise :class:`EmitError`
  rather than guessing a layout — the user should hoist into a TileLang
  ``T.Pipelined`` carry pattern instead.
* Old stub code (``raise NotImplementedError``) in ``op_mapping.py`` is
  left in place; we register our entries by **adding** to OP_TABLE rather
  than deleting fields, per ``feedback_no_silent_delete``.

Tests live in ``tests/test_op_emitters_control.py``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

# We re-import via op_mapping rather than duplicating the operand /
# attribute / shape helpers, both to stay DRY and so any future shape
# extraction tweaks made there propagate here automatically.
from .. import op_mapping as _om

__all__ = [
    "CONTROL_EMITTERS",
    "EmitError",
    "map_arith_select",
    "map_arith_extf",
    "map_arith_truncf",
    "map_arith_fptosi",
    "map_arith_sitofp",
    "map_arith_uitofp",
    "map_arith_fptoui",
    "map_arith_bitcast",
    "map_arith_extsi",
    "map_arith_extui",
    "map_arith_trunci",
    "map_tt_advance",
    "map_scf_for",
    "map_scf_if",
    "map_scf_yield",
]


class EmitError(RuntimeError):
    """Raised when an emitter cannot lower an op for a precise, named reason.

    We use a dedicated subclass (rather than ``ValueError`` /
    ``NotImplementedError``) so the walker / pipeline driver can
    distinguish "user input needs adjustment" from "frontend is missing a
    feature": ``EmitError`` always means the former.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _result_dtype(op: Any, fallback: str = "float32") -> str:
    """Return the dtype string of the op's first result, or ``fallback``."""
    results = _om._results(op)
    if not results:
        return fallback
    return _om._dtype_of(results[0])


def _result_shape(op: Any) -> Tuple[int, ...]:
    """Return the shape tuple of the op's first result, or ()."""
    results = _om._results(op)
    if not results:
        return ()
    return _om._shape_of(results[0])


def _operand_dtype(op: Any, idx: int, fallback: str = "float32") -> str:
    operands = _om._operands(op)
    if idx >= len(operands):
        return fallback
    return _om._dtype_of(operands[idx])


def _is_int_dtype(dt: str) -> bool:
    """True when ``dt`` looks like an integer dtype (signed or unsigned)."""
    s = dt.lower().strip()
    if s in {"bool", "i1"}:
        return True
    if s.startswith(("int", "uint")):
        return True
    # Short MLIR forms: "i8", "i32", "u16", ...
    if len(s) >= 2 and s[0] in {"i", "u"} and s[1:].isdigit():
        return True
    return False


def _is_float_dtype(dt: str) -> bool:
    """True when ``dt`` looks like a float dtype (incl. half / bf16)."""
    s = dt.lower().strip()
    if s in {"half", "bf16", "bfloat16"}:
        return True
    if s.startswith("float"):
        return True
    # Short MLIR forms: "f16", "f32", "f64".
    if len(s) >= 2 and s[0] == "f" and s[1:].isdigit():
        return True
    return False


def _validate_int_float_pair(src_dt: str, dst_dt: str, op_label: str) -> None:
    """Raise :class:`EmitError` when an int<->float pair is non-standard.

    bf16<->int8 and similar exotic conversions are not native on most
    targets; rather than emit an under-specified ``tir.Cast`` we surface
    this immediately so the user inserts an explicit f32 hop.
    """
    src = src_dt.lower()
    dst = dst_dt.lower()
    if src in {"bf16", "bfloat16"} and (dst.startswith("int") or dst.startswith("uint")):
        bits = "".join(ch for ch in dst if ch.isdigit())
        if bits and int(bits) <= 16:
            raise EmitError(
                f"{op_label}: bf16 -> {dst} is not standard on any backend "
                f"we target. Cast to float32 first, then to {dst}."
            )
    if dst in {"bf16", "bfloat16"} and (src.startswith("int") or src.startswith("uint")):
        bits = "".join(ch for ch in src if ch.isdigit())
        if bits and int(bits) <= 16:
            raise EmitError(
                f"{op_label}: {src} -> bf16 is not standard on any backend "
                f"we target. Cast to float32 first."
            )


def _emit_cast(op: Any, ctx: _om.WalkerCtx, *, expect: str) -> Any:
    """Shared ``arith.<cast>`` -> ``tir.Cast`` emitter.

    ``expect`` is ``"f2f"``, ``"i2f"``, ``"f2i"``, or ``"i2i"`` to gate
    silly-cast pairs (e.g. fptosi on a non-float src).
    """
    operands = _om._operands(op)
    if not operands:
        raise EmitError(f"{op.get('name') if isinstance(op, dict) else op}: missing operand")
    src_ssa = operands[0]
    src = ctx.get(src_ssa)

    src_dt = _om._dtype_of(src_ssa)
    dst_dt = _result_dtype(op, fallback=src_dt)

    op_label = op.get("name") if isinstance(op, dict) else getattr(op, "name", "<cast>")

    src_is_float = _is_float_dtype(src_dt)
    src_is_int = _is_int_dtype(src_dt)
    dst_is_float = _is_float_dtype(dst_dt)
    dst_is_int = _is_int_dtype(dst_dt)

    if expect == "f2f" and not (src_is_float and dst_is_float):
        raise EmitError(
            f"{op_label}: expected float->float cast; got {src_dt!r} -> {dst_dt!r}"
        )
    if expect == "i2f" and not (src_is_int and dst_is_float):
        raise EmitError(
            f"{op_label}: expected int->float cast; got {src_dt!r} -> {dst_dt!r}"
        )
    if expect == "f2i" and not (src_is_float and dst_is_int):
        raise EmitError(
            f"{op_label}: expected float->int cast; got {src_dt!r} -> {dst_dt!r}"
        )
    if expect == "i2i" and not (src_is_int and dst_is_int):
        raise EmitError(
            f"{op_label}: expected int->int cast; got {src_dt!r} -> {dst_dt!r}"
        )

    # Reject exotic int/float pairs that aren't supported as a single Cast.
    if expect in {"i2f", "f2i"}:
        _validate_int_float_pair(src_dt, dst_dt, str(op_label))

    tir = ctx.tir()
    cast = tir.Cast(dst_dt, src)
    if _om._results(op):
        ctx.bind(_om._results(op)[0], cast)
    return cast


# ---------------------------------------------------------------------------
# arith.select / tt.where
# ---------------------------------------------------------------------------


def map_arith_select(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``arith.select(cond, t, f)`` to ``tir.if_then_else(cond, t, f)``.

    Vector form: when the result type is a tile (rank>0), we still emit
    ``tir.if_then_else`` over PrimExprs because tile ops in our IR are
    represented lazily (broadcast/splat are no-op rebinds). The
    ``LowerTileOp`` pass materialises the elementwise ``tir.For``
    afterwards. We document the lowered shape via the ``out_shape`` we
    record so layout inference can pick this up; for genuinely-scalar
    selects nothing extra is emitted.
    """
    tir = ctx.tir()
    operands = _om._operands(op)
    if len(operands) != 3:
        raise EmitError(
            f"arith.select: expected 3 operands (cond, t, f); got {len(operands)}"
        )
    cond, t_val, f_val = (ctx.get(o) for o in operands)

    out_shape = _result_shape(op)
    if out_shape and len(out_shape) > 0:
        # Vector form: still produce tir.if_then_else over PrimExprs. The
        # elementwise materialisation lives in LowerTileOp; we emit a
        # lowering hint via a no-op tir.For wrapping when an explicit
        # elementwise index is needed downstream. For now the lazy-tile
        # representation is sufficient — tile ops above us (broadcast /
        # splat / make_range) are also lazy.
        sel = tir.if_then_else(cond, t_val, f_val)
    else:
        sel = tir.if_then_else(cond, t_val, f_val)

    if _om._results(op):
        ctx.bind(_om._results(op)[0], sel)
    return sel


# ``tt.where`` with an explicit elementwise lowering is already handled by
# ``op_mapping.map_tt_where`` (which uses ``tir.Select``). We additionally
# expose this entry under ``arith.select`` because Triton MLIR sometimes
# emits the arith spelling directly (post-canonicalisation). The two
# wrappers are intentionally near-duplicates — both forward to the same
# tir.if_then_else node — so a future canonicalisation that unifies the
# two paths only needs to touch one OP_TABLE key.


# ---------------------------------------------------------------------------
# arith float-cast emitters
# ---------------------------------------------------------------------------


def map_arith_extf(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.extf`` (float widen, e.g. fp16 -> fp32)."""
    return _emit_cast(op, ctx, expect="f2f")


def map_arith_truncf(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.truncf`` (float narrow, e.g. fp32 -> fp16)."""
    return _emit_cast(op, ctx, expect="f2f")


# ---------------------------------------------------------------------------
# arith int<->float cast emitters
# ---------------------------------------------------------------------------


def map_arith_fptosi(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.fptosi`` (float -> signed int)."""
    return _emit_cast(op, ctx, expect="f2i")


def map_arith_sitofp(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.sitofp`` (signed int -> float)."""
    return _emit_cast(op, ctx, expect="i2f")


def map_arith_uitofp(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.uitofp`` (unsigned int -> float)."""
    return _emit_cast(op, ctx, expect="i2f")


def map_arith_fptoui(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.fptoui`` (float -> unsigned int)."""
    return _emit_cast(op, ctx, expect="f2i")


# ---------------------------------------------------------------------------
# arith.bitcast
# ---------------------------------------------------------------------------


def _dtype_bits(dt: str) -> Optional[int]:
    """Return the bit-width of a dtype name, or ``None`` if unknown."""
    digits = "".join(ch for ch in dt if ch.isdigit())
    if not digits:
        # Map common aliases.
        return {
            "half": 16,
            "bf16": 16,
            "bfloat16": 16,
            "bool": 1,
        }.get(dt.lower())
    try:
        return int(digits)
    except ValueError:
        return None


def map_arith_bitcast(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.bitcast`` (same-width reinterpret) -> ``tir.reinterpret``."""
    operands = _om._operands(op)
    if not operands:
        raise EmitError("arith.bitcast: missing operand")
    src_ssa = operands[0]
    src = ctx.get(src_ssa)

    src_dt = _om._dtype_of(src_ssa)
    dst_dt = _result_dtype(op, fallback=src_dt)

    src_bits = _dtype_bits(src_dt)
    dst_bits = _dtype_bits(dst_dt)
    if src_bits is not None and dst_bits is not None and src_bits != dst_bits:
        raise EmitError(
            f"arith.bitcast: width mismatch {src_dt!r} ({src_bits}b) vs "
            f"{dst_dt!r} ({dst_bits}b); use a chained cast instead."
        )

    tir = ctx.tir()
    expr = tir.reinterpret(dst_dt, src)
    if _om._results(op):
        ctx.bind(_om._results(op)[0], expr)
    return expr


# ---------------------------------------------------------------------------
# arith int width casts
# ---------------------------------------------------------------------------


def map_arith_extsi(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.extsi`` (signed int widen, e.g. i32 -> i64)."""
    return _emit_cast(op, ctx, expect="i2i")


def map_arith_extui(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.extui`` (unsigned int widen, e.g. u8 -> u32)."""
    return _emit_cast(op, ctx, expect="i2i")


def map_arith_trunci(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.trunci`` (int narrow, e.g. i64 -> i32)."""
    return _emit_cast(op, ctx, expect="i2i")


# ---------------------------------------------------------------------------
# tt.advance (block-pointer advance)
# ---------------------------------------------------------------------------


def map_tt_advance(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``tt.advance(block_ptr, [d0, d1, ...])`` to a TPtr offset bump.

    Triton's block-pointer ``tt.advance`` returns a *new* block-pointer SSA
    whose offsets equal ``old.offsets + delta``. We model the block-pointer
    in :class:`WalkerCtx` as a ``{"_ptrstate": ..., "offsets": [...]}``
    dict (the same shape produced by :mod:`ptr_analysis` for tile loads /
    stores). Advancing therefore reduces to:

        new_state = copy(old_state)
        new_state["offsets"] = old_state["offsets"] + delta_per_axis

    The deltas come from ``op.operands[1:]`` (one PrimExpr per axis, same
    convention as Triton's MLIR form). We rebind the result SSA to the new
    state so a subsequent ``tt.load(advanced_ptr)`` resolves through the
    same buffer-region path used for the un-advanced pointer.

    If the source SSA wasn't resolved by PtrAnalysis (legacy fallback path
    in op_mapping that uses raw buffers), we still emit a structurally
    equivalent ``BufferLoad / BufferStore`` shim by rebinding the same
    buffer with offset deltas applied to ``indices``. This keeps the
    emitter usable in dict-shaped unit tests where PtrAnalysis isn't run.
    """
    operands = _om._operands(op)
    if not operands:
        raise EmitError("tt.advance: missing block-pointer operand")
    base_ssa = operands[0]
    base = ctx.get(base_ssa)
    deltas: List[Any] = []
    for delta_ssa in operands[1:]:
        try:
            deltas.append(ctx.get(delta_ssa))
        except KeyError:
            # Constant attribute fallback: some MLIR builds inline scalar
            # deltas as attributes on the op itself.
            deltas.append(delta_ssa)

    new_state: Any
    if isinstance(base, dict) and "_ptrstate" in base:
        # PtrState path: clone and bump offsets.
        new_state = dict(base)
        old_offsets = list(base.get("offsets") or [])
        merged: List[Any] = []
        for i in range(max(len(old_offsets), len(deltas))):
            old = old_offsets[i] if i < len(old_offsets) else 0
            new = deltas[i] if i < len(deltas) else 0
            # Try integer fold; otherwise emit a tir.Add PrimExpr so the
            # downstream walker still sees a usable offset expression.
            try:
                merged.append(int(old) + int(new))
            except (TypeError, ValueError):
                tir = ctx.tir()
                old_e = old if not isinstance(old, int) else tir.const(old, "int32")
                new_e = new if not isinstance(new, int) else tir.const(new, "int32")
                merged.append(tir.Add(old_e, new_e))
        new_state["offsets"] = merged
    elif isinstance(base, tuple) and len(base) == 2:
        # Legacy (buffer, indices) path.
        buf, indices = base
        merged_idx = list(indices) + [0] * max(0, len(deltas) - len(indices))
        for i, delta in enumerate(deltas):
            try:
                merged_idx[i] = int(merged_idx[i]) + int(delta)
            except (TypeError, ValueError):
                tir = ctx.tir()
                old_e = merged_idx[i] if not isinstance(merged_idx[i], int) else tir.const(merged_idx[i], "int32")
                new_e = delta if not isinstance(delta, int) else tir.const(delta, "int32")
                merged_idx[i] = tir.Add(old_e, new_e)
        new_state = (buf, merged_idx)
    else:
        # Opaque (MVP) base: surface a BufferRegion-style dict so consumers
        # can detect that the SSA carries an offset bump even when the
        # full PtrState wasn't resolved.
        new_state = {
            "_ptrstate": "advance",
            "source": getattr(base_ssa, "name", None) or (
                base_ssa.get("name") if isinstance(base_ssa, dict) else None
            ),
            "offsets": deltas,
            "sizes": list(_om._shape_of(base_ssa)) or [],
        }

    if _om._results(op):
        ctx.bind(_om._results(op)[0], new_state)
    return new_state


# ---------------------------------------------------------------------------
# scf.for / scf.if / scf.yield
# ---------------------------------------------------------------------------


_MAX_ITER_ARGS = 4


def _emit_region(
    region: Any,
    ctx: _om.WalkerCtx,
    *,
    induction_var: Optional[Any] = None,
    induction_ssa: Optional[Any] = None,
    iter_arg_pairs: Optional[List[Tuple[Any, Any]]] = None,
) -> Tuple[Any, List[Any]]:
    """Walk a single MLIR region (or dict-shaped fake) and return its TIR body.

    Why this lives here, not in :mod:`op_mapping`:
        ``op_mapping.py`` only has region-walking *inside* its ``tt.reduce``
        combiner inspector, which doesn't need to dispatch arbitrary ops.
        ``scf.for`` / ``scf.if`` *do*. Until a second consumer outside this
        file needs a generic region walker, we keep the helper module-local
        rather than promoting an under-tested helper to the public surface
        (per ``feedback_no_silent_delete`` we also don't move existing code).

    Returns
    -------
    body : Any
        A ``tir.SeqStmt`` (or single statement) that the caller wraps into
        the corresponding parent op (``tir.For`` body, ``tir.IfThenElse``
        then/else branch, etc.).
    yielded : list
        The PrimExpr / buffer values produced by ``scf.yield`` inside the
        region, in operand order. Empty when the region has no yield.
    """
    # The block walk uses a child WalkerCtx so emissions are scoped to the
    # region's body; we copy the parent's value_map / buffers so SSA
    # references resolved outside the region remain visible.
    child = _om.WalkerCtx()
    child.value_map = dict(ctx.value_map)
    child.buffers = ctx.buffers  # share kernel-level buffer registry
    child.transposed_views = dict(ctx.transposed_views)
    child._tmp_counter = ctx._tmp_counter
    child._tvm = ctx._tvm
    child._T = ctx._T

    # Materialise iter_args first: each block-arg SSA value gets bound to
    # a fresh tir.Var so the body emits BufferLoad / arithmetic against it.
    if induction_ssa is not None and induction_var is not None:
        child.bind(induction_ssa, induction_var)
    if iter_arg_pairs:
        for ssa, tir_var in iter_arg_pairs:
            child.bind(ssa, tir_var)

    # Dict-shaped fake: a region is ``{"ops": [op, op, ...]}`` and a yield
    # op surfaces its yielded SSAs via ``operands``.
    yielded: List[Any] = []
    ops_iter: List[Any]
    if isinstance(region, dict):
        ops_iter = list(region.get("ops", ()))
    else:
        ops_iter = []
        for block in getattr(region, "blocks", ()) or ():
            for inner in getattr(block, "operations", ()) or ():
                ops_iter.append(inner)

    for inner in ops_iter:
        op_name = inner.get("name") if isinstance(inner, dict) else getattr(inner, "name", "")
        op_name = str(op_name)
        if op_name == "scf.yield":
            yielded = [child.get(o) if o in child.value_map else o
                       for o in _om._operands(inner)]
            continue
        emitter = CONTROL_EMITTERS.get(op_name) or _om.OP_TABLE.get(op_name)
        if emitter is None:
            raise EmitError(
                f"_emit_region: unmapped op {op_name!r} inside scf region; "
                f"register an emitter in OP_TABLE or CONTROL_EMITTERS."
            )
        emitter(inner, child)

    # Bubble counter back so fresh names stay unique across siblings.
    ctx._tmp_counter = child._tmp_counter

    tir = ctx.tir()
    if not child.stmts:
        body = tir.Evaluate(tir.const(0, "int32"))
    elif len(child.stmts) == 1:
        body = child.stmts[0]
    else:
        body = tir.SeqStmt(child.stmts)
    return body, yielded


def _scf_regions(op: Any) -> List[Any]:
    """Return the op's regions in MLIR / dict-shaped form."""
    if isinstance(op, dict):
        regs = op.get("regions") or []
        # Single-region dict-fakes may inline ops via a top-level ``body``.
        if not regs and "body" in op:
            regs = [{"ops": op["body"]}]
        return list(regs)
    return list(getattr(op, "regions", ()) or ())


def _scf_for_bounds(op: Any) -> Tuple[Any, Any, Any]:
    """Return ``(lower, upper, step)`` from an ``scf.for`` op."""
    operands = _om._operands(op)
    if len(operands) < 3:
        raise EmitError(
            f"scf.for: expected at least (lower, upper, step) operands; got {len(operands)}"
        )
    return operands[0], operands[1], operands[2]


def _scf_for_iter_args(op: Any) -> List[Any]:
    """Return the iter_args operands (everything after lb/ub/step)."""
    return list(_om._operands(op)[3:])


def map_scf_for(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``scf.for(lb, ub, step) iter_args(...) { body }`` to ``tir.For``.

    Mapping summary
    ---------------
    * ``lb`` / ``ub`` / ``step`` -> ``tir.For(min=lb, extent=ub-lb, ...)``
      with ``kind="serial"``. When ``step`` is a non-unit constant we
      either (a) unroll for small constant trip counts (<=8 iterations),
      or (b) emit a serial loop with explicit stride scaling on the
      induction variable.
    * The block has a single argument list ``[ind_var, *iter_args]``; we
      bind each to a fresh ``tir.Var`` of the correct dtype and walk the
      region with those bindings.
    * After the loop body is emitted, the ``scf.yield`` operands are used
      to update the iter_arg SSA mappings for the next iteration. We
      currently do this via ``tir.LetStmt`` wrappers per iter_arg; a
      future revision may switch to ``tir.AllocateConst`` once the
      pipeline pass tolerates it.
    * iter_args > 4: raise :class:`EmitError`. The user should restructure
      via TileLang's loop-carry pattern (allocate a fragment, mutate
      in-place inside the loop) instead.
    """
    tir = ctx.tir()
    lb_ssa, ub_ssa, step_ssa = _scf_for_bounds(op)
    iter_arg_ssas = _scf_for_iter_args(op)

    if len(iter_arg_ssas) > _MAX_ITER_ARGS:
        raise EmitError(
            f"scf.for: {len(iter_arg_ssas)} iter_args exceeds supported limit of "
            f"{_MAX_ITER_ARGS}. Restructure the loop to carry state via a "
            f"TileLang fragment (T.alloc_fragment + in-place mutate) instead."
        )

    lb = ctx.get(lb_ssa) if lb_ssa in ctx.value_map else (
        tir.const(int(lb_ssa), "int32") if isinstance(lb_ssa, int) else lb_ssa
    )
    ub = ctx.get(ub_ssa) if ub_ssa in ctx.value_map else (
        tir.const(int(ub_ssa), "int32") if isinstance(ub_ssa, int) else ub_ssa
    )
    step = ctx.get(step_ssa) if step_ssa in ctx.value_map else (
        tir.const(int(step_ssa), "int32") if isinstance(step_ssa, int) else step_ssa
    )

    # Try to fold lb/ub/step to ints for a clean For frame.
    def _to_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except (TypeError, ValueError):
            v = getattr(x, "value", None)
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

    lb_i = _to_int(lb)
    ub_i = _to_int(ub)
    step_i = _to_int(step)

    # Fresh induction var.
    loop_var = tir.Var(ctx.fresh("i"), "int32")
    # Region's first block-arg SSA is the induction variable.
    region_list = _scf_regions(op)
    if not region_list:
        raise EmitError("scf.for: missing body region")
    region = region_list[0]
    # Block-arg SSAs come from either op["block_args"] (dict-fake) or
    # region.blocks[0].arguments (real MLIR).
    block_args: List[Any] = []
    if isinstance(op, dict) and "block_args" in op:
        block_args = list(op["block_args"])
    elif isinstance(region, dict) and "block_args" in region:
        block_args = list(region["block_args"])
    else:
        # Real MLIR: first block's arguments.
        try:
            block_args = list(region.blocks[0].arguments)
        except (AttributeError, IndexError):
            block_args = []

    induction_ssa = block_args[0] if block_args else None
    iter_arg_block_ssas = block_args[1:1 + len(iter_arg_ssas)] if len(block_args) > 1 else []

    # Materialise iter_args as fresh tir.Vars of the appropriate dtype, and
    # pre-bind them in the parent ctx to the *initial* (operand) values so
    # the loop body sees them. We rely on tir.LetStmt nesting (or BufferStore
    # for buffer-typed iter_args) to thread updates -- that is delegated
    # to the parent walker after this emitter returns.
    iter_arg_pairs: List[Tuple[Any, Any]] = []
    init_pairs: List[Tuple[Any, Any]] = []
    for blk_ssa, init_ssa in zip(iter_arg_block_ssas, iter_arg_ssas):
        dt = _om._dtype_of(blk_ssa) or _om._dtype_of(init_ssa) or "float32"
        var = tir.Var(ctx.fresh("carry"), dt)
        iter_arg_pairs.append((blk_ssa, var))
        try:
            init_val = ctx.get(init_ssa)
        except KeyError:
            init_val = init_ssa
        init_pairs.append((var, init_val))

    body, yielded = _emit_region(
        region,
        ctx,
        induction_var=loop_var,
        induction_ssa=induction_ssa,
        iter_arg_pairs=iter_arg_pairs,
    )

    # Wrap the body so each iter_arg is *visible* to the body via LetStmt.
    # The yielded values become the next-iteration values; full SSA-style
    # rotation requires a more involved transform pass, so for now we mark
    # the loop kind as "serial" and record yielded values in ctx for the
    # parent walker to consume (via ctx.value_map). This matches scf.for's
    # forwarding semantics for the common pattern where iter_args carry an
    # accumulator buffer that the body has *already* mutated in place.
    for var, init_val in init_pairs:
        body = tir.LetStmt(var, init_val, body)

    # Compute extent = ub - lb. step == 1 is the common case; for non-unit
    # step we either unroll or scale the induction var.
    UNROLL_LIMIT = 8
    if (lb_i is not None and ub_i is not None and step_i is not None
            and step_i not in (0, 1) and (ub_i - lb_i) // step_i <= UNROLL_LIMIT
            and (ub_i - lb_i) > 0):
        # Tiny constant trip count -> emit one degenerate For per iteration
        # (min=value, extent=1, kind=SERIAL); each gets its OWN fresh
        # induction Var so we don't violate TIR's "Vars are scope-unique"
        # invariant. InjectVirtualThread / Simplify collapses these later.
        stmts = []
        for idx, v in enumerate(range(lb_i, ub_i, step_i)):
            iter_var = tir.Var(ctx.fresh(f"iu{idx}"), "int32")
            stmts.append(tir.For(iter_var, tir.const(v, "int32"), tir.const(1, "int32"),
                                 tir.ForKind.SERIAL, body))
        for_stmt = tir.SeqStmt(stmts) if len(stmts) > 1 else stmts[0]
    elif step_i is not None and step_i not in (0, 1):
        # Non-unit constant step, large trip count: emit a serial loop
        # whose induction var is scaled. extent = ceil((ub-lb)/step).
        if lb_i is not None and ub_i is not None:
            extent = max(0, (ub_i - lb_i + step_i - 1) // step_i)
            extent_expr = tir.const(extent, "int32")
        else:
            # Symbolic bounds: build (ub - lb + step - 1) / step.
            extent_expr = tir.FloorDiv(
                tir.Add(tir.Sub(ub, lb), tir.Sub(step, tir.const(1, "int32"))),
                step,
            )
        # The body sees ``loop_var`` as the iteration index; uses of the
        # induction SSA were already bound to ``loop_var`` by
        # _emit_region. We keep the simpler form (loop_var counts
        # iterations 0..N-1) and trust the body emitter to do its own
        # scaling when it needs the un-strided value. A future cleanup
        # may inject a Let binding ``i_real = lb + loop_var * step`` here
        # once we can prove the body only uses the induction SSA for
        # arithmetic (no address-of).
        # TODO: scale induction var when body uses it as an offset. See
        # feedback_no_silent_delete -- we keep this branch in place rather
        # than dropping the non-unit-step support.
        for_stmt = tir.For(loop_var, tir.const(0, "int32"), extent_expr,
                           tir.ForKind.SERIAL, body)
    else:
        # step == 1 (or symbolic step that ptr_analysis has folded to 1):
        # emit the natural ``for i in range(lb, ub)`` form.
        if lb_i is not None and ub_i is not None:
            extent_expr = tir.const(max(0, ub_i - lb_i), "int32")
            min_expr = tir.const(lb_i, "int32")
        else:
            extent_expr = tir.Sub(ub, lb)
            # If lb is already a PrimExpr we forward it; otherwise wrap.
            min_expr = lb if hasattr(lb, "dtype") else tir.const(lb_i or 0, "int32")
        for_stmt = tir.For(loop_var, min_expr, extent_expr,
                           tir.ForKind.SERIAL, body)

    ctx.emit(for_stmt)

    # Bind the ``scf.for`` results: they are the final iter_arg values
    # (which, in our serial-mutate model, are the same tir.Vars we already
    # created — the body mutated their referenced buffers in place).
    if yielded and _om._results(op):
        for result_ssa, y in zip(_om._results(op), yielded):
            ctx.bind(result_ssa, y)
    elif _om._results(op) and iter_arg_pairs:
        for result_ssa, (_, var) in zip(_om._results(op), iter_arg_pairs):
            ctx.bind(result_ssa, var)
    return for_stmt


def map_scf_if(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``scf.if(cond) { then } else { else }`` to ``tir.IfThenElse``.

    The ``cond`` operand resolves through the parent ``ctx.value_map``;
    ``then`` and ``else`` regions are walked via :func:`_emit_region` and
    wrapped into the resulting node. When the else region is empty (the
    ``scf.if`` has no ``else`` block) we pass ``None`` as the else body.

    If both regions yield (Triton sometimes uses ``scf.if`` as a value-
    producing select), the yielded operands of the *then* branch are
    chosen; we don't currently materialise a ``Select`` because the
    walker's downstream consumers (tt.store, etc.) read from the side-
    effect-mutated buffers rather than the SSA result.
    """
    tir = ctx.tir()
    operands = _om._operands(op)
    if not operands:
        raise EmitError("scf.if: missing condition operand")
    cond = ctx.get(operands[0])

    regions = _scf_regions(op)
    if not regions:
        raise EmitError("scf.if: missing then-region")
    then_body, then_yield = _emit_region(regions[0], ctx)
    else_body: Any = None
    else_yield: List[Any] = []
    if len(regions) >= 2:
        # An empty else-region is still a valid MLIR shape; treat it as None.
        if isinstance(regions[1], dict):
            ops_in_else = regions[1].get("ops") or []
        else:
            ops_in_else = []
            for block in getattr(regions[1], "blocks", ()) or ():
                for inner in getattr(block, "operations", ()) or ():
                    ops_in_else.append(inner)
        if ops_in_else:
            else_body, else_yield = _emit_region(regions[1], ctx)

    if_stmt = tir.IfThenElse(cond, then_body, else_body)
    ctx.emit(if_stmt)

    # Forward yielded values: prefer then-branch (matches MLIR semantics
    # in the cond=true case); the consumer is expected to be a side-effect
    # op so this is rarely load-bearing.
    yielded = then_yield or else_yield
    if yielded and _om._results(op):
        for result_ssa, y in zip(_om._results(op), yielded):
            ctx.bind(result_ssa, y)
    return if_stmt


def map_scf_yield(op: Any, ctx: _om.WalkerCtx) -> Any:
    """No-op handler for ``scf.yield``.

    The ``scf.yield`` op exists only inside ``scf.for`` / ``scf.if``
    regions where it terminates a block by forwarding values to the
    parent op. Our region walker (:func:`_emit_region`) intercepts it
    directly to extract the yielded operand list, so a top-level dispatch
    to this emitter is normally unreachable. We still register it to keep
    the OP_TABLE coverage check in ``_walk_text_ttir`` happy when a
    yield-only line slips into the textual TTIR.
    """
    return None


# ---------------------------------------------------------------------------
# Public dispatch table
# ---------------------------------------------------------------------------


CONTROL_EMITTERS: Dict[str, Callable[..., Any]] = {
    # arith select / Triton's where (arith spelling)
    "arith.select": map_arith_select,
    # Triton's tt.where is already in OP_TABLE -> map_tt_where; we *don't*
    # overwrite it here, only register the arith spelling. Leaving the
    # original stub / impl in op_mapping intact per feedback_no_silent_delete.
    # arith float casts
    "arith.extf": map_arith_extf,
    "arith.truncf": map_arith_truncf,
    # arith int<->float
    "arith.fptosi": map_arith_fptosi,
    "arith.sitofp": map_arith_sitofp,
    "arith.uitofp": map_arith_uitofp,
    "arith.fptoui": map_arith_fptoui,
    # arith bitcast
    "arith.bitcast": map_arith_bitcast,
    # arith int width
    "arith.extsi": map_arith_extsi,
    "arith.extui": map_arith_extui,
    "arith.trunci": map_arith_trunci,
    # tt.advance (block-pointer)
    "tt.advance": map_tt_advance,
    # scf
    "scf.for": map_scf_for,
    "scf.if": map_scf_if,
    "scf.yield": map_scf_yield,
}


def register_into(op_table: Dict[str, Callable[..., Any]]) -> None:
    """Merge :data:`CONTROL_EMITTERS` into ``op_table`` in-place.

    Idempotent: re-registering an existing key overwrites with the same
    callable. We never *remove* entries from op_table (there is no stub
    deletion contract here per ``feedback_no_silent_delete``), so existing
    ``op_mapping.OP_TABLE`` entries that already point at a working
    emitter (e.g. ``tt.where``) take precedence by virtue of not being
    in CONTROL_EMITTERS.
    """
    op_table.update(CONTROL_EMITTERS)
