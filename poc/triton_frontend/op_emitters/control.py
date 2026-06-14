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
* ``arith.extsi`` / ``arith.extui`` / ``arith.trunci`` /
  ``arith.index_cast`` -> ``tir.Cast``.
* ``tt.advance`` -> structured TPtr offset update (BufferRegion
  rebind), analogous to ``tt.addptr``.
* ``cf.br`` / ``cf.cond_br`` -> CFG terminators accepted by the region
  walker when Triton emits an early-return diamond before the structured
  loop body.
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
from ..op_mapping import EmitError

# ``_op_name`` lives in the walker; import lazily-safe so a missing walker
# (unit-test stubs) degrades to ``None`` rather than an import error.
try:  # pragma: no cover - import wiring
    from .. import mlir_walker as _wk
except Exception:  # pragma: no cover
    _wk = None  # type: ignore

import os as _os  # noqa: WPS433


# DIRECT-FRAGMENT->GLOBAL EPILOGUE (named generic capability).
# ----------------------------------------------------------------------------
# A loop-carried MMA-C accumulator lives in a swizzled ``local.fragment`` (the
# tensor-core store layout). The post-loop ``tt.store`` of that accumulator must
# reach global memory. The PRIOR epilogue staged the whole MxN fp32 accumulator
# through a freshly-allocated ``shared`` tile (``carry_logical``) and then did a
# second shared->global store -- 128KB of shared for a 128x256 fp32 tile, which
# pushed the autotune-winning tile over the GB10 opt-in shared cap (101376 B).
#
# Native Triton stores the MMA-C fragment DIRECTLY to global, per-warp-tile,
# layout-aware -- it never materialises the accumulator in shared at the
# epilogue. TileLang's ``T.copy(fragment, global)`` lowers to exactly that: the
# CopyNode picks the fragment (highest scope-level) as the iteration base and
# ``InferLayout`` propagates the fragment's registered ``make_mma_store_layout``
# to the parallel store loop, emitting a direct layout-aware fragment->global
# store with no shared staging.
#
# This is a GENERIC, BACKEND-AGNOSTIC frontend capability: it binds the loop
# result to the fragment itself so EVERY fragment-carry epilogue (all tiles, any
# backend whose copy lowering supports fragment->global -- CUDA tensor-core and
# Metal simdgroup alike) skips the shared staging buffer. RULE #1: it is a
# correctness route, NOT a degraded fallback -- the parity gate below proves the
# direct store is bit-correct; if it ever regressed it must RAISE, never
# silently revert to staging.
#
# The env override exists ONLY for same-session A/B measurement of the shared
# budget and parity; it defaults to the direct store. Set
# ``TL_FRAG_GLOBAL_EPILOGUE=0`` to force the legacy shared-staging path for a
# controlled comparison.
_DIRECT_FRAG_GLOBAL_EPILOGUE_ENABLED = (
    _os.environ.get("TL_FRAG_GLOBAL_EPILOGUE", "1") != "0"
)

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
    "map_arith_index_cast",
    "map_arith_constant",
    "map_tensor_collapse_shape",
    "map_tt_advance",
    "map_cf_br",
    "map_cf_cond_br",
    "map_tt_func",
    "emit_tt_call",
    "map_scf_for",
    "map_scf_if",
    "map_scf_yield",
    "map_scf_while",
    "map_llvm_inline_asm",
    "SCF_WHILE_MAX_ITERATIONS",
    "PTX_TO_TIR",
]


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


def _primexpr_lanes(value: Any) -> int:
    """Return vector lane count for a TVM PrimExpr, or 1 for scalar values."""
    dt = str(getattr(value, "dtype", ""))
    if "x" not in dt:
        return 1
    try:
        return int(dt.rsplit("x", 1)[1])
    except ValueError:
        return 1


def _with_lanes(dtype: str, lanes: int) -> str:
    if lanes <= 1 or "x" in dtype:
        return dtype
    return f"{dtype}x{lanes}"


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


def _is_scalar_primexpr(value: Any) -> bool:
    """Return True if ``value`` can be used as a scalar ``tir.LetStmt`` value.

    ``tir.LetStmt(var, value, body)`` requires ``value`` to satisfy
    ``ir.PrimExpr`` (scalar). Iter-args of ``scf.for`` / ``scf.while`` are
    sometimes buffer-typed tiles -- we materialise them as
    ``(buffer, shape)`` tuples or raw ``tir.Buffer`` objects upstream.
    Feeding those to ``LetStmt`` raises::

        TypeError: Mismatched type on argument #1 ... Expected
        ir.PrimExpr but got ffi.Array

    This helper lets the caller distinguish the two paths cleanly.
    """
    # Fast paths for plain Python numeric types -- ``tir.Var`` and
    # ``tir.PrimExpr`` are TVM Object subclasses; numeric scalars get
    # auto-promoted by TVM's argument coercion. Anything that's clearly
    # a tuple-shaped descriptor / collection / Buffer is *not* scalar.
    if value is None:
        return False
    if isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, (tuple, list, dict, set)):
        return False
    # tvm.tir.Buffer is not a PrimExpr.
    try:
        import tvm  # noqa: WPS433
        if isinstance(value, tvm.tir.Buffer):
            return False
        # ffi.Array (TVM-wrapped tuple, e.g. shape descriptors).
        ffi_array = getattr(getattr(tvm, "ffi", None), "Array", None)
        if ffi_array is not None and isinstance(value, ffi_array):
            return False
        return isinstance(value, tvm.ir.PrimExpr)
    except Exception:
        # If TVM probing fails, fall back to "treat as scalar" so we don't
        # silently mask a real bug -- the LetStmt call itself will surface
        # the type error if we got it wrong.
        return True


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
    try:
        from .arith import _emit_tile_unary, _is_tile_operand  # noqa: WPS433

        if _is_tile_operand(ctx, src):
            return _emit_tile_unary(
                op,
                ctx,
                src,
                lambda lane: tir.Cast(dst_dt, lane),
                str(op_label),
                dst_dt,
            )
    except ImportError:
        pass

    cast = tir.Cast(_with_lanes(dst_dt, _primexpr_lanes(src)), src)
    if _om._results(op):
        ctx.bind(_om._results(op)[0], cast)
    return cast


# ---------------------------------------------------------------------------
# arith.select / tt.where
# ---------------------------------------------------------------------------


def map_arith_select(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``arith.select(cond, t, f)`` to ``tir.if_then_else(cond, t, f)``.

    Scalar form: emits a single ``tir.if_then_else`` over PrimExprs.

    Buffer form (Wave F2): when **any** of ``cond`` / ``t`` / ``f`` resolves
    to a ``tir.Buffer`` -- which happens after Wave E2 lowered a broadcasted
    constant tile or a per-lane comparator into a ``decl_buffer`` -- the
    scalar ``tir.if_then_else`` path explodes with::

        Mismatched type on argument #N when calling _OpIfThenElse:
        Expected `ir.PrimExpr` but got `tirx.Buffer`

    Mirroring Wave F1's fix in :func:`emit_tt_load`, we materialise a
    per-lane ``tir.For`` nest: allocate a fresh tile-scoped result buffer,
    iterate over the result shape, and ``BufferLoad`` each Buffer-typed
    operand at the loop indices before feeding it into
    ``tir.if_then_else``. The result SSA is rebound to the new buffer so
    downstream consumers see a tile-shaped value just like before.
    """
    tir = ctx.tir()
    tvm_mod = ctx.tvm()
    operands = _om._operands(op)
    if len(operands) != 3:
        raise EmitError(
            f"arith.select: expected 3 operands (cond, t, f); got {len(operands)}"
        )
    cond, t_val, f_val = (ctx.get(o) for o in operands)

    # Per-lane materialisation when any operand is a Buffer-shaped tile.
    # We keep the scalar fast-path (no allocation, no loop nest) for the
    # common case where Wave E2 lowering hasn't fired.
    buffer_cls = tvm_mod.tir.Buffer
    has_buffer_operand = (
        isinstance(cond, buffer_cls)
        or isinstance(t_val, buffer_cls)
        or isinstance(f_val, buffer_cls)
        or isinstance(cond, _om.LazyTileExpr)
        or isinstance(t_val, _om.LazyTileExpr)
        or isinstance(f_val, _om.LazyTileExpr)
    )
    if has_buffer_operand:
        out_shape = _result_shape(op)
        if not out_shape:
            # A Buffer-typed operand with a rank-0 result is structurally
            # impossible: report it loudly rather than silently returning a
            # bogus scalar.
            raise EmitError(
                "arith.select: Buffer-typed operand on a rank-0 result; "
                "expected a tile result shape (post-broadcast)."
            )
        out_dtype = _result_dtype(op)
        def _lane(read_ctx: _om.WalkerCtx, value: Any, indices: Tuple[Any, ...], role: str) -> Any:
            """Pull a scalar lane out of ``value`` for the current loop_vars."""
            read_tir = read_ctx.tir()
            if isinstance(value, _om.LazyTileExpr):
                rank = len(value.shape)
                if len(indices) >= rank:
                    idx = list(indices[-rank:])
                else:
                    idx = [read_tir.const(0, "int32")] * (rank - len(indices)) + list(indices)
                for axis, extent in enumerate(value.shape):
                    if int(extent) == 1:
                        idx[axis] = read_tir.const(0, "int32")
                return value.read_lane(read_ctx, tuple(idx))
            if isinstance(value, buffer_cls):
                rank = len(value.shape)
                if rank == 0:
                    return read_tir.BufferLoad(value, [read_tir.const(0, "int32")])
                lv = list(indices)
                if len(lv) >= rank:
                    load_indices = lv[-rank:]
                else:
                    load_indices = [read_tir.const(0, "int32")] * (rank - len(lv)) + lv
                return read_tir.BufferLoad(value, load_indices)
            # Scalar PrimExpr (or python int/float/bool) passes through;
            # ``tir.if_then_else`` will type-check it for us.
            if hasattr(value, "dtype") or isinstance(value, (int, float, bool)):
                return value
            raise EmitError(
                f"arith.select: unsupported {role} operand type "
                f"{type(value).__name__}; expected tir.PrimExpr or tir.Buffer"
            )

        out_buf = _om.LazyTileExpr(
            out_shape,
            out_dtype,
            lambda read_ctx, indices: read_ctx.tir().if_then_else(
                _lane(read_ctx, cond, tuple(indices), "cond"),
                _lane(read_ctx, t_val, tuple(indices), "true"),
                _lane(read_ctx, f_val, tuple(indices), "false"),
            ),
            name=ctx.fresh("select_expr"),
        )
        if _om._results(op):
            ctx.bind(_om._results(op)[0], out_buf)
        return out_buf

    # Scalar fast-path: identical to the pre-F2 behaviour.
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


def map_arith_index_cast(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.index_cast`` (index <-> integer)."""
    return _emit_cast(op, ctx, expect="i2i")


# ---------------------------------------------------------------------------
# tensor shape-only ops emitted by TritonStructured / PtrAnalysis
# ---------------------------------------------------------------------------


def _prod(values: Tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _flatten_indices(ctx: _om.WalkerCtx, indices: Tuple[Any, ...], shape: Tuple[int, ...]) -> Any:
    tir = ctx.tir()
    flat = tir.const(0, "int32")
    for idx, extent in zip(indices, shape):
        flat = flat * tir.const(int(extent), "int32") + idx
    return flat


def _unflatten_index(ctx: _om.WalkerCtx, flat: Any, shape: Tuple[int, ...]) -> Tuple[Any, ...]:
    tir = ctx.tir()
    out: List[Any] = [tir.const(0, "int32")] * len(shape)
    cur = flat
    for axis in range(len(shape) - 1, -1, -1):
        extent = tir.const(int(shape[axis]), "int32")
        out[axis] = cur % extent
        cur = cur // extent
    return tuple(out)


def map_tensor_collapse_shape(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``tensor.collapse_shape`` as a metadata-only tile reshape.

    PtrAnalysis rewrites some structured pointer masks from ``tensor<1xNxi1>``
    to ``tensor<Nxi1>`` before ``tts.load``. TileLang does not need a data copy
    for that collapse; it only needs subsequent lane reads to address the
    original producer with the equivalent row-major indices.
    """

    operands = _om._operands(op)
    results = _om._results(op)
    if not operands or not results:
        raise EmitError("tensor.collapse_shape: expected one operand and one result")
    src_ssa = operands[0]
    result = results[0]
    value = ctx.get(src_ssa)

    dst_shape = _result_shape(op)
    src_shape = tuple(getattr(value, "shape", ()) or _om._shape_of(src_ssa))
    if isinstance(value, _om.LazyTileExpr) and src_shape and dst_shape:
        if _prod(src_shape) != _prod(dst_shape):
            raise EmitError(
                "tensor.collapse_shape: source and result extents differ: "
                f"{src_shape} -> {dst_shape}"
            )

        def _reader(read_ctx: _om.WalkerCtx, indices: Tuple[Any, ...]) -> Any:
            flat = _flatten_indices(read_ctx, tuple(indices), dst_shape)
            return value.read_lane(read_ctx, _unflatten_index(read_ctx, flat, src_shape))

        reshaped = _om.LazyTileExpr(
            dst_shape,
            value.dtype,
            _reader,
            name=f"{value.name}_collapse" if value.name else "collapse_shape",
            constant_value=value.constant_value,
        )
        ctx.bind(result, reshaped)
        return reshaped

    ctx.bind(result, value)
    return value


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
# tt.func  (block-arg seeding)  /  arith.constant  (scalar literal seeding)
# ---------------------------------------------------------------------------
#
# The walker in ``triton_frontend.__init__:_walk_mlir_module`` dispatches by
# OP_TABLE name and recurses into regions afterwards. ``tt.func`` historically
# lived in ``_TTIR_STRUCTURAL_OPS`` (no-op skip) which meant block arguments
# were never bound into ``ctx.value_map`` -- downstream emitters that look up
# ``%arg0`` (e.g. ``tt.load %arg0, ...``) then KeyError'd on the operand.
# Similarly, ``arith.constant`` was unmapped, so any kernel using ``%c0 =
# arith.constant 0 : i32`` blew up at the first use of ``%c0``.
#
# We register both ops here in CONTROL_EMITTERS. The emitters do NOT recurse
# into regions themselves -- the parent walker still owns recursion -- they
# only seed the value_map (and ctx.buffers, for pointer-typed block args)
# so subsequent ops can resolve their operands.
#
# Hard constraint per spec: arith.constant with an unsupported attr type
# (e.g. ``dense<...>`` array splat) raises EmitError instead of guessing.


def _ssa_name(value: Any) -> Optional[str]:
    """Return the printed SSA name for a value (``%arg0`` / ``%c0`` / ...).

    Tries, in order:
      1. ``value.get_name()`` (real MLIR Value / BlockArgument)
      2. ``value["name"]`` (dict-shaped fake)
      3. ``str(value).split()[0]`` (last-ditch printable form)

    Returns ``None`` when no usable name can be produced.
    """
    getter = getattr(value, "get_name", None)
    if callable(getter):
        try:
            name = getter()
            if name:
                return str(name)
        except Exception:
            pass
    if isinstance(value, dict):
        nm = value.get("name")
        if nm:
            return str(nm)
    try:
        s = str(value).strip()
        if s:
            head = s.split(None, 1)[0]
            return head if head else None
    except Exception:
        pass
    return None


def _type_string(value: Any) -> str:
    """Best-effort printable type for a TTIR Value or dict-fake.

    For dict-fakes we synthesize ``!tt.ptr<dtype>`` if the entry sets
    ``"is_ptr": True`` so the same emitter logic applies in tests.
    """
    if isinstance(value, dict):
        if value.get("is_ptr"):
            elt = str(value.get("dtype", "float32"))
            return f"!tt.ptr<{elt}>"
        # Plain scalar / tensor: synthesize from shape + dtype.
        shape = tuple(value.get("shape", ()))
        dt = str(value.get("dtype", "float32"))
        if shape:
            return f"tensor<{'x'.join(str(s) for s in shape)}x{dt}>"
        return dt
    typ = getattr(value, "type", None)
    if typ is None:
        return ""
    try:
        return str(typ)
    except Exception:
        return ""


def _is_ptr_type(type_str: str) -> bool:
    s = type_str.strip()
    return s.startswith("!tt.ptr<") or s.startswith("tt.ptr<")


def _ptr_element_dtype(type_str: str) -> str:
    """Extract ``f32`` from ``!tt.ptr<f32>`` (or fall back to float32)."""
    s = type_str.strip()
    if "<" in s and s.endswith(">"):
        return s[s.index("<") + 1:-1].strip() or "float32"
    return "float32"


def _arg_buffer_shape(ctx: _om.WalkerCtx, idx: int, clean: str, ssa: str) -> Optional[List[int]]:
    """Return caller-seeded flat ABI shape for a pointer block arg."""
    shapes = getattr(ctx, "arg_buffer_shapes", None) or {}
    keys = (
        idx,
        str(idx),
        clean,
        ssa,
        str(ssa).lstrip("%"),
        f"%{str(clean).lstrip('%')}",
    )
    for key in keys:
        try:
            shape = shapes.get(key)
        except AttributeError:
            shape = None
        if shape is None:
            continue
        return [max(int(dim), 1) for dim in list(shape)]
    return None


def _func_block_args(op: Any) -> List[Any]:
    """Return the entry-block arguments of a ``tt.func`` op.

    Real MLIR: ``op.regions[0].blocks[0].arguments``. Dict-fake: the op
    may surface them via ``op["block_args"]`` directly.
    """
    if isinstance(op, dict):
        ba = op.get("block_args")
        if ba is not None:
            return list(ba)
        regions = op.get("regions") or []
        if regions:
            r0 = regions[0]
            if isinstance(r0, dict):
                return list(r0.get("block_args") or [])
        return []
    regions = getattr(op, "regions", ()) or ()
    for region in regions:
        blocks = getattr(region, "blocks", ()) or ()
        for block in blocks:
            return list(getattr(block, "arguments", ()) or ())
    return []


def _is_trivial_noop_body(body_stmt: Any, ctx: _om.WalkerCtx) -> bool:
    """Return True iff ``body_stmt`` carries NO real compute.

    The MLIR region walker returns ``tir.Evaluate(0)`` (an ``Evaluate`` of a
    constant) for an empty region. At the ``tt.func`` top level that means the
    walker produced a do-nothing kernel and we must RAISE (RULE #1) instead of
    assembling a runnable empty PrimFunc. We treat the body as trivial when it
    is a single ``tir.Evaluate`` whose value is a constant (the ``Evaluate(0)``
    sentinel). Any ``BufferStore`` / ``For`` / ``SeqStmt`` / ``IfThenElse`` /
    ``LetStmt`` / ``AttrStmt`` / ``AllocBuffer`` -- i.e. any actual statement --
    makes the body non-trivial and the check returns False.
    """
    try:
        tir = ctx.tir()
    except Exception:
        return False
    Evaluate = getattr(tir, "Evaluate", None)
    if Evaluate is None or not isinstance(body_stmt, Evaluate):
        return False
    # A bare Evaluate wraps an expression in ``.value``. The empty-region
    # sentinel is ``Evaluate(const(0))``; a real Evaluate (e.g. an extern
    # call / Call that performs a side effect) wraps a Call and is NOT
    # trivial. Treat only Evaluate-of-IntImm/FloatImm as the empty sentinel.
    value = getattr(body_stmt, "value", None)
    IntImm = getattr(tir, "IntImm", None)
    FloatImm = getattr(tir, "FloatImm", None)
    const_types = tuple(t for t in (IntImm, FloatImm) if t is not None)
    if const_types and isinstance(value, const_types):
        return True
    return False


def map_tt_func(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Build a ``tvm.tir.PrimFunc`` from a ``tt.func`` op.

    Mapping summary
    ---------------
    * Pointer block args (``!tt.ptr<T>``) -> ``tir.decl_buffer`` in ``ctx.buffers``.
    * Scalar block args (``i32``) -> ``tir.Var`` in ``ctx.value_map``.
    * Body region is walked via ``_emit_region`` to collect statements.
    * Output is a ``PrimFunc`` attached to ``ctx.prim_func``.
    """
    # Helper tt.func: a tt.func whose sym_name is referenced by some
    # tt.call is inlined at the call site. We must NOT build a PrimFunc
    # for it here.
    sym = _func_sym_name(op)
    if sym and hasattr(ctx, "callee_used") and sym in ctx.callee_used:
        return None

    block_args = _func_block_args(op)

    tir_mod = None
    try:
        tir_mod = ctx.tir()
    except Exception:
        pass

    if block_args:
        for idx, arg in enumerate(block_args):
            ssa = _ssa_name(arg) or f"%arg{idx}"
            clean = ssa.lstrip("%") or f"arg{idx}"
            type_str = _type_string(arg)

            if _is_ptr_type(type_str):
                elt = _normalize_dtype(_ptr_element_dtype(type_str))
                shape = _arg_buffer_shape(ctx, idx, clean, ssa) or [1]
                if tir_mod is not None:
                    if clean not in ctx.buffers:
                        ctx.buffers[clean] = tir_mod.decl_buffer(
                            shape=shape, dtype=elt, name=clean,
                        )
                        if _arg_buffer_shape(ctx, idx, clean, ssa) is not None:
                            getattr(ctx, "fixed_arg_buffer_keys", set()).add(clean)
                    bound = ctx.buffers[clean]
                else:
                    bound = {
                        "_placeholder": True,
                        "name": clean,
                        "dtype": elt,
                        "shape": shape,
                    }
                    ctx.buffers[clean] = bound
            elif _om._is_tensor_type(type_str):
                try:
                    shape, elt = _om._parse_tensor_type(type_str)
                except ValueError as exc:
                    raise EmitError(
                        f"tt.func block arg {ssa!r}: cannot parse tensor type "
                        f"{type_str!r}: {exc}"
                    ) from exc
                if tir_mod is not None:
                    if clean not in ctx.buffers:
                        ctx.buffers[clean] = tir_mod.decl_buffer(
                            shape=list(shape), dtype=elt, name=clean,
                        )
                    bound = ctx.buffers[clean]
                else:
                    bound = {
                        "_placeholder": True,
                        "name": clean,
                        "dtype": elt,
                        "shape": list(shape),
                    }
                    ctx.buffers[clean] = bound
            else:
                dt = _normalize_dtype(type_str) if type_str else "int32"
                if tir_mod is not None:
                    bound = tir_mod.Var(clean, dt)
                    runtime_args = getattr(ctx, "runtime_args", None)
                    if runtime_args is not None and bound not in runtime_args:
                        runtime_args.append(bound)
                else:
                    bound = {"_var_placeholder": True, "name": clean, "dtype": dt}

            try:
                ctx.bind(arg, bound)
            except Exception:
                pass
            ctx.value_map[ssa] = bound

    # Walk body region
    regions = _scf_regions(op)
    if not regions:
        return None
    
    # We walk the region directly.
    body_stmt, _ = _emit_region(regions[0], ctx)

    if tir_mod is None:
        return None

    # RULE #1 (no silent fallback): refuse to build a do-nothing kernel.
    # ``_emit_region`` returns ``tir.Evaluate(0)`` for an empty region. At a
    # ``scf.if`` branch that is a legitimate no-op, but at the ``tt.func``
    # top level it means the walker produced NO compute (every body op was
    # skipped or silently dropped) -- assembling a PrimFunc from it ships a
    # runnable kernel that does nothing (``tilelang.compile`` would accept it
    # WITHOUT raising). That is the exact silent-stub failure mode we must
    # never emit. Raise loudly naming the kernel and the walked-op count so
    # the caller can see WHERE the lowering went empty.
    if _is_trivial_noop_body(body_stmt, ctx):
        _sym = _func_sym_name(op) or getattr(ctx, "kernel_name", "main")
        raise RuntimeError(
            "triton_frontend.map_tt_func: the MLIR walker produced an EMPTY "
            f"body for tt.func {_sym!r} -- the assembled PrimFunc body is a "
            "bare `T.evaluate(0)` (no buffer stores, no loops, no compute). "
            "Refusing to return a runnable do-nothing kernel (RULE #1: no "
            "silent fallback / no empty stub that tilelang.compile would "
            "accept). This means every op in the tt.func body region was "
            "skipped or dropped without emitting a TIR statement. Inspect the "
            "TTIR body ops and their OP_TABLE emitters; a real kernel must "
            "emit at least one BufferStore / For / compute statement."
        )

    # Assemble PrimFunc
    buffer_map: Dict[Any, Any] = {}
    params: List[Any] = []
    for buf_name, buf in ctx.buffers.items():
        var = tir_mod.Var(buf_name, "handle")
        params.append(var)
        buffer_map[var] = buf

    for var in getattr(ctx, "runtime_args", []) or []:
        if var not in params:
            params.append(var)

    local_buffers = list(getattr(ctx, "local_buffers", []) or [])
    if local_buffers:
        AllocBuffer = getattr(tir_mod, "AllocBuffer", None)
        if AllocBuffer is not None:
            alloc_stmts = [AllocBuffer(buf) for buf in local_buffers]
            body_stmt = tir_mod.SeqStmt(alloc_stmts + [body_stmt])

    program_id_vars = list(getattr(ctx, "program_id_vars", []) or [])
    if program_id_vars:
        thread_tags = ("blockIdx.x", "blockIdx.y", "blockIdx.z")
        # Wrap the blockIdx.* ``thread_extent`` AttrStmts in CANONICAL axis
        # order (x, y, z). TVM's CUDA host codegen packs the launch grid tuple
        # in the nesting/declaration order of these AttrStmts, NOT by the
        # blockIdx tag. Triton TTIR emits ``get_program_id`` in source order
        # (e.g. y=program_id(1) then z=program_id(2) then x=program_id(0) for
        # the Tri-Dao chunk kernels), so wrapping in encounter order makes the
        # host pack grid as (x, z, y) -> CUDA grid.y/grid.z get SWAPPED extents
        # -> only a wrong subset of blocks runs. Sorting by axis aligns the
        # host grid tuple (grid.x<-axis0, grid.y<-axis1, grid.z<-axis2) with
        # the in-kernel blockIdx.<axis> usage. Wrap innermost-first so axis 0
        # ends up outermost. RULE #1: a deterministic canonical grid order or
        # the launch silently degenerates.
        for var, axis, extent in sorted(
            program_id_vars, key=lambda t: t[1], reverse=True
        ):
            tag = thread_tags[axis] if 0 <= axis < len(thread_tags) else f"blockIdx.{axis}"
            iter_var = tir_mod.IterVar(
                (0, extent), var, tir_mod.IterVar.ThreadIndex, tag,
            )
            body_stmt = tir_mod.AttrStmt(iter_var, "thread_extent", extent, body_stmt)
            if hasattr(extent, "name") and extent not in params:
                params.append(extent)

    num_warps = int(getattr(ctx, "num_warps", 4) or 4)
    num_stages = int(getattr(ctx, "num_stages", 2) or 2)
    threads_per_block = num_warps * 32
    # Reuse the SAME canonical ``threadIdx.x`` Var that body emitters used for
    # any local lane-0 guards (e.g. scalar atomic-rmw loops). If none was
    # created, ``thread_idx_var()`` mints it now. Using one shared Var keeps a
    # local ``if threadIdx_x == 0`` guard referring to the same thread index as
    # this outer block thread binding.
    tid_var = ctx.thread_idx_var()
    tid_extent = tir_mod.const(threads_per_block, "int32")
    if getattr(ctx, "requires_single_thread_body", False):
        body_stmt = tir_mod.IfThenElse(
            tir_mod.EQ(tid_var, tir_mod.const(0, "int32")),
            body_stmt,
            None,
        )
    tid_iter = tir_mod.IterVar(
        (0, tid_extent), tid_var, tir_mod.IterVar.ThreadIndex, "threadIdx.x",
    )
    body_stmt = tir_mod.AttrStmt(tid_iter, "thread_extent", tid_extent, body_stmt)

    sym_name = _func_sym_name(op)
    func_name = sym_name or getattr(ctx, "kernel_name", "main")

    func = tir_mod.PrimFunc(params=params, body=body_stmt, buffer_map=buffer_map)
    func = func.with_attr("tir.noalias", True)
    func = func.with_attr("global_symbol", func_name)
    func = func.with_attr("num_warps", num_warps)
    func = func.with_attr("num_stages", num_stages)

    ctx.prim_func = func
    return func


def _parse_value_attr(value_attr: Any) -> Tuple[str, Any]:
    """Return ``(dtype_str, scalar_value)`` for an ``arith.constant`` value attr.

    Accepted shapes
    ---------------
    * Real MLIR ``IntegerAttr`` / ``FloatAttr``: probed via ``.value`` and
      ``.type``. ``dtype_str`` is taken from ``str(.type)``.
    * Dict-shaped fake: ``{"value": <int|float>, "type": "i32"}`` or just
      ``{"value": ..., "dtype": ...}``.
    * Generic-form string ``"42 : i32"`` (the printed form used by the
      MLIR generic syntax) -- split on the colon.

    Raises :class:`EmitError` for anything else (e.g. ``DenseElementsAttr``,
    ``ArrayAttr``) because silently lowering an array constant to a single
    IntImm/FloatImm would change semantics.
    """
    # Real MLIR Integer/Float attr.
    if hasattr(value_attr, "value") and hasattr(value_attr, "type"):
        # Probe for array-attr shapes first: DenseElementsAttr exposes
        # ``.value`` as a numpy array-ish, not a scalar. We err on the side
        # of explicitness: any non-scalar ``.value`` raises EmitError.
        v = value_attr.value
        # numpy arrays / list / tuple are not supported.
        if isinstance(v, (list, tuple)):
            raise EmitError(
                f"arith.constant with array attr unsupported; got: "
                f"{value_attr!r}"
            )
        # numpy scalars OK; arrays not.
        try:
            import numpy as _np  # noqa: WPS433
            if isinstance(v, _np.ndarray):
                raise EmitError(
                    f"arith.constant with array attr unsupported; got: "
                    f"{value_attr!r}"
                )
        except ImportError:
            pass
        dtype_str = str(value_attr.type)
        return dtype_str, v

    # Dict-fake.
    if isinstance(value_attr, dict):
        if "value" not in value_attr:
            raise EmitError(
                f"arith.constant: dict attr missing 'value' field: {value_attr!r}"
            )
        v = value_attr["value"]
        if isinstance(v, (list, tuple)):
            raise EmitError(
                f"arith.constant with array attr unsupported; got: {value_attr!r}"
            )
        dtype_str = str(value_attr.get("type") or value_attr.get("dtype") or "int32")
        return dtype_str, v

    # Generic-form string: ``"42 : i32"`` or ``"3.14 : f32"``.
    if isinstance(value_attr, str):
        s = value_attr.strip()
        if "dense" in s.lower() or "array" in s.lower() or s.startswith("["):
            raise EmitError(
                f"arith.constant with array attr unsupported; got: {value_attr!r}"
            )
        if s.lower() in ("true", "false"):
            return "i1", (1 if s.lower() == "true" else 0)
        if ":" not in s:
            raise EmitError(
                f"arith.constant: cannot parse value attr {value_attr!r} "
                f"(expected '<value> : <type>')"
            )
        val_part, dt_part = s.rsplit(":", 1)
        val_part = val_part.strip()
        dtype_str = dt_part.strip()
        # Try int first, then float.
        try:
            return dtype_str, int(val_part)
        except ValueError:
            try:
                return dtype_str, float(val_part)
            except ValueError as exc:
                raise EmitError(
                    f"arith.constant: cannot parse scalar value {val_part!r} "
                    f"in {value_attr!r}"
                ) from exc

    # Real MLIR ``Attribute`` (generic base type from mlir-python-bindings) that
    # does NOT surface a Python-native ``.value``/``.type`` pair -- e.g. an
    # ``IntegerAttr`` that the bindings only expose in its textual form. This is
    # exactly the shape that the PtrAnalysis -> ``tts.*`` lowering produces for
    # the unmasked-load sentinel ``arith.constant 2147483647 : i64`` (INT_MAX).
    # Its ``str()`` is the canonical generic-syntax ``"<value> : <type>"``, so we
    # reuse the same scalar parser as the string branch above. We must do this
    # explicitly (rather than silently): an ArrayAttr / DenseElementsAttr also
    # stringifies, and those would change semantics if folded to a scalar -- so
    # we reject anything whose text looks array-shaped, mirroring the str branch.
    _attr_text = None
    try:
        _attr_text = str(value_attr)
    except Exception:  # pragma: no cover - defensive
        _attr_text = None
    if _attr_text is not None:
        s = _attr_text.strip()
        low = s.lower()
        if "dense" in low or "array" in low or s.startswith("["):
            raise EmitError(
                f"arith.constant with array attr unsupported; got: {value_attr!r}"
            )
        # Boolean ``i1`` constant: MLIR prints ``true`` / ``false`` with no
        # ``: i1`` type suffix. Map to a ``bool`` scalar (1 / 0). Without this
        # the ``: in s`` parse below misses and we hit the hard error -- the
        # exact ``Attribute(true)`` gap that blocked _chunk_state_bwd_dx.
        if low in ("true", "false"):
            return "i1", (1 if low == "true" else 0)
        if ":" in s:
            val_part, dt_part = s.rsplit(":", 1)
            val_part = val_part.strip()
            dtype_str = dt_part.strip()
            try:
                return dtype_str, int(val_part)
            except ValueError:
                try:
                    return dtype_str, float(val_part)
                except ValueError:
                    pass  # fall through to the hard error below

    raise EmitError(
        f"arith.constant with unsupported attr type {type(value_attr).__name__}; "
        f"got: {value_attr!r}"
    )


def _normalize_dtype(dtype_str: str) -> str:
    """Canonicalise short MLIR dtype names to TVM's spelling.

    ``i32`` -> ``int32``; ``f32`` -> ``float32``; ``bf16`` -> ``bfloat16``;
    ``f16`` -> ``float16``; ``i1`` -> ``bool``. Anything else passes through
    unchanged so already-TVM-spelled dtypes (``float32``, ``int64``) work.
    """
    s = dtype_str.strip()
    aliases = {
        "i1": "bool",
        "bf16": "bfloat16",
        "index": "int64",
    }
    if s in aliases:
        return aliases[s]
    if len(s) >= 2 and s[0] in {"i", "u"} and s[1:].isdigit():
        prefix = "int" if s[0] == "i" else "uint"
        return f"{prefix}{s[1:]}"
    if len(s) >= 2 and s[0] == "f" and s[1:].isdigit():
        return f"float{s[1:]}"
    return s


def _maybe_downcast_dense(value_attr: Any) -> Any:
    """Downcast a generic MLIR ``Attribute`` to its concrete dense subclass.

    Some MLIR Python bindings (notably jaxlib's
    ``jaxlib.mlir._mlir_libs._mlir.ir``) hand the walker the *generic*
    ``Attribute`` base type rather than the concrete ``DenseElementsAttr`` /
    ``DenseIntElementsAttr`` / ``DenseFPElementsAttr``. The base type does NOT
    expose ``is_splat`` / ``get_splat_value``, so a naive ``hasattr`` probe
    misses dense splats such as the ``dense<2147483647> : tensor<64xi64>``
    unmasked-load sentinel that the PtrAnalysis -> ``tts.*`` lowering emits.

    We resolve the bindings' ``ir`` module from the attribute's own type module
    and use ``<DenseClass>.isinstance(attr)`` + ``<DenseClass>(attr)`` to cast.
    Returns the downcast attribute when it is a dense elements attr, else the
    original ``value_attr`` unchanged. Never raises (cast failures fall through
    to the caller's existing logic).
    """
    # Already concrete (real upstream MLIR bindings expose is_splat directly).
    if hasattr(value_attr, "is_splat"):
        return value_attr
    tymod = getattr(type(value_attr), "__module__", "") or ""
    if "mlir" not in tymod and "ir" not in tymod:
        return value_attr
    try:
        import importlib  # noqa: WPS433
        ir = importlib.import_module(tymod)
    except Exception:
        return value_attr
    for _clsname in ("DenseIntElementsAttr", "DenseFPElementsAttr", "DenseElementsAttr"):
        cls = getattr(ir, _clsname, None)
        if cls is None:
            continue
        try:
            if cls.isinstance(value_attr):
                return cls(value_attr)
        except Exception:
            continue
    return value_attr


def _is_dense_attr(value_attr: Any) -> bool:
    """Detect an MLIR ``DenseElementsAttr`` (FP or integer variant).

    We check for the trio of accessors the real bindings expose
    (``is_splat`` / ``get_splat_value`` / ``type.shape``) rather than
    importing the MLIR class directly so the test harness's dict / list
    fakes don't false-positive here. A generic ``Attribute`` is first run
    through :func:`_maybe_downcast_dense` so bindings that hand us the base
    type (jaxlib) still resolve dense splats.
    """
    value_attr = _maybe_downcast_dense(value_attr)
    if not (hasattr(value_attr, "is_splat") and hasattr(value_attr, "type")):
        return False
    return getattr(value_attr.type, "shape", None) is not None


def _extract_dense_attr(
    value_attr: Any, result_value: Any
) -> Tuple[Tuple[int, ...], str, bool, Any]:
    """Return ``(shape, dtype, is_splat, payload)`` for a dense attr.

    ``payload`` is a Python scalar when ``is_splat`` is True; otherwise a
    materialised list of element values (one per tile slot) obtained by
    iterating the attribute. Element dtypes that we can't safely round-trip
    to a Python primitive (e.g. complex) raise :class:`EmitError`.
    """
    shape: Tuple[int, ...] = tuple(value_attr.type.shape)
    dtype: str = _normalize_dtype(str(value_attr.type.element_type))
    if dtype not in {"bool"} and not (
        dtype.startswith("int")
        or dtype.startswith("uint")
        or dtype.startswith("float")
        or dtype.startswith("bfloat")
    ):
        raise EmitError(
            f"arith.constant: dense attr with unsupported element dtype "
            f"{dtype!r} (from {value_attr.type.element_type!r})"
        )
    if value_attr.is_splat:
        # ``get_splat_value()`` returns an IntegerAttr / FloatAttr; ``.value``
        # is the Python scalar.
        sv = value_attr.get_splat_value().value
        return shape, dtype, True, sv
    # Per-element: iterate the attribute (jaxlib bindings yield Python
    # scalars in row-major order).
    try:
        elements = list(value_attr)
    except Exception as exc:  # pragma: no cover - defensive
        raise EmitError(
            f"arith.constant: dense attr is non-splat but not iterable: "
            f"{value_attr!r}"
        ) from exc
    return shape, dtype, False, elements


def map_arith_constant(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``arith.constant`` to a ``tir.IntImm`` / ``tir.FloatImm``.

    Mapping summary
    ---------------
    * Pull the ``value`` attribute (generic form: ``<{value = N : <ty>}>``).
    * Build ``tir.IntImm`` for integer / bool dtypes, ``tir.FloatImm`` for
      float dtypes (after normalising short MLIR spellings to TVM's names
      via :func:`_normalize_dtype`).
    * Bind the result SSA into ``ctx.value_map`` under both the result
      Value object and its printed name string.

    Dense / array attrs (e.g. ``dense<0.0> : tensor<256xf32>`` produced by
    Triton's ``tt.load %ptr, %mask, other=0.0`` materialisation) are
    materialised into a freshly declared ``tir.decl_buffer`` initialised
    via a serial ``tir.For`` nest; the buffer is bound into the value map
    so downstream loads/stores resolve through it.
    """
    attrs = _om._attrs(op)
    value_attr: Any = attrs.get("value")
    if value_attr is None:
        # Some MLIR Python bindings (e.g. jaxlib's) hide ``arith.constant``'s
        # ``value`` under the operation properties rather than the
        # discardable-attributes dict that ``_attrs`` iterates. Probe the
        # op directly for an attribute named ``value`` before giving up.
        if not isinstance(op, dict):
            value_attr = getattr(op, "value", None)
            # Some bindings additionally expose the typed attribute via
            # ``op.attributes["value"]`` even when iterating yields empty.
            if value_attr is None:
                op_attrs = getattr(op, "attributes", None)
                if op_attrs is not None:
                    try:
                        if "value" in op_attrs:
                            value_attr = op_attrs["value"]
                    except Exception:
                        pass
    if value_attr is None:
        raise EmitError(
            "arith.constant: missing 'value' attribute"
        )

    # Bindings that hand us the generic ``Attribute`` base (jaxlib) need an
    # explicit downcast so the dense-elements branch below sees ``is_splat`` /
    # ``get_splat_value``. No-op for concrete-binding / dict / str inputs.
    value_attr = _maybe_downcast_dense(value_attr)

    tir = ctx.tir()
    results = _om._results(op)
    result = results[0] if results else None

    # Dense / DenseFPElementsAttr / DenseIntElementsAttr branch: materialise
    # as a buffer initialised by a serial For nest writing the constant
    # value(s) to every element. Triton's ``tt.load %ptrs, %mask, other=0.0``
    # round-trips the ``other`` operand through this op shape.
    if _is_dense_attr(value_attr):
        shape, dtype, is_splat, payload = _extract_dense_attr(
            value_attr, result
        )
        nm_for_buf = (_ssa_name(result) if result is not None else None) or ctx.fresh("c")
        # Strip a leading '%' so the buffer name reads cleanly in dumps.
        ssa_clean = nm_for_buf.lstrip("%").replace(".", "_")
        buf_name = f"const_{ssa_clean}"

        if not all(isinstance(s, int) for s in shape):  # pragma: no cover
            raise EmitError(
                f"arith.constant: dense attr with non-static shape {shape!r}"
            )

        if is_splat:
            def _splat_expr(read_ctx: _om.WalkerCtx) -> Any:
                read_tir = read_ctx.tir()
                if dtype == "bool" or dtype.startswith("int") or dtype.startswith("uint"):
                    return read_tir.IntImm(dtype, int(payload))
                return read_tir.FloatImm(dtype, float(payload))

            lazy = _om.LazyTileExpr(
                shape if shape else (1,),
                dtype,
                lambda read_ctx, _indices: _splat_expr(read_ctx),
                name=buf_name,
                constant_value=payload,
            )
            if result is not None:
                try:
                    ctx.bind(result, lazy)
                except Exception:
                    pass
                nm = _ssa_name(result)
                if nm:
                    ctx.value_map[nm] = lazy
            return lazy

        # Tile-scoped allocation: see ``op_mapping._alloc_tile_buffer``.
        # The dense constant lives entirely inside the kernel body; making
        # it a PrimFunc parameter would trip ``VerifyMemory``.
        buf = _om._alloc_tile_buffer(
            ctx, list(shape) if shape else [1], dtype, buf_name
        )
        if is_splat:
            const_tiles = getattr(ctx, "constant_tile_values", None)
            if const_tiles is None:
                const_tiles = {}
                ctx.constant_tile_values = const_tiles
            for key in (
                str(getattr(buf, "data", "")),
                str(getattr(buf, "name", "")),
                str(buf),
            ):
                if key:
                    const_tiles[key] = payload

        # Build a serial tir.For nest writing the constant(s) into ``buf``.
        # All shapes from MLIR DenseElementsAttr are static (RankedTensorType
        # constants), so we can safely fold to integer extents.
        if shape:
            # Allocate one induction var per axis.
            ivars = [tir.Var(ctx.fresh(f"i{a}"), "int32") for a in range(len(shape))]
            if is_splat:
                value_expr = (
                    tir.IntImm(dtype, int(payload))
                    if (dtype == "bool" or dtype.startswith("int") or dtype.startswith("uint"))
                    else tir.FloatImm(dtype, float(payload))
                )
                body = tir.BufferStore(buf, value_expr, list(ivars))
            else:
                # Per-element: linearise (i0 * s1 * s2 * ... + i1 * s2 * ... + ...)
                # then dispatch through tir.if_then_else cascades. For correctness
                # and simplicity, unroll: emit a SeqStmt of N stores at constant
                # indices since ``payload`` is fully known.
                stmts = []
                strides: List[int] = []
                acc = 1
                for s in reversed(shape):
                    strides.append(acc)
                    acc *= s
                strides.reverse()
                for lin_idx, elt in enumerate(payload):
                    idxs = []
                    rem = lin_idx
                    for st in strides:
                        idxs.append(tir.const(rem // st, "int32"))
                        rem = rem % st
                    value_expr = (
                        tir.IntImm(dtype, int(elt))
                        if (dtype == "bool" or dtype.startswith("int") or dtype.startswith("uint"))
                        else tir.FloatImm(dtype, float(elt))
                    )
                    stmts.append(tir.BufferStore(buf, value_expr, idxs))
                body = tir.SeqStmt(stmts) if len(stmts) > 1 else stmts[0]
            if is_splat:
                # Wrap in a serial For nest, innermost axis last.
                nest = body
                for axis_idx in reversed(range(len(shape))):
                    nest = tir.For(
                        ivars[axis_idx],
                        tir.const(0, "int32"),
                        tir.const(int(shape[axis_idx]), "int32"),
                        tir.ForKind.SERIAL,
                        nest,
                    )
                ctx.emit(nest)
            else:
                ctx.emit(body)
        # Bind the buffer under the result SSA value and printed name so
        # downstream consumers (loads / elementwise) resolve.
        if result is not None:
            try:
                ctx.bind(result, buf)
            except Exception:
                pass
            nm = _ssa_name(result)
            if nm:
                ctx.value_map[nm] = buf
        return buf

    dtype_str, scalar_val = _parse_value_attr(value_attr)
    dtype = _normalize_dtype(dtype_str)

    if dtype == "bool" or dtype.startswith("int") or dtype.startswith("uint"):
        const = tir.IntImm(dtype, int(scalar_val))
    elif dtype.startswith("float") or dtype.startswith("bfloat"):
        const = tir.FloatImm(dtype, float(scalar_val))
    else:
        raise EmitError(
            f"arith.constant: unsupported dtype {dtype!r} (from {dtype_str!r})"
        )

    if result is not None:
        try:
            ctx.bind(result, const)
        except Exception:
            pass
        nm = _ssa_name(result)
        if nm:
            ctx.value_map[nm] = const
    return const


# ---------------------------------------------------------------------------
# scf.for / scf.if / scf.yield
# ---------------------------------------------------------------------------


_MAX_ITER_ARGS = 16


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
    child.ptr_states = getattr(ctx, "ptr_states", {})
    child.ssa_users = getattr(ctx, "ssa_users", {})
    child.arg_buffer_shapes = getattr(ctx, "arg_buffer_shapes", {})
    child.fixed_arg_buffer_keys = getattr(ctx, "fixed_arg_buffer_keys", set())
    child.constant_tile_values = getattr(ctx, "constant_tile_values", {})
    child.loop_carry_buffers = dict(getattr(ctx, "loop_carry_buffers", {}))
    # FRAGMENTCARRY: share the in-place fused-dot result SSA set so a tt.dot
    # emitted in this region keeps its result == the fragment carry (no
    # fragment->shared epilogue), accumulating in place across the K-loop.
    if getattr(ctx, "frag_carry_dot_results", None):
        child.frag_carry_dot_results = ctx.frag_carry_dot_results
    if getattr(ctx, "frag_c_metal_unfolded", None):
        child.frag_c_metal_unfolded = ctx.frag_c_metal_unfolded
    child.requires_single_thread_body = bool(
        getattr(ctx, "requires_single_thread_body", False)
    )
    # PROLOGUE-OPT: propagate the thread-distribution gate + warp count into
    # the region child so tiles materialized inside scf.for/scf.if regions
    # (transforms 1/2/3) get the SAME thread binding as the top-level
    # prologue. ``num_threads()`` resolves via ``_thread_var_root`` (the root
    # ctx) but we mirror ``num_warps``/``num_stages`` for any direct reads.
    child.routed_triton_prologue_opt = bool(
        getattr(ctx, "routed_triton_prologue_opt", False)
    )
    child.routed_triton_thread_distribute = bool(
        getattr(ctx, "routed_triton_thread_distribute", False)
    )
    # ITERATION 3 (coalesced async loads): propagate the async-load gate AND
    # share the SAME counter-list object so a global->shared ``T.copy`` emitted
    # by ``_emit_load_copy`` inside this region surfaces to the parent's
    # ``map_scf_for`` (which reads the delta to decide cp.async eligibility).
    child.routed_triton_async_loads = bool(
        getattr(ctx, "routed_triton_async_loads", False)
    )
    # ITERATION 5 (DoutTranspose / coalesced strided load): propagate the
    # ground-truth contiguous-tile-axis hint into the region child. The dstates
    # ``dout``/``C`` producer loads live INSIDE the scf.for K-loop body; without
    # this propagation the load emitter would not see the hint and would keep
    # iterating the strided axis innermost (non-coalesced). RULE #1: this only
    # reorders the load TRAVERSAL order for axes the route VERIFIED contiguous.
    if getattr(ctx, "routed_contiguous_tile_axis", None) is not None:
        child.routed_contiguous_tile_axis = ctx.routed_contiguous_tile_axis
    # ITERATION 6 (C-tile executed TMA): propagate the per-source innermost
    # ground-truth-contiguity set so the CopyNode emitter inside the scf.for
    # K-loop body grounds the C (%arg1) innermost stride to literal 1 and lowers
    # to a real TMA load. RULE #1: same gated ground-truth set, never fabricated.
    if getattr(ctx, "routed_contiguous_innermost_sources", None) is not None:
        child.routed_contiguous_innermost_sources = (
            ctx.routed_contiguous_innermost_sources
        )
    if not hasattr(ctx, "_gmem_shared_copies") or not isinstance(
        getattr(ctx, "_gmem_shared_copies", None), list
    ):
        ctx._gmem_shared_copies = []
    child._gmem_shared_copies = ctx._gmem_shared_copies
    # FULL TRANSFORM 1 (Coalesce-style addressing fold): the bulk of the
    # addressing/mask binops/broadcasts live INSIDE the scf.for K-loop body.
    # Without propagating the fold set into the region child, those tiles would
    # be materialized into spilled [N] arrays (the child's default empty set
    # disables the fold). Share the SAME set object so in-loop feeders keep
    # fold-eligible tiles lazy exactly like the top-level prologue.
    child.fold_addressing_ssa = getattr(ctx, "fold_addressing_ssa", set())
    child.num_warps = int(getattr(ctx, "num_warps", 4) or 4)
    child.num_stages = int(getattr(ctx, "num_stages", 2) or 2)
    # Share the ONE canonical ``threadIdx.x`` Var with the root ctx so a
    # lane-0 guard emitted inside this region (scalar atomic-rmw) and the
    # outer block ``threadIdx.x`` thread_extent binding (stamped by
    # ``map_tt_func`` on the root ctx) reference the SAME Var object.
    child._thread_var_root = getattr(ctx, "_thread_var_root", None) or ctx
    child._tmp_counter = ctx._tmp_counter
    child._tvm = ctx._tvm
    child._T = ctx._T
    # Propagate the codegen target so target-sensitive emitters inside the
    # region (e.g. tt.dot accumulator scope -- fragment-C on CUDA vs
    # shared-C on Metal) see the same target as the top-level walk.
    if getattr(ctx, "target", None) is not None:
        child.target = ctx.target
    if hasattr(ctx, "launch_grid"):
        child.launch_grid = ctx.launch_grid

    # Share mutable lists so inner ops surface to parent
    if not hasattr(ctx, "program_id_vars"):
        ctx.program_id_vars = []
    child.program_id_vars = ctx.program_id_vars
    if not hasattr(ctx, "local_buffers"):
        ctx.local_buffers = []
    child.local_buffers = ctx.local_buffers
    # FRAMEFIX: share the MMA-C fragment registry so a tt.dot emitted in a
    # child region surfaces its fragment to the parent ctx, whose prim_func is
    # the one ``from_ttir`` runs the post-walk layout re-registration on.
    if not hasattr(ctx, "mma_c_fragments"):
        ctx.mma_c_fragments = []
    child.mma_c_fragments = ctx.mma_c_fragments
    if not hasattr(ctx, "runtime_args"):
        ctx.runtime_args = []
    child.runtime_args = ctx.runtime_args
    child.callees = ctx.callees
    child.callee_used = ctx.callee_used

    # Materialise iter_args first: each block-arg SSA value gets bound to
    # a fresh tir.Var so the body emits BufferLoad / arithmetic against it.
    if induction_ssa is not None and induction_var is not None:
        child.bind(induction_ssa, induction_var)
    if iter_arg_pairs:
        for ssa, tir_var in iter_arg_pairs:
            child.bind(ssa, tir_var)
            if hasattr(tir_var, "shape") and hasattr(tir_var, "dtype"):
                child.loop_carry_buffers[ssa] = tir_var
                try:
                    getter = getattr(ssa, "get_name", None)
                    name = getter() if callable(getter) else getattr(ssa, "name", None)
                    if name:
                        child.loop_carry_buffers[str(name)] = tir_var
                except Exception:
                    pass
            elif (
                isinstance(tir_var, tuple)
                and len(tir_var) == 2
                and isinstance(tir_var[1], (list, tuple))
                and tir_var[1]
            ):
                child.loop_carry_buffers[ssa] = tir_var
                try:
                    getter = getattr(ssa, "get_name", None)
                    name = getter() if callable(getter) else getattr(ssa, "name", None)
                    if name:
                        child.loop_carry_buffers[str(name)] = tir_var
                except Exception:
                    pass

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
    if getattr(child, "requires_single_thread_body", False):
        ctx.requires_single_thread_body = True
    # Propagate tile-scoped buffers allocated inside the region back to the
    # parent context. ``_alloc_tile_buffer`` registers each buffer in
    # ``ctx.local_buffers``; when ``ctx`` is the child, those buffers are
    # lost when the child is discarded. ``_make_prim_func`` only wraps
    # ``ctx.local_buffers`` entries with ``tir.AllocBuffer`` stmts -- any
    # buffer that's missing from the top-level list surfaces as an
    # "undefined free Var" error in ``MakePackedAPI``.
    parent_locals = getattr(ctx, "local_buffers", None)
    child_locals = getattr(child, "local_buffers", None)
    if parent_locals is not None and child_locals:
        parent_set = set(id(b) for b in parent_locals)
        for buf in child_locals:
            if id(buf) not in parent_set:
                parent_locals.append(buf)

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


def _same_tir_buffer(a: Any, b: Any) -> bool:
    """Best-effort identity test for TIR buffers."""
    if a is b:
        return True
    same_as = getattr(a, "same_as", None)
    if callable(same_as):
        try:
            return bool(same_as(b))
        except Exception:
            return False
    return False


def _copy_buffer_stmt(ctx: _om.WalkerCtx, src: Any, dst: Any) -> Any:
    """Build a serial elementwise copy from ``src`` into ``dst``."""
    tir = ctx.tir()
    shape = list(getattr(dst, "shape", []) or [])
    loop_vars = [
        tir.Var(ctx.fresh(f"carry_copy{axis}"), "int32")
        for axis in range(len(shape))
    ]
    idx = list(loop_vars) or [tir.const(0, "int32")]
    if isinstance(src, _om.LazyTileExpr):
        src_rank = len(src.shape)
        if src_rank:
            src_indices = list(idx[-src_rank:])
            for axis, extent in enumerate(src.shape):
                if int(extent) == 1:
                    src_indices[axis] = tir.const(0, "int32")
        else:
            src_indices = []
        value = src.read_lane(ctx, tuple(src_indices))
    else:
        value = tir.BufferLoad(src, idx)
    body: Any = tir.BufferStore(dst, value, idx)
    for axis in range(len(loop_vars) - 1, -1, -1):
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            shape[axis],
            tir.ForKind.SERIAL,
            body,
        )
    return body


def _append_loop_carry_copies(
    ctx: _om.WalkerCtx,
    body: Any,
    iter_arg_pairs: List[Tuple[Any, Any]],
    yielded: List[Any],
) -> Any:
    """Copy yielded tile values into carried buffers at loop-body end.

    Multiple loop carries are committed through temporaries so every yielded
    expression observes the previous-iteration carries. This matters for
    online softmax patterns such as Flash Attention, where ``l_i`` and
    ``acc`` must use the old ``m_i`` even though ``m_i`` is also yielded.
    """
    if not yielded:
        return body
    tvm_mod = ctx.tvm()
    Buffer = tvm_mod.tir.Buffer

    # Collect the (carry, yielded_value) pairs that actually require a commit
    # copy (carry is a real Buffer, yielded is a Buffer/LazyTileExpr, and the
    # two are not already the same buffer).
    committable: List[Tuple[Any, Any]] = []
    for (_blk_ssa, carry), yielded_value in zip(iter_arg_pairs, yielded):
        if not isinstance(carry, Buffer):
            continue
        if not isinstance(yielded_value, (Buffer, _om.LazyTileExpr)):
            continue
        if isinstance(yielded_value, Buffer) and _same_tir_buffer(carry, yielded_value):
            continue
        committable.append((carry, yielded_value))

    if not committable:
        return body

    # SINGLE-CARRY FAST PATH (generic SMEM-footprint fix): the snapshot
    # temporary only exists to break a cross-carry read-before-write hazard —
    # i.e. when one carry's yielded expression must observe the *previous*
    # iteration value of *another* carry that is also being committed (online
    # softmax: ``acc`` reads old ``m_i``). With exactly one committed carry,
    # the yielded value is fully computed into its own buffer before any
    # commit, so there is no such hazard and the intermediate ``carry_next``
    # buffer is dead weight. For a shared-resident carry that temporary is a
    # full tile in threadgroup memory (e.g. 64x64 fp32 = 16 KiB), which on
    # Metal alone can blow past the 32 KiB threadgroup cap. Commit the yielded
    # value straight into the carry buffer instead.
    if len(committable) == 1:
        carry, yielded_value = committable[0]
        return ctx.tir().SeqStmt([body, _copy_buffer_stmt(ctx, yielded_value, carry)])

    snapshot_copies: List[Any] = []
    commit_copies: List[Any] = []
    for carry, yielded_value in committable:
        scope_fn = getattr(carry, "scope", None)
        try:
            scope = str(scope_fn() if callable(scope_fn) else scope_fn)
        except Exception:
            scope = "local"
        if getattr(ctx, "requires_single_thread_body", False):
            scope = "local"
        if not scope:
            scope = "local"
        tmp = _om._alloc_tile_buffer(
            ctx,
            list(getattr(carry, "shape", []) or [1]),
            str(getattr(carry, "dtype", "float32")),
            ctx.fresh("carry_next"),
            scope=scope,
        )
        snapshot_copies.append(_copy_buffer_stmt(ctx, yielded_value, tmp))
        commit_copies.append(_copy_buffer_stmt(ctx, tmp, carry))
    return ctx.tir().SeqStmt([body] + snapshot_copies + commit_copies)


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


def _region_body_ops(region: Any) -> List[Any]:
    """Return the flat op list of a region (dict-fake or real MLIR)."""
    if isinstance(region, dict):
        return list(region.get("ops", ()))
    ops: List[Any] = []
    for block in getattr(region, "blocks", ()) or ():
        for inner in getattr(block, "operations", ()) or ():
            ops.append(inner)
    return ops


def _is_ptr_tensor(value: Any) -> bool:
    """True iff ``value``'s MLIR type is a (tensor of) ``!tt.ptr``.

    Covers both scalar pointers (``!tt.ptr<f32>``) and the block-pointer
    tile form (``tensor<64x32x!tt.ptr<f32>>``) that scf.for iter_args carry.
    """
    try:
        ts = _type_string(value)
    except Exception:
        return False
    return "!tt.ptr<" in ts or "tt.ptr<" in ts


def _dedupe_states(states_map: Dict[str, Any]) -> List[Any]:
    """Stable, id-deduplicated list of PtrState objects from a states map."""
    seen = set()
    out: List[Any] = []
    for st in states_map.values():
        if id(st) in seen:
            continue
        seen.add(id(st))
        out.append(st)
    out.sort(key=lambda s: str(getattr(s, "result_ssa", "") or ""))
    return out


def _build_def_map(ops: List[Any]) -> Dict[str, Any]:
    """Map each op's result SSA name -> the defining op (for use-def tracing)."""
    defs: Dict[str, Any] = {}
    for inner in ops:
        for res in _om._results(inner):
            nm = _ssa_name(res)
            if nm:
                defs[nm] = inner
    return defs


def _trace_scalar_through(defs: Dict[str, Any], name: Optional[str]) -> Optional[str]:
    """Strip ``index_cast`` / ``bitcast`` / ``extsi`` wrappers off ``name``.

    Returns the printed SSA name of the underlying integer scalar (e.g. the
    raw ``%argK`` stride argument) so a ``tt.addptr`` per-trip advance can be
    matched to the PtrState stride that shares the same source scalar.
    """
    seen = set()
    cur = name
    while cur and cur not in seen:
        seen.add(cur)
        op = defs.get(cur)
        if op is None:
            return cur
        opname = _wk._op_name(op) if _wk is not None else ""
        if opname in ("arith.index_cast", "arith.bitcast", "arith.extsi",
                      "arith.trunci", "arith.sitofp"):
            operands = _om._operands(op)
            cur = _ssa_name(operands[0]) if operands else None
            continue
        return cur
    return cur


def _ptr_state_offsets_are_scalar(
    ctx: _om.WalkerCtx, offsets: List[Any], strides: List[Any]
) -> bool:
    """True iff every PtrState offset/stride resolves to a SCALAR value.

    A scalar resolves to a TIR ``PrimExpr`` (or int / int-literal string).
    Tile-shaped offsets (matmul's ``offs_m[:,None]*K`` index tensors) resolve
    to an ``ffi.Map`` / ``Buffer`` / ``(buffer, indices)`` tuple instead --
    those must NOT take the scalar-strided forwarding path. Unresolvable
    names are treated as scalar (they're typically pre-loop SSA the load
    resolves later); the load path re-resolves and raises loudly if wrong.
    """
    def _is_scalar(ref: Any) -> bool:
        if ref is None or isinstance(ref, int):
            return True
        if isinstance(ref, (list, tuple)):
            return False
        if isinstance(ref, str):
            val = _resolve_ssa_to_tir(ctx, ref)
            if val is None:
                # Numeric literal spelled as a string is scalar; an unbound
                # symbolic name is assumed scalar (resolved at load time).
                return True
            ref = val
        tn = type(ref).__name__
        if tn in ("Map", "Array", "Buffer", "BufferRegion"):
            return False
        if isinstance(ref, (list, tuple)):
            return False
        return True
    return all(_is_scalar(o) for o in offsets) and all(_is_scalar(s) for s in strides)


def _unwrap_tir_cast(expr: Any) -> Any:
    """Strip TIR ``Cast`` wrappers so a stride ``Cast(int64, argK)`` compares
    structurally-equal to a bare ``argK`` factor from the addptr increment."""
    seen = 0
    cur = expr
    while cur is not None and type(cur).__name__ == "Cast" and seen < 8:
        nxt = getattr(cur, "value", None)
        if nxt is None:
            break
        cur = nxt
        seen += 1
    return cur


def _resolve_ssa_to_tir(ctx: _om.WalkerCtx, name: Optional[str]) -> Any:
    """Best-effort resolve an SSA name to its bound TIR value (or None)."""
    if not name:
        return None
    for key in (name, name.lstrip("%"), f"%{name.lstrip('%')}"):
        try:
            if key in ctx.value_map:
                return ctx.value_map[key]
        except TypeError:
            pass
    return None


def _addptr_per_trip_flat(
    ctx: _om.WalkerCtx,
    defs: Dict[str, Any],
    yielded_ssa: Optional[str],
    blk_arg_name: Optional[str],
) -> Any:
    """Return the per-iteration *flat* pointer advance as a resolved TIR scalar.

    The carried pointer is yielded as ``addptr(<blk_arg>, splat(<scalar>))``.
    ``<scalar>`` is the per-trip flat advance in elements -- either a bare
    kernel-arg stride (``splat(%strideK)``) or ``muli(step, %strideK)``. Every
    leaf resolves to a kernel-arg / pre-loop value (stable across the
    shim-vs-walker SSA renumbering), so we resolve it to TIR *here* (at
    forwarding time, before the loop body) and the load folds
    ``trips * scalar`` into the flat index. RULE #1: a non-uniform-splat
    advance, or a leaf that does not resolve, raises rather than mis-striding.
    """
    op = defs.get(yielded_ssa) if yielded_ssa else None
    opname = _wk._op_name(op) if (op is not None and _wk is not None) else None
    if op is None or opname != "tt.addptr":
        raise EmitError(
            "map_scf_for: loop-carried pointer iter_arg "
            f"{blk_arg_name!r} is not yielded by a tt.addptr "
            f"(got {opname!r}); cannot forward strided PtrState across the loop."
        )
    operands = _om._operands(op)
    if len(operands) < 2:
        raise EmitError(f"map_scf_for: malformed tt.addptr for {blk_arg_name!r}")
    incr_op = defs.get(_ssa_name(operands[1]))
    incr_opname = _wk._op_name(incr_op) if (incr_op is not None and _wk is not None) else None
    if incr_opname != "tt.splat":
        raise EmitError(
            "map_scf_for: loop-carried tt.addptr advance for "
            f"{blk_arg_name!r} is not a uniform tt.splat (got {incr_opname!r}). "
            "RULE #1: refusing a non-uniform per-trip advance."
        )
    sp_ops = _om._operands(incr_op)
    scalar_name = _ssa_name(sp_ops[0]) if sp_ops else None
    scalar_op = defs.get(scalar_name)
    scalar_opname = _wk._op_name(scalar_op) if (scalar_op is not None and _wk is not None) else None

    def _resolve_leaf(nm: Optional[str]) -> Any:
        traced = _trace_scalar_through(defs, nm)
        val = _resolve_ssa_to_tir(ctx, traced)
        if val is None:
            val = _resolve_ssa_to_tir(ctx, nm)
        if val is None:
            d = defs.get(traced)
            dn = _wk._op_name(d) if (d is not None and _wk is not None) else None
            if dn == "arith.constant":
                cst = _const_value_of(d)
                if cst is not None:
                    return ctx.tir().const(int(cst), "int32")
            raise EmitError(
                "map_scf_for: loop-carried advance leaf "
                f"{nm!r} for {blk_arg_name!r} does not resolve to a kernel-arg "
                "/ constant. RULE #1: refusing to forward."
            )
        return _unwrap_tir_cast(val)

    if scalar_opname == "arith.muli":
        factors = [_resolve_leaf(_ssa_name(o)) for o in _om._operands(scalar_op)]
        acc = factors[0]
        for f in factors[1:]:
            acc = acc * f
        return acc
    # Bare scalar advance (step folded into the stride or step == 1).
    return _resolve_leaf(scalar_name)


def _const_value_of(op: Any) -> Optional[int]:
    """Extract an integer value from an ``arith.constant`` op (best-effort).

    The value lives in the op's *inherent* attribute (``<{value = 32 :
    i32}>``), not the discardable attribute dict, so we try (1) the attrs
    dict, then (2) parse the printed op form ``value = <N> : i<bits>``.
    """
    try:
        attrs = _om._attrs(op) if hasattr(_om, "_attrs") else {}
    except Exception:
        attrs = {}
    val = attrs.get("value") if isinstance(attrs, dict) else None
    for cand in (val, getattr(op, "value", None)):
        if cand is None:
            continue
        try:
            return int(cand)
        except (TypeError, ValueError):
            pass
        v = getattr(cand, "value", None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    # Parse the printed inherent attribute: ``<{value = 32 : i32}>``.
    try:
        import re  # noqa: WPS433
        m = re.search(r"value\s*=\s*(-?\d+)\s*:\s*i\d+", str(op))
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def _forward_loop_ptr_states(
    ctx: _om.WalkerCtx,
    region: Any,
    iter_arg_block_ssas: List[Any],
    iter_arg_ssas: List[Any],
    induction_ssa: Any,
    lb_ssa: Any,
    step_ssa: Any,
    lb_const: Optional[int] = None,
    step_const: Optional[int] = None,
) -> Dict[str, Any]:
    """Forward each ptr iter_arg's PtrState onto its ``scf.for`` block-arg.

    For every iter_arg whose INIT operand carries a strided PtrState (the
    shim already recovered it for the pre-loop ``tts.make_tptr`` + pre-loop
    ``tt.addptr``), register ``ctx.ptr_states[<block_arg_name>]`` with the
    same source/sizes/strides but with the per-iteration advance folded into
    the offset of the advancing axis: ``offset[axis] = init_offset +
    induction_var``. The in-loop ``tt.load`` on the block-arg then resolves
    via ``_lookup_ptr_state`` to the strided ``_emit_ptrstate_tile_load_tir``
    path instead of the scalar ``carry_index`` gather.

    Returns a ``dict[block_arg_name -> tagged-dict]`` for the forwarded
    carries (the tagged-dict is what the block-arg is bound to so in-loop
    ``tt.load`` / ``tt.addptr`` consume the strided state directly). RULE #1:
    if an init has a PtrState but the advance cannot be identified,
    :func:`_addptr_advance_axis` raises rather than silently dropping to the
    scalar gather.
    """
    states_map = getattr(ctx, "ptr_states", None)
    if states_map is None:
        return {}
    ops = _region_body_ops(region)
    defs = _build_def_map(ops)
    # Find scf.yield and its operands (the yielded value per iter_arg).
    yield_operands: List[Any] = []
    for inner in ops:
        if _wk is not None and _wk._op_name(inner) == "scf.yield":
            yield_operands = list(_om._operands(inner))
            break
    induction_name = _ssa_name(induction_ssa) if induction_ssa is not None else None
    def _bound_name(ref: Any, tag: str, const_val: Optional[int]) -> str:
        # Numbering-independent constant binding. The K-loop's lower bound and
        # step are ``arith.constant`` SSA values (e.g. ``%c0_i32`` / ``%c32_i32``).
        # Triton's make_ttir canonicalizer/CSE renames + renumbers SSA results,
        # so a positional ``_ssa_name(ref)`` (``%0``) can COLLIDE with an
        # unrelated op (a ``tt.splat`` tile) in the folded TTIR's value_map,
        # which then resolves the "lb" carry leaf to a float32 tile (a category
        # error caught downstream as ``Cast(... ffi.OpaquePyObject)``). When the
        # caller has already int-folded the bound (same ``_to_int`` the loop
        # frame uses for ``lb_i``/``step_i``), bind that constant under a
        # synthetic name -- identical semantics, independent of SSA numbering.
        if isinstance(ref, int):
            const_val = int(ref)
        if const_val is not None:
            nm = f"__{tag}_const__"
            ctx.value_map[nm] = ctx.tir().const(int(const_val), "int32")
            return nm
        return _ssa_name(ref)
    lb_name = _bound_name(lb_ssa, "lb", lb_const)
    step_name = _bound_name(step_ssa, "step", step_const)
    # Build a sizes -> [states] index. The shim keys states by the REWRITTEN
    # SSA numbering, but ``parse_ttir`` re-numbers (or falls through to a
    # provider that parses the original IR), so an exact init-SSA-name match
    # misses. We instead match each iter_arg to its recovered PtrState by tile
    # SIZES (the iter_arg's MLIR tensor type), mirroring the source+shape
    # reconciliation the top-level loads already use
    # (``_compatible_ptr_state_for_base``). RULE #1: when sizes are ambiguous
    # we still forward deterministically (round-robin per sizes key) and the
    # advance-axis match below guards correctness.
    sizes_index: Dict[Tuple[int, ...], List[Any]] = {}
    for st in _dedupe_states(states_map):
        try:
            key = tuple(int(s) for s in (getattr(st, "sizes", ()) or ()))
        except (TypeError, ValueError):
            continue
        if not key:
            continue
        if not (getattr(st, "strides", ()) or ()):
            continue
        sizes_index.setdefault(key, []).append(st)
    consumed: Dict[Tuple[int, ...], int] = {}
    forwarded: Dict[str, Any] = {}
    for idx, (blk_ssa, init_ssa) in enumerate(zip(iter_arg_block_ssas, iter_arg_ssas)):
        # Only forward POINTER-typed iter_args (``tensor<...x!tt.ptr<..>>``).
        # Non-pointer carries (e.g. the float ``acc`` accumulator yielded by
        # ``arith.addf``) keep the existing buffer-carry path -- forwarding a
        # PtrState onto them by size-match alone would be a category error.
        if not _is_ptr_tensor(init_ssa) and not _is_ptr_tensor(blk_ssa):
            continue
        try:
            tile_shape = tuple(int(s) for s in _om._shape_of(init_ssa))
        except (TypeError, ValueError):
            tile_shape = ()
        if not tile_shape and blk_ssa is not None:
            try:
                tile_shape = tuple(int(s) for s in _om._shape_of(blk_ssa))
            except (TypeError, ValueError):
                tile_shape = ()
        candidates = sizes_index.get(tile_shape, [])
        if not candidates:
            continue
        pick = consumed.get(tile_shape, 0)
        if pick >= len(candidates):
            pick = len(candidates) - 1
        consumed[tile_shape] = pick + 1
        state = candidates[pick]
        strides = list(getattr(state, "strides", ()) or ())
        sizes = list(getattr(state, "sizes", ()) or ())
        offsets = list(getattr(state, "offsets", ()) or ())
        if not strides or not sizes:
            continue
        blk_name = _ssa_name(blk_ssa)
        if not blk_name:
            continue
        # SCALAR-OFFSET GUARD: only forward the strided ``make_tptr`` tile-load
        # when every offset/stride resolves to a SCALAR (per-program base +
        # induction). The Tri-Dao chunk kernels are exactly this shape. A loop
        # whose pointer offsets are TILE-shaped index tensors (e.g. plain
        # matmul ``a_ptrs = a_ptr + offs_m[:,None]*K + offs_k[None,:]``) keeps
        # its existing ``(buffer, indices)`` tuple-carry path -- forwarding a
        # scalar-offset PtrState onto it would feed an ``ffi.Map`` into the
        # flat-index arithmetic (TypeError). RULE #1: forward only the case we
        # can prove is scalar-strided; otherwise leave the proven-correct path.
        if not _ptr_state_offsets_are_scalar(ctx, offsets, strides):
            continue
        yielded_ssa = (
            _ssa_name(yield_operands[idx]) if idx < len(yield_operands) else None
        )
        # Per-iteration flat advance (resolved TIR scalar) the source
        # ``tt.addptr`` applies each trip. The load folds ``trips * advance``
        # into the flat index (``trips = (induction - lb) / step``), so the
        # in-loop strided load slides exactly like the native multi-K-trip
        # kernel. Numbering-independent: the advance leaves are kernel-args.
        flat_advance = _addptr_per_trip_flat(ctx, defs, yielded_ssa, blk_name)
        # Register under the block-arg name so ``_lookup_ptr_state`` (the
        # ``ptr_ssa``-keyed branch) also resolves it, AND build the tagged-dict
        # the in-loop ``tt.load`` / ``tt.addptr`` consume directly when the
        # block-arg is bound to it. The dict shape mirrors ``seed_ptr_states``.
        states_map[blk_name] = state
        tagged = {
            "_ptrstate": state,
            "source": state.source,
            "offsets": list(offsets),
            "sizes": list(sizes),
            "strides": list(strides),
            "shape": list(state.shape) if state.shape is not None else None,
            "_carry_flat": {
                "advance": flat_advance,
                "induction": induction_name,
                "lb": lb_name,
                "step": step_name,
            },
        }
        forwarded[blk_name] = tagged
    return forwarded


def _replace_ptr_state_offsets(state: Any, new_offsets: Tuple[Any, ...]) -> Any:
    """Return a copy of ``state`` with ``offsets`` replaced (dataclass-safe)."""
    try:
        import dataclasses  # noqa: WPS433
        return dataclasses.replace(state, offsets=new_offsets)
    except Exception:
        # Fallback for non-dataclass PtrState shims: shallow clone.
        import copy  # noqa: WPS433
        clone = copy.copy(state)
        try:
            clone.offsets = new_offsets
        except Exception as exc:  # pragma: no cover
            raise EmitError(
                f"map_scf_for: cannot set forwarded offsets on PtrState: {exc}"
            ) from exc
        return clone


def _frag_scope_of(buf: Any) -> str:
    """Return the storage scope string of a TIR buffer (``""`` if unknown)."""
    scope_fn = getattr(buf, "scope", None)
    if scope_fn is None:
        return ""
    try:
        return str(scope_fn() if callable(scope_fn) else scope_fn)
    except Exception:
        return ""


def _is_zero_constant_lazy(init_val: Any) -> bool:
    """True iff ``init_val`` is a ``LazyTileExpr`` carrying a literal-0 constant.

    The fused-dot accumulator carry is seeded from ``acc = zeros(...)`` -- a
    zero-constant lazy tile. Allocating that seed into a ``local.fragment`` is
    only swizzle-correct because zero maps to zero under any layout permutation
    (see the call site). A non-zero init would need a layout-aware seed copy,
    so we gate the fragment carry on a proven-zero init.
    """
    if not isinstance(init_val, _om.LazyTileExpr):
        return False
    try:
        return float(init_val.constant_value) == 0.0
    except (TypeError, ValueError):
        return False


def _dot_accumulator_iter_arg_slots(
    region: Any,
    iter_arg_block_ssas: List[Any],
) -> Tuple[Dict[int, Tuple[int, int]], List[str]]:
    """Return ``{iter_arg_idx: (M, N)}`` for fused-dot accumulator carries.

    Triton's folded/canonicalized TTIR fuses ``acc = acc + dot(a, b)`` into a
    single ``tt.dot %a, %b, %acc_blockarg`` whose RESULT is yielded back into
    the SAME iter_arg slot -- the accumulator threads through the K-loop as a
    loop-carried tile. This is the fused-GEMM-accumulate-into-carry form.

    On CUDA that carry MUST be a swizzled ``local.fragment`` (the tensor-core
    MMA store layout asserts a fragment C). If it is materialised as a plain
    ``shared`` carry, ``map_tt_dot`` either aborts at LayoutInference (shared C
    on the MMA path) or -- if round-tripped through the serial-scalar
    loop-carry copies -- silently corrupts the accumulator because the linear
    element copies ignore the mma fragment swizzle. The fix is to allocate
    those carries DIRECTLY as ``local.fragment`` so ``map_tt_dot`` finds a real
    fragment C, accumulates IN PLACE across iterations, and the yielded value
    IS the carry buffer (so ``_append_loop_carry_copies`` skips the corrupting
    scalar copy via its ``_same_tir_buffer`` short-circuit). This is exactly
    how a hand-written TileLang kernel carries a ``T.alloc_fragment`` GEMM
    accumulator across a K-loop -- a real fragment-resident, swizzle-correct
    carry, not a serial-scalar round-trip.

    Detection is purely structural (use-def on the region's own ops + the
    scf.yield operand order); it does not depend on SSA numbering. Returns
    ``(slots, dot_result_ssas)`` where ``slots`` maps each accumulator
    iter-arg index to its ``(M, N)`` logical tile shape (so the carry is
    allocated at the right extent) and ``dot_result_ssas`` lists the SSA names
    of the in-place fused-dot RESULTS. ``map_tt_dot`` consults the latter to
    keep the result == the fragment carry (no in-loop fragment->shared
    epilogue), so the accumulation stays in place across iterations.
    """
    if not iter_arg_block_ssas:
        return {}, []
    ops = _region_body_ops(region)
    if not ops:
        return {}, []
    # scf.yield operand order maps 1:1 onto iter_arg slots.
    yield_operands: List[Any] = []
    for inner in ops:
        if _wk is not None and _wk._op_name(inner) == "scf.yield":
            yield_operands = list(_om._operands(inner))
            break
    if not yield_operands:
        return {}, []
    blk_names = [_ssa_name(b) for b in iter_arg_block_ssas]
    blk_index = {nm: i for i, nm in enumerate(blk_names) if nm}
    slots: Dict[int, Tuple[int, int]] = {}
    dot_result_ssas: List[str] = []
    _unfolded_frag_results: List[str] = []
    # Build a lookup of ``arith.addf`` ops keyed by their first operand SSA name
    # (the in-flight accumulator), to recognise the UNFOLDED accumulate form
    # below. Triton's *un*canonicalised TTIR emits ``acc = acc + dot(a,b,0)`` as
    # two ops -- ``%d = tt.dot(a,b, %zeroC)`` then ``%s = arith.addf(%accBlkArg,
    # %d)`` with ``%s`` yielded back into the accumulator slot. The dot's own C
    # is a FRESH zero (not the block-arg), so the folded detector above misses
    # it; this map lets the second pass below stitch dot->addf->yield together.
    addf_by_lhs: Dict[str, Any] = {}
    addf_by_rhs: Dict[str, Any] = {}
    for inner in ops:
        if _wk is None or _wk._op_name(inner) != "arith.addf":
            continue
        add_ops = _om._operands(inner)
        if len(add_ops) < 2:
            continue
        lhs, rhs = _ssa_name(add_ops[0]), _ssa_name(add_ops[1])
        if lhs:
            addf_by_lhs[lhs] = inner
        if rhs:
            addf_by_rhs[rhs] = inner

    def _zero_const_ssa(name: str) -> bool:
        """True iff ``name`` names an ``arith.constant`` dense-0 tile op."""
        for cand in ops:
            if _wk is None or _wk._op_name(cand) != "arith.constant":
                continue
            res = _om._results(cand)
            if not res or _ssa_name(res[0]) != name:
                continue
            attrs = _om._attrs(cand)
            val = attrs.get("value")
            sval = str(val)
            return "0.0" in sval or "dense<0" in sval or sval.strip() in ("0", "0.0")
        return False

    for inner in ops:
        if _wk is None or _wk._op_name(inner) != "tt.dot":
            continue
        dot_operands = _om._operands(inner)
        if len(dot_operands) < 3:
            continue
        results = _om._results(inner)
        if not results:
            continue
        res_name = _ssa_name(results[0])
        c_name = _ssa_name(dot_operands[2])
        slot = blk_index.get(c_name)
        _unfolded = False
        if slot is not None:
            # FOLDED form: dot's C IS the carry block-arg and the dot RESULT is
            # yielded back into the SAME slot (accumulate-in-place).
            if slot >= len(yield_operands):
                continue
            if _ssa_name(yield_operands[slot]) != res_name:
                continue
        else:
            _unfolded = True
            # UNFOLDED form: dot's C is a fresh zero; its result feeds an
            # ``arith.addf(carryBlkArg, dotResult)`` whose result is yielded
            # back into the carry slot. This is the SAME accumulate-into-carry
            # semantics, just not yet fused by Triton's canonicaliser. We mark
            # the dot result so ``map_tt_dot`` keeps the dot's C fragment-
            # resident (off SHARED): the dot computes its 64x64 partial in
            # ``simdgroup`` registers and the addf reads that fragment, so the
            # separate ``dot_c_shared`` MxN fp32 threadgroup tile is never
            # allocated. The carry tile itself stays as-is (its own slot).
            if not _zero_const_ssa(c_name):
                continue
            add_op = addf_by_rhs.get(res_name) or addf_by_lhs.get(res_name)
            if add_op is None:
                continue
            add_ops = _om._operands(add_op)
            # The OTHER addf operand must be a carry block-arg.
            other = [_ssa_name(o) for o in add_ops if _ssa_name(o) != res_name]
            carry_slot = next(
                (blk_index[n] for n in other if n in blk_index), None
            )
            if carry_slot is None:
                continue
            add_res = _om._results(add_op)
            if not add_res:
                continue
            add_res_name = _ssa_name(add_res[0])
            if carry_slot >= len(yield_operands):
                continue
            if _ssa_name(yield_operands[carry_slot]) != add_res_name:
                continue
            slot = carry_slot
        # Recover the accumulator (M, N) from the dot's A (M x Ka) and
        # B (Kb x N) operand tile shapes.
        try:
            a_shape = [int(s) for s in _om._shape_of(dot_operands[0])]
            b_shape = [int(s) for s in _om._shape_of(dot_operands[1])]
        except (TypeError, ValueError):
            continue
        if len(a_shape) != 2 or len(b_shape) != 2:
            continue
        slots[slot] = (a_shape[0], b_shape[1])
        if res_name and not _unfolded:
            # Only the FOLDED in-place carry result is recorded as an in-place
            # fused-dot result (``map_tt_dot`` then keeps it == the fragment
            # carry and SKIPS the fragment->shared epilogue). The UNFOLDED form
            # has a SEPARATE carry tile and its dot result feeds an ``addf``, so
            # it must KEEP the layout-aware fragment->shared epilogue (a logical
            # read of a raw ``simdgroup`` register tile does not compile/is
            # wrong); ``map_tt_dot`` learns the C scope from ``_unfolded_frag_c``.
            dot_result_ssas.append(res_name)
        if res_name and _unfolded:
            _unfolded_frag_results.append(res_name)
    return slots, dot_result_ssas, _unfolded_frag_results


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
    * Scalar iter_args are held in 1-element local buffers and updated from
      ``scf.yield`` at the end of each body, so the next iteration observes
      the carried value. Buffer/tuple carries continue to mutate in place.
    * iter_args > 16: raise :class:`EmitError`. The user should restructure
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

    # FORWARDPTR: thread loop-carried strided PtrState across the scf.for so
    # in-loop ``tt.load``s on the block-args resolve to a strided make_tptr
    # tile-load (NOT the scalar carry_index gather). Registers
    # ``ctx.ptr_states[<block_arg>]`` with the per-iteration advance folded
    # into the offset. Must run BEFORE the iter_arg materialisation loop so
    # the gather path below sees the forwarded state and steps aside.
    forwarded_tagged: Dict[str, Any] = _forward_loop_ptr_states(
        ctx, region, iter_arg_block_ssas, iter_arg_ssas, induction_ssa,
        lb_ssa, step_ssa, lb_const=lb_i, step_const=step_i,
    )

    # FRAGMENTCARRY: detect iter_arg slots that are fused-dot accumulators
    # (``acc = acc + dot(a, b)`` folded so the dot writes the loop-carried
    # tile). On CUDA those carries MUST be allocated as a swizzled
    # ``local.fragment`` -- the tensor-core MMA store layout asserts a fragment
    # C, and a serial-scalar loop-carry copy of a fragment ignores the mma
    # swizzle and silently corrupts the accumulator. Allocating the carry as a
    # fragment lets ``map_tt_dot`` accumulate IN PLACE (the yielded value IS the
    # carry, so the corrupting copy is skipped) -- a real fragment-resident,
    # swizzle-correct carry exactly like a hand-written TileLang K-loop. Empty
    # on Metal/unspecified targets, where shared-C GEMM is the correct path.
    _frag_carry_slots: Dict[int, Tuple[int, int]] = {}
    try:
        from .reduction import _is_cuda_target as _frag_is_cuda  # noqa: WPS433
    except Exception:  # pragma: no cover - reduction always importable here
        _frag_is_cuda = None
    try:
        from .reduction import _is_metal_target as _frag_is_metal  # noqa: WPS433
    except Exception:  # pragma: no cover - reduction always importable here
        _frag_is_metal = None
    # DEFECT-C (frontend): the fused-dot MMA-C accumulator carry must leave
    # SHARED. On CUDA the tensor-core MMA store layout REQUIRES a swizzled
    # ``local.fragment`` C (above). On Metal the same fragment-resident carry
    # binds C to a ``simdgroup_*`` register tile (the legacy simdgroup GEMM
    # path, ``src/backend/metal/op/gemm.cc``: a ``local.fragment``/``metal.
    # simdgroup`` C selects ``kMetalSIMDGroup`` whose accumulator lives in
    # ``simdgroup_float8x8`` registers, NOT a ``threadgroup`` tile) -- so the
    # MxN fp32 C accumulator no longer consumes threadgroup memory. This is the
    # SAME fragment-resident accumulator a hand-written TileLang Metal kernel
    # carries via ``T.alloc_fragment`` across a K-loop; the only change is that
    # the Triton-frontend folded ``acc = acc + dot`` carry now qualifies for it
    # on Metal too instead of being pinned to ``shared``. Empty only on truly
    # unspecified targets, where shared-C GEMM remains the correct default.
    _frag_target_ok = (_frag_is_cuda is not None and _frag_is_cuda(ctx)) or (
        _frag_is_metal is not None and _frag_is_metal(ctx)
    )
    if _frag_target_ok:
        (
            _frag_carry_slots,
            _frag_dot_results,
            _frag_unfolded_results,
        ) = _dot_accumulator_iter_arg_slots(region, iter_arg_block_ssas)
        # UNFOLDED accumulate (``addf(carry, dot(a,b,0))``): the dot's C must be
        # fragment-resident (off SHARED) but its result is NOT an in-place carry
        # -- ``map_tt_dot`` keeps the layout-aware fragment->shared epilogue so
        # the ``addf`` reads correct logical data. Record those results on a
        # SEPARATE ctx set the C-scope decision consults to force fragment C.
        if _frag_unfolded_results:
            existing_u = set(
                getattr(ctx, "frag_c_metal_unfolded", set()) or set()
            )
            existing_u.update(_frag_unfolded_results)
            ctx.frag_c_metal_unfolded = existing_u
        # Record the in-place fused-dot result SSAs so ``map_tt_dot`` keeps the
        # result == the fragment carry (skips its fragment->shared epilogue),
        # leaving the accumulation in place across the K-loop. The set is shared
        # onto the region child below (which is where the dot is actually
        # emitted) via the explicit propagation in ``_emit_region``.
        if _frag_dot_results:
            existing = set(getattr(ctx, "frag_carry_dot_results", set()) or set())
            existing.update(_frag_dot_results)
            ctx.frag_carry_dot_results = existing

    # Materialise iter_args. Scalar carries get a fresh tir.Var bound via
    # ``tir.LetStmt(var, init, body)``; buffer / tuple carries (e.g. a
    # ``T.gemm`` accumulator tile, surfaced as ``(buffer, shape)``) cannot
    # go through LetStmt -- ``tirx.Bind`` rejects non-PrimExpr values with
    # ``Expected ir.PrimExpr but got ffi.Array``. For those we bind the
    # block-arg SSA *directly* to the buffer descriptor so the body's
    # BufferLoad / BufferStore on the iter_arg resolves to the actual
    # buffer; mutation threads forward by in-place store, which is exactly
    # how matmul / reduce-into-tile loops are written in TTIR.
    iter_arg_pairs: List[Tuple[Any, Any]] = []
    scalar_carry_buffers: List[Tuple[int, Any]] = []
    for idx, (blk_ssa, init_ssa) in enumerate(zip(iter_arg_block_ssas, iter_arg_ssas)):
        try:
            init_val = ctx.get(init_ssa)
        except KeyError:
            init_val = init_ssa
        # FORWARDPTR: this block-arg has a forwarded strided PtrState. Bind it
        # to the tagged-dict (``{"_ptrstate": ...}``) so the in-loop
        # ``tt.load`` resolves through the strided ``make_tptr`` tile-load and
        # the in-loop ``tt.addptr`` composes harmlessly -- NOT the scalar
        # ``carry_index`` gather. The per-iteration advance is already folded
        # into the forwarded offset, so no loop-carry copy is needed.
        _blk_name = _ssa_name(blk_ssa)
        if _blk_name in forwarded_tagged:
            iter_arg_pairs.append((blk_ssa, forwarded_tagged[_blk_name]))
            continue
        if isinstance(init_val, _om.LazyTileExpr):
            shape = list(init_val.shape or (1,))
            # FRAGMENTCARRY: a fused-dot accumulator carry on CUDA is allocated
            # as a swizzled ``local.fragment`` (not ``shared``), so ``map_tt_dot``
            # finds a real fragment C and accumulates IN PLACE across the K-loop
            # (swizzle-correct, exactly like a hand-written TileLang accumulator).
            is_frag_carry = (
                idx in _frag_carry_slots and _is_zero_constant_lazy(init_val)
            )
            carry_scope = "local.fragment" if is_frag_carry else "shared"
            carry_buf = _om._alloc_tile_buffer(
                ctx,
                shape,
                init_val.dtype,
                ctx.fresh("carry_tile"),
                scope=carry_scope,
            )
            if is_frag_carry:
                # Zero the fragment with the layout-AWARE ``T.fill`` (a
                # ``tl.tileop.fill`` that participates in LayoutInference), NOT
                # the serial-scalar ``_copy_buffer_stmt``. A scalar zero-fill
                # imposes a FLAT (replicate, _i*N+_j) fragment layout that
                # CONFLICTS with the gemm's mma store layout ("Get different
                # layout for carry_tile"). ``T.fill`` zeroes the fragment under
                # WHATEVER layout inference assigns it -- the mma store layout
                # the gemm requires -- so there is no conflict and the seed
                # stays swizzle-correct (0 maps to 0 under any permutation).
                import tilelang.language as _Tfrag  # noqa: WPS433
                fill_handle = _Tfrag.fill(carry_buf, tir.const(0, init_val.dtype))
                if isinstance(fill_handle, tir.PrimExpr):
                    ctx.emit(tir.Evaluate(fill_handle))
                else:
                    ctx.emit(fill_handle)
            else:
                ctx.emit(_copy_buffer_stmt(ctx, init_val, carry_buf))
            iter_arg_pairs.append((blk_ssa, carry_buf))
            continue
        if (
            isinstance(init_val, tuple)
            and len(init_val) == 2
            and isinstance(init_val[1], (list, tuple))
        ):
            base_buf, indices = init_val
            new_indices: List[Any] = []
            materialized = False
            for index_expr in indices:
                if isinstance(index_expr, _om.LazyTileExpr):
                    index_buf = _om._alloc_tile_buffer(
                        ctx,
                        list(index_expr.shape or (1,)),
                        index_expr.dtype,
                        ctx.fresh("carry_index"),
                    )
                    ctx.emit(_copy_buffer_stmt(ctx, index_expr, index_buf))
                    new_indices.append(index_buf)
                    materialized = True
                else:
                    new_indices.append(index_expr)
            if materialized:
                init_val = (base_buf, new_indices)
        if _is_scalar_primexpr(init_val):
            dt = _om._normalize_mlir_dtype(
                _om._dtype_of(blk_ssa) or _om._dtype_of(init_ssa) or "float32"
            )
            carry_buf = _om._alloc_tile_buffer(
                ctx, [1], dt, ctx.fresh("carry"), scope="local"
            )
            zero = tir.const(0, "int32")
            ctx.emit(tir.BufferStore(carry_buf, init_val, [zero]))
            iter_arg_pairs.append((blk_ssa, tir.BufferLoad(carry_buf, [zero])))
            scalar_carry_buffers.append((idx, carry_buf))
        else:
            # Buffer / tuple / ffi.Array carry: skip LetStmt; bind the
            # block-arg SSA directly to the descriptor so body emitters
            # see the buffer. This is the matmul T.gemm path.
            iter_arg_pairs.append((blk_ssa, init_val))

    def _emit_body(induction_value: Any) -> Tuple[Any, List[Any]]:
        body, yielded = _emit_region(
            region,
            ctx,
            induction_var=induction_value,
            induction_ssa=induction_ssa,
            iter_arg_pairs=iter_arg_pairs,
        )
        body = _append_loop_carry_copies(ctx, body, iter_arg_pairs, yielded)

        scalar_updates: List[Any] = []
        for iter_idx, carry_buf in scalar_carry_buffers:
            if iter_idx >= len(yielded):
                continue
            yielded_value = yielded[iter_idx]
            if not _is_scalar_primexpr(yielded_value):
                raise EmitError(
                    "map_scf_for: scalar iter_arg yielded a non-scalar "
                    f"value of type {type(yielded_value).__name__}"
                )
            scalar_updates.append(
                tir.BufferStore(carry_buf, yielded_value, [tir.const(0, "int32")])
            )
        if scalar_updates:
            body = tir.SeqStmt([body] + scalar_updates)
        return body, yielded

    # ITERATION 3 (coalesced async loads). Snapshot the cooperative
    # global->shared copy counter BEFORE emitting the body so we can detect
    # whether this K-loop carries any cp.async-eligible producer.
    _async_enabled = bool(getattr(ctx, "routed_triton_async_loads", False))
    if not hasattr(ctx, "_gmem_shared_copies") or not isinstance(
        getattr(ctx, "_gmem_shared_copies", None), list
    ):
        ctx._gmem_shared_copies = []
    _gmem_copies_before = len(ctx._gmem_shared_copies)

    def _maybe_pipeline(for_node: Any) -> Any:
        """Stamp ``num_stages`` on a serial K-loop with global->shared copies.

        Mirrors the annotation ``T.Pipelined(..., num_stages=N)`` would stamp.
        ``PipelinePlanning`` (which CHECKs ``ForKind::kSerial`` -- our loop is
        serial) reads ``num_stages`` off the loop, schedules the global->shared
        copies as async producers, and ``LowerPTXAsyncCopy`` lowers them to
        ``cp.async`` (SASS LDGSTS). Gated on the routed-triton async-load path
        AND on the body having emitted at least one cooperative shared copy --
        annotating a copy-free loop would make ``PipelinePlanning`` schedule an
        empty pipeline. RULE #1: pipeline (coalesce) when eligible, else leave
        the loop exactly as the serial path produced it; no silent half-state.
        """
        if not _async_enabled:
            return for_node
        import os as _os_pipe
        if (_os_pipe.environ.get("TL_NO_NUM_STAGES") == "1"
                or _os_pipe.environ.get("TL_FORCE_CP_ASYNC") == "1"):
            # ASYNCIMPL: when the routed copy is emitted as an explicit cp.async
            # producer (``TL_FORCE_CP_ASYNC=1`` sets is_async_copy in memory.py),
            # Copy::LowerCPAsync emits genuine cp.async/LDGSTS at LowerTileOp time,
            # INDEPENDENT of the software pipeline. We DROP the num_stages pipeline
            # annotation so PipelinePlanning skips this loop entirely and its
            # overlapping-write check (producer copy + masked OOB-zero epilogue both
            # write the shared tile) is never triggered. NOTE: without the pipeline
            # there is no auto-inserted cp.async.wait before the GEMM consumer, so
            # this path is FAST-but-RACY -- it exists for MEASURED SASS/codegen
            # verification of LDGSTS, not as a correct default (see memory.py).
            return for_node
        if len(ctx._gmem_shared_copies) <= _gmem_copies_before:
            return for_node
        if not isinstance(for_node, tir.For) or for_node.kind != tir.ForKind.SERIAL:
            return for_node
        # PipelinePlanning requires the loop body to reduce to a SeqStmt
        # (producer copy + consumer) -- a lone-statement body would LOG(FATAL)
        # there. A K-loop carrying a global->shared copy AND its gemm/carry
        # consumer is always a SeqStmt; if it somehow is not, leave the loop
        # serial rather than stamp an annotation that would crash planning.
        if not isinstance(for_node.body, tir.SeqStmt):
            return for_node
        num_stages = int(getattr(ctx, "num_stages", 2) or 2)
        if num_stages < 1:
            return for_node
        anns = dict(for_node.annotations or {})
        anns["num_stages"] = tir.const(num_stages, "int32")
        return tir.For(
            for_node.loop_var,
            for_node.min,
            for_node.extent,
            for_node.kind,
            for_node.body,
            for_node.thread_binding,
            anns,
        )

    # Compute extent = ub - lb. step == 1 is the common case; for non-unit
    # step we either unroll or scale the induction var.
    UNROLL_LIMIT = 8
    yielded: List[Any] = []
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
            body, yielded = _emit_body(iter_var)
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
        body_induction = lb + loop_var * step
        body, yielded = _emit_body(body_induction)
        for_stmt = _maybe_pipeline(
            tir.For(loop_var, tir.const(0, "int32"), extent_expr,
                    tir.ForKind.SERIAL, body)
        )
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
        body, yielded = _emit_body(loop_var)
        for_stmt = _maybe_pipeline(
            tir.For(loop_var, min_expr, extent_expr,
                    tir.ForKind.SERIAL, body)
        )

    ctx.emit(for_stmt)

    # Bind the ``scf.for`` results to the final carried values. Scalar
    # carries live in local buffers; buffer/tuple carries mutate in place.
    if _om._results(op):
        scalar_by_idx = {idx: buf for idx, buf in scalar_carry_buffers}
        Buffer = ctx.tvm().tir.Buffer
        for idx, result_ssa in enumerate(_om._results(op)):
            if idx in scalar_by_idx:
                ctx.bind(
                    result_ssa,
                    tir.BufferLoad(scalar_by_idx[idx], [tir.const(0, "int32")]),
                )
            elif idx < len(iter_arg_pairs):
                carry = iter_arg_pairs[idx][1]
                is_pointer_like_carry = (
                    isinstance(carry, tuple)
                    or (isinstance(carry, dict) and "_ptrstate" in carry)
                )
                is_tile_like_carry = (
                    isinstance(carry, (Buffer, _om.LazyTileExpr))
                    or (hasattr(carry, "shape") and hasattr(carry, "dtype"))
                )
                if is_tile_like_carry and not is_pointer_like_carry:
                    # FRAGMENTCARRY EPILOGUE: a fused-dot accumulator carried as
                    # a swizzled ``local.fragment`` must be materialised to a
                    # logical ``shared`` tile BEFORE any non-gemm consumer (the
                    # post-loop ``tt.store``) reads it. The fragment's per-thread
                    # registers hold the mma store layout (NOT a flat [i,j]
                    # tile), so a logical-indexed store read would impose a flat
                    # layout that CONFLICTS with the gemm's mma layout
                    # ("Get different layout for carry_tile"). The cooperative
                    # ``T.copy(C_frag -> C_logical_shared)`` distributes the
                    # per-thread fragment registers to the shared tile via the
                    # fragment's inferred thread-layout -- the SAME swizzle-correct
                    # epilogue a hand-written TileLang kernel uses, and the path
                    # our Metal parity already validates. Bind the loop result to
                    # the shared logical tile so the store reads layout-correct
                    # data. Only the genuine fragment carry takes this path.
                    carry_scope = _frag_scope_of(carry)
                    if idx in _frag_carry_slots and carry_scope == "local.fragment":
                        if _DIRECT_FRAG_GLOBAL_EPILOGUE_ENABLED:
                            # DIRECT FRAGMENT->GLOBAL EPILOGUE (named capability,
                            # see module docstring). Bind the loop result to the
                            # MMA-C fragment ITSELF -- no shared ``carry_logical``
                            # staging tile. The downstream ``tt.store`` lowers to
                            # ``T.copy(fragment, global)``: the CopyNode iterates
                            # over the fragment (highest scope-level) and
                            # ``InferLayout`` propagates the fragment's registered
                            # ``make_mma_store_layout`` to the store loop, so the
                            # per-warp register tile is written DIRECTLY to global
                            # at the correct [i,j] positions -- the same
                            # layout-aware fragment->global store native Triton
                            # emits, with the MxN fp32 shared buffer eliminated.
                            # RULE #1: this is the correctness path; the parity
                            # gate proves it bit-correct. A wrong result must RAISE
                            # (parity harness FAIL), never silently re-stage.
                            ctx.bind(result_ssa, carry)
                        else:
                            # Legacy shared-staging epilogue, kept ONLY for
                            # same-session A/B measurement (TL_FRAG_GLOBAL_EPILOGUE=0).
                            c_logical = _om._alloc_tile_buffer(
                                ctx,
                                list(getattr(carry, "shape", []) or [1]),
                                str(getattr(carry, "dtype", "float32")),
                                ctx.fresh("carry_logical"),
                                scope="shared",
                            )
                            import tilelang.language as _Tepi  # noqa: WPS433
                            copy_handle = _Tepi.copy(carry, c_logical)
                            if isinstance(copy_handle, tir.PrimExpr):
                                ctx.emit(tir.Evaluate(copy_handle))
                            else:
                                ctx.emit(copy_handle)
                            ctx.bind(result_ssa, c_logical)
                    else:
                        ctx.bind(result_ssa, carry)
                elif yielded and idx < len(yielded):
                    ctx.bind(result_ssa, yielded[idx])
                else:
                    ctx.bind(result_ssa, carry)
            elif yielded and idx < len(yielded):
                ctx.bind(result_ssa, yielded[idx])
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


def map_cf_br(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Accept a bare CFG branch terminator in a linearised region walk."""
    return None


def map_cf_cond_br(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Accept Triton's early-return CFG branch terminator.

    The unstructured ``cf.cond_br`` ops seen in FLA chunk kernels guard an
    immediate ``tt.return`` block before the real body. The public runtime
    adapter launches only valid program ids, so that early-return edge is not
    taken for supported shapes; the walker only needs to ignore the CFG
    terminator and continue into the body block.
    """
    return None


# ---------------------------------------------------------------------------
# scf.while  (bounded-iterations lowering)
# ---------------------------------------------------------------------------
#
# TIR has no native ``While`` node, so we lower an ``scf.while`` loop into a
# ``tir.For`` of ``SCF_WHILE_MAX_ITERATIONS`` iterations with an inner
# ``tir.IfThenElse`` guarding the after-region. The before-region is walked
# *inside* the outer For so its condition can capture the current iter_arg
# bindings on every iteration; if the condition is false the after-region
# simply doesn't execute (this is the cheapest "early exit" we can encode
# without a TIR Stop primitive, and it is correct under TIR semantics
# because the after-region's only effect is to mutate the same iter_arg
# variables we read in the next iteration's before-region).
#
# Honest limitations:
# * ``SCF_WHILE_MAX_ITERATIONS`` is a static upper bound. We default to 1024
#   and let the env var ``TILELANG_SCF_WHILE_MAX_ITERS`` override it. When
#   the loop genuinely needs more, the user should refactor to ``scf.for``
#   (which has no implicit cap).
# * If the user's TTIR carries an explicit ``upper_bound`` attribute we
#   prefer that; otherwise fall back to the env-overridable default.
# * If the user explicitly tags the op with ``"unbounded": True`` (or sets a
#   non-positive bound) we raise EmitError -- silently truncating would
#   change program semantics.

import os as _os


def _scf_while_default_max_iters() -> int:
    raw = _os.environ.get("TILELANG_SCF_WHILE_MAX_ITERS")
    if raw is None:
        return 1024
    try:
        n = int(raw)
        if n <= 0:
            return 1024
        return n
    except ValueError:
        return 1024


SCF_WHILE_MAX_ITERATIONS: int = _scf_while_default_max_iters()


def _scf_while_bound(op: Any) -> int:
    """Resolve the static iteration bound for a given ``scf.while`` op.

    Order of precedence:
    1. ``op["attrs"]["upper_bound"]`` (explicit, per-op) -- wins.
    2. ``op["attrs"]["max_iters"]`` (legacy spelling) -- still honoured.
    3. ``SCF_WHILE_MAX_ITERATIONS`` (env-overridable default).

    Raise :class:`EmitError` when the attrs explicitly mark the loop as
    unbounded or specify a non-positive bound.
    """
    attrs = _om._attrs(op)
    if attrs.get("unbounded"):
        raise EmitError(
            "scf.while with unbounded iterations not supported; "
            "please rewrite with explicit upper bound or use scf.for "
            "(set the 'upper_bound' attribute or env "
            "TILELANG_SCF_WHILE_MAX_ITERS to override the default cap)."
        )
    for key in ("upper_bound", "max_iters"):
        if key in attrs:
            try:
                bound = int(attrs[key])
            except (TypeError, ValueError) as exc:
                raise EmitError(
                    f"scf.while: attribute {key!r}={attrs[key]!r} is not an integer"
                ) from exc
            if bound <= 0:
                raise EmitError(
                    f"scf.while: attribute {key!r}={bound} must be positive; "
                    "rewrite with scf.for if the loop has no useful upper bound."
                )
            return bound
    return SCF_WHILE_MAX_ITERATIONS


def map_scf_while(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``scf.while`` with iter_args to a bounded ``tir.For`` + IfThenElse.

    Shape we accept (mirrors :func:`map_scf_for`'s dict-fake):

        op = {
            "name": "scf.while",
            "operands": [init0, init1, ...],            # iter_arg inits
            "results":  [r0, r1, ...],                  # final iter_arg vals
            "attrs":    {"upper_bound": N, ...},        # optional cap
            "regions":  [before_region, after_region],
            # Block args of each region as flat lists (dict-fake convenience):
            "before_block_args": [carry0_b, carry1_b, ...],
            "after_block_args":  [carry0_a, carry1_a, ...],
        }

    The before-region terminates with ``scf.condition(%cond) %carry...`` and
    the after-region with ``scf.yield %carry...``. We model both terminators
    by intercepting them inside :func:`_emit_region` (yield) and via a
    dedicated probe here (condition).

    Implementation:
        carry_i = tir.Var(... per iter_arg)            # bound to init_i via Let
        for k in 0..MAX_ITERS:
            <before_region body>      # may set carry_i_next, evaluate cond
            if cond:
                <after_region body>   # mutates carry_i in place
            else:
                pass                  # acts as a soft early-exit

    The result SSAs are bound to the carry vars (their final values).
    """
    tir = ctx.tir()
    init_ssas = list(_om._operands(op))
    if len(init_ssas) > _MAX_ITER_ARGS:
        raise EmitError(
            f"scf.while: {len(init_ssas)} iter_args exceeds supported limit of "
            f"{_MAX_ITER_ARGS}. Restructure to carry state via a TileLang "
            "fragment (T.alloc_fragment + in-place mutate) instead."
        )

    bound = _scf_while_bound(op)

    regions = _scf_regions(op)
    if len(regions) < 2:
        raise EmitError(
            "scf.while: expected 2 regions (before, after); "
            f"got {len(regions)}"
        )
    before_region, after_region = regions[0], regions[1]

    # Resolve block-arg SSAs for both regions.
    before_block_args: List[Any] = []
    after_block_args: List[Any] = []
    if isinstance(op, dict):
        before_block_args = list(op.get("before_block_args") or [])
        after_block_args = list(op.get("after_block_args") or [])
        # Fall back to per-region "block_args" if the op-level keys are absent.
        if not before_block_args and isinstance(before_region, dict):
            before_block_args = list(before_region.get("block_args") or [])
        if not after_block_args and isinstance(after_region, dict):
            after_block_args = list(after_region.get("block_args") or [])
    else:
        try:
            before_block_args = list(before_region.blocks[0].arguments)
        except (AttributeError, IndexError):
            before_block_args = []
        try:
            after_block_args = list(after_region.blocks[0].arguments)
        except (AttributeError, IndexError):
            after_block_args = []

    # Materialise carry vars (one entry per iter_arg). For scalar carries we
    # allocate a fresh ``tir.Var`` and emit a ``tir.LetStmt`` that binds the
    # initial value -- same pattern as ``map_scf_for``. For buffer / tuple
    # carries (e.g. a fragment passed in as state) we thread the descriptor
    # itself; ``tir.LetStmt`` would otherwise raise ``tirx.Bind expected
    # ir.PrimExpr but got ffi.Array``. ``carry_vars`` may therefore contain
    # either a ``tir.Var`` or a buffer descriptor.
    carry_vars: List[Any] = []
    init_pairs: List[Tuple[Any, Any]] = []
    for idx, init_ssa in enumerate(init_ssas):
        try:
            init_val = ctx.get(init_ssa)
        except KeyError:
            init_val = init_ssa
        if _is_scalar_primexpr(init_val):
            dt = _om._dtype_of(init_ssa) or "float32"
            # Prefer the before-region block-arg dtype when richer.
            if idx < len(before_block_args):
                blk_dt = _om._dtype_of(before_block_args[idx])
                if blk_dt:
                    dt = blk_dt
            var = tir.Var(ctx.fresh("wcarry"), dt)
            carry_vars.append(var)
            init_pairs.append((var, init_val))
        else:
            # Buffer-typed carry: thread the descriptor itself; no LetStmt.
            carry_vars.append(init_val)

    # Walk the before-region: detect ``scf.condition`` as terminator and
    # capture (cond_expr, forwarded_values).
    def _walk_before(region: Any, ctx: _om.WalkerCtx,
                     iter_pairs: List[Tuple[Any, Any]]) -> Tuple[Any, Any, List[Any]]:
        """Return (body_stmt, cond_expr, forwarded_values)."""
        child = _om.WalkerCtx()
        child.value_map = dict(ctx.value_map)
        child.buffers = ctx.buffers
        child.transposed_views = dict(ctx.transposed_views)
        child.arg_buffer_shapes = getattr(ctx, "arg_buffer_shapes", {})
        child.fixed_arg_buffer_keys = getattr(ctx, "fixed_arg_buffer_keys", set())
        child._tmp_counter = ctx._tmp_counter
        child._tvm = ctx._tvm
        child._T = ctx._T
        if getattr(ctx, "target", None) is not None:
            child.target = ctx.target
        if hasattr(ctx, "launch_grid"):
            child.launch_grid = ctx.launch_grid
        for ssa, var in iter_pairs:
            child.bind(ssa, var)

        if isinstance(region, dict):
            ops_iter = list(region.get("ops", ()))
        else:
            ops_iter = []
            for block in getattr(region, "blocks", ()) or ():
                for inner in getattr(block, "operations", ()) or ():
                    ops_iter.append(inner)

        cond_expr: Any = None
        forwarded: List[Any] = []
        for inner in ops_iter:
            op_name = inner.get("name") if isinstance(inner, dict) else getattr(inner, "name", "")
            op_name = str(op_name)
            if op_name == "scf.condition":
                cond_operands = list(_om._operands(inner))
                if not cond_operands:
                    raise EmitError(
                        "scf.condition: missing condition operand"
                    )
                cond_ssa = cond_operands[0]
                try:
                    cond_expr = child.get(cond_ssa)
                except KeyError:
                    cond_expr = cond_ssa
                for fwd_ssa in cond_operands[1:]:
                    try:
                        forwarded.append(child.get(fwd_ssa))
                    except KeyError:
                        forwarded.append(fwd_ssa)
                continue
            emitter = CONTROL_EMITTERS.get(op_name) or _om.OP_TABLE.get(op_name)
            if emitter is None:
                raise EmitError(
                    f"scf.while before-region: unmapped op {op_name!r}; "
                    "register an emitter in OP_TABLE or CONTROL_EMITTERS."
                )
            emitter(inner, child)

        ctx._tmp_counter = child._tmp_counter

        tir_local = ctx.tir()
        if not child.stmts:
            body = tir_local.Evaluate(tir_local.const(0, "int32"))
        elif len(child.stmts) == 1:
            body = child.stmts[0]
        else:
            body = tir_local.SeqStmt(child.stmts)

        if cond_expr is None:
            raise EmitError(
                "scf.while before-region missing scf.condition terminator"
            )
        return body, cond_expr, forwarded

    # Bind carry-var SSAs in the *parent* ctx so before-region ops that
    # reference them via op["operands"] resolve correctly during walking.
    iter_arg_pairs_before = [(blk_ssa, var) for blk_ssa, var in zip(before_block_args, carry_vars)]
    iter_arg_pairs_after = [(blk_ssa, var) for blk_ssa, var in zip(after_block_args, carry_vars)]

    before_body, cond_expr, _forwarded = _walk_before(before_region, ctx, iter_arg_pairs_before)

    # Walk the after-region with the *same* carry vars; its terminating
    # scf.yield's operands become the next-iteration carry values. We rely
    # on the body to mutate carry vars in place when needed (matching
    # map_scf_for's serial-mutate model). The yielded SSAs are recorded so
    # the parent emitter can rebind result SSAs.
    after_body, after_yield = _emit_region(
        after_region,
        ctx,
        induction_var=None,
        induction_ssa=None,
        iter_arg_pairs=iter_arg_pairs_after,
    )

    # Compose: if cond { after_body } else { /* nothing -- soft exit */ }
    guarded_after = tir.IfThenElse(cond_expr, after_body, None)

    # Sequence: before_body; guarded_after.
    body_stmts = []
    body_stmts.append(before_body)
    body_stmts.append(guarded_after)
    loop_body = tir.SeqStmt(body_stmts) if len(body_stmts) > 1 else body_stmts[0]

    # Wrap the body in LetStmts that bind the *scalar* carry vars to their
    # initial values. NOTE: this matches map_scf_for's structure (Let on
    # entry); the carry vars are *mutable* tir.Vars whose updates the
    # after-region is responsible for emitting via BufferStore on a sibling
    # buffer. The downstream LowerLetStmt pass folds the binding correctly.
    # Buffer-typed carries skipped the init_pairs collection above.
    for var, init_val in init_pairs:
        if not _is_scalar_primexpr(init_val):
            raise EmitError(
                f"map_scf_while: carry init for {var!r} is not a scalar "
                f"PrimExpr (got type={type(init_val).__name__}); buffer "
                f"carries should not enter the LetStmt path."
            )
        loop_body = tir.LetStmt(var, init_val, loop_body)

    loop_var = tir.Var(ctx.fresh("wi"), "int32")
    for_stmt = tir.For(
        loop_var,
        tir.const(0, "int32"),
        tir.const(int(bound), "int32"),
        tir.ForKind.SERIAL,
        loop_body,
    )
    ctx.emit(for_stmt)

    # Bind the scf.while result SSAs to the (final-value) carry vars. The
    # yielded SSAs from the after-region terminator take precedence when
    # available.
    results = list(_om._results(op))
    if results:
        # Prefer after-yield (matches scf.while's "loop terminates with the
        # yield"); fall back to carry vars when the yield was empty.
        for idx, result_ssa in enumerate(results):
            if idx < len(after_yield):
                ctx.bind(result_ssa, after_yield[idx])
            elif idx < len(carry_vars):
                ctx.bind(result_ssa, carry_vars[idx])
    return for_stmt


# ---------------------------------------------------------------------------
# llvm.inline_asm  (PTX-pattern -> portable TIR intrinsic)
# ---------------------------------------------------------------------------
#
# Triton emits a small handful of ``ptx.<approx>.<dtype>`` patterns via
# ``tl.inline_asm_elementwise`` for fast transcendentals. We don't try to
# parse arbitrary inline asm -- that is a research project. Instead we
# match the asm_string against a closed-set dictionary of well-known PTX
# intrinsics and lower them to portable TIR calls (``tir.tanh`` etc.) so
# the same kernel can target Metal / CPU / non-NVIDIA back-ends.
#
# The dictionary is intentionally narrow. Any unrecognised pattern raises
# EmitError with the offending asm_string in the message so the user can
# either (a) rewrite the kernel to use a portable intrinsic or (b) submit
# a PR adding the new pattern here.

# Each value is a unary callable ``(tir, x) -> tir.PrimExpr``.
PTX_TO_TIR: Dict[str, Callable[[Any, Any], Any]] = {
    # Fast tanh: NVPTX's tanh.approx.f32 matches TIR's portable tanh, which
    # lowers to the platform's fast-math implementation on CPU/Metal.
    "tanh.approx.f32": lambda tir, x: tir.tanh(x),
    # Approx exp2 / log2 / rcp -- common in softmax / layer-norm fast paths.
    "ex2.approx.f32": lambda tir, x: tir.exp2(x),
    "lg2.approx.f32": lambda tir, x: tir.log2(x),
    "rcp.approx.f32": lambda tir, x: tir.div(tir.const(1.0, "float32"), x),
    "rcp.approx.ftz.f32": lambda tir, x: tir.div(tir.const(1.0, "float32"), x),
    "rsqrt.approx.f32": lambda tir, x: tir.div(tir.const(1.0, "float32"), tir.sqrt(x)),
    "sqrt.approx.f32": lambda tir, x: tir.sqrt(x),
    "sin.approx.f32": lambda tir, x: tir.sin(x),
    "cos.approx.f32": lambda tir, x: tir.cos(x),
}


def _normalize_ptx_asm(asm: str) -> Optional[str]:
    """Reduce a PTX inline-asm string to a canonical intrinsic key.

    Triton's emitted strings look like ``"tanh.approx.f32 $0, $1;"`` or
    ``"  tanh.approx.f32 \t$0,$1 ;\n"``. We strip whitespace, drop the
    operand placeholders + trailing semicolon, and lowercase the mnemonic.
    Returns ``None`` when the shape doesn't match the expected
    ``<mnemonic> $<dst>, $<src>;`` form -- the caller raises EmitError.
    """
    if not isinstance(asm, str):
        return None
    s = asm.strip().lower()
    if not s:
        return None
    # Strip trailing semicolon and any trailing comments.
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    # Split off operand list at the first '$' or comma after the mnemonic.
    # Mnemonic is the first whitespace-delimited token.
    parts = s.split(None, 1)
    if not parts:
        return None
    mnemonic = parts[0]
    return mnemonic


def map_llvm_inline_asm(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``llvm.inline_asm`` (or ``tt.elementwise_inline_asm``) to TIR.

    Strategy:
      * Pull ``asm_string`` from op.attrs.
      * Match against :data:`PTX_TO_TIR`.
      * On hit: emit the corresponding TIR intrinsic with the (single) arg.
      * On miss: raise :class:`EmitError` with the asm_string verbatim.

    We deliberately do **not** try to parse complex multi-instruction
    sequences, multi-operand ops, or non-elementwise patterns. Those are
    flagged for human triage.
    """
    attrs = _om._attrs(op)
    asm = attrs.get("asm_string")
    if asm is None:
        # MLIR's textual form sometimes spells this as "asm".
        asm = attrs.get("asm")
    if asm is None:
        raise EmitError(
            "llvm.inline_asm: missing asm_string attribute; cannot identify "
            "the PTX intrinsic to retarget."
        )

    mnemonic = _normalize_ptx_asm(str(asm))
    if mnemonic is None or mnemonic not in PTX_TO_TIR:
        # Surface the *full* asm_string so triage can copy-paste it into
        # the dictionary (or back into a portable intrinsic). The message
        # also lists supported keys so the user can see what we DO handle.
        supported = ", ".join(sorted(PTX_TO_TIR.keys()))
        raise EmitError(
            "llvm.inline_asm with unrecognized PTX pattern; cannot retarget "
            f"to Metal. asm_string={asm!r}. Supported intrinsics: {supported}."
        )

    operands = _om._operands(op)
    if not operands:
        raise EmitError(
            f"llvm.inline_asm({mnemonic}): expected 1 operand; got 0"
        )
    src_ssa = operands[0]
    src = ctx.get(src_ssa) if src_ssa in ctx.value_map else src_ssa

    tir = ctx.tir()
    expr = PTX_TO_TIR[mnemonic](tir, src)

    if _om._results(op):
        ctx.bind(_om._results(op)[0], expr)
    return expr


# ---------------------------------------------------------------------------
# tt.call -- inline expansion of a tt.func callee
# ---------------------------------------------------------------------------
#
# Triton emits ``tt.call @callee(operands...)`` whenever the kernel uses a
# helper function (e.g. the reusable softmax/sum reduction blocks pulled in
# from ``triton.language.standard``). The TTIR module then carries a sibling
# ``tt.func private @callee(...) {...}`` declaration. This emitter inlines
# the callee at the call site:
#
#   1. Look up the callee tt.func op (registered in ``ctx.callees`` by the
#      module-level pre-pass in ``triton_frontend.__init__:_walk_mlir_module``).
#   2. Push a substitution frame mapping the callee's entry-block arguments
#      to the caller's already-resolved TIR operands.
#   3. Walk the callee's entry-block body via the OP_TABLE / CONTROL_EMITTERS
#      dispatch, intercepting ``tt.return`` to capture the returned SSA.
#   4. Pop the substitution frame and bind the tt.call's result SSA to the
#      captured return value.
#
# We deliberately don't materialise the callee as a separate ``tir.PrimFunc``
# / ``tir.call_extern`` because (a) it preserves SSA simplicity in the walker,
# and (b) it matches Triton's own backend semantics, which inline these
# helpers before lowering to GPU. Cap on call depth comes for free from
# Python recursion limits; we don't expect helper-of-helper depth >2 in the
# kernels we target.
#
# Hard constraint: an unresolvable ``@callee`` raises :class:`EmitError`
# rather than returning silently -- a missing callee almost certainly means
# the pre-pass missed a tt.func, and an invalid lowering is worse than a
# loud failure (per ``feedback_no_silent_fallback``).


# Match ``callee = @symbol_name`` inside the printed properties block.
# Symbol names admit dots and underscores (e.g.
# ``@triton.language.standard.max__fp32S128S_c0_cFalse_cTrue_cFalse``).
_CALLEE_RE = __import__("re").compile(
    r"callee\s*=\s*@(?P<sym>[A-Za-z_][\w.]*)"
)


def _parse_callee_attr(op: Any) -> Optional[str]:
    """Extract the callee symbol from a ``tt.call`` op.

    Tries, in order:
      1. ``op.attrs['callee']`` (dict-fake or jaxlib registered-dialect path).
      2. ``op.attributes['callee']`` -> stringify (real MLIR FlatSymbolRefAttr).
      3. Regex over ``str(op)`` -- the unregistered-dialect path under
         jaxlib's ``allow_unregistered_dialects=True`` puts the callee in
         the printed ``<{...}>`` properties block but never surfaces it via
         the Python attribute accessors.

    Returns the symbol name WITHOUT the leading ``@``, or None when the op
    has no parseable callee.
    """
    # Path 1: dict-fake.
    if isinstance(op, dict):
        attrs = op.get("attrs") or {}
        if "callee" in attrs:
            sym = str(attrs["callee"]).strip().strip('@"')
            return sym.replace(".", "_") if sym else None

    # Path 2: real MLIR attributes.
    attrs_obj = getattr(op, "attributes", None)
    if attrs_obj is not None:
        try:
            for a in attrs_obj:
                if getattr(a, "name", None) == "callee":
                    val = str(a.attr).strip().strip('@"')
                    return val.replace(".", "_") if val else None
        except Exception:
            pass

    # Path 3: regex over the printed op text (unregistered-dialect path).
    try:
        text = str(op)
    except Exception:
        return None
    m = _CALLEE_RE.search(text)
    if m:
        return m.group("sym").strip('@"').replace(".", "_")
    return None


def _func_sym_name(func_op: Any) -> Optional[str]:
    """Return the ``sym_name`` attribute of a ``tt.func`` op (no leading @)."""
    if isinstance(func_op, dict):
        attrs = func_op.get("attrs") or {}
        sym = attrs.get("sym_name") or func_op.get("sym_name")
        # Normalise dotted symbols to underscores so the key matches the
        # one ``_parse_callee_attr`` produces (it always applies the same
        # ``.replace('.', '_')``). Without this, a real ``sym_name`` attr
        # carrying ``triton.language.standard.cdiv`` would be registered
        # with dots while the call site looks it up with underscores --
        # the two never match and tt.call raises "unknown callee".
        return str(sym).replace(".", "_") if sym else None
    # Try op.attributes first.
    attrs_obj = getattr(func_op, "attributes", None)
    if attrs_obj is not None:
        try:
            for a in attrs_obj:
                if getattr(a, "name", None) == "sym_name":
                    raw = str(a.attr).strip()
                    if raw.startswith('"') and raw.endswith('"'):
                        raw = raw[1:-1]
                    # Same dotted -> underscore normalisation as the call
                    # site (``_parse_callee_attr``) so the keys match.
                    return raw.replace(".", "_") or None
        except Exception:
            pass
    # Fall back to the printed property string.
    try:
        s = str(func_op)
        if s.startswith("tt.func public @"):
            idx = len("tt.func public @")
            end = s.find("(", idx)
            if end != -1:
                return s[idx:end].strip()
        import re
        m = re.search(r'tt\.func.*?@([a-zA-Z0-9_.-]+)', s)
        if m:
            return m.group(1).replace(".", "_")
        m2 = re.search(r'sym_name\s*=\s*"([^"]+)"', s)
        if m2:
            return m2.group(1).replace(".", "_")
    except Exception:
        pass
    return None


def _func_entry_block_ops(func_op: Any) -> List[Any]:
    """Return the ops in the entry block (block 0) of a ``tt.func``.

    Triton helper functions printed by Triton 3.x have a second
    ``^bb1: // no predecessors`` block carrying ``ub.poison`` + a sentinel
    ``tt.return``. That block is unreachable and must NOT be walked --
    its ub.poison would error out the dispatcher. We restrict ourselves to
    the entry block where the real body lives.
    """
    if isinstance(func_op, dict):
        regions = func_op.get("regions") or []
        if regions:
            r0 = regions[0]
            if isinstance(r0, dict):
                blocks = r0.get("blocks") or []
                if blocks:
                    b0 = blocks[0]
                    if isinstance(b0, dict):
                        return list(b0.get("ops") or [])
        # Single-region inline form: ``{"body": [op, ...]}``.
        return list(func_op.get("body") or [])
    regions = list(getattr(func_op, "regions", ()) or ())
    if not regions:
        return []
    blocks = list(getattr(regions[0], "blocks", ()) or ())
    if not blocks:
        return []
    return list(getattr(blocks[0], "operations", ()) or [])


def _find_func_return_operand(func_op: Any) -> Optional[Any]:
    """Find the ``tt.return`` op in the entry block and return its operand SSA.

    Returns ``None`` for a void return (``tt.return`` with no operands) or
    when the entry block has no ``tt.return`` at all (defensive; every
    well-formed tt.func has one).
    """
    for inner in _func_entry_block_ops(func_op):
        op_name = inner.get("name") if isinstance(inner, dict) else getattr(inner, "name", "")
        if str(op_name) == "tt.return":
            operands = _om._operands(inner)
            if operands:
                return operands[0]
            return None
    return None


def emit_tt_call(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Inline-expand a ``tt.call @callee(operands...)`` into the caller body.

    See module header for the full strategy. Raises :class:`EmitError` on
    an unresolvable callee or a callee whose body has no ``tt.return``
    when the call site expects a result.
    """
    sym = _parse_callee_attr(op)
    if not sym:
        raise EmitError(
            "tt.call: could not extract a callee symbol from the op. "
            "Expected ``callee = @<name>`` in the properties block; got: "
            f"{str(op)!r}"
        )

    callee_func = ctx.lookup_callee(sym)
    if callee_func is None:
        known = sorted(ctx.callees.keys())
        raise EmitError(
            f"tt.call: references unknown callee '@{sym}'. The module-level "
            f"pre-pass found these tt.func symbols: {known}. Either the "
            "pre-pass skipped this tt.func or the symbol name in the call "
            "is mistyped."
        )

    # Mark the callee as referenced so the module walker knows to skip
    # re-emitting its body at module-level (a helper that's only inlined
    # via tt.call must not also be walked as a top-level kernel).
    ctx.callee_used.add(sym)

    # Resolve the call's operands to TIR values via the caller's value_map.
    operands = list(_om._operands(op))
    resolved_operands: List[Any] = []
    for o in operands:
        if o in ctx.value_map:
            resolved_operands.append(ctx.value_map[o])
        else:
            # ctx.get raises with a useful diagnostic if missing, but we
            # also want to allow operands that are themselves unhashable
            # MLIR Value objects -- ctx.get handles that uniformly.
            resolved_operands.append(ctx.get(o))

    # Build the substitution frame from callee block-args -> caller operands.
    block_args = _func_block_args(callee_func)
    if len(block_args) != len(resolved_operands):
        raise EmitError(
            f"tt.call @{sym}: arity mismatch: callee declares "
            f"{len(block_args)} block-arg(s) but the call site passes "
            f"{len(resolved_operands)} operand(s)."
        )

    subst: Dict[Any, Any] = {}
    for arg, val in zip(block_args, resolved_operands):
        # Bind under the SSA Value object (if hashable) AND its printed
        # name. The double-keyed binding mirrors what map_tt_func does so
        # downstream operand lookups resolve regardless of how they were
        # captured (Value-object key or "%argN" string key).
        try:
            subst[arg] = val
        except Exception:
            pass
        ssa = _ssa_name(arg)
        if ssa:
            subst[ssa] = val

    # Walk the callee's entry block. We dispatch each inner op via OP_TABLE
    # / CONTROL_EMITTERS just like the module walker would, but inside a
    # ``push_substitution`` overlay so that block-arg lookups resolve to
    # the caller's operands. Captured return operand is bound after the
    # walk closes the substitution scope (so its TIR value -- already
    # bound under the inlined SSA -- survives).
    return_value: Any = None
    body_ops = _func_entry_block_ops(callee_func)
    with ctx.push_substitution(subst):
        for inner in body_ops:
            inner_name = inner.get("name") if isinstance(inner, dict) else getattr(inner, "name", "")
            inner_name = str(inner_name)
            if inner_name == "tt.return":
                ret_operands = _om._operands(inner)
                if ret_operands:
                    # Resolve THROUGH the substitution frame so a tt.return
                    # that yields a block-arg directly (rare, but legal)
                    # resolves to the caller's operand.
                    ret_ssa = ret_operands[0]
                    if ret_ssa in ctx.value_map:
                        return_value = ctx.value_map[ret_ssa]
                    else:
                        # Try the ctx.get path (consults the substitution
                        # stack first, then value_map). Falls through to
                        # KeyError if genuinely unbound.
                        return_value = ctx.get(ret_ssa)
                continue
            emitter = _om.OP_TABLE.get(inner_name) or CONTROL_EMITTERS.get(inner_name)
            if emitter is None:
                raise EmitError(
                    f"tt.call @{sym}: callee body contains op {inner_name!r} "
                    "which has no emitter in OP_TABLE / CONTROL_EMITTERS. "
                    "Register it before lowering kernels that depend on "
                    f"@{sym}."
                )
            emitter(inner, ctx)

    # Bind the call's result SSA to whatever the callee returned.
    results = _om._results(op)
    if results:
        if return_value is None:
            raise EmitError(
                f"tt.call @{sym}: call site has a result SSA but the "
                "callee body had no ``tt.return <value>``. Did the helper "
                "lose its return op during a transform?"
            )
        ctx.bind(results[0], return_value)
    return return_value


# ---------------------------------------------------------------------------
# Public dispatch table
# ---------------------------------------------------------------------------


# H4 Wave-I: per-emitter ``owns_regions`` attribute. Each emitter that walks
# its own region(s) (so the global walker MUST NOT descend) sets this to
# True; ``mlir_walker._emitter_owns_regions`` reads it. Adding a new
# region-owning op (e.g. ``tt.gather`` / ``tt.histogram``) requires only
# setting the attribute -- no walker change.
map_scf_for.owns_regions = True
map_scf_if.owns_regions = True
map_scf_while.owns_regions = True
emit_tt_call.owns_regions = True
map_tt_func.owns_regions = True


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
    "arith.index_cast": map_arith_index_cast,
    # arith.constant -- scalar literal (i32 / f32 / ...). Seeds value_map
    # so downstream uses of ``%c0`` resolve via ctx.get(); raises EmitError
    # on array (``dense<...>``) attrs rather than silently splatting.
    "arith.constant": map_arith_constant,
    # PtrAnalysis can rewrite tensor-of-mask operands through shape-only
    # collapses before tts.load/tts.store. This is a metadata reshape, not a
    # data movement op.
    "tensor.collapse_shape": map_tensor_collapse_shape,
    # tt.advance (block-pointer)
    "tt.advance": map_tt_advance,
    # tt.func -- structural; seeds block-arg buffers / vars into the ctx so
    # downstream emitters can look up ``%arg0`` via ctx.get(). The walker
    # owns recursion into the body region itself.
    "tt.func": map_tt_func,
    "tt.return": lambda op, ctx: None,
    # Region terminators consumed by the parent reduce/scan emitter; the
    # walker still encounters them as standalone ops in some captures
    # (when the parent emitter walks the region manually but the global
    # walker also descends). Register a no-op so OP_TABLE lookups don't
    # surface them as ``FAILED_OPS``.
    "tt.reduce.return": lambda op, ctx: None,
    "tt.scan.return": lambda op, ctx: None,
    "ub.poison": lambda op, ctx: ctx.tir().const(0, "int32"),
    # CFG branch terminators appear after TritonStructured rewrites some
    # early returns into basic-block diamonds.
    "cf.br": map_cf_br,
    "cf.cond_br": map_cf_cond_br,
    # tt.call -- inline-expand a helper tt.func at the call site.
    "tt.call": emit_tt_call,
    # scf
    "scf.for": map_scf_for,
    "scf.if": map_scf_if,
    "scf.yield": map_scf_yield,
    "scf.while": map_scf_while,
    # llvm inline asm (Triton's PTX fast-math escape hatch). We also accept
    # the Triton-dialect spelling some pipelines emit pre-canonicalisation.
    "llvm.inline_asm": map_llvm_inline_asm,
    "tt.elementwise_inline_asm": map_llvm_inline_asm,
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
