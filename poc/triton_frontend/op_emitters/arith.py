"""Arithmetic / math emitters for the Triton TTIR -> TileLang TIR walker.

This module ships the value-producing emitters for the MLIR ``arith.*`` and
``math.*`` dialects (plus ``tt.fma``) that appear in Triton TTIR after the
upstream Triton frontend has lowered Python source to MLIR. The op names
match the canonical MLIR dialect spellings used by upstream Triton (e.g.
``arith.addf``, ``math.sqrt``); see ``llvm-project`` /
``mlir/include/mlir/Dialect/Arith/IR/ArithOps.td`` and ``MathOps.td``.

Each emitter takes ``(op, ctx)`` exactly like the existing emitters in
``poc.triton_frontend.op_mapping`` -- ``op`` is either an ``mlir.ir.Operation``
or a dict ``{"name", "operands", "results", "attrs"}`` (the unit-test shape),
and ``ctx`` is a :class:`~poc.triton_frontend.op_mapping.WalkerCtx`. Emitters
return a ``tvm.tir.PrimExpr`` (the resulting value) and ``ctx.bind`` the SSA
result so downstream consumers can pick it up.

The dispatch table :data:`ARITH_EMITTERS` is merged into
:data:`poc.triton_frontend.op_mapping.OP_TABLE` at module import time so the
walker resolves ``arith.addf`` / ``math.exp`` / ``tt.fma`` exactly like it
resolves ``tt.load`` and friends.

Design notes
------------
* We never silently fall back: if an op is invoked on a dtype the lowering
  doesn't support (e.g. ``arith.addi`` on a float type, or ``math.exp`` on
  an integer), we raise :class:`EmitError` with the offending dtype.
* Comparisons (``arith.cmpf`` / ``arith.cmpi``) read MLIR's ``predicate``
  attribute (the integer-keyed enum used by ``arith.CmpIPredicate`` /
  ``arith.CmpFPredicate``) and pick the corresponding ``tvm.tir`` node.
* ``tt.fma`` prefers a hardware-mappable intrinsic when available
  (``tvm.tir.call_intrin("<dtype>", "tir.fma", a, b, c)``), with a clean
  ``(a*b) + c`` fallback when the intrinsic name isn't registered (this
  also keeps ``structural_equal`` happy in unit tests where we feed scalar
  PrimExprs directly).

Tile-shape contract (vector_add e2e fix)
----------------------------------------
Triton TTIR for ``vector_add``-class kernels combines tile-shaped operands
that surface in the walker as **mixed shapes**:

* ``tt.make_range``/``tt.load`` -> ``tir.Buffer`` (when the lane count
  exceeds the vector-width cap or the tile fell back to the per-element
  ``# DEGRADED:`` path).
* ``tt.splat`` -> ``tir.Broadcast`` (a vector ``PrimExpr`` whose ``.value``
  is the scalar source).
* ``tt.broadcast`` (vec->tile) -> ``tir.Buffer``.

Before this fix, ``arith.addi(splat, make_range)`` raised
``TypeError: tirx.Add expected ir.PrimExpr but got tirx.Buffer`` because
``ctx.tir().Add`` is scalar-only. The :func:`_emit_tile_binop` helper
below detects tile operands (Buffer / Broadcast / Ramp) and lowers the op
to a per-lane ``tir.For`` that writes into a fresh result Buffer. The
emitter returns the Buffer so downstream tile-aware ops (``tt.store``,
follow-on ``arith.*``) keep composing without per-emitter shape probes.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

# Shared helper for Triton 3.6 ``<{predicate = N : i64}>`` Properties.
# jaxlib's mlir.ir under ``allow_unregistered_dialects=True`` exposes
# property-only attrs as an empty ``op.attributes`` dict, so we have to
# fall back to parsing the printed op text. The same helper is used by
# ``op_emitters/memory.py`` for ``tt.make_range`` (Wave C2 fix).
from ..op_mapping import (
    LazyTileExpr,
    _attrs_with_properties_shared,
    _constant_tile_bool,
    EmitError,
    materialize_lazy_tile,
    should_fold_addressing,
)

# We import op_mapping lazily inside emitters when we need to reach into
# WalkerCtx machinery; the type alias below is just for static readers.
EmitContext = Any  # poc.triton_frontend.op_mapping.WalkerCtx


__all__ = [
    "ARITH_EMITTERS",
    "EmitError",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _operands(op: Any) -> Tuple[Any, ...]:
    """Operand tuple, MLIR-or-dict shape agnostic. Mirror of op_mapping._operands."""
    if isinstance(op, dict):
        return tuple(op.get("operands", ()))
    return tuple(op.operands)


def _results(op: Any) -> Tuple[Any, ...]:
    if isinstance(op, dict):
        return tuple(op.get("results", ()))
    return tuple(op.results)


def _attrs(op: Any) -> Dict[str, Any]:
    if isinstance(op, dict):
        return dict(op.get("attrs", {}))
    return {a.name: a.attr for a in op.attributes} if hasattr(op, "attributes") else {}


def _dtype_of(value: Any) -> str:
    """Best-effort element dtype for an SSA value or PrimExpr operand."""
    if isinstance(value, dict):
        return str(value.get("dtype", "float32"))
    # PrimExpr / Buffer: ``.dtype`` is a string-like.
    dt = getattr(value, "dtype", None)
    if dt is not None:
        return str(dt)
    typ = getattr(value, "type", None)
    if typ is None:
        return "float32"
    elt = getattr(typ, "element_type", None)
    if elt is None:
        return "float32"
    return str(elt)


_FLOAT_DTYPES = {
    "float16", "f16", "bfloat16", "bf16",
    "float32", "f32",
    "float64", "f64",
}


def _is_float_dtype(dt: str) -> bool:
    s = str(dt).lower()
    if s in _FLOAT_DTYPES:
        return True
    return s.startswith(("float", "bfloat", "f16", "f32", "f64", "bf16"))


def _is_integer_dtype(dt: str) -> bool:
    s = str(dt).lower()
    if s in _FLOAT_DTYPES:
        return False
    return s.startswith(("int", "uint", "i", "u")) and not s.startswith(("inf",))


def _bind_result(op: Any, ctx: EmitContext, value: Any) -> Any:
    """Bind ``value`` to the op's first result SSA, if any. Returns ``value``."""
    res = _results(op)
    if res:
        ctx.bind(res[0], value)
    return value


def _resolve_two(op: Any, ctx: EmitContext) -> Tuple[Any, Any]:
    operands = _operands(op)
    if len(operands) != 2:
        raise EmitError(
            f"{op.get('name') if isinstance(op, dict) else getattr(op, 'name', '?')}: "
            f"expected 2 operands, got {len(operands)}"
        )
    return ctx.get(operands[0]), ctx.get(operands[1])


def _resolve_one(op: Any, ctx: EmitContext) -> Any:
    operands = _operands(op)
    if len(operands) != 1:
        raise EmitError(
            f"{op.get('name') if isinstance(op, dict) else getattr(op, 'name', '?')}: "
            f"expected 1 operand, got {len(operands)}"
        )
    return ctx.get(operands[0])


# ---------------------------------------------------------------------------
# Tile-shape helpers (vector_add e2e fix)
# ---------------------------------------------------------------------------
#
# Triton's TTIR commonly hands us tile-shaped operands at the arith layer:
# ``arith.addi(splat<256xi32>, range<256xi32>)``. The splat is a
# ``tir.Broadcast`` (a vector PrimExpr), the range may be a ``tir.Buffer``
# (when it spilled past the vector-width cap in ``tt.make_range``) or a
# ``tir.Ramp`` (small range). Mixed shapes (Buffer + Broadcast) cannot
# feed ``ctx.tir().Add`` directly -- TVM's scalar binop nodes reject Buffer
# operands. The helpers here detect tile operands and emit a per-lane
# ``tir.For`` over a fresh result Buffer.


def _is_tile_operand(ctx: EmitContext, value: Any) -> bool:
    """True iff ``value`` is a tile (Buffer, Broadcast, or Ramp)."""
    tvm_mod = ctx.tvm()
    tir = tvm_mod.tir
    if isinstance(value, LazyTileExpr):
        return True
    if isinstance(value, tvm_mod.tir.Buffer):
        return True
    for cls_name in ("Broadcast", "Ramp"):
        cls = getattr(tir, cls_name, None)
        if cls is not None and isinstance(value, cls):
            return True
    # Vector-typed PrimExpr (dtype like ``int32x256``): treat as tile too.
    dt = getattr(value, "dtype", None)
    if dt is not None and "x" in str(dt):
        try:
            lanes = int(str(dt).rsplit("x", 1)[-1])
        except ValueError:
            lanes = 1
        if lanes > 1:
            return True
    return False


def _tile_lanes(ctx: EmitContext, value: Any) -> int:
    """Return the number of lanes for a tile operand, or 1 for a scalar."""
    tvm_mod = ctx.tvm()
    if isinstance(value, tvm_mod.tir.Buffer):
        # Single-axis tile (vector_add); higher-rank handled by _tile_shape.
        try:
            return int(value.shape[0]) if len(value.shape) >= 1 else 1
        except Exception:
            return 1
    lanes = getattr(value, "lanes", None)
    if lanes is not None:
        try:
            return int(lanes)
        except Exception:
            pass
    dt = getattr(value, "dtype", None)
    if dt is not None and "x" in str(dt):
        try:
            return int(str(dt).rsplit("x", 1)[-1])
        except ValueError:
            return 1
    return 1


def _tile_shape(ctx: EmitContext, value: Any) -> Tuple[int, ...]:
    """Return the multi-dim shape for a tile operand (rank>=1)."""
    tvm_mod = ctx.tvm()
    if isinstance(value, LazyTileExpr):
        return value.shape
    if isinstance(value, tvm_mod.tir.Buffer):
        try:
            return tuple(int(s) for s in value.shape)
        except Exception:
            return (_tile_lanes(ctx, value),)
    return (_tile_lanes(ctx, value),)


def _tile_dtype(ctx: EmitContext, value: Any) -> str:
    """Return the per-lane dtype for a tile/scalar operand."""
    tvm_mod = ctx.tvm()
    if isinstance(value, LazyTileExpr):
        return value.dtype
    if isinstance(value, tvm_mod.tir.Buffer):
        return str(value.dtype)
    dt = getattr(value, "dtype", None)
    if dt is None:
        return "float32"
    s = str(dt)
    if "x" in s:
        s = s.rsplit("x", 1)[0]
    return s


def _read_lane(ctx: EmitContext, value: Any, indices: Tuple[Any, ...]) -> Any:
    """Per-lane read from a tile operand at multi-dim ``indices``.

    * ``Buffer`` -> ``BufferLoad``.
    * ``Broadcast`` -> the splat value (constant per lane).
    * ``Ramp`` -> ``base + stride * idx`` (rank-1 only).
    * scalar PrimExpr -> the value itself (broadcast semantics).
    """
    tvm_mod = ctx.tvm()
    tir = tvm_mod.tir
    if isinstance(value, LazyTileExpr):
        return value.read_lane(ctx, indices)
    if isinstance(value, tvm_mod.tir.Buffer):
        # Truncate / pad indices to the buffer rank.
        rank = len(value.shape)
        idx = list(indices[-rank:]) if rank else [tir.const(0, "int32")]
        if not idx:
            idx = [tir.const(0, "int32")]
        # DOUTTRANSPOSE16B: a producer tile registered as PHYSICALLY transposed
        # ([K, hd] in memory while the LOGICAL operand is [hd, K]) is read by
        # consumers (the lazy ``dout*exp`` AND the register-A MMA fragment fill)
        # with LOGICAL [m=hd, k] indices -> swap to physical [k, m] so the VALUE
        # read is bit-identical to the [hd, K] layout. RULE #1: pure index
        # relabel keyed on a route-registered transposed tile; never reorders an
        # untransposed buffer.
        _tset = getattr(ctx, "transposed_phys_tiles", None)
        if _tset and rank == 2:
            try:
                _datakey = getattr(value, "data", value)
            except Exception:
                _datakey = value
            if _datakey in _tset:
                idx = [idx[1], idx[0]]
        return tir.BufferLoad(value, idx)
    bcast_cls = getattr(tir, "Broadcast", None)
    if bcast_cls is not None and isinstance(value, bcast_cls):
        return value.value
    ramp_cls = getattr(tir, "Ramp", None)
    if ramp_cls is not None and isinstance(value, ramp_cls):
        # rank-1 Ramp; use the last (innermost) loop var.
        last = indices[-1] if indices else tir.const(0, "int32")
        return value.base + value.stride * last
    dt = getattr(value, "dtype", None)
    if dt is not None and "x" in str(dt):
        last = indices[-1] if indices else tir.const(0, "int32")
        binop_pyops = {
            "Add": lambda x, y: x + y,
            "Sub": lambda x, y: x - y,
            "Mul": lambda x, y: x * y,
            "Div": lambda x, y: x / y,
            "Mod": lambda x, y: x % y,
            "FloorDiv": lambda x, y: x // y,
            "FloorMod": lambda x, y: x % y,
        }
        for cls_name, pyop in binop_pyops.items():
            cls = getattr(tir, cls_name, None)
            if cls is not None and isinstance(value, cls):
                return pyop(
                    _read_lane(ctx, value.a, (last,)),
                    _read_lane(ctx, value.b, (last,)),
                )
        cast_cls = getattr(tir, "Cast", None)
        if cast_cls is not None and isinstance(value, cast_cls):
            scalar_dt = str(value.dtype).rsplit("x", 1)[0]
            return tir.Cast(scalar_dt, _read_lane(ctx, value.value, (last,)))
    # Scalar PrimExpr: broadcast semantics.
    return value


def _is_zero_constant_tile(ctx: EmitContext, value: Any) -> bool:
    """True when ``value`` is a dense splat-zero tile materialized earlier."""
    const_tiles = getattr(ctx, "constant_tile_values", {}) or {}
    keys = (
        str(getattr(value, "data", "")),
        str(getattr(value, "name", "")),
        str(value),
    )
    for key in keys:
        if not key or key not in const_tiles:
            continue
        try:
            return float(const_tiles[key]) == 0.0
        except Exception:
            return False
    return False


def _loop_carry_target(ctx: EmitContext, op: Any, value: Any) -> Any:
    """Return the loop-carried output buffer when ``value`` aliases one."""
    carries = getattr(ctx, "loop_carry_buffers", {}) or {}
    operands = _operands(op)
    if operands:
        first = operands[0]
        if first in carries:
            return carries[first]
        try:
            getter = getattr(first, "get_name", None)
            name = getter() if callable(getter) else getattr(first, "name", None)
            if name and str(name) in carries:
                return carries[str(name)]
        except Exception:
            pass
    for carried in carries.values():
        if carried is value:
            return carried
    return None


def _ctx_is_cuda_target(ctx: EmitContext) -> bool:
    """True iff the codegen target is explicitly CUDA / NVIDIA.

    Mirrors ``reduction._is_cuda_target`` -- prefers ``ctx.target`` (set by
    ``from_ttir`` before the tilelang lowering passes establish
    ``Target.current()``), then the ambient target, then False. Used to gate
    the REGSCALECOLLAPSE register-resident A-operand path to the CUDA
    tensor-core MMA target (the Metal simdgroup path applies the analogous
    fragment scale via its own emitter, the SIMT/scalar paths never use it).
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


def _feeds_tensorcore_dot_as_register_a(
    ctx: EmitContext, op: Any, out_shape: Tuple[int, ...], out_dtype: str
) -> bool:
    """True iff this binop's result feeds a tensor-core ``tt.dot`` EXCLUSIVELY.

    REGSCALECOLLAPSE gate (RULE #1: scoped + fail-closed). Returns True only
    when ALL of:
      * CUDA / NVIDIA target (tensor-core MMA; gemm_rs reads A from registers).
      * The result SSA's recorded consumers are EXACTLY ``{"tt.dot"}`` -- the
        product is consumed only by the GEMM, so keeping it lazy starves no
        other reader. (``ctx.ssa_users`` is the module-prepass use-def map; it
        bridges scf.for iter-args, so an in-loop ``dout*exp -> dot`` is caught.)
      * The result is a rank-2 tile whose M/K are MMA-tileable (``% 8 == 0``):
        the same ``can_use_tile_gemm_shape`` gate the dot emitter uses to pick
        the tensor-core path. A non-MMA shape would route to the scalar dot,
        which cannot consume a fragment A operand.

    When True, the binop stays lazy; ``reduction._stage_operand_to_shared``
    then materialises it straight into a ``local.fragment`` (MMA-A layout), so
    the scale is a register op on the A fragment -- native's
    ``ldmatrix -> *scale in registers -> mma`` with no shared round-trip.
    """
    if not _ctx_is_cuda_target(ctx):
        return False
    # Rank-2 MMA-tileable result only (matches the dot emitter's
    # ``can_use_tile_gemm_shape`` so the dot actually takes the MMA path).
    if len(out_shape) != 2:
        return False
    try:
        m, k = int(out_shape[0]), int(out_shape[1])
    except Exception:
        return False
    if m % 8 != 0 or k % 8 != 0:
        return False
    ssa_users = getattr(ctx, "ssa_users", None)
    if not ssa_users:
        return False
    from ..op_mapping import _result_ssa_name  # noqa: WPS433

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
    # EXCLUSIVELY consumed by tt.dot: no other op reads the scaled product, so
    # keeping it lazy (folding the scale into the fragment fill) starves no
    # reader. RULE #1: fail-closed -- any non-dot consumer keeps the
    # materialise-to-shared path.
    return set(users) == {"tt.dot"}


def _emit_tile_binop(
    op: Any,
    ctx: EmitContext,
    a: Any,
    b: Any,
    scalar_combine: Callable[[Any, Any], Any],
    op_label: str,
    out_dtype: str,
) -> Any:
    """Emit a per-lane ``tir.For`` for a binop whose operands include tiles.

    Returns the freshly allocated result ``tir.Buffer``. ``scalar_combine``
    receives per-lane PrimExprs and returns the per-lane result PrimExpr.
    """
    # Pick the larger shape between the two operands; tile-vs-scalar broadcasts.
    shape_a = _tile_shape(ctx, a) if _is_tile_operand(ctx, a) else ()
    shape_b = _tile_shape(ctx, b) if _is_tile_operand(ctx, b) else ()
    out_shape = shape_a if len(shape_a) >= len(shape_b) else shape_b
    if not out_shape:
        # Defensive: caller should not reach here without a tile operand.
        raise EmitError(
            f"{op_label}: _emit_tile_binop called without tile operand"
        )
    lazy = LazyTileExpr(
        out_shape,
        out_dtype,
        lambda read_ctx, indices: scalar_combine(
            _read_lane(read_ctx, a, tuple(indices)),
            _read_lane(read_ctx, b, tuple(indices)),
        ),
        name=ctx.fresh("tile_expr"),
    )
    # FULL TRANSFORM 1 (Coalesce-style addressing fold): when this binop's
    # result is consumed ONLY as addressing/mask for a tt.load/tt.store, keep
    # it lazy so its per-lane value folds into the load/store loop body instead
    # of being materialized into a spilled [N] array. Consumers
    # (``_resolve_lane_operand`` / ``_scalarize_tile_index_base`` /
    # ``_read_lane`` / strided store rhs) already read a LazyTileExpr per-lane.
    if should_fold_addressing(ctx, op):
        return _bind_result(op, ctx, lazy)
    # REGSCALECOLLAPSE (register/fragment-resident A-operand scale, native
    # mirror): when this elementwise binop's result feeds a tensor-core GEMM
    # as its A operand (e.g. the Mamba ``dout * exp(dA)`` decay scale that feeds
    # ``tt.dot``), keep it LAZY instead of materialising the scaled product into
    # a ``local`` tile + a cooperative SHARED round-trip (``dot_a_logical``).
    # The GEMM emitter (``reduction._stage_operand_to_shared``) then materialises
    # this lazy expr DIRECTLY into a ``local.fragment`` in the MMA-A load layout:
    # the fragment-fill reads the raw operand from SHARED (the cp.async-staged
    # ``dout``) and applies the per-K scale per-element, so the scaled value
    # lives in the A fragment REGISTERS at MMA time -- exactly native's
    # ``ldmatrix dout -> *exp(dA) in registers -> mma`` (gemm_rs). This removes
    # the ``dout_a_logical`` shared buffer AND the ``__syncthreads`` its write
    # needed, per K-step. RULE #1: only kept lazy when the result feeds a
    # tensor-core dot EXCLUSIVELY (proven below) so no other consumer is
    # starved; otherwise materialise as before.
    if _feeds_tensorcore_dot_as_register_a(ctx, op, out_shape, out_dtype):
        return _bind_result(op, ctx, lazy)
    out = materialize_lazy_tile(
        ctx,
        lazy,
        out_shape,
        out_dtype,
        name="tile_binop",
    )
    return _bind_result(op, ctx, out)


def _emit_tile_unary(
    op: Any,
    ctx: EmitContext,
    x: Any,
    scalar_apply: Callable[[Any], Any],
    op_label: str,
    out_dtype: str,
) -> Any:
    """Emit a per-lane ``tir.For`` for a unary op on a tile operand.

    Mirrors :func:`_emit_tile_binop` but for arity-1 ops (math intrinsics
    plus arith.* casts). Returns the freshly allocated result ``tir.Buffer``;
    ``scalar_apply`` receives the per-lane PrimExpr and returns the per-lane
    result PrimExpr. Used by ``math.exp`` / ``math.sqrt`` / ``math.log`` and
    cast emitters when the input resolves to a tile.
    """
    out_shape = _tile_shape(ctx, x) if _is_tile_operand(ctx, x) else ()
    if not out_shape:
        raise EmitError(
            f"{op_label}: _emit_tile_unary called without tile operand"
        )
    lazy = LazyTileExpr(
        out_shape,
        out_dtype,
        lambda read_ctx, indices: scalar_apply(_read_lane(read_ctx, x, tuple(indices))),
        name=ctx.fresh("tile_expr"),
    )
    return _bind_result(op, ctx, lazy)


def _maybe_tile_binop(
    op: Any,
    ctx: EmitContext,
    a: Any,
    b: Any,
    scalar_combine: Callable[[Any, Any], Any],
    op_label: str,
    out_dtype: str,
) -> Any:
    """Dispatch: tile path when either operand is a tile, scalar otherwise.

    Returns ``None`` when the scalar fast-path applies (caller emits the
    PrimExpr directly); returns the bound result Buffer when the tile
    path was taken.
    """
    if _is_tile_operand(ctx, a) or _is_tile_operand(ctx, b):
        return _emit_tile_binop(op, ctx, a, b, scalar_combine, op_label, out_dtype)
    return None


# ---------------------------------------------------------------------------
# Float arithmetic (arith.*f)
# ---------------------------------------------------------------------------


def _emit_addf(op: Any, ctx: EmitContext) -> Any:
    # REGCARRYCOLLAPSE: the UNFOLDED dstates recurrence emits a SEPARATE
    # ``%acc = arith.addf %carry_blockarg, %dot_result`` after the dot. By the
    # time we reach this addf, ``map_tt_dot`` has already accumulated the dot
    # IN PLACE into the carry fragment (C = the carry, clear_accum=False), so
    # the carry fragment ALREADY holds ``carry + dot``. Both operands here are
    # bound to that same carry fragment, so emitting ``carry + carry`` would
    # DOUBLE the accumulator. ``map_scf_for`` recorded this addf's result SSA in
    # ``frag_carry_unfold_add`` after proving the structural recurrence; rebind
    # the add result straight to the carry fragment (no compute) so the yielded
    # value IS the carry -> ``_append_loop_carry_copies`` short-circuits (no
    # carry_next) and there is no shared dot_c_logical / tile_binop. RULE #1:
    # only the proven unfold-add result reaches this no-op rebind.
    _unfold_add = getattr(ctx, "frag_carry_unfold_add", None) or {}
    if _unfold_add:
        from ..op_mapping import _result_ssa_name  # noqa: WPS433
        _add_res = _result_ssa_name(op)
        _carry_ssa = _unfold_add.get(_add_res) if _add_res else None
        if _carry_ssa is not None:
            try:
                _carry_buf = ctx.get(_carry_ssa)
            except KeyError:
                _carry_buf = (getattr(ctx, "loop_carry_buffers", {}) or {}).get(
                    _carry_ssa
                )
            if _carry_buf is None:
                raise EmitError(
                    "arith.addf REGCARRYCOLLAPSE: unfolded-recurrence carry "
                    f"block-arg {_carry_ssa!r} for add result {_add_res!r} is "
                    "not bound to a fragment buffer; the dot must have routed "
                    "its in-place accumulate into this carry first. Refusing to "
                    "emit a doubling carry+carry add (RULE #1)."
                )
            return _bind_result(op, ctx, _carry_buf)
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.addf on non-float type {dt!r}; use arith.addi")
    if _loop_carry_target(ctx, op, a) is None and _is_zero_constant_tile(ctx, a):
        return _bind_result(op, ctx, b)
    if _is_zero_constant_tile(ctx, b):
        return _bind_result(op, ctx, a)
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().Add(x, y),
                             "arith.addf", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().Add(a, b))


def _emit_subf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.subf on non-float type {dt!r}; use arith.subi")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().Sub(x, y),
                             "arith.subf", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().Sub(a, b))


def _emit_mulf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.mulf on non-float type {dt!r}; use arith.muli")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().Mul(x, y),
                             "arith.mulf", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().Mul(a, b))


def _emit_divf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.divf on non-float type {dt!r}; use arith.divsi/divui")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().Div(x, y),
                             "arith.divf", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().Div(a, b))


def _emit_negf(op: Any, ctx: EmitContext) -> Any:
    a = _resolve_one(op, ctx)
    dt = _dtype_of(a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.negf on non-float type {dt!r}")
    tir = ctx.tir()
    zero = tir.const(0, dt)
    return _bind_result(op, ctx, tir.Sub(zero, a))


# ---------------------------------------------------------------------------
# Integer arithmetic (arith.*i)
# ---------------------------------------------------------------------------


def _emit_addi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.addi on float type {dt!r}; use arith.addf")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().Add(x, y),
                             "arith.addi", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().Add(a, b))


def _emit_subi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.subi on float type {dt!r}; use arith.subf")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().Sub(x, y),
                             "arith.subi", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().Sub(a, b))


def _emit_muli(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.muli on float type {dt!r}; use arith.mulf")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().Mul(x, y),
                             "arith.muli", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().Mul(a, b))


def _emit_divsi(op: Any, ctx: EmitContext) -> Any:
    """Signed integer division -- TIR ``truncdiv`` matches MLIR ``arith.divsi``."""
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.divsi on float type {dt!r}; use arith.divf")
    tir = ctx.tir()
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: tir.truncdiv(x, y),
                             "arith.divsi", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, tir.truncdiv(a, b))


def _emit_divui(op: Any, ctx: EmitContext) -> Any:
    """Unsigned integer division -- ``floordiv`` for non-negative operands.

    MLIR ``arith.divui`` is well-defined only for non-negative operands; the
    backend dtype (``uint*``) carries that contract. ``floordiv`` lowers to
    the same hardware divide on every backend we target.
    """
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.divui on float type {dt!r}; use arith.divf")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().floordiv(x, y),
                             "arith.divui", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().floordiv(a, b))


def _emit_remsi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.remsi on float type {dt!r}")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().truncmod(x, y),
                             "arith.remsi", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().truncmod(a, b))


def _emit_remui(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.remui on float type {dt!r}")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().floormod(x, y),
                             "arith.remui", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().floormod(a, b))


# ---------------------------------------------------------------------------
# Bitwise / logical (arith.andi / arith.ori / arith.xori)
# ---------------------------------------------------------------------------
#
# In MLIR ``arith.andi`` is the canonical spelling for both bitwise AND on
# integer tensors and logical AND on ``i1`` boolean tensors (Triton's TTIR
# emits this for compound boundary masks such as
# ``mask_row_lt_T & mask_col_lt_K`` produced by ``tt.make_block_ptr`` +
# ``tl.load(..., boundary_check=(0, 1))``). FLA's
# ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` lowers each boundary
# check into a chain of ``arith.andi : tensor<...xi1>`` ops; without this
# emitter the reducer stalls at FAILED_OPS on ``arith.andi``.
#
# TVM's ``tir.bitwise_and`` covers both the integer-bitwise and
# i1-logical paths (TIR boolean ops are represented as 8-bit ints and
# bitwise-AND on an i1 tile is semantically logical-AND).


def _emit_andi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.andi on float type {dt!r}")
    tir = ctx.tir()
    # PROLOGUE-OPT transform (3) cont.: ``guard_true & x`` -> ``x``. The
    # int32-overflow guards (folded to constant-true tiles in ``_emit_cmpi``)
    # would otherwise materialize a bool[] AND loop here. Folding the identity
    # keeps the real predicate lazy and removes the loop. Gated.
    if getattr(ctx, "routed_triton_prologue_opt", False):
        a_true = _constant_tile_bool(a) is True
        b_true = _constant_tile_bool(b) is True
        if a_true and not b_true:
            return _bind_result(op, ctx, b)
        if b_true and not a_true:
            return _bind_result(op, ctx, a)
        if a_true and b_true:
            return _bind_result(op, ctx, a)
    tile = _maybe_tile_binop(op, ctx, a, b,
                             lambda x, y: tir.bitwise_and(x, y),
                             "arith.andi", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, tir.bitwise_and(a, b))


def _emit_ori(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.ori on float type {dt!r}")
    tir = ctx.tir()
    tile = _maybe_tile_binop(op, ctx, a, b,
                             lambda x, y: tir.bitwise_or(x, y),
                             "arith.ori", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, tir.bitwise_or(a, b))


def _emit_xori(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.xori on float type {dt!r}")
    tir = ctx.tir()
    tile = _maybe_tile_binop(op, ctx, a, b,
                             lambda x, y: tir.bitwise_xor(x, y),
                             "arith.xori", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, tir.bitwise_xor(a, b))


# ---------------------------------------------------------------------------
# Min / max
# ---------------------------------------------------------------------------


def _emit_minimumf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.minimumf on non-float type {dt!r}; use arith.minsi")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().Min(x, y),
                             "arith.minimumf", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().Min(a, b))


def _emit_maximumf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.maximumf on non-float type {dt!r}; use arith.maxsi")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().Max(x, y),
                             "arith.maximumf", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().Max(a, b))


def _emit_minsi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.minsi on float type {dt!r}; use arith.minimumf")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().Min(x, y),
                             "arith.minsi", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().Min(a, b))


def _emit_maxsi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.maxsi on float type {dt!r}; use arith.maximumf")
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: ctx.tir().Max(x, y),
                             "arith.maxsi", dt)
    if tile is not None:
        return tile
    return _bind_result(op, ctx, ctx.tir().Max(a, b))


# ---------------------------------------------------------------------------
# Math intrinsics (math.*)
# ---------------------------------------------------------------------------


def _math_unary(name: str, tir_fn_name: str):
    """Build an emitter that maps math.<name> to ``tvm.tir.<tir_fn_name>``.

    Tile-shape contract: when ``x`` is a tile operand (``tir.Buffer`` /
    ``Broadcast`` / ``Ramp`` / vector PrimExpr), the scalar ``tir.<fn>``
    constructor would raise ``tirx.convert: Expected PrimExpr but got
    tirx.Buffer``. We instead allocate a fresh tile buffer and emit a
    serial ``tir.For`` nest that applies the scalar intrinsic per-lane,
    matching the pattern used by ``_emit_tile_binop`` for binary arith.
    """

    def _emit(op: Any, ctx: EmitContext) -> Any:
        x = _resolve_one(op, ctx)
        dt = _tile_dtype(ctx, x)  # per-lane scalar dtype (handles Buffer/vector)
        if not _is_float_dtype(dt):
            raise EmitError(
                f"math.{name} on non-float type {dt!r}; "
                f"Triton frontends emit math.* only on float operands"
            )
        tir = ctx.tir()
        fn = getattr(tir, tir_fn_name, None)

        def _scalar_apply(scalar: Any) -> Any:
            if fn is None:
                return tir.call_intrin(dt, f"tir.{tir_fn_name}", scalar)
            return fn(scalar)

        if _is_tile_operand(ctx, x):
            return _emit_tile_unary(op, ctx, x, _scalar_apply, f"math.{name}", dt)

        return _bind_result(op, ctx, _scalar_apply(x))

    _emit.__name__ = f"_emit_math_{name}"
    return _emit


_emit_math_sqrt = _math_unary("sqrt", "sqrt")
_emit_math_exp = _math_unary("exp", "exp")
# ``math.exp2`` / ``math.log2`` are emitted by Triton's frontend whenever the
# user code calls ``tl.math.exp2`` / ``tl.exp2`` / ``tl.math.log2`` /
# ``tl.log2``. flash-linear-attention's gated-delta-rule kernels (and the
# ``fla.ops.utils.op.exp2`` shim that wraps ``tl.math.exp2``) lean on these
# heavily; without the emitter the OP_TABLE-membership probe in
# ``run_corpus`` reports the kernel as FAILED_OPS even though the upstream
# ``math.exp`` / ``math.log`` cousins lower fine.
_emit_math_exp2 = _math_unary("exp2", "exp2")
_emit_math_log = _math_unary("log", "log")
_emit_math_log2 = _math_unary("log2", "log2")
# ``math.rsqrt`` shows up in normalisation kernels (LayerNorm, RMSNorm) and
# is part of the same flash-linear-attention surface that needs ``exp2``.
_emit_math_rsqrt = _math_unary("rsqrt", "rsqrt")
_emit_math_sin = _math_unary("sin", "sin")
_emit_math_cos = _math_unary("cos", "cos")
_emit_math_tanh = _math_unary("tanh", "tanh")
# ``math.erf`` is needed for the GELU-erf activation; ``math.floor`` /
# ``math.ceil`` appear in indexing math after constant folding.
_emit_math_erf = _math_unary("erf", "erf")
_emit_math_floor = _math_unary("floor", "floor")
_emit_math_ceil = _math_unary("ceil", "ceil")


def _emit_math_absf(op: Any, ctx: EmitContext) -> Any:
    x = _resolve_one(op, ctx)
    dt = _tile_dtype(ctx, x)
    if not _is_float_dtype(dt):
        raise EmitError(f"math.absf on non-float type {dt!r}; use math.absi")
    tir = ctx.tir()
    abs_fn = getattr(tir, "abs", None)

    def _apply(scalar: Any) -> Any:
        if abs_fn is None:
            return tir.call_intrin(dt, "tir.fabs", scalar)
        return abs_fn(scalar)

    if _is_tile_operand(ctx, x):
        return _emit_tile_unary(op, ctx, x, _apply, "math.absf", dt)
    return _bind_result(op, ctx, _apply(x))


# ---------------------------------------------------------------------------
# Comparisons (arith.cmpf / arith.cmpi)
# ---------------------------------------------------------------------------

# MLIR arith.CmpFPredicate enum -> tvm.tir comparison node factory.
# References:
#   llvm-project/mlir/include/mlir/Dialect/Arith/IR/ArithBase.td
# Both numeric values and string mnemonics are accepted because dict-shaped
# fakes typically pass strings while real MLIR exposes a typed Attribute.
_CMPF_PREDICATES = {
    # 0 false - never true; we expand to (x != x) which is False for non-NaN.
    # Triton frontend never emits this so we error out instead.
    "false":  None,
    "oeq":    "EQ",
    "ogt":    "GT",
    "oge":    "GE",
    "olt":    "LT",
    "ole":    "LE",
    "one":    "NE",
    "ord":    None,   # ordered: !isNaN(x) && !isNaN(y); we don't lower NaN checks.
    "ueq":    "EQ",   # unordered-or-equal: equivalent for non-NaN inputs.
    "ugt":    "GT",
    "uge":    "GE",
    "ult":    "LT",
    "ule":    "LE",
    "une":    "NE",
    "uno":    None,
    "true":   None,
}

_CMPF_NUMERIC = [
    "false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord",
    "ueq",   "ugt", "uge", "ult", "ule", "une", "uno", "true",
]


_CMPI_PREDICATES = {
    "eq":  "EQ",
    "ne":  "NE",
    "slt": "LT",
    "sle": "LE",
    "sgt": "GT",
    "sge": "GE",
    "ult": "LT",
    "ule": "LE",
    "ugt": "GT",
    "uge": "GE",
}

_CMPI_NUMERIC = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]


def _normalize_predicate(raw: Any, table: Dict[str, Any], numeric: list) -> str:
    """Coerce raw predicate (int / str / IntegerAttr) to a string key in ``table``."""
    # Integer enum value -> name.
    if isinstance(raw, bool):
        # Bool is a subclass of int; reject explicitly to avoid accidents.
        raise EmitError(f"comparison predicate must be int/str, got bool {raw}")
    if isinstance(raw, int):
        if 0 <= raw < len(numeric):
            return numeric[raw]
        raise EmitError(f"comparison predicate index {raw} out of range")
    s = str(raw).strip().lower()
    # MLIR sometimes prefixes with the enum name ("CmpFPredicate.olt").
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    if s.startswith("predicate"):
        s = s[len("predicate"):]
    if s not in table:
        raise EmitError(f"unknown comparison predicate {raw!r}")
    return s


def _emit_cmpf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.cmpf on non-float type {dt!r}; use arith.cmpi")
    # Real Triton 3.6 ops store ``predicate`` as a Property which jaxlib's
    # bindings hide from ``op.attributes`` -- route through the shared
    # parser so we transparently pick it up from the ``<{...}>`` block.
    attrs = _attrs_with_properties_shared(op)
    raw = attrs.get("predicate", attrs.get("kind"))
    if raw is None:
        raise EmitError("arith.cmpf: missing 'predicate' attribute")
    pred = _normalize_predicate(raw, _CMPF_PREDICATES, _CMPF_NUMERIC)
    cls_name = _CMPF_PREDICATES[pred]
    if cls_name is None:
        raise EmitError(
            f"arith.cmpf predicate {pred!r} (false/true/ord/uno) not lowerable; "
            f"Triton frontend should not emit it"
        )
    tir = ctx.tir()
    cls = getattr(tir, cls_name)
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: cls(x, y),
                             "arith.cmpf", "bool")
    if tile is not None:
        return tile
    return _bind_result(op, ctx, cls(a, b))


_INT32_MAX = 2147483647
_INT32_MIN = -2147483648


def _imm_int(value: Any) -> Any:
    """Return the python int for an IntImm-like scalar PrimExpr, else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    # ``tir.Broadcast`` of an IntImm: peel one ``.value`` layer.
    raw = getattr(value, "value", None)
    if raw is not None and hasattr(raw, "dtype") and not hasattr(raw, "value"):
        raw = None
    if raw is not None and hasattr(raw, "value") and not hasattr(getattr(raw, "value"), "dtype"):
        try:
            return int(raw.value)
        except Exception:
            pass
    raw = getattr(value, "value", None)
    if raw is not None and not hasattr(raw, "dtype"):
        try:
            return int(raw)
        except Exception:
            return None
    return None


def _const_int_operand(value: Any) -> Any:
    """Return the python int for a scalar/splat constant operand, else ``None``.

    Recognizes a raw python int, a ``tir.IntImm``, a ``tir.Broadcast`` of an
    IntImm, AND a constant-splat ``LazyTileExpr`` (Triton's ``tt.splat`` of an
    int literal -- e.g. the int32 overflow bound widened to a tile). For the
    LazyTileExpr case we probe ``read_lane`` with a zero index: a constant
    splat's reader returns the same IntImm regardless of lane, so the value is
    safe to fold. A genuine per-lane expression returns a non-constant and is
    correctly rejected.
    """
    direct = _imm_int(value)
    if direct is not None:
        return direct
    if isinstance(value, LazyTileExpr):
        cv = getattr(value, "constant_value", None)
        if cv is not None and not isinstance(cv, bool):
            try:
                return int(cv)
            except Exception:
                return None
        # Probe the reader at lane 0 with a throwaway minimal ctx; a constant
        # splat ignores ``read_ctx``/indices and returns its IntImm.
        try:
            import tvm  # noqa: WPS433

            class _Probe:
                @staticmethod
                def tir():
                    return tvm.tir

            zero = (tvm.tir.const(0, "int32"),) * max(len(value.shape), 1)
            lane = value.read_lane(_Probe, zero)
            return _imm_int(lane)
        except Exception:
            return None
    return None


def _is_int32_overflow_guard(ctx: EmitContext, pred: str, a: Any, b: Any) -> bool:
    """Recognize Triton's always-true int64<->int32 overflow bounds.

    PROLOGUE-OPT transform (3). Triton emits, around every int64 address
    expression, the pair ``addr <= 2147483647`` (``sle``) and
    ``-2147483648 <= addr`` (``sle`` with the const on the LHS). For the
    Tri-Dao chunk shapes these are statically TRUE, but the walker would
    otherwise materialize them as ``bool[64]`` / ``bool[4096]`` arrays plus a
    serial loop each, then ``&`` them together. We canonicalize the pair to a
    constant-true tile so neither the array nor the loop is emitted.

    Returns True only for the SLE/SGE bound forms whose constant operand is
    exactly the int32 extreme on the side that makes the predicate vacuous.
    Gated to the routed-triton path so non-routed parity stays byte-identical.
    """
    if not getattr(ctx, "routed_triton_prologue_opt", False):
        return False
    if pred not in ("sle", "sge"):
        return False
    ca = _const_int_operand(a)
    cb = _const_int_operand(b)
    # The "address" side is any NON-constant integer expression: a tile
    # (Buffer / LazyTileExpr / Ramp), OR a scalar ``int64`` address PrimExpr
    # (``Mul``/``Add``/``Sub`` of blockIdx-scaled offsets) read per-lane
    # inside ``materialize_lazy_tile``. Either way Triton's guard against it
    # is vacuous for the Tri-Dao chunk shapes. We require the OTHER side to be
    # exactly the int32 extreme and this side to NOT be a python int constant.
    def _is_addr_side(value: Any, const: Any) -> bool:
        if const is not None:
            return False  # a literal constant is never the address side
        if _is_tile_operand(ctx, value):
            return True
        dt = str(getattr(value, "dtype", ""))
        # scalar int address expression (no vector lanes suffix).
        return dt.startswith("int") and "x" not in dt
    if pred == "sle":
        # ``addr <= INT32_MAX``  -> always true.
        if cb == _INT32_MAX and _is_addr_side(a, ca):
            return True
        # ``INT32_MIN <= addr``  -> always true.
        if ca == _INT32_MIN and _is_addr_side(b, cb):
            return True
    if pred == "sge":
        # ``addr >= INT32_MIN``  -> always true.
        if cb == _INT32_MIN and _is_addr_side(a, ca):
            return True
        # ``INT32_MAX >= addr``  -> always true.
        if ca == _INT32_MAX and _is_addr_side(b, cb):
            return True
    return False


def _const_true_tile(ctx: EmitContext, shape: Any) -> Any:
    """Build a constant-true ``bool`` LazyTileExpr over ``shape`` (no buffer)."""
    tir = ctx.tir()
    return LazyTileExpr(
        tuple(int(s) for s in shape) or (1,),
        "bool",
        lambda read_ctx, indices: read_ctx.tir().const(True, "bool"),
        name=ctx.fresh("guard_true"),
        constant_value=True,
    )


def _emit_cmpi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _tile_dtype(ctx, a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.cmpi on float type {dt!r}; use arith.cmpf")
    # Real Triton 3.6 ops store ``predicate`` as a Property which jaxlib's
    # bindings hide from ``op.attributes`` -- route through the shared
    # parser so we transparently pick it up from the ``<{...}>`` block.
    attrs = _attrs_with_properties_shared(op)
    raw = attrs.get("predicate", attrs.get("kind"))
    if raw is None:
        raise EmitError("arith.cmpi: missing 'predicate' attribute")
    pred = _normalize_predicate(raw, _CMPI_PREDICATES, _CMPI_NUMERIC)
    cls_name = _CMPI_PREDICATES[pred]
    tir = ctx.tir()
    cls = getattr(tir, cls_name)
    # PROLOGUE-OPT transform (3): fold always-true int32-overflow guards to a
    # constant-true tile (no bool[] array, no serial loop). ``_emit_andi``
    # then drops them when ``&``-ing with a real predicate.
    if _is_int32_overflow_guard(ctx, pred, a, b):
        # Tile operand -> constant-true tile (folds the bool[] array + loop).
        # Scalar address operand (read per-lane inside materialize_lazy_tile)
        # -> scalar ``True`` so the per-lane guard body collapses to a const.
        if _is_tile_operand(ctx, a) or _is_tile_operand(ctx, b):
            out_shape = _tile_shape(ctx, a) if _is_tile_operand(ctx, a) else _tile_shape(ctx, b)
            return _bind_result(op, ctx, _const_true_tile(ctx, out_shape))
        return _bind_result(op, ctx, tir.const(True, "bool"))
    tile = _maybe_tile_binop(op, ctx, a, b, lambda x, y: cls(x, y),
                             "arith.cmpi", "bool")
    if tile is not None:
        return tile
    return _bind_result(op, ctx, cls(a, b))


# ---------------------------------------------------------------------------
# Fused multiply-add
# ---------------------------------------------------------------------------


def _emit_fma(op: Any, ctx: EmitContext) -> Any:
    operands = _operands(op)
    if len(operands) != 3:
        raise EmitError(f"tt.fma: expected 3 operands (a, b, c); got {len(operands)}")
    a, b, c = (ctx.get(o) for o in operands)
    dt = _dtype_of(a)
    if not _is_float_dtype(dt):
        raise EmitError(f"tt.fma on non-float type {dt!r}")
    tir = ctx.tir()
    # Prefer a real fma intrinsic when registered. Probe the IR's Op registry
    # before issuing the call so we don't crash builders that lack the op.
    fma_op = None
    try:
        op_get = getattr(getattr(tir, "op", None), "Op", None)
        if op_get is not None and hasattr(op_get, "get"):
            try:
                fma_op = op_get.get("tir.fma")
            except Exception:  # pragma: no cover -- registry miss
                fma_op = None
    except Exception:  # pragma: no cover -- defensive
        fma_op = None
    if fma_op is not None:
        result = tir.call_intrin(dt, fma_op, a, b, c)
    else:
        result = tir.Add(tir.Mul(a, b), c)
    return _bind_result(op, ctx, result)


# ---------------------------------------------------------------------------
# Dispatch table (exported & merged into op_mapping.OP_TABLE at import time)
# ---------------------------------------------------------------------------

ARITH_EMITTERS: Dict[str, Callable[[Any, EmitContext], Any]] = {
    # Float arithmetic
    "arith.addf": _emit_addf,
    "arith.subf": _emit_subf,
    "arith.mulf": _emit_mulf,
    "arith.divf": _emit_divf,
    "arith.negf": _emit_negf,
    # Integer arithmetic
    "arith.addi": _emit_addi,
    "arith.subi": _emit_subi,
    "arith.muli": _emit_muli,
    "arith.divsi": _emit_divsi,
    "arith.divui": _emit_divui,
    "arith.remsi": _emit_remsi,
    "arith.remui": _emit_remui,
    # Bitwise / logical (i1 boolean masks + integer bitops)
    "arith.andi": _emit_andi,
    "arith.ori": _emit_ori,
    "arith.xori": _emit_xori,
    # Min / max
    "arith.minimumf": _emit_minimumf,
    "arith.maximumf": _emit_maximumf,
    "arith.minnumf": _emit_minimumf,
    "arith.maxnumf": _emit_maximumf,
    "arith.minsi": _emit_minsi,
    "arith.maxsi": _emit_maxsi,
    # Math intrinsics
    "math.sqrt": _emit_math_sqrt,
    "math.rsqrt": _emit_math_rsqrt,
    "math.exp": _emit_math_exp,
    # ``math.exp2`` / ``math.log2`` cover the ``tl.math.exp2`` / ``tl.exp2``
    # path used by flash-linear-attention's gated-delta-rule kernels
    # (``fla.ops.utils.op.exp2``) -- enabling these unblocks the chunk-h
    # forward kernel from FAILED_OPS to LOWERED_DEGRADED in the reducer
    # corpus.
    "math.exp2": _emit_math_exp2,
    "math.log": _emit_math_log,
    "math.log2": _emit_math_log2,
    "math.sin": _emit_math_sin,
    "math.cos": _emit_math_cos,
    "math.tanh": _emit_math_tanh,
    "math.erf": _emit_math_erf,
    "math.floor": _emit_math_floor,
    "math.ceil": _emit_math_ceil,
    "math.absf": _emit_math_absf,
    # Comparisons
    "arith.cmpf": _emit_cmpf,
    "arith.cmpi": _emit_cmpi,
    # FMA
    "tt.fma": _emit_fma,
}
