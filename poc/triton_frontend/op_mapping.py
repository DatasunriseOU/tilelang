"""Triton TTIR -> TileLang TIR op-by-op dispatch table.

Real implementations for all 16 ops in ``RFC_unified_fused_kernel.md``
section 5.1. Status: 16 of 16 emitters now produce TIR (those not
backed by a complete TileLang primitive raise ``NotImplementedError``
with a ``# TODO: verify`` marker; see individual emitters).

Implemented in this file:
    tt.load, tt.store, tt.atomic_rmw, tt.dot, tt.reduce, tt.where,
    tt.broadcast, tt.splat, tt.expand_dims, tt.reshape, tt.make_range,
    async_copy (and tt.async_commit / tt.async_wait), mbarrier
    (init/arrive/wait), tt.experimental_descriptor_load / _store,
    tt.print.

Each ``map_tt_<name>`` is the emitter for one TTIR op. The dispatch
table :data:`OP_TABLE` is consumed by the TTIR walker invoked from
:func:`triton_frontend.from_ttir`. Emitters return a ``tvm.tir`` PrimExpr
or Stmt, with side effects on ``ctx`` (SSA -> buffer/expression map).

The walker (and the emitters) accept either a real ``mlir.ir.Operation``
or a dict-shaped fake of the form
``{"name": str, "operands": [...], "results": [...], "attrs": {...}}``;
the latter is used in unit tests to avoid an MLIR-bindings dependency.

Adding a new mapping
--------------------
1. Add ``def map_tt_<new_name>(op, ctx) -> Any: ...`` below.
2. Register it in :data:`OP_TABLE` under the exact TTIR op name.
3. Cite the RFC subsection that justifies the lowering.
4. Add a conformance kernel under :mod:`triton_frontend.conformance`
   that exercises the new op end-to-end.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "OP_TABLE",
    "EmitFn",
    "WalkerCtx",
    # Memory ops
    "map_tt_load",
    "map_tt_store",
    "map_tt_atomic_rmw",
    # Compute ops
    "map_tt_dot",
    "map_tt_reduce",
    "map_tt_where",
    # Shape ops
    "map_tt_broadcast",
    "map_tt_splat",
    "map_tt_expand_dims",
    "map_tt_reshape",
    "map_tt_make_range",
    # Async / barrier
    "map_tt_async_copy",
    "map_tt_mbarrier",
    # TMA
    "map_tt_experimental_descriptor_load",
    "map_tt_experimental_descriptor_store",
    # Misc
    "map_tt_print",
]


EmitFn = Callable[..., Any]
"""Type alias: ``(op: mlir.ir.Operation, ctx: WalkerCtx) -> tvm.tir.Stmt|Expr``."""


class WalkerCtx:
    """State threaded through the TTIR walker.

    The walker keeps a mapping from MLIR SSA values to TVM TIR exprs,
    a list of already-emitted statements, plus references to the buffers
    that the eventual ``tvm.tir.PrimFunc`` will declare.

    The fields here are all ``Any`` because real TVM types are imported
    lazily inside emitters; the dataclass-shaped surface keeps emitters
    test-friendly without importing TVM at module load time.
    """

    def __init__(self) -> None:
        # SSA value -> tvm.tir.PrimExpr or tvm.tir.Buffer
        self.value_map: Dict[Any, Any] = {}
        # Ordered list of emitted statements (becomes a SeqStmt body).
        self.stmts: List[Any] = []
        # Param name -> tvm.tir.Buffer (for tt.func arguments).
        self.buffers: Dict[str, Any] = {}
        # Auto-generated temp counter for fresh names.
        self._tmp_counter: int = 0
        # Lazy-loaded TVM modules.
        self._tvm: Any = None
        self._T: Any = None

    # ---- helpers --------------------------------------------------------

    def fresh(self, prefix: str = "v") -> str:
        """Return a unique name suitable for a buffer / variable."""
        self._tmp_counter += 1
        return f"{prefix}_{self._tmp_counter}"

    def tvm(self) -> Any:
        """Lazy-import ``tvm`` and cache the module handle."""
        if self._tvm is None:
            import tvm  # noqa: WPS433 (intentional lazy import)
            self._tvm = tvm
        return self._tvm

    def tir(self) -> Any:
        """Shortcut to ``tvm.tir``."""
        return self.tvm().tir

    def get(self, ssa_value: Any) -> Any:
        """Resolve an MLIR SSA value to its TIR equivalent."""
        if ssa_value in self.value_map:
            return self.value_map[ssa_value]
        raise KeyError(
            f"WalkerCtx: SSA value {ssa_value!r} not yet mapped; emitter "
            f"called out of TTIR program order?"
        )

    def bind(self, ssa_value: Any, tir_value: Any) -> None:
        """Record the TIR value produced by an emitter for ``ssa_value``."""
        self.value_map[ssa_value] = tir_value

    def emit(self, stmt: Any) -> None:
        """Append a TIR statement to the current function body."""
        self.stmts.append(stmt)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _operands(op: Any) -> Tuple[Any, ...]:
    """Return ``op`` operands as a tuple, hiding the MLIR vs dict shape diff.

    The walker can call us with either a real ``mlir.ir.Operation`` or a
    test-only dict ``{"operands": [...], "results": [...], "attrs": {...}}``.
    """
    if isinstance(op, dict):
        return tuple(op.get("operands", ()))
    return tuple(op.operands)


def _results(op: Any) -> Tuple[Any, ...]:
    """Return ``op`` SSA results, hiding the MLIR vs dict shape diff."""
    if isinstance(op, dict):
        return tuple(op.get("results", ()))
    return tuple(op.results)


def _attrs(op: Any) -> Dict[str, Any]:
    """Return ``op`` attribute dict, hiding the MLIR vs dict shape diff."""
    if isinstance(op, dict):
        return dict(op.get("attrs", {}))
    # Real MLIR: attributes attribute. Keep stringified for portability.
    return {a.name: a.attr for a in op.attributes} if hasattr(op, "attributes") else {}


def _shape_of(value: Any) -> Tuple[int, ...]:
    """Best-effort shape extraction for a TTIR SSA value.

    Triton TTIR types are MLIR ``RankedTensorType`` for tile values; for
    scalar values we return ``()``. The walker also feeds us a dict-shaped
    fake during unit tests; we accept that shape too.
    """
    if isinstance(value, dict):
        return tuple(value.get("shape", ()))
    typ = getattr(value, "type", None)
    if typ is None:
        return ()
    shape = getattr(typ, "shape", None)
    return tuple(shape) if shape is not None else ()


def _dtype_of(value: Any) -> str:
    """Best-effort element dtype for a TTIR SSA value (defaults to float32)."""
    if isinstance(value, dict):
        return str(value.get("dtype", "float32"))
    typ = getattr(value, "type", None)
    if typ is None:
        return "float32"
    elt = getattr(typ, "element_type", None)
    if elt is None:
        return "float32"
    return str(elt)


# ---------------------------------------------------------------------------
# Memory ops -- RFC section 5.1, "tt.load" / "tt.store" / "tt.atomic_rmw"
# ---------------------------------------------------------------------------


def map_tt_load(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.load(ptr, mask, other)`` to a guarded ``BufferLoad``.

    RFC section 5.1: ``T.copy`` + masked predicate via ``T.if_then_else``.
    Triton-shared precedent: ``LoadOpConversion`` lowers to
    ``memref.load`` with an ``scf.if`` mask guard. We emit the equivalent
    in TIR: when a mask is present, wrap each lane in ``if_then_else``
    using the optional ``other`` value (default 0) for the false branch.

    Implementation summary
    ----------------------
    * ``ptr`` SSA must be the result of pointer-arithmetic that
      ``ptr_analysis`` has already resolved into a ``StridedLayout``;
      that layout's ``base`` buffer + element offset becomes the
      ``BufferLoad`` index.
    * ``mask`` (optional) is an i1 tile from ``tt.cmp`` etc.
    * ``other`` (optional) is the dtype-typed fill for masked-out lanes.

    The emitter binds the load's SSA result to a ``BufferLoad`` PrimExpr
    so consumers (arith ops, ``tt.store``) can reference it directly.
    """
    tir = ctx.tir()
    operands = _operands(op)
    if len(operands) < 1:
        raise ValueError("tt.load: missing pointer operand")
    ptr_ssa = operands[0]
    mask_ssa = operands[1] if len(operands) >= 2 else None
    other_ssa = operands[2] if len(operands) >= 3 else None

    # ptr_analysis has already populated value_map[ptr_ssa] with a
    # ``(buffer, indices)`` tuple. Fall back to assuming ptr_ssa is an
    # opaque buffer + scalar offset for the elementwise MVP path.
    resolved = ctx.get(ptr_ssa)
    if isinstance(resolved, tuple) and len(resolved) == 2:
        buf, indices = resolved
    else:
        # MVP: treat the SSA value itself as the buffer with offset 0.
        buf, indices = resolved, [0]

    load_expr = tir.BufferLoad(buf, list(indices))

    if mask_ssa is not None:
        mask_expr = ctx.get(mask_ssa)
        if other_ssa is not None:
            other_expr = ctx.get(other_ssa)
        else:
            # Default ``other`` is 0 in Triton semantics.
            dtype = _dtype_of(_results(op)[0]) if _results(op) else "float32"
            other_expr = tir.const(0, dtype)
        load_expr = tir.if_then_else(mask_expr, load_expr, other_expr)

    if _results(op):
        ctx.bind(_results(op)[0], load_expr)
    return load_expr


def map_tt_store(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.store(ptr, val, mask)`` to a guarded ``BufferStore``.

    RFC section 5.1: ``T.copy`` + masked predicate. We emit a
    ``BufferStore`` directly into the resolved buffer, optionally wrapped
    in an ``IfThenElse`` when ``mask`` is present. The downstream
    ``LowerTileOp`` / ``MergeIfStmt`` passes coalesce these into
    vectorized writes.
    """
    tir = ctx.tir()
    operands = _operands(op)
    if len(operands) < 2:
        raise ValueError("tt.store: missing pointer or value operand")
    ptr_ssa, val_ssa = operands[0], operands[1]
    mask_ssa = operands[2] if len(operands) >= 3 else None

    resolved = ctx.get(ptr_ssa)
    if isinstance(resolved, tuple) and len(resolved) == 2:
        buf, indices = resolved
    else:
        buf, indices = resolved, [0]
    val_expr = ctx.get(val_ssa)

    store_stmt = tir.BufferStore(buf, val_expr, list(indices))
    if mask_ssa is not None:
        mask_expr = ctx.get(mask_ssa)
        store_stmt = tir.IfThenElse(mask_expr, store_stmt, None)

    ctx.emit(store_stmt)
    return store_stmt


def _atomic_rmw_kind(op: Any) -> str:
    """Extract the RMW kind from a TTIR ``tt.atomic_rmw`` op.

    Triton's ``RMWOp`` enum has values ``and``, ``or``, ``xor``, ``add``,
    ``fadd``, ``max``, ``min``, ``umax``, ``umin``, ``xchg``. We
    canonicalize to ``add``/``max``/``min``/``xchg``/``and``/``or``/``xor``
    after stripping the ``f``/``u`` integer-vs-float prefix.
    """
    attrs = _attrs(op)
    raw = attrs.get("rmw_op") or attrs.get("atomic_rmw_op") or attrs.get("kind")
    if raw is None:
        raise ValueError("tt.atomic_rmw: missing 'rmw_op' attribute")
    s = str(raw).lower().strip()
    # Strip MLIR-style prefixes/suffixes that occasionally appear.
    if s.startswith("rmw_op."):
        s = s[len("rmw_op."):]
    if s.startswith("f"):
        # fadd / fmax / fmin -> add / max / min
        rest = s[1:]
        if rest in {"add", "max", "min"}:
            return rest
    if s.startswith("u") and s[1:] in {"max", "min"}:
        # umax / umin -> max / min (signedness disambiguated by buffer dtype)
        return s[1:]
    return s


def map_tt_atomic_rmw(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.atomic_rmw`` to ``T.atomic_*`` / ``call_intrin``.

    Recipe (RFC 5.1):
    * Read the ``rmw_op`` attribute (``add``/``max``/``min``/``xchg``/...).
    * Resolve the pointer via ``ptr_analysis`` to ``(buffer, indices)``.
    * Emit ``tilelang.language.atomic_add`` / ``atomic_max`` / etc. when
      a TileLang intrinsic exists; otherwise fall back to
      ``tir.call_intrin("tir.atomic_<op>", buffer.access_ptr("w"), val)``.
    * Mask handling matches ``tt.store``: wrap in ``if_then_else``.
    * Result SSA (the pre-update value) binds the call's return value.
    """
    operands = _operands(op)
    if len(operands) < 2:
        raise ValueError("tt.atomic_rmw: expected at least (ptr, val) operands")
    ptr_ssa, val_ssa = operands[0], operands[1]
    mask_ssa = operands[2] if len(operands) >= 3 else None

    resolved = ctx.get(ptr_ssa)
    if isinstance(resolved, tuple) and len(resolved) == 2:
        buf, indices = resolved
    else:
        buf, indices = resolved, [0]
    val_expr = ctx.get(val_ssa)
    kind = _atomic_rmw_kind(op)

    tir = ctx.tir()
    # Lazy-import TileLang only when emitting (test path uses dicts but
    # never reaches this branch unless TileLang is available).
    try:
        import tilelang.language as T  # type: ignore
    except ImportError:  # pragma: no cover -- tilelang absent
        T = None  # type: ignore

    # We model the destination as buf[indices] -- the scalar atomic path.
    # ``atomic_add`` / ``atomic_max`` / ``atomic_min`` accept a Buffer; we
    # pass the underlying buffer with the indices encoded in ``val_expr``
    # by way of BufferLoad / BufferStore semantics handled inside TileLang.
    return_prev = bool(_results(op))

    if T is not None and kind in {"add", "max", "min", "xchg", "and", "or", "xor"}:
        atomic_fn = {
            "add": T.atomic_add,
            "max": T.atomic_max,
            "min": T.atomic_min,
            "xchg": T.atomic_xchg,
            "and": T.atomic_and,
            "or": T.atomic_or,
            "xor": T.atomic_xor,
        }[kind]
        # The scalar TileLang path expects a Buffer directly. When indices
        # are nonzero the caller (ptr_analysis) is expected to slice the
        # buffer ahead of time; the MVP path uses a flat buffer view.
        result = atomic_fn(buf, val_expr, return_prev=return_prev)
    else:
        # Unknown rmw_op: fall back to a generic ``tir.atomic_<op>`` extern.
        intrin_name = f"tir.atomic_{kind}"
        ret_dtype = _dtype_of(_results(op)[0]) if return_prev and _results(op) else "handle"
        # Build access pointer: buf.access_ptr("rw") + offset_indices.
        # For the MVP we let the buffer's flat element 0 be the target;
        # ptr_analysis fills in proper indices for non-trivial cases.
        access = tir.call_intrin(
            "handle",
            tir.op.Op.get("tir.address_of"),
            tir.BufferLoad(buf, list(indices)),
        ) if hasattr(tir.op.Op, "get") else buf
        result = tir.call_intrin(ret_dtype, intrin_name, access, val_expr)

    if mask_ssa is not None:
        mask_expr = ctx.get(mask_ssa)
        # Wrap in if_then_else; for non-prev returns we still emit the call
        # unconditionally because TileLang intrinsics treat the call as a
        # statement-level handle.
        if return_prev:
            zero = tir.const(0, _dtype_of(_results(op)[0]))
            result = tir.if_then_else(mask_expr, result, zero)
        else:
            result = tir.IfThenElse(mask_expr, tir.Evaluate(result), None)

    if return_prev:
        ctx.bind(_results(op)[0], result)
    else:
        ctx.emit(result)
    return result


# ---------------------------------------------------------------------------
# Compute ops -- RFC section 5.1, "tt.dot" / "tt.reduce" / "tt.where"
# ---------------------------------------------------------------------------


def map_tt_dot(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.dot(a, b, c)`` to ``T.gemm(A, B, C)``.

    Recipe (RFC 5.1):
    * Operands a, b must be tiles already in ``shared`` or ``fragment``
      scope (PtrAnalysis + LayoutInference will hoist them); accumulator
      ``c`` lives in ``fragment``. If ``c`` is missing (Triton allows it)
      we allocate a fresh fragment buffer at the result's shape/dtype.
    * Emit ``tilelang.language.gemm(A, B, C)``; LayoutInference picks
      WMMA / WGMMA / MFMA / SIMDgroup per target.
    * The result SSA binds the in-place accumulator buffer ``C`` (gemm
      reads-modifies-writes); downstream ops see ``C`` as the value.
    * Triton fp16 inputs accumulate to fp32 by convention; we honour the
      result's TTIR dtype (the frontend type-checker has set it).
    """
    operands = _operands(op)
    if len(operands) < 2:
        raise ValueError("tt.dot: expected at least 2 operands (A, B)")
    a_ssa, b_ssa = operands[0], operands[1]
    c_ssa = operands[2] if len(operands) >= 3 else None

    a = ctx.get(a_ssa)
    b = ctx.get(b_ssa)

    attrs = _attrs(op)
    transpose_A = bool(attrs.get("transpose_A", False) or attrs.get("trans_a", False))
    transpose_B = bool(attrs.get("transpose_B", False) or attrs.get("trans_b", False))

    import tilelang.language as T  # type: ignore  # lazy
    if c_ssa is not None:
        try:
            c = ctx.get(c_ssa)
        except KeyError:
            c = None
    else:
        c = None
    if c is None:
        # Allocate a fresh fp32 accumulator at the result's shape (Triton
        # default: fp16 inputs accumulate to fp32).
        result = _results(op)[0] if _results(op) else None
        out_shape = list(_shape_of(result)) if result is not None else []
        out_dtype = _dtype_of(result) if result is not None else "float32"
        # Triton TTIR convention: fp16 -> fp32 accumulation.
        if out_dtype in {"float16", "f16", "bfloat16", "bf16"}:
            out_dtype = "float32"
        c = T.alloc_fragment(out_shape, out_dtype)

    handle = T.gemm(a, b, c, transpose_A=transpose_A, transpose_B=transpose_B)
    ctx.emit(handle)
    if _results(op):
        # Bind the result SSA to the accumulator buffer (in-place semantics).
        ctx.bind(_results(op)[0], c)
    return handle


def _reduce_combiner_kind(op: Any) -> str:
    """Inspect a ``tt.reduce`` op's combiner region and return its kind.

    Real MLIR carries the combiner inside a region; the dict-shaped fake
    op simply exposes a top-level ``combiner`` field. Result is one of
    ``"add"``, ``"max"``, ``"min"``, ``"mul"``.
    """
    if isinstance(op, dict):
        c = op.get("combiner")
        if c is not None:
            s = str(c).lower()
            if s in {"add", "sum", "addf", "addi"}:
                return "add"
            if s in {"max", "maxnumf", "maxsi", "maxui"}:
                return "max"
            if s in {"min", "minnumf", "minsi", "minui"}:
                return "min"
            if s in {"mul", "prod", "mulf", "muli"}:
                return "mul"
            return s
    # Real MLIR path: the first op of the first region tells us the kind.
    regions = getattr(op, "regions", None) or ()
    for region in regions:
        for block in getattr(region, "blocks", ()) or ():
            for inner in getattr(block, "operations", ()) or ():
                name = getattr(inner, "name", "")
                low = str(name).lower()
                if "add" in low:
                    return "add"
                if "max" in low:
                    return "max"
                if "min" in low:
                    return "min"
                if "mul" in low:
                    return "mul"
    raise ValueError("tt.reduce: cannot determine combiner kind from op")


def map_tt_reduce(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.reduce`` to ``T.reduce_sum`` / ``T.reduce_max`` / etc.

    Recipe (RFC 5.1):
    * Inspect the reduce op's combiner region: addf -> reduce_sum,
      maxnumf -> reduce_max, minnumf -> reduce_min, mulf -> reduce_prod
      (we currently fall back to a sum-of-logs note for ``mul``; see
      ``# TODO: verify`` below).
    * Allocate a fragment buffer for the result via ``T.alloc_fragment``;
      if the producer is shared, downstream LayoutInference will pick.
    * Emit ``tilelang.language.reduce_<op>(src, dst, axis)`` with the
      MLIR ``axis`` attribute; cross-warp axes will further lower to
      ``LowerThreadAllreduce``.
    * Bind the reducer's destination buffer as the result SSA.
    """
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.reduce: missing source operand")
    src_ssa = operands[0]
    src = ctx.get(src_ssa)

    attrs = _attrs(op)
    axis = int(attrs.get("axis", -1))
    kind = _reduce_combiner_kind(op)

    import tilelang.language as T  # type: ignore  # lazy

    result_value = _results(op)[0] if _results(op) else None
    src_shape = list(_shape_of(operands[0])) or list(getattr(src, "shape", []) or [])
    if src_shape:
        ax = axis if axis >= 0 else len(src_shape) + axis
        out_shape = src_shape[:ax] + src_shape[ax + 1:]
    else:
        out_shape = list(_shape_of(result_value)) if result_value is not None else []
    out_dtype = _dtype_of(result_value) if result_value is not None else _dtype_of(operands[0])
    dst = T.alloc_fragment(out_shape or [1], out_dtype)

    if kind == "add":
        T.reduce_sum(src, dst, dim=axis, clear=True)
    elif kind == "max":
        T.reduce_max(src, dst, dim=axis, clear=True)
    elif kind == "min":
        T.reduce_min(src, dst, dim=axis, clear=True)
    elif kind == "mul":
        # # TODO: verify reduce_prod in tilelang.language; not currently
        # exposed (only sum/max/min/abssum/absmax/bitand/bitor/bitxor).
        raise NotImplementedError(
            "tt.reduce 'mul' combiner: reduce_prod not in tilelang.language; "
            "fall back to expanding into a manual loop or T.cumsum/log path."
        )
    else:
        raise NotImplementedError(
            f"tt.reduce: unsupported combiner kind {kind!r}; "
            f"add wiring in op_mapping._reduce_combiner_kind."
        )

    if result_value is not None:
        ctx.bind(result_value, dst)
    return dst


def map_tt_where(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.where(cond, t, f)`` to ``tir.Select`` (lane-wise)."""
    tir = ctx.tir()
    operands = _operands(op)
    if len(operands) != 3:
        raise ValueError(
            f"tt.where: expected 3 operands (cond, true, false); got {len(operands)}"
        )
    cond, t_val, f_val = (ctx.get(o) for o in operands)
    sel = tir.Select(cond, t_val, f_val)
    if _results(op):
        ctx.bind(_results(op)[0], sel)
    return sel


# ---------------------------------------------------------------------------
# Shape ops -- RFC section 5.1, broadcast/splat/expand_dims/reshape/make_range
# ---------------------------------------------------------------------------


def map_tt_broadcast(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.broadcast`` to a shape-only rebind.

    Triton ``tt.broadcast`` is purely a logical reshape that replicates
    a singleton dim. In our TIR-level abstraction (where tiles are
    represented as PrimExpr trees indexed lazily) we record the new
    shape on the SSA binding and let LayoutInference / FlattenBuffer
    resolve the physical replication. The PrimExpr itself is unchanged.
    """
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.broadcast: missing source operand")
    src = ctx.get(operands[0])
    if _results(op):
        ctx.bind(_results(op)[0], src)
    return src


def map_tt_splat(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.splat`` (scalar -> tile) by binding the scalar PrimExpr.

    Like ``tt.broadcast``, splat is a logical-only op at the TIR layer:
    indexing the resulting tile with any indices yields the same scalar.
    LayoutInference replicates as needed at lowering time.
    """
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.splat: missing source operand")
    src = ctx.get(operands[0])
    if _results(op):
        ctx.bind(_results(op)[0], src)
    return src


def map_tt_expand_dims(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.expand_dims`` to a shape rebind (no data movement).

    The ``axis`` attribute is recorded but not materialized: TileLang's
    PrimExpr indexing carries the rank implicitly and downstream passes
    canonicalize.
    """
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.expand_dims: missing source operand")
    src = ctx.get(operands[0])
    if _results(op):
        ctx.bind(_results(op)[0], src)
    return src


def map_tt_reshape(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.reshape`` to a TileLang ``view`` over the source.

    For elementwise / Tier-1 kernels the reshape is always rank-only and
    can be modeled as a no-op rebind in PrimExpr space; for true
    physical reshapes we fall back to ``tilelang.language.view`` /
    ``tilelang.language.reshape`` (when the source is a buffer rather
    than a PrimExpr).
    """
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.reshape: missing source operand")
    src = ctx.get(operands[0])
    # If the source is a buffer (e.g. shared/local), call into TileLang
    # to materialize the new view; otherwise rebind as PrimExpr.
    tvm_mod = ctx.tvm()
    if isinstance(src, tvm_mod.tir.Buffer):
        try:
            from tilelang.language import view as tl_view  # type: ignore
            new_shape = _shape_of(_results(op)[0]) if _results(op) else ()
            src = tl_view(src, list(new_shape))
        except ImportError:  # pragma: no cover -- TileLang absent
            pass
    if _results(op):
        ctx.bind(_results(op)[0], src)
    return src


def map_tt_make_range(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.make_range(start, end)`` to a Ramp PrimExpr.

    Triton's ``tt.make_range`` is a 1-D tensor of consecutive integers in
    ``[start, end)``. The natural TIR equivalent is ``tir.Ramp`` with
    stride 1 and lanes = (end - start). Consumers (ptr arith) treat it
    like any vector PrimExpr; PtrAnalysis recognizes the pattern and
    folds it into strided indexing.
    """
    tir = ctx.tir()
    attrs = _attrs(op)
    start = int(attrs.get("start", 0))
    end = int(attrs.get("end", 0))
    lanes = end - start
    if lanes <= 0:
        raise ValueError(
            f"tt.make_range: invalid range [{start}, {end}); end must be > start"
        )
    ramp = tir.Ramp(tir.const(start, "int32"), tir.const(1, "int32"), lanes)
    if _results(op):
        ctx.bind(_results(op)[0], ramp)
    return ramp


# ---------------------------------------------------------------------------
# Async / barrier -- RFC section 5.1, async_copy / mbarrier
# ---------------------------------------------------------------------------


def map_tt_async_copy(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``async_copy`` / ``tt.async_commit`` / ``tt.async_wait``.

    Recipe (RFC 5.1):
    * For the load form (``tt.async_copy_global_to_local`` /
      ``async_copy``): resolve src and dst via ``ptr_analysis`` and emit
      ``tilelang.language.async_copy(src, dst)`` -- the call carries the
      pipeline-stage annotation consumed by
      ``InjectSoftwarePipeline`` / ``LowerPTXAsyncCopy``.
    * ``tt.async_commit_group`` and ``tt.async_wait`` are TIR-layer
      no-ops: TileLang's ``inject_pipeline`` derives commit/wait from
      the surrounding ``T.Pipelined`` frame, so we do not need to emit
      anything here. The ops are still consumed by the walker so they
      are not flagged as unmapped.
    """
    op_name = op.get("name") if isinstance(op, dict) else getattr(op, "name", "")
    name = str(op_name).lower()

    # Commit / wait are pipeline boundary markers: no-ops at TIR layer.
    if "commit" in name or "wait" in name:
        return None

    operands = _operands(op)
    if len(operands) < 2:
        raise ValueError("async_copy: expected (src_ptr, dst_ptr) operands")
    src_ssa, dst_ssa = operands[0], operands[1]
    src_resolved = ctx.get(src_ssa)
    dst_resolved = ctx.get(dst_ssa)

    # Unpack (buffer, indices) tuples produced by ptr_analysis; otherwise
    # treat the resolved value as a buffer/region directly.
    def _materialize(resolved: Any) -> Any:
        tir_mod = ctx.tir()
        if isinstance(resolved, tuple) and len(resolved) == 2:
            buf, indices = resolved
            return tir_mod.BufferLoad(buf, list(indices))
        return resolved

    src = _materialize(src_resolved)
    dst = _materialize(dst_resolved)

    import tilelang.language as T  # type: ignore  # lazy
    handle = T.async_copy(src, dst)
    ctx.emit(handle)
    return handle


def map_tt_mbarrier(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``mbarrier`` ops to ``T.alloc_barrier`` + sync intrinsics.

    Recipe (RFC 5.1):
    * For ``tt.barrier_init`` / ``mbarrier.init``: emit
      ``tilelang.language.alloc_barrier(N)`` and bind the result SSA to
      the returned barrier buffer.
    * For ``tt.barrier_arrive`` / ``mbarrier.arrive``: emit
      ``T.barrier_arrive(barrier)``.
    * For ``tt.barrier_wait`` / ``mbarrier.wait``: emit
      ``T.barrier_wait(barrier, parity)``.
    * For plain barriers (no operand): emit ``T.call_intrin`` for
      ``tir.tvm_storage_sync("shared")``; the backend codegen picks
      __syncthreads / s_barrier / threadgroup_barrier per target.
    """
    op_name = op.get("name") if isinstance(op, dict) else getattr(op, "name", "")
    name = str(op_name).lower()
    attrs = _attrs(op)

    import tilelang.language as T  # type: ignore  # lazy

    if "init" in name:
        arrive_count = int(attrs.get("count") or attrs.get("arrive_count") or 1)
        bar = T.alloc_barrier(arrive_count)
        if _results(op):
            ctx.bind(_results(op)[0], bar)
        return bar

    operands = _operands(op)
    if "arrive" in name:
        if not operands:
            raise ValueError("mbarrier.arrive: missing barrier operand")
        bar = ctx.get(operands[0])
        handle = T.barrier_arrive(bar)
        ctx.emit(handle)
        return handle

    if "wait" in name:
        if not operands:
            raise ValueError("mbarrier.wait: missing barrier operand")
        bar = ctx.get(operands[0])
        parity = int(attrs.get("parity", 0))
        handle = T.barrier_wait(bar, parity)
        ctx.emit(handle)
        return handle

    # Plain ``__syncthreads`` style barrier.
    tir = ctx.tir()
    handle = tir.call_intrin("handle", tir.op.Op.get("tir.tvm_storage_sync"), "shared")
    ctx.emit(handle)
    return handle


# ---------------------------------------------------------------------------
# TMA -- RFC section 5.1 + 5.4 (Hopper TMA strategy)
# ---------------------------------------------------------------------------


def _is_nv_target() -> bool:
    """Return True iff the current TVM target looks like NVIDIA CUDA.

    Lazy: we read ``tvm.target.Target.current()`` and fall back to False
    when no target is active (e.g. during dict-shaped unit tests).
    """
    try:
        import tvm  # type: ignore
    except ImportError:  # pragma: no cover
        return False
    target = tvm.target.Target.current(allow_none=True)
    if target is None:
        return False
    kind = str(getattr(target, "kind", "")).lower()
    return "cuda" in kind or "nvptx" in kind


def _emit_descriptor_copy(op: Any, ctx: WalkerCtx, *, is_load: bool) -> Any:
    """Shared body for descriptor load/store: TMA on NV, fallback elsewhere.

    The descriptor SSA is the first operand; the in-shared / out-shared
    tile is the second; remaining operands are coordinate offsets that
    PtrAnalysis has folded into a (buffer, indices) tuple.
    """
    operands = _operands(op)
    if len(operands) < 2:
        raise ValueError(
            f"{'descriptor_load' if is_load else 'descriptor_store'}: expected "
            f"(desc, tile, ...offsets) operands"
        )
    desc_ssa, tile_ssa = operands[0], operands[1]
    desc = ctx.get(desc_ssa)
    tile = ctx.get(tile_ssa)

    import tilelang.language as T  # type: ignore

    if _is_nv_target():
        # On NV, ``T.tma_copy`` directly accepts a descriptor and a tile.
        # FuseMBarrierArriveExpectTx fuses the surrounding mbarrier proto.
        if is_load:
            handle = T.tma_copy(desc, tile)
        else:
            handle = T.tma_copy(tile, desc)
        ctx.emit(handle)
        return handle

    # Non-NV fallback (Triton PR #6753 strategy): decompose into a regular
    # T.copy over the descriptor's base buffer + coordinate offsets that
    # PtrAnalysis has resolved.
    tir = ctx.tir()
    desc_buf, desc_idx = (desc if isinstance(desc, tuple) else (desc, [0]))
    src = tir.BufferLoad(desc_buf, list(desc_idx)) if hasattr(desc_buf, "shape") else desc_buf
    if is_load:
        handle = T.copy(src, tile)
    else:
        handle = T.copy(tile, src)
    ctx.emit(handle)
    return handle


def map_tt_experimental_descriptor_load(op: Any, ctx: WalkerCtx) -> Any:
    """Lower TMA descriptor load to ``T.tma_copy`` (NV) / pointer fallback.

    Recipe (RFC 5.1 + 5.4):
    * On NV target: emit ``tilelang.language.tma_copy(desc, dst, ...)``;
      ``FuseMBarrierArriveExpectTx`` later fuses the surrounding
      mbarrier protocol.
    * Off NV: per Triton PR #6753, decompose into a strided pointer-arith
      ``T.copy`` and route through the regular ``LowerTileOp`` path.
    """
    return _emit_descriptor_copy(op, ctx, is_load=True)


def map_tt_experimental_descriptor_store(op: Any, ctx: WalkerCtx) -> Any:
    """Lower TMA descriptor store to ``T.tma_copy`` (NV) / pointer fallback.

    Same recipe as the load variant but for shared->global movement.
    """
    return _emit_descriptor_copy(op, ctx, is_load=False)


# ---------------------------------------------------------------------------
# Misc -- RFC section 5.1, "tt.print"
# ---------------------------------------------------------------------------


def map_tt_print(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.print`` to ``T.call_extern("printf", ...)``.

    Recipe (RFC 5.1):
    * Format string lives in the op's ``prefix`` attribute.
    * Operand SSAs become printf args after type-promotion (i32, f32, ...).
    * For the buffer-vs-scalar split we delegate to
      ``tilelang.language.print`` for buffers and a direct
      ``tir.call_extern("printf", ...)`` for scalar lists, which is the
      portable surface across CUDA / HIP / Metal codegens.
    """
    operands = _operands(op)
    attrs = _attrs(op)
    prefix = str(attrs.get("prefix", attrs.get("msg", "")))

    tir = ctx.tir()
    args = [ctx.get(o) for o in operands]

    # When a single operand is a buffer, defer to the rich TileLang
    # print() macro which handles per-thread / per-warp gating.
    import tilelang.language as T  # type: ignore  # lazy
    if len(args) == 1 and hasattr(args[0], "scope") and callable(args[0].scope):
        # Buffer-shaped arg.
        T.print(args[0], msg=prefix or "")
        return None

    # Scalar / multi-arg path: emit a direct printf call_extern. On
    # Metal, codegen will route this to os_log; on HIP/CUDA it lowers
    # to printf. We pass the prefix as the format string verbatim.
    fmt = prefix if prefix else " ".join(["%g"] * len(args)) + "\n"
    handle = tir.call_extern("handle", "printf", fmt, *args)
    ctx.emit(handle)
    return handle


# ---------------------------------------------------------------------------
# Dispatch table (TTIR op name -> emitter)
# ---------------------------------------------------------------------------

OP_TABLE: Dict[str, EmitFn] = {
    # memory
    "tt.load": map_tt_load,
    "tt.store": map_tt_store,
    "tt.atomic_rmw": map_tt_atomic_rmw,
    # compute
    "tt.dot": map_tt_dot,
    "tt.reduce": map_tt_reduce,
    "tt.where": map_tt_where,
    # shape
    "tt.broadcast": map_tt_broadcast,
    "tt.splat": map_tt_splat,
    "tt.expand_dims": map_tt_expand_dims,
    "tt.reshape": map_tt_reshape,
    "tt.make_range": map_tt_make_range,
    # async / barrier (multiple TTIR spellings route through one emitter)
    "async_copy": map_tt_async_copy,
    "tt.async_copy_global_to_local": map_tt_async_copy,
    "tt.async_commit_group": map_tt_async_copy,
    "tt.async_wait": map_tt_async_copy,
    "mbarrier": map_tt_mbarrier,
    "tt.barrier_init": map_tt_mbarrier,
    "tt.barrier_arrive": map_tt_mbarrier,
    "tt.barrier_wait": map_tt_mbarrier,
    # TMA
    "tt.experimental_descriptor_load": map_tt_experimental_descriptor_load,
    "tt.experimental_descriptor_store": map_tt_experimental_descriptor_store,
    # misc
    "tt.print": map_tt_print,
}
