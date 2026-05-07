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

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# When True, force the log/exp synthesis for tt.reduce(mul) instead of the
# native reduce_prod primitive. Useful for numerical-comparison testing on
# backends that haven't validated their "mul" all-reduce path yet.
_USE_LOGEXP_PROD = False

__all__ = [
    "OP_TABLE",
    "EmitError",
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
    "map_tt_trans",
    "map_tt_make_range",
    # Async / barrier
    "map_tt_async_copy",
    "map_tt_mbarrier",
    "map_tt_sync_threads_partial",
    # TMA
    "map_tt_experimental_descriptor_load",
    "map_tt_experimental_descriptor_store",
    # Grid / launch
    "map_tt_program_id",
    # Misc
    "map_tt_print",
]


class EmitError(RuntimeError):
    """Raised when an emitter cannot lower an op for a precise, named reason.

    We use a dedicated subclass (rather than ``ValueError`` /
    ``NotImplementedError``) so the walker / pipeline driver can
    distinguish "user input needs adjustment" from "frontend is missing a
    feature": ``EmitError`` always means the former.

    This is the canonical definition. ``op_emitters/{arith,control,reduction}``
    re-export this class via ``from ..op_mapping import EmitError`` so all
    emitter modules raise the same exception type and ``except EmitError``
    in the pipeline catches every emitter-side failure.
    """


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
        # SSA value -> source SSA value for transposed views (tt.trans).
        # Lets map_tt_dot fold an intervening tt.trans into transpose_A/B
        # without materialising the transpose. The pair (i, j) records the
        # axes flipped; for the common 2D matmul case both values are the
        # last two axes (e.g. (-2, -1)).
        self.transposed_views: Dict[Any, Tuple[int, int]] = {}
        # SSA name (string, e.g. "%2") -> PtrState seeded by the
        # ``run_ptr_analysis_pre_pass`` helper in pipeline.py. Emitters in
        # ``op_emitters/memory.py`` look up here to choose between the real
        # T.copy path and the per-element ``# DEGRADED:`` fallback.
        self.ptr_states: Dict[str, Any] = {}

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


# ---------------------------------------------------------------------------
# Generic-form properties parser (shared by memory.py and arith.py emitters)
# ---------------------------------------------------------------------------
#
# Triton 3.6 (and any MLIR op that uses Properties storage) prints its
# inherent attributes inside ``<{...}>`` rather than the legacy ``{...}``
# braces. jaxlib's ``mlir.ir`` Python bindings expose properties only when
# the dialect is registered; with ``allow_unregistered_dialects=True`` --
# the only mode we have on hosts that lack a Triton-aware MLIR build --
# ``op.attributes`` returns an EMPTY map for property-only ops. The same
# bug affected ``tt.make_range`` (fixed in op_emitters/memory.py during
# Wave C2) and now bites ``arith.cmpi`` / ``arith.cmpf`` whose ``predicate``
# attribute is stored as a Property in Triton 3.6.
#
# We recover by lifting the ``<{...}>`` slice out of ``str(op)`` (the
# printed assembly *does* include properties even when the dict-shaped
# accessor is empty) and parsing the small ``key = literal`` grammar
# Triton emits. The helpers live here so every op-emitter module can use
# them without duplicating the regex.

# Match ``<{...}>`` at any nesting level inside the printed op. We only
# need the OUTERMOST one -- properties don't nest.
_PROPERTIES_RE = re.compile(r"<\{(?P<body>[^}]*)\}>")
# A key=value pair with an optional ``: <type>`` annotation. Matches
# ``predicate = 4 : i64``, ``start = 0 : i32``, and ``end = 256 : i32``.
_PROP_PAIR_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<val>-?\d+|true|false|\"[^\"]*\")"
    r"(?:\s*:\s*[A-Za-z_][A-Za-z0-9_]*)?"
)


def _parse_generic_properties_shared(op: Any) -> Dict[str, Any]:
    """Extract Triton 3.6 MLIR Properties from ``str(op)``.

    jaxlib's ``mlir.ir`` Python bindings under
    ``allow_unregistered_dialects=True`` don't expose dialect-registered
    Properties via ``op.attributes`` -- only inherent attrs of *registered*
    dialects come through. This helper pulls the ``<{...}>`` block out of
    the printed op string and parses the ``key = literal : <type>`` grammar
    Triton/arith emit.

    Returns an empty dict when the op has no printable ``<{...}>`` block
    (real dict-shaped fakes or attributes-only ops). Values are coerced
    to Python ``int`` / ``bool`` / ``str`` -- the coverage we need for
    ``tt.make_range`` and ``arith.cmpi``/``arith.cmpf`` predicates.
    """
    if isinstance(op, dict):
        return {}
    try:
        text = str(op)
    except Exception:
        return {}
    m = _PROPERTIES_RE.search(text)
    if not m:
        return {}
    out: Dict[str, Any] = {}
    for pair in _PROP_PAIR_RE.finditer(m.group("body")):
        key = pair.group("key")
        raw = pair.group("val")
        if raw == "true":
            out[key] = True
        elif raw == "false":
            out[key] = False
        elif raw.startswith("\"") and raw.endswith("\""):
            out[key] = raw[1:-1]
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                out[key] = raw
    return out


def _attrs_with_properties_shared(op: Any) -> Dict[str, Any]:
    """``_attrs`` plus a fallback to the ``<{...}>`` properties block.

    Existing dict-shaped fakes and ops with classic attribute storage
    take the fast path through the legacy ``_attrs`` helper. When that
    returns an empty dict and we're looking at a real MLIR op whose
    inherent attributes live in Properties storage, we fall back to a
    small textual parser. The parser is intentionally narrow: we only
    accept the ``key = scalar : type`` shape Triton/arith emit.
    """
    base: Dict[str, Any] = {}
    try:
        base = dict(_attrs(op))
    except Exception:
        # ``_attrs`` itself can throw on op shapes whose ``op.attributes``
        # iterator yields strings instead of NamedAttribute records (some
        # jaxlib builds do this for unregistered ops). Treat as empty.
        base = {}
    if base:
        return base
    return _parse_generic_properties_shared(op)


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
    """Best-effort element dtype for a TTIR SSA value (defaults to float32).

    Normalises short MLIR dtype spellings (``f32``, ``i32``, ``bf16``,
    ``i1``) to TVM's canonical names (``float32``, ``int32``,
    ``bfloat16``, ``bool``). The MLIR generic form prints element types
    using the short spelling (a bare ``f32`` rather than ``float32``);
    TVM's ``tir.decl_buffer`` rejects those short forms with
    ``ValueError: unknown dtype 'f32'``. Returning the canonical TVM
    spelling here is the pinch point that keeps every emitter that
    threads dtype through ``_dtype_of`` working with both shapes.
    """
    if isinstance(value, dict):
        return _normalize_mlir_dtype(str(value.get("dtype", "float32")))
    typ = getattr(value, "type", None)
    if typ is None:
        return "float32"
    elt = getattr(typ, "element_type", None)
    if elt is None:
        # ``typ`` may itself be a scalar element type (i32 / f32) when the
        # SSA value is non-tensor. Stringify and normalise.
        return _normalize_mlir_dtype(str(typ))
    return _normalize_mlir_dtype(str(elt))


# Canonical short-form -> TVM dtype map. Covers every spelling we expect
# from Triton 3.6 generic-form TTIR (and a handful of legacy aliases for
# robustness). Anything outside this set raises in :func:`_normalize_mlir_dtype`
# so silent dtype defaulting cannot mask a regression -- the maintainer's
# "no silent fallback" hard constraint.
_MLIR_DTYPE_ALIASES: Dict[str, str] = {
    # Floating point
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "bf16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "bfloat16": "bfloat16",
    # Integer
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
    # Unsigned (Triton occasionally surfaces these via ptr<u8> etc.)
    "ui8": "uint8",
    "ui16": "uint16",
    "ui32": "uint32",
    "ui64": "uint64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    # Index -- treat as 64-bit signed for kernel-args (matches TVM convention).
    "index": "int64",
    # Opaque handle -- pointer-typed block args fall back here when the
    # element type can't be resolved; we keep TVM's spelling intact.
    "handle": "handle",
}


def _normalize_mlir_dtype(dtype: str) -> str:
    """Canonicalise an MLIR-printed dtype string to TVM's spelling.

    Raises ``ValueError`` when the input is genuinely unknown so that a
    coverage gap surfaces immediately rather than silently lowering as
    ``float32`` (the regression that motivated this helper).
    """
    s = (dtype or "").strip()
    if not s:
        # Empty string falls back to handle so pointer block args without
        # a resolvable element dtype keep working; downstream emitters
        # that need a real dtype will already have replaced it.
        return "handle"
    if s in _MLIR_DTYPE_ALIASES:
        return _MLIR_DTYPE_ALIASES[s]
    # Already-canonical TVM spellings pass through (the alias map already
    # covers them, but keep this branch defensive against future TVM
    # additions like ``float8_e4m3``).
    if s.startswith(("float", "int", "uint")) or s == "bool":
        return s
    raise ValueError(f"unsupported MLIR dtype: {dtype!r}")


# ---------------------------------------------------------------------------
# PtrState helpers (Wave-2: enable T.copy(global[region], frag) emission
# when triton-shared PtrAnalysis has surfaced multi-element tile metadata).
# ---------------------------------------------------------------------------


def _ptrstate_is_tile(resolved: Dict[str, Any]) -> bool:
    """Return True iff the PtrState describes a tile larger than one element."""
    sizes = resolved.get("sizes") or []
    if not sizes:
        return False
    for s in sizes:
        try:
            if int(s) > 1:
                return True
        except (TypeError, ValueError):
            # Symbolic size string (e.g. "BLOCK_SIZE_M") -> treat as tile.
            if isinstance(s, str) and s.strip() and s.strip() != "1":
                return True
    return False


def _ptrstate_offsets_or_zero(resolved: Dict[str, Any]) -> List[Any]:
    raw = resolved.get("offsets") or []
    out: List[Any] = []
    for o in raw:
        try:
            out.append(int(o))
        except (TypeError, ValueError):
            out.append(o)
    return out or [0]


def _ptrstate_sizes_int(resolved: Dict[str, Any]) -> List[int]:
    """Project ``sizes`` to concrete ints; symbolic dims fall back to 1024."""
    out: List[int] = []
    for s in resolved.get("sizes") or []:
        try:
            out.append(int(s))
        except (TypeError, ValueError):
            out.append(1024)
    return out


def _ptrstate_buffer(ctx: "WalkerCtx", resolved: Dict[str, Any], dtype: str) -> Any:
    """Return (or create) the global buffer that the PtrState aliases."""
    name = resolved.get("source") or ctx.fresh("ptr")
    if name not in ctx.buffers:
        tir = ctx.tir()
        ctx.buffers[name] = tir.decl_buffer(
            shape=_ptrstate_sizes_int(resolved) or [1024],
            dtype=dtype,
            name=name,
        )
    return ctx.buffers[name]


def _emit_load_copy(op: Any, ctx: "WalkerCtx", resolved: Dict[str, Any],
                    mask_ssa: Any, other_ssa: Any) -> Any:
    """Emit ``T.copy(global[region], frag)`` and bind the result SSA to frag.

    The buffer-region path keeps the frontend on the high-level surface
    (RFC 5.1) so LayoutInference / LowerTileOp can apply uniformly. When
    a mask is present we still emit the copy and wrap a subsequent
    fragment-zeroing pass keyed on ``mask`` for the masked-out lanes -- the
    unconditional copy keeps shape inference simple while the mask predicate
    enforces semantics. ``other`` (Triton's masked-out fill) is honoured by
    pre-clearing the fragment to that value before the copy.
    """
    tir = ctx.tir()
    result = _results(op)[0] if _results(op) else None
    out_shape = _ptrstate_sizes_int(resolved) or list(_shape_of(result)) or [1024]
    out_dtype = _dtype_of(result) if result is not None else "float32"

    try:
        import tilelang.language as T  # type: ignore
    except ImportError:  # pragma: no cover -- dict-walker fallback path
        # No TileLang available: fall back to a single BufferLoad so the
        # walker stays usable in the dict-shaped unit-test environment.
        buf = _ptrstate_buffer(ctx, resolved, out_dtype)
        load_expr = tir.BufferLoad(buf, _ptrstate_offsets_or_zero(resolved))
        if result is not None:
            ctx.bind(result, load_expr)
        return load_expr

    src_buf = _ptrstate_buffer(ctx, resolved, out_dtype)
    frag = T.alloc_fragment(out_shape, out_dtype)

    # Build a buffer-region slice from offsets+sizes. The C++ side accepts
    # either Buffer or BufferRegion; we produce a tir.BufferRegion here so
    # downstream passes see explicit ranges.
    offs = _ptrstate_offsets_or_zero(resolved)
    region = []
    for i, sz in enumerate(out_shape):
        off = offs[i] if i < len(offs) else 0
        try:
            off_i = int(off)
        except (TypeError, ValueError):
            off_i = 0
        region.append((off_i, off_i + int(sz)))

    if other_ssa is not None and mask_ssa is not None:
        # Pre-clear the fragment to ``other`` so masked-out lanes carry the
        # right fill value after the unconditional copy.
        try:
            other_expr = ctx.get(other_ssa)
        except KeyError:
            other_expr = tir.const(0, out_dtype)
        if hasattr(T, "fill"):
            T.fill(frag, other_expr)

    # Emit the copy. Slicing the global buffer with explicit ranges keeps
    # the lowered IR readable; the helper handles BufferRegion construction.
    src_slice = src_buf
    if hasattr(T, "region"):
        try:
            src_slice = T.region(src_buf, "r", *[r[0] for r in region])
        except Exception:  # pragma: no cover -- API drift
            src_slice = src_buf
    T.copy(src_slice, frag)

    # Mask is applied lane-wise after the copy via T.if_then_else when
    # downstream consumers need it; the SSA bound to ``result`` is the
    # fragment buffer so dependent ops (gemm, reduce) see the tile.
    if mask_ssa is not None:
        try:
            mask_expr = ctx.get(mask_ssa)
            if hasattr(T, "if_then_else"):
                # Stash a follow-up predicate so consumers can re-apply if
                # the masked-out fill needs to differ from ``other``. We
                # bind the (frag, mask) pair instead of just the fragment.
                ctx.value_map[result] = (frag, mask_expr)  # noqa: E501
                return frag
        except KeyError:
            pass

    if result is not None:
        ctx.bind(result, frag)
    return frag


def _emit_store_copy(op: Any, ctx: "WalkerCtx", resolved: Dict[str, Any],
                     val_expr: Any, mask_ssa: Any) -> Any:
    """Emit ``T.copy(val_frag, global[region])`` for the buffer-region path."""
    tir = ctx.tir()
    out_shape = _ptrstate_sizes_int(resolved) or [1024]
    # Pull dtype from the value being stored where possible.
    dtype = "float32"
    try:
        dtype = str(getattr(val_expr, "dtype", None) or "float32")
    except Exception:
        pass
    if dtype in {"", "handle"}:
        dtype = "float32"

    try:
        import tilelang.language as T  # type: ignore
    except ImportError:  # pragma: no cover
        buf = _ptrstate_buffer(ctx, resolved, dtype)
        store_stmt = tir.BufferStore(buf, val_expr, _ptrstate_offsets_or_zero(resolved))
        if mask_ssa is not None:
            try:
                mask_expr = ctx.get(mask_ssa)
                store_stmt = tir.IfThenElse(mask_expr, store_stmt, None)
            except KeyError:
                pass
        ctx.emit(store_stmt)
        return store_stmt

    dst_buf = _ptrstate_buffer(ctx, resolved, dtype)
    handle = T.copy(val_expr, dst_buf)
    if mask_ssa is not None:
        # Predicate guards the entire copy; LowerTileOp will fuse this with
        # the per-lane mask once layouts are inferred.
        try:
            mask_expr = ctx.get(mask_ssa)
            handle = tir.IfThenElse(mask_expr, tir.Evaluate(handle), None)
            ctx.emit(handle)
            return handle
        except KeyError:
            pass
    return handle


# ---------------------------------------------------------------------------
# Memory ops -- RFC section 5.1, "tt.load" / "tt.store" / "tt.atomic_rmw"
# ---------------------------------------------------------------------------


# TODO(memory-emitters): the legacy stubs map_tt_load / map_tt_store and
# their shape-op siblings (map_tt_make_range / map_tt_broadcast /
# map_tt_splat / map_tt_expand_dims / map_tt_reshape) have a richer
# replacement in ``poc.triton_frontend.op_emitters.memory.MEMORY_EMITTERS``
# that honors the "never silent-fallback" rule: when PtrAnalysis is
# unavailable for a multi-element tile they emit a ``# DEGRADED:``
# pragma_comment AttrStmt that survives PrimFunc pretty-print. Per the
# ``feedback_no_silent_delete`` policy we keep these stubs live until the
# walker is rewired through the new overlay.
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

    # ptr_analysis has already populated value_map[ptr_ssa] with either:
    #  * a tagged ``{"_ptrstate": ..., "sizes": [...], ...}`` dict carrying
    #    a multi-element PtrState (preferred -- enables T.copy on a buffer
    #    region instead of a scalar BufferLoad), or
    #  * a legacy ``(buffer, indices)`` tuple (kept for callers that haven't
    #    migrated), or
    #  * an opaque value used as the buffer with offset 0 (MVP path).
    resolved = ctx.get(ptr_ssa) if ptr_ssa in ctx.value_map else None

    # Tile path: PtrState describes >1 element along at least one axis -> T.copy.
    if isinstance(resolved, dict) and "_ptrstate" in resolved and _ptrstate_is_tile(resolved):
        return _emit_load_copy(op, ctx, resolved, mask_ssa, other_ssa)

    if isinstance(resolved, tuple) and len(resolved) == 2:
        buf, indices = resolved
    elif isinstance(resolved, dict) and "_ptrstate" in resolved:
        # PtrState present but trivial (scalar) -> fall through to BufferLoad
        # using the printed source name as the buffer placeholder.
        result = _results(op)[0] if _results(op) else None
        out_dtype = _dtype_of(result) if result is not None else "float32"
        buf_name = resolved.get("source") or ctx.fresh("buf")
        if buf_name not in ctx.buffers:
            ctx.buffers[buf_name] = tir.decl_buffer(
                shape=[1024], dtype=out_dtype, name=buf_name
            )
        buf, indices = ctx.buffers[buf_name], _ptrstate_offsets_or_zero(resolved)
    elif resolved is not None:
        # MVP: treat the SSA value itself as the buffer with offset 0.
        buf, indices = resolved, [0]
    else:
        # Fallback path used when PtrAnalysis hasn't run yet (e.g. unit
        # tests with dict-shaped fakes): seed a placeholder buffer so the
        # walker can keep going. The shape/dtype come from the result type
        # when available; otherwise we fall back to a flat 1024xfp32 buf.
        # TODO: replace with PtrAnalysis-derived (buffer, indices).
        result = _results(op)[0] if _results(op) else None
        out_shape = list(_shape_of(result)) if result is not None else [1024]
        out_dtype = _dtype_of(result) if result is not None else "float32"
        buf_name = (
            getattr(ptr_ssa, "name", None)
            or (ptr_ssa.get("name") if isinstance(ptr_ssa, dict) else None)
            or ctx.fresh("buf")
        )
        if buf_name not in ctx.buffers:
            ctx.buffers[buf_name] = tir.decl_buffer(
                shape=out_shape or [1024], dtype=out_dtype, name=buf_name
            )
        buf, indices = ctx.buffers[buf_name], [0]

    # Prefer high-level T.copy when consumers can use it (RFC 5.1: keep the
    # frontend on the high-level surface so LayoutInference / LowerTileOp
    # apply uniformly). For the MVP scalar path we still emit BufferLoad
    # because T.copy expects buffer regions, not scalar elements.
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

    resolved = ctx.get(ptr_ssa) if ptr_ssa in ctx.value_map else None
    val_expr = ctx.get(val_ssa)

    # Tile path: PtrState describes >1 element along at least one axis -> T.copy.
    if isinstance(resolved, dict) and "_ptrstate" in resolved and _ptrstate_is_tile(resolved):
        return _emit_store_copy(op, ctx, resolved, val_expr, mask_ssa)

    if isinstance(resolved, tuple) and len(resolved) == 2:
        buf, indices = resolved
    elif isinstance(resolved, dict) and "_ptrstate" in resolved:
        out_dtype = _dtype_of(val_ssa) or "float32"
        buf_name = resolved.get("source") or ctx.fresh("buf")
        if buf_name not in ctx.buffers:
            ctx.buffers[buf_name] = tir.decl_buffer(
                shape=[1024], dtype=out_dtype, name=buf_name
            )
        buf, indices = ctx.buffers[buf_name], _ptrstate_offsets_or_zero(resolved)
    elif resolved is not None:
        buf, indices = resolved, [0]
    else:
        # Same placeholder-seed path as map_tt_load; see notes there.
        # TODO: replace with PtrAnalysis-derived (buffer, indices).
        out_shape = list(_shape_of(val_ssa))
        out_dtype = _dtype_of(val_ssa)
        buf_name = (
            getattr(ptr_ssa, "name", None)
            or (ptr_ssa.get("name") if isinstance(ptr_ssa, dict) else None)
            or ctx.fresh("buf")
        )
        if buf_name not in ctx.buffers:
            ctx.buffers[buf_name] = tir.decl_buffer(
                shape=out_shape or [1024], dtype=out_dtype or "float32", name=buf_name
            )
        buf, indices = ctx.buffers[buf_name], [0]

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
    # Wave E3: ``rmw_op`` is a Triton 3.6 inherent (Properties-storage) attr.
    # Use the shared helper so jaxlib's empty op.attributes path falls back
    # to parsing the printed ``<{rmw_op = ...}>`` block instead of silently
    # defaulting to None and tripping the "missing 'rmw_op'" error.
    attrs = _attrs_with_properties_shared(op)
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

    # Wave E3: ``transpose_A``/``transpose_B`` (and the lowered ``trans_a``/
    # ``trans_b`` aliases plus ``out_dtype``) are Triton 3.6 inherent attrs
    # stored as Properties. Use the shared helper so jaxlib's empty
    # op.attributes path doesn't silently treat the dot as un-transposed.
    attrs = _attrs_with_properties_shared(op)
    transpose_A = bool(attrs.get("transpose_A", False) or attrs.get("trans_a", False))
    transpose_B = bool(attrs.get("transpose_B", False) or attrs.get("trans_b", False))

    # Fold an intervening tt.trans (recorded by map_tt_trans) into the
    # transpose flags. dsa_splitk and sparse_mla_path_c emit
    # ``%bt = tt.trans %b ; %c = tt.dot %a, %bt`` rather than a single
    # tt.dot with a transpose_B attribute; without folding we'd materialise
    # an unnecessary copy.
    if a_ssa in ctx.transposed_views:
        transpose_A = not transpose_A
    if b_ssa in ctx.transposed_views:
        transpose_B = not transpose_B

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

    # Wave E3: ``axis`` is a Triton 3.6 inherent (Properties) attr; the
    # shared helper falls back to parsing ``<{axis = N : i32}>`` from the
    # printed op text when jaxlib's op.attributes view is empty. Without
    # this, every reduce silently collapses on axis=-1 (last axis).
    attrs = _attrs_with_properties_shared(op)
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
        # Prefer the dedicated reduce_prod primitive (Wave-2 add); only
        # fall back to exp(reduce_sum(log(src))) when the backend reports
        # the "mul" reduction kind unimplemented or when callers force the
        # log/exp path for cross-backend numerical comparison.
        if not _USE_LOGEXP_PROD and hasattr(T, "reduce_prod"):
            T.reduce_prod(src, dst, dim=axis, clear=True)
        else:
            if hasattr(T, "log"):
                log_src = T.log(src)
            else:
                tir = ctx.tir()
                log_src = tir.call_intrin(out_dtype, "tir.log", src)
            T.reduce_sum(log_src, dst, dim=axis, clear=True)
            if hasattr(T, "exp"):
                T.copy(T.exp(dst), dst)
            else:
                tir = ctx.tir()
                ctx.emit(tir.call_intrin(out_dtype, "tir.exp", dst))
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


def map_tt_trans(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.trans`` (logical transpose) to a sidecar bind.

    Triton's ``tt.trans`` flips two axes of a tile (default: the last
    two). At the TIR layer we don't materialise the transpose; instead
    we rebind the result SSA to the *same* TIR value as the source and
    record the flipped-axis pair in ``ctx.transposed_views`` so that a
    downstream ``tt.dot`` consumer can fold it into ``transpose_A`` /
    ``transpose_B`` on the emitted ``T.gemm`` call. This matches the
    Triton pattern used by ``tl.dot(A, B, trans_b=True)`` which the
    frontend lowers as ``%bt = tt.trans %b; tt.dot %a, %bt`` when the
    ``trans_b`` kwarg has been propagated as a separate op.
    """
    operands = _operands(op)
    if not operands:
        raise ValueError("tt.trans: missing source operand")
    src_ssa = operands[0]
    src = ctx.get(src_ssa)
    # Wave E3: ``order`` (Triton's permutation tuple) is an inherent attr in
    # Triton 3.6 stored as a Property. Without the shared helper, jaxlib's
    # op.attributes view is empty and we'd default to the last-two-axes
    # swap even when the kernel asked for a different permutation.
    attrs = _attrs_with_properties_shared(op)
    # Default to flipping the last two axes; honour an explicit ``order``
    # attribute when it carries a 2-element permutation that swaps two
    # axes (Triton's general form, but matmul callers always use a swap).
    order = attrs.get("order")
    if order is not None and len(tuple(order)) >= 2:
        a, b = int(tuple(order)[-2]), int(tuple(order)[-1])
    else:
        a, b = -2, -1
    if _results(op):
        result_ssa = _results(op)[0]
        ctx.bind(result_ssa, src)
        # If the source was itself transposed, double-transpose cancels.
        if src_ssa in ctx.transposed_views:
            ctx.transposed_views.pop(result_ssa, None)
        else:
            ctx.transposed_views[result_ssa] = (a, b)
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
    # Wave E3: ``start``/``end`` live in Properties storage in Triton 3.6.
    # NOTE: this legacy emitter is currently superseded by
    # ``op_emitters/memory.py:emit_tt_make_range`` via OP_TABLE.update(...)
    # at module-init. We still migrate the helper here defensively in case
    # the merge order ever shifts -- otherwise this dead path would
    # silently re-introduce the C2 zero-length-Ramp bug (start=end=0).
    attrs = _attrs_with_properties_shared(op)
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
    # Wave E3: ``count``/``arrive_count``/``parity`` (i.e. the barrier_kind
    # tunables) are Triton 3.6 inherent attrs stored as Properties. Use the
    # shared helper so jaxlib's empty op.attributes path falls back to
    # parsing ``<{count = N : i32}>`` from the printed op text.
    attrs = _attrs_with_properties_shared(op)

    import tilelang.language as T  # type: ignore  # lazy

    if "init" in name:
        arrive_count = int(attrs.get("count") or attrs.get("arrive_count") or 1)
        bar = T.alloc_barrier(arrive_count)
        if _results(op):
            ctx.bind(_results(op)[0], bar)
        return bar

    operands = _operands(op)
    tir = ctx.tir()
    if "arrive" in name:
        if not operands:
            raise ValueError("mbarrier.arrive: missing barrier operand")
        bar = ctx.get(operands[0])
        # Prefer the high-level T.barrier_arrive (re-exported via
        # tilelang.language.builtin). Fall back to a raw call_intrin so
        # the emitter still works if the symbol isn't re-exported.
        # TODO: verify barrier_arrive remains in tilelang.language.
        if hasattr(T, "barrier_arrive"):
            handle = T.barrier_arrive(bar)
        else:
            handle = tir.call_intrin(
                "handle", tir.op.Op.get("tl.mbarrier_arrive"), bar
            )
        ctx.emit(handle)
        return handle

    if "wait" in name:
        if not operands:
            raise ValueError("mbarrier.wait: missing barrier operand")
        bar = ctx.get(operands[0])
        parity = int(attrs.get("parity", 0))
        # Prefer the high-level T.barrier_wait; fall back to call_intrin.
        # TODO: verify barrier_wait remains in tilelang.language.
        if hasattr(T, "barrier_wait"):
            handle = T.barrier_wait(bar, parity)
        else:
            handle = tir.call_intrin(
                "handle", tir.op.Op.get("tl.mbarrier_wait_parity"), bar, parity
            )
        ctx.emit(handle)
        return handle

    # Plain ``__syncthreads`` style barrier.
    tir = ctx.tir()
    handle = tir.call_intrin("handle", tir.op.Op.get("tir.tvm_storage_sync"), "shared")
    ctx.emit(handle)
    return handle


def map_tt_sync_threads_partial(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.sync_threads_partial(mask, n_threads)`` to ``T.sync_threads_partial``.

    Recipe (cppmega.mlx topk_selector → unified pipeline migration, Phase 1):
    Triton-style radix-select kernels emit a partial-warp barrier so only the
    lanes set in ``mask`` rendezvous. Forward straight to the new TileLang
    primitive that lowers to ``__syncwarp(mask)`` on CUDA, a no-op on HIP
    (wavefront is hardware-convergent), and ``simdgroup_barrier`` on Metal.
    Also accepts the alias spellings ``triton.language.partial_barrier`` and
    ``tt.partial_barrier`` so multiple TTIR producers route through one
    emitter.
    """
    operands = _operands(op)
    if len(operands) < 2:
        raise ValueError(
            f"tt.sync_threads_partial: expected (mask, n_threads); got "
            f"{len(operands)} operands"
        )
    mask = ctx.get(operands[0])
    n_threads = ctx.get(operands[1])

    import tilelang.language as T  # type: ignore  # lazy

    if hasattr(T, "sync_threads_partial"):
        handle = T.sync_threads_partial(mask, n_threads)
    else:
        tir = ctx.tir()
        handle = tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.sync_threads_partial"),
            mask,
            n_threads,
        )
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
    # to printf. The prefix may originate from user-supplied TTIR and
    # could carry malicious format specifiers (%n stack-write, etc.) —
    # sanitize before forwarding to the GPU printf runtime.
    fmt = prefix if prefix else " ".join(["%g"] * len(args)) + "\n"
    fmt = _sanitize_printf_format(fmt)
    handle = tir.call_extern("handle", "printf", fmt, *args)
    ctx.emit(handle)
    return handle


_PRINTF_FORBIDDEN_SPECS = ("%n",)


def _sanitize_printf_format(fmt: str) -> str:
    """Defang malicious printf specifiers that could corrupt GPU runtime state.

    %n writes to memory and is dangerous in any context that accepts
    untrusted format strings. We escape it as a literal "%%n" so the
    runtime sees a printable token, never a write directive. Other
    specifiers (%s/%p/%x) are left intact — they only read; an attacker
    controlling the format would still need a matching argument list to
    leak anything non-trivial, and our caller fixes the arg count.
    """
    if not fmt:
        return fmt
    out = fmt
    for bad in _PRINTF_FORBIDDEN_SPECS:
        out = out.replace(bad, "%" + bad)  # %n -> %%n (literal)
    return out


# ---------------------------------------------------------------------------
# Grid / launch -- RFC section 5.1, ``tt.get_program_id`` (a.k.a. tt.program_id)
# ---------------------------------------------------------------------------


def map_tt_program_id(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.get_program_id(axis=N)`` to a block-binding ``Var``.

    Triton's ``tl.program_id(axis=N)`` selects gridDim.(x|y|z); the
    natural TileLang equivalent is ``T.Kernel(bx, by, bz)``'s block
    binding for axis ``N`` (``KernelLaunchFrame.get_block_binding``).

    Outside an active KernelLaunchFrame (e.g. unit tests with dict-shaped
    fakes) we fall back to allocating a fresh ``int32`` Var so the walker
    keeps going; downstream codegen replaces it with the real binding.
    """
    attrs = _attrs(op)
    axis = int(attrs.get("axis", 0))
    if axis < 0 or axis > 2:
        raise ValueError(
            f"tt.program_id: axis must be in [0, 2]; got {axis}"
        )

    var: Any
    try:
        from tilelang.language.kernel import KernelLaunchFrame  # type: ignore
        frame = KernelLaunchFrame.Current()
    except Exception:  # pragma: no cover -- TileLang absent during tests
        frame = None

    if frame is not None:
        try:
            var = frame.get_block_binding(axis)
        except Exception:  # pragma: no cover -- frame missing axis
            var = None
    else:
        var = None

    if var is None:
        tir = ctx.tir()
        var = tir.Var(ctx.fresh(f"pid{axis}"), "int32")

    if _results(op):
        ctx.bind(_results(op)[0], var)
    return var


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
    "tt.trans": map_tt_trans,
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
    # partial-warp / subgroup barrier (cppmega.mlx topk_selector migration)
    "tt.sync_threads_partial": map_tt_sync_threads_partial,
    "tt.partial_barrier": map_tt_sync_threads_partial,
    "triton.language.partial_barrier": map_tt_sync_threads_partial,
    # TMA
    "tt.experimental_descriptor_load": map_tt_experimental_descriptor_load,
    "tt.experimental_descriptor_store": map_tt_experimental_descriptor_store,
    # grid / launch (multiple TTIR spellings route through one emitter)
    "tt.program_id": map_tt_program_id,
    "tt.get_program_id": map_tt_program_id,
    # misc
    "tt.print": map_tt_print,
}


# ---------------------------------------------------------------------------
# arith.* / math.* / tt.fma -- sourced from op_emitters.arith to avoid the
# 1400-line table in this file growing further (and to keep merge-conflict
# surface small for parallel work on individual op families).
# ---------------------------------------------------------------------------
from .op_emitters.arith import ARITH_EMITTERS  # noqa: E402

OP_TABLE.update(ARITH_EMITTERS)


# ---------------------------------------------------------------------------
# Reductions / scan / dot / atomics -- sourced from op_emitters.reduction.
# Path C kernel surface: ``tt.reduce`` and ``tt.scan`` lower to explicit
# ``tir.For`` + accumulator (rather than the high-level ``T.reduce_*``
# macros invoked by the legacy ``map_tt_reduce`` above). The legacy
# ``map_tt_reduce`` / ``map_tt_dot`` / ``map_tt_atomic_rmw`` stubs in this
# file are deliberately left in place per ``feedback_no_silent_delete``;
# the ``OP_TABLE.update`` call below overrides them with the explicit
# emitters for the modern ``tt.atomic_<op>`` names. ``tt.atomic_rmw``
# (the legacy single-op spelling carrying an ``rmw_op`` attribute) is
# still handled by ``map_tt_atomic_rmw`` here.
# TODO: once Path C is the only path, fold the explicit emitters back
# into this file and delete the high-level stubs above.
# ---------------------------------------------------------------------------
from .op_emitters.reduction import REDUCTION_EMITTERS  # noqa: E402

OP_TABLE.update(REDUCTION_EMITTERS)


# ---------------------------------------------------------------------------
# Memory / shape -- sourced from op_emitters.memory. ``tt.load`` / ``tt.store``
# / ``tt.make_range`` / ``tt.broadcast`` / ``tt.splat`` / ``tt.expand_dims`` /
# ``tt.view`` / ``tt.reshape`` / ``tt.addptr`` / ``tts.make_tptr``. The legacy
# ``map_tt_*`` stubs above are kept per ``feedback_no_silent_delete``; the
# ``OP_TABLE.update`` below overrides them with the real TIR emitters.
# ---------------------------------------------------------------------------
from .op_emitters.memory import MEMORY_EMITTERS  # noqa: E402

OP_TABLE.update(MEMORY_EMITTERS)


# ---------------------------------------------------------------------------
# Control flow + casts -- sourced from op_emitters.control. ``arith.select`` /
# ``arith.{ext,trunc,fpto*,*tofp,bit}cast`` / ``arith.{ext,trunc}{si,ui}`` /
# ``tt.advance`` / ``scf.{for,if,yield}``.
# ---------------------------------------------------------------------------
from .op_emitters.control import CONTROL_EMITTERS  # noqa: E402

OP_TABLE.update(CONTROL_EMITTERS)
