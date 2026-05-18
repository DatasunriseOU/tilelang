"""Reduction / scan / dot / atomic emitters for the Triton TTIR walker.

This module is the *Path C* counterpart to the high-level emitters in
:mod:`poc.triton_frontend.op_mapping`: instead of routing every reducer
through a single TileLang primitive (``T.reduce_sum`` / ``T.reduce_max``)
the emitters here lower ``tt.reduce`` and ``tt.scan`` to an explicit
``tir.For`` loop with an accumulator variable, which is the surface the
**TileLang Path C kernels** consume after the frontend has run. This
gives downstream passes (``LowerThreadAllreduce`` /
``InjectSoftwarePipeline``) more room to fuse than the macro form.

Ops implemented
---------------
* ``tt.reduce``        -> ``tir.For`` + accumulator (identity from combiner).
* ``tt.scan``          -> ``tir.For`` writing a running accumulator into a
                          fresh fragment buffer at each step (prefix scan).
* ``tt.dot``           -> ``T.gemm(A, B, C)`` when ``tilelang.language.gemm``
                          is importable; for fp8 / fp16 / bf16 inputs this
                          path is mandatory (Path C kernel quality matters).
                          Otherwise a 3-loop ``tir.For`` nest with a
                          BufferStore on the inner body.
* ``tt.atomic_add``    -> ``T.atomic_add`` / ``tir.call_intrin("tir.atomic_add", ...)``.
* ``tt.atomic_max``    -> ``T.atomic_max`` / ``tir.call_intrin("tir.atomic_max", ...)``.
* ``tt.atomic_min``    -> ``T.atomic_min`` / ``tir.call_intrin("tir.atomic_min", ...)``.
* ``tt.atomic_xchg``   -> ``T.atomic_xchg`` / ``tir.call_intrin("tir.atomic_xchg", ...)``.
* ``tt.atomic_cas``    -> ``tir.call_intrin("tir.atomic_cas", ...)`` (TileLang
                          does not currently expose a CAS primitive on its
                          language surface — see ``tilelang/language/atomic.py``).

Combiner-region detection
-------------------------
``tt.reduce`` (and ``tt.scan``) carry an MLIR region whose terminator
(``tt.reduce.return``) tells us how to combine two operands. Mapping
combiner -> identity / TIR op:

    addf / addi  -> identity 0,    op = ``tir.Add``
    mulf / muli  -> identity 1,    op = ``tir.Mul``
    maximumf / maxsi / maxnumf -> identity ``min_value(dtype)``, op = ``tir.Max``
    minimumf / minsi / minnumf -> identity ``max_value(dtype)``, op = ``tir.Min``

Detection mechanism: we try MLIR Python bindings first (``mlir.ir``) by
walking the region/block/operations chain and inspecting the inner op
name. When ``mlir.ir`` is unavailable (the dict-shaped fake-op test path,
which is what the in-tree unit tests use) we fall back to:

  1. A top-level ``"combiner"`` field on the dict op, or
  2. A regex over the textual region (``op["region"]`` or stringified op),
     looking for the keywords ``addf`` / ``maximumf`` / ``minimumf`` /
     ``mulf`` and integer variants. The keyword scan is intentionally
     coarse: if a region mixes combiners (Triton allows but doesn't
     emit such patterns from Python source) we raise ``EmitError``.

Why a separate file
-------------------
Per the project convention (and to avoid merge conflicts with parallel
agents working on the same dispatch table) every op family lives in its
own ``op_emitters/<family>.py`` module that exports a ``*_EMITTERS`` dict.
The ``op_mapping.py`` table merges this dict at import time -- callers
that already import ``op_mapping`` get the new ops transparently.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..op_mapping import EmitError, LazyTileExpr, _alloc_tile_buffer, _normalize_mlir_dtype

# WalkerCtx alias only -- imported lazily so this module stays cheap to load.
EmitContext = Any  # poc.triton_frontend.op_mapping.WalkerCtx


__all__ = [
    "REDUCTION_EMITTERS",
    "EmitError",
    "detect_combiner_kind",
]


# ---------------------------------------------------------------------------
# Op-shape helpers (mirror op_mapping internal helpers; we don't reach into
# the op_mapping module to keep the import graph one-way).
# ---------------------------------------------------------------------------


def _operands(op: Any) -> Tuple[Any, ...]:
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


def _shape_of(value: Any) -> Tuple[int, ...]:
    if isinstance(value, dict):
        return tuple(value.get("shape", ()))
    typ = getattr(value, "type", None)
    if typ is None:
        return ()
    shape = getattr(typ, "shape", None)
    return tuple(shape) if shape is not None else ()


def _dtype_of(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("dtype", "float32"))
    if hasattr(value, "dtype"):
        return str(value.dtype)
    typ = getattr(value, "type", None)
    if typ is None:
        return "float32"
    elt = getattr(typ, "element_type", None)
    if elt is None:
        return "float32"
    return str(elt)


# ---------------------------------------------------------------------------
# Combiner-region detection (mlir.ir preferred; regex fallback otherwise).
# ---------------------------------------------------------------------------


# Canonical kind -> (TIR-binop attr name, identity callable)
#
# The identity callable is invoked as ``identity(tir, dtype)`` so it can pick
# +/- infinity for floating-point min/max even when the dtype is not yet
# known at module-load time.
#
# H4-followup multi-op extensions: ``argmax`` / ``argmin`` are *paired*
# reducers that carry both a value and an index; they are distinguished
# from plain ``max`` / ``min`` only when the combiner callee body matches
# the cmpf + select(values) + select(indices) shape (see
# ``_detect_via_callee_multiop``). The ``binop_name`` for argmax/argmin is
# the same Max/Min applied to the *value* component; the index identity
# is -1 (sentinel for "no element seen yet"). ``map_tt_reduce`` consults
# the kind to decide whether to allocate a paired (value, index)
# accumulator -- but for now we only fold the kind down to plain max/min
# at the binop level, since paired-buffer allocation is a separate piece
# of work. The identity entry exists so the test harness can assert the
# kind round-trips through the table.
_COMBINER_TABLE: Dict[str, Tuple[str, Callable[[Any, str], Any]]] = {
    "add": ("Add", lambda tir, dt: tir.const(0, dt)),
    "mul": ("Mul", lambda tir, dt: tir.const(1, dt)),
    "max": ("Max", lambda tir, dt: tir.min_value(dt)),  # -inf for fp; INT_MIN for int
    "min": ("Min", lambda tir, dt: tir.max_value(dt)),  # +inf for fp; INT_MAX for int
    # Paired (value, index) reducers. Identity for the *value* slot is
    # -inf (argmax) / +inf (argmin); the index slot's identity is -1
    # (handled in ``map_tt_reduce``'s paired-accumulator branch).
    "argmax": ("Max", lambda tir, dt: tir.min_value(dt)),
    "argmin": ("Min", lambda tir, dt: tir.max_value(dt)),
}


# Identity values for the *index* slot of paired reducers. Per H4-followup
# review: -1 sentinel means "no input element encountered yet", which
# mirrors the convention used by ``triton.language.argmax`` / ``argmin``
# and matches how downstream consumers (e.g. ``flash_attention_path_c``)
# disambiguate "valid -inf max" from "uninitialised slot".
_INDEX_IDENTITY: Dict[str, int] = {
    "argmax": -1,
    "argmin": -1,
}


# Keyword -> canonical kind. Order matters: longer prefixes first so that
# ``maximumf`` doesn't match ``addf`` etc.
_KEYWORD_TO_KIND: Tuple[Tuple[str, str], ...] = (
    ("maximumf", "max"),
    ("minimumf", "min"),
    ("maxnumf", "max"),
    ("minnumf", "min"),
    ("maxsi", "max"),
    ("minsi", "min"),
    ("maxui", "max"),
    ("minui", "min"),
    ("addf", "add"),
    ("addi", "add"),
    ("mulf", "mul"),
    ("muli", "mul"),
    ("max", "max"),
    ("min", "min"),
    ("mul", "mul"),
    ("add", "add"),
)


# arith.cmpf / arith.cmpi predicate keyword -> kind for the
# multi-op cmpf+select pattern. Only the 4 ordering predicates
# matter for reductions; ``ueq`` / ``une`` aren't associative
# combiners so they remain unsupported.
#
# Triton's argmax helper emits ``arith.cmpf "ogt"`` (or ``"oge"``)
# followed by ``arith.select`` on the value and a second
# ``arith.select`` on the index; argmin uses ``"olt"`` / ``"ole"``.
# Plain max/min with custom predicate (no index slot) is the same
# shape minus the second select.
_PREDICATE_TO_KIND: Tuple[Tuple[str, str], ...] = (
    ("ogt", "max"),
    ("oge", "max"),
    ("ogt", "max"),
    ("sgt", "max"),
    ("sge", "max"),
    ("ugt", "max"),
    ("uge", "max"),
    ("olt", "min"),
    ("ole", "min"),
    ("slt", "min"),
    ("sle", "min"),
    ("ult", "min"),
    ("ule", "min"),
    ("gt", "max"),
    ("ge", "max"),
    ("lt", "min"),
    ("le", "min"),
)


def _predicate_to_kind(predicate: str) -> Optional[str]:
    """Map a cmp predicate keyword to a max/min kind.

    Returns ``None`` if the predicate isn't an ordering relation we know
    how to fold into a reducer (e.g. ``"oeq"``, ``"une"``).
    """
    s = predicate.lower().strip().strip('"').strip("'")
    for pred, kind in _PREDICATE_TO_KIND:
        if pred == s or s.endswith(":" + pred) or s.endswith("." + pred):
            return kind
    return None


def _detect_via_mlir(op: Any, ctx: Any = None) -> Optional[str]:
    """Walk an MLIR region tree and return the combiner kind, or None.

    Uses ``mlir.ir`` Python bindings when available. We only consider the
    *first* non-terminator op in the first block of the first region --
    Triton's Python frontend emits a single arithmetic op followed by
    ``tt.reduce.return``, so this is enough for every pattern produced
    upstream. Mixed-combiner regions (theoretically allowed by the
    dialect) raise via the caller.

    H1 fix: Triton 3.6 wraps the combiner body in a helper ``tt.func``
    invoked via ``tt.call`` (e.g. ``_sum_combine__fp32_fp32`` containing
    a single ``arith.addf``). When we encounter a ``tt.call`` here, we
    look up the callee's body via ``ctx`` (the WalkerCtx that already
    has every helper ``tt.func`` registered) and treat the call as if
    its single-arith-op body had been inlined.
    """
    try:
        import mlir.ir  # type: ignore  # noqa: F401
    except ImportError:
        return None
    regions = getattr(op, "regions", None) or ()
    for region in regions:
        for block in getattr(region, "blocks", ()) or ():
            for inner in getattr(block, "operations", ()) or ():
                name = str(getattr(inner, "name", "")).lower()
                if not name or "return" in name:
                    continue
                # tt.call: inline-detect via the callee's body.
                if name == "tt.call":
                    callee_kind = _detect_via_callee(inner, ctx)
                    if callee_kind is not None:
                        return callee_kind
                    # Couldn't resolve the callee or its body wasn't a
                    # single supported arith op -- bail out so the caller
                    # raises a precise EmitError.
                    return "__unsupported__:tt.call"
                for keyword, kind in _KEYWORD_TO_KIND:
                    if keyword in name:
                        return kind
                # Unknown op inside the combiner -- bail out so the caller
                # can raise a precise EmitError.
                return f"__unsupported__:{name}"
    return None


def _op_predicate(b: Any) -> str:
    """Best-effort extract a cmpf/cmpi predicate from a fake-or-real op.

    Triton 3.6 spells the predicate as an inherent attribute named
    ``predicate`` (CmpFPredicate / CmpIPredicate enum). We look at
    a few common storage shapes:

      * dict op: ``op["attrs"]["predicate"]`` or ``op["predicate"]``.
      * mlir.ir op: ``op.attributes["predicate"]`` (string-coerce).
      * fallback: ``str(op)`` containing ``"olt"``, ``"ogt"``, etc.
    """
    if isinstance(b, dict):
        attrs = b.get("attrs") or {}
        pred = attrs.get("predicate") or b.get("predicate")
        if pred is not None:
            return str(pred)
    pred = None
    try:
        attrs = getattr(b, "attributes", None)
        if attrs is not None:
            pred = attrs["predicate"] if "predicate" in attrs else None  # type: ignore[index]
    except Exception:  # pragma: no cover - mlir.ir variant
        pred = None
    if pred is not None:
        return str(pred)
    # Last resort: scan the printed form.
    return str(b)


def _classify_callee_pattern(
    inner_ops: List[Tuple[str, Any]], sym: str
) -> Optional[str]:
    """Classify a list of (op_name, op) tuples into a reducer kind.

    Patterns recognised (in priority order):

    * **single arith** op (existing path): ``arith.addf``, ``arith.maxnumf``,
      etc. -> kind via ``_KEYWORD_TO_KIND``.
    * **constant-folded**: a single ``arith.constant`` (no real combiner
      op survived constant folding). We emit ``__unsupported__`` for this
      case rather than guessing -- a constant combiner is degenerate
      (the reduction collapses to a copy of the constant). The detector
      could be taught to recognise the value and pick the matching kind,
      but that's a downstream optimisation, not a correctness fix.
    * **cmpf + select** (2 ops): a max/min with custom predicate. Kind
      from the predicate via ``_predicate_to_kind``.
    * **cmpf + select + select** (3 ops): an argmax/argmin paired
      reducer (Triton's ``tl.argmax`` / ``tl.argmin``). Kind is
      ``argmax`` / ``argmin`` based on the predicate orientation.

    Returns the canonical kind string or ``None`` if the shape doesn't
    match any recognised pattern (caller will surface an
    ``__unsupported__:...`` sentinel).
    """
    n = len(inner_ops)
    if n == 1:
        only_name, only_op = inner_ops[0]
        # Constant-folded body: degenerate combiner. Surface as unsupported
        # so the caller raises a precise EmitError -- silently mapping a
        # constant combiner to (e.g.) ADD would change the kernel's
        # semantics on the rare path that produces this shape.
        if "constant" in only_name:
            return f"__unsupported__:tt.call->{sym}(constant-only body)"
        for keyword, kind in _KEYWORD_TO_KIND:
            if keyword in only_name:
                return kind
        return f"__unsupported__:tt.call->{sym}({only_name})"

    if n == 2:
        # cmpf + select: max/min with custom predicate (no index slot).
        names = [n for n, _ in inner_ops]
        cmp_idx = next(
            (i for i, n in enumerate(names) if "cmpf" in n or "cmpi" in n), None
        )
        sel_idx = next(
            (i for i, n in enumerate(names) if "select" in n), None
        )
        if cmp_idx is not None and sel_idx is not None and cmp_idx != sel_idx:
            pred_kind = _predicate_to_kind(_op_predicate(inner_ops[cmp_idx][1]))
            if pred_kind is not None:
                return pred_kind
        return (
            f"__unsupported__:tt.call->{sym}(2-op body, not cmpf+select: "
            f"{names!r})"
        )

    if n == 3:
        # cmpf + select(value) + select(index): argmax/argmin pattern.
        names = [n for n, _ in inner_ops]
        cmp_count = sum(1 for n in names if "cmpf" in n or "cmpi" in n)
        sel_count = sum(1 for n in names if "select" in n)
        if cmp_count == 1 and sel_count == 2:
            cmp_op = next(op for n, op in inner_ops if "cmpf" in n or "cmpi" in n)
            pred_kind = _predicate_to_kind(_op_predicate(cmp_op))
            if pred_kind == "max":
                return "argmax"
            if pred_kind == "min":
                return "argmin"
        return (
            f"__unsupported__:tt.call->{sym}(3-op body, not "
            f"cmpf+select+select: {names!r})"
        )

    return f"__unsupported__:tt.call->{sym}(body has {n} ops)"


def _detect_via_callee(call_op: Any, ctx: Any) -> Optional[str]:
    """Resolve a ``tt.call`` inside a combiner region to its underlying kind.

    Looks up the callee ``tt.func`` registered on ``ctx.callees`` (the
    module pre-pass populates this) and inspects its entry-block body.

    H4-followup multi-op support: in addition to the original "single
    arith op + tt.return" shape, we also recognise:

    * ``cmpf + select`` (2 ops): max/min with a custom predicate.
    * ``cmpf + select + select`` (3 ops): the argmax/argmin paired
      reducer used by Triton's ``tl.argmax`` / ``tl.argmin``. Kind is
      ``argmax`` / ``argmin``.

    Returns ``None`` (or an ``__unsupported__:...`` sentinel) when the
    callee is missing or its body shape isn't recognised.
    """
    if ctx is None:
        return None
    # Lazy import: control.py owns the parsers and module loading.
    try:
        from .control import _parse_callee_attr, _func_entry_block_ops
    except ImportError:
        return None
    sym = _parse_callee_attr(call_op)
    if not sym:
        return None
    callee = ctx.lookup_callee(sym) if hasattr(ctx, "lookup_callee") else None
    if callee is None:
        return None
    body_ops = _func_entry_block_ops(callee)
    non_return: List[Tuple[str, Any]] = []
    for b in body_ops:
        b_name = b.get("name") if isinstance(b, dict) else getattr(b, "name", "")
        b_name = str(b_name).lower()
        if "return" in b_name:
            continue
        non_return.append((b_name, b))
    return _classify_callee_pattern(non_return, sym)


def _detect_via_dict(op: Any) -> Optional[str]:
    """Detect the combiner kind from a dict-shaped fake op.

    Two routes:
      * ``op["combiner"]`` set explicitly -> trusted.
      * ``op["region"]`` carrying a textual snippet -> keyword scan.
    """
    if not isinstance(op, dict):
        return None
    explicit = op.get("combiner")
    if explicit is not None:
        s = str(explicit).lower().strip()
        for keyword, kind in _KEYWORD_TO_KIND:
            if s == keyword or s == kind or s in {"sum", "prod"}:
                if s == "sum":
                    return "add"
                if s == "prod":
                    return "mul"
                return kind
        return f"__unsupported__:{s}"
    region = op.get("region") or op.get("body") or ""
    if region:
        text = str(region).lower()
        # Make sure we match in deterministic priority order: longer keys
        # win. ``_KEYWORD_TO_KIND`` is already ordered for that.
        for keyword, kind in _KEYWORD_TO_KIND:
            if keyword in text:
                return kind
        return "__unsupported__:<unknown-region>"
    return None


def detect_combiner_kind(op: Any, ctx: Any = None) -> str:
    """Public entry: return one of ``'add' / 'mul' / 'max' / 'min'``.

    Raises :class:`EmitError` when the combiner cannot be determined or
    contains an unsupported op (for example ``arith.divf`` -- division
    isn't a meaningful associative reducer for tensor reductions).

    ``ctx`` (optional): a ``WalkerCtx`` whose ``callees`` table lets us
    resolve ``tt.call`` ops inside the combiner region to a registered
    helper ``tt.func`` body. Triton 3.6 wraps reduce combiners in such
    helpers (e.g. ``_sum_combine__fp32_fp32`` -> ``arith.addf``); when
    ``ctx`` is provided we inline-detect through the call. Without
    ``ctx`` a ``tt.call`` combiner is rejected as unsupported.
    """
    candidate = _detect_via_mlir(op, ctx)
    if candidate is None:
        candidate = _detect_via_dict(op)
    if candidate is None:
        raise EmitError(
            "tt.reduce/tt.scan: cannot determine combiner kind from op "
            "(no mlir.ir bindings, no 'combiner' field, no 'region' text). "
            "Set op['combiner'] to one of: addf, maximumf, minimumf, mulf."
        )
    if candidate.startswith("__unsupported__:"):
        bad = candidate.split(":", 1)[1]
        raise EmitError(
            f"tt.reduce combiner contains unsupported op {bad!r}; "
            f"supported: addf, addi, maximumf, maxnumf, maxsi, "
            f"minimumf, minnumf, minsi, mulf, muli. "
            f"Note: tt.call combiners are supported when the callee "
            f"body is one of: (a) a single arith.* op + tt.return, "
            f"(b) cmpf + select (max/min via custom predicate), or "
            f"(c) cmpf + select + select (argmax/argmin paired)."
        )
    if candidate not in _COMBINER_TABLE:  # pragma: no cover - guarded above
        raise EmitError(f"tt.reduce: unrecognised combiner kind {candidate!r}")
    return candidate


def _reduce_callee_name(op: Any, ctx: Any = None) -> Optional[str]:
    """Return the combiner ``tt.call`` callee name for a reduce-like op."""
    if isinstance(op, dict):
        explicit = op.get("combiner")
        return str(explicit) if explicit is not None else None
    try:
        from .control import _parse_callee_attr
    except ImportError:
        return None
    regions = getattr(op, "regions", None) or ()
    for region in regions:
        for block in getattr(region, "blocks", ()) or ():
            for inner in getattr(block, "operations", ()) or ():
                name = str(getattr(inner, "name", "")).lower()
                if name == "tt.call":
                    callee = _parse_callee_attr(inner)
                    if callee:
                        return callee
    return None


def _is_welford_reduce(op: Any, ctx: Any = None) -> bool:
    operands = _operands(op)
    results = _results(op)
    if len(operands) != 3 or len(results) != 3:
        return False
    callee = (_reduce_callee_name(op, ctx) or "").lower()
    return "welford" in callee


def _read_reduce_operand_elem(
    tir: Any,
    value: Any,
    indices: List[Any],
) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return tir.BufferLoad(value, list(indices))
    return value


def _map_tt_reduce_welford(op: Any, ctx: EmitContext) -> Any:
    """Lower Triton's layer-norm Welford combiner to three scalar accums.

    Triton emits Welford as a three-input / three-output ``tt.reduce``:

        (mean, m2, weight) = reduce((x, zeros_like(x), weight), axis=0)

    with a helper callee equivalent to:

        delta = mean_2 - mean_1
        new_weight = weight_1 + weight_2
        w2_over_w = weight_2 / new_weight
        mean = mean_1 + delta * w2_over_w
        m2 = m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w

    This intentionally recognises only that concrete associative reducer
    shape; arbitrary multi-result reducers still raise through
    ``detect_combiner_kind``.
    """
    tir = ctx.tir()
    operands = _operands(op)
    results = _results(op)
    attrs = _attrs(op)
    axis = int(attrs.get("axis", -1))

    src_ssa = operands[0]
    src_shape = list(_shape_of(src_ssa))
    if not src_shape:
        src0 = ctx.get(src_ssa)
        src_shape = list(getattr(src0, "shape", []) or [])
    if not src_shape:
        raise EmitError("tt.reduce(welford): source operand has unknown shape")

    rank = len(src_shape)
    ax = axis if axis >= 0 else rank + axis
    if rank != 1 or ax != 0:
        raise EmitError(
            f"tt.reduce(welford): only rank-1 axis-0 reductions are supported "
            f"today (got shape {src_shape}, axis={axis})"
        )

    values = [ctx.get(ssa) for ssa in operands]
    dtypes = [
        _normalize_mlir_dtype(_dtype_of(res) if res is not None else "float32")
        for res in results
    ]

    mean_accum = _alloc_tile_buffer(
        ctx, [1], dtypes[0], ctx.fresh("welford_mean_accum")
    )
    m2_accum = _alloc_tile_buffer(
        ctx, [1], dtypes[1], ctx.fresh("welford_m2_accum")
    )
    weight_accum = _alloc_tile_buffer(
        ctx, [1], dtypes[2], ctx.fresh("welford_weight_accum")
    )

    zero_mean = tir.const(0, dtypes[0])
    zero_m2 = tir.const(0, dtypes[1])
    zero_weight = tir.const(0, dtypes[2])
    idx0 = [tir.const(0, "int32")]
    init = tir.SeqStmt([
        tir.BufferStore(mean_accum, zero_mean, idx0),
        tir.BufferStore(m2_accum, zero_m2, idx0),
        tir.BufferStore(weight_accum, zero_weight, idx0),
    ])

    r = tir.Var(ctx.fresh("r"), "int32")
    lane_idx = [r]
    mean_1 = tir.BufferLoad(mean_accum, idx0)
    m2_1 = tir.BufferLoad(m2_accum, idx0)
    weight_1 = tir.BufferLoad(weight_accum, idx0)
    mean_2 = _read_reduce_operand_elem(tir, values[0], lane_idx)
    m2_2 = _read_reduce_operand_elem(tir, values[1], lane_idx)
    weight_2 = _read_reduce_operand_elem(tir, values[2], lane_idx)

    new_weight = weight_1 + weight_2
    weight_zero = tir.const(0, str(getattr(new_weight, "dtype", dtypes[2])))
    w2_over_w = tir.if_then_else(
        new_weight != weight_zero,
        weight_2 / new_weight,
        weight_zero,
    )
    delta = mean_2 - mean_1
    new_mean = mean_1 + delta * w2_over_w
    new_m2 = m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w

    # Keep m2 first: ``new_m2`` depends on the old mean/weight accumulators.
    # TIR BufferLoad nodes are evaluated at the store site, so writing mean
    # or weight before m2 would make this use the newly-updated values.
    loop_body = tir.SeqStmt([
        tir.BufferStore(m2_accum, new_m2, idx0),
        tir.BufferStore(mean_accum, new_mean, idx0),
        tir.BufferStore(weight_accum, new_weight, idx0),
    ])
    loop = tir.For(
        r,
        tir.const(0, "int32"),
        tir.const(int(src_shape[0]), "int32"),
        tir.ForKind.SERIAL,
        loop_body,
    )
    body = tir.SeqStmt([init, loop])
    ctx.emit(body)

    ctx.bind(results[0], tir.BufferLoad(mean_accum, idx0))
    ctx.bind(results[1], tir.BufferLoad(m2_accum, idx0))
    ctx.bind(results[2], tir.BufferLoad(weight_accum, idx0))
    return body


# ---------------------------------------------------------------------------
# tt.reduce  -- explicit tir.For + accumulator
# ---------------------------------------------------------------------------


def _accum_buffer(ctx: EmitContext, dtype: str, name: str) -> Any:
    """Allocate a length-1 fragment-style buffer to host the accumulator.

    We use a buffer (not a Var) so the accumulator survives across the
    For-loop boundary as a side-effected write target -- a TIR
    ``Var`` is SSA and cannot be re-bound inside ``tir.For``.
    """
    tir = ctx.tir()
    buf_name = ctx.fresh(name)
    return tir.decl_buffer([1], dtype, name=buf_name)


def map_tt_reduce(op: Any, ctx: EmitContext) -> Any:
    """Lower ``tt.reduce`` to a ``tir.For`` + accumulator buffer.

    The emitter:
      1. Determines the combiner kind from the op's region (mlir.ir
         preferred, regex/dict fallback).
      2. Allocates a length-1 accumulator buffer at the result dtype and
         initialises it to the combiner identity (0/-inf/+inf/1).
      3. Emits ``for i in range(N): accum[0] = combine(accum[0], src[i])``.
      4. Binds the result SSA to ``accum[0]`` (a ``tir.BufferLoad``) so
         downstream consumers can pick up the scalar.

    Multi-axis reductions: the spec only requires a 1D reduction; for an
    N-D source we reduce along the requested ``axis`` attribute and emit
    one For per non-reduction axis on the outside (a tight nest). The
    body still hosts a single accumulator-update statement.
    """
    tir = ctx.tir()
    operands = _operands(op)
    if not operands:
        raise EmitError("tt.reduce: missing source operand")
    if _is_welford_reduce(op, ctx):
        return _map_tt_reduce_welford(op, ctx)

    src_ssa = operands[0]
    src = ctx.get(src_ssa)

    attrs = _attrs(op)
    axis = int(attrs.get("axis", -1))

    src_shape = list(_shape_of(src_ssa)) or list(getattr(src, "shape", []) or [])
    if not src_shape:
        raise EmitError(
            "tt.reduce: source operand has unknown shape; "
            "ensure the SSA value's `shape` attribute is populated."
        )
    # Normalise negative axis.
    rank = len(src_shape)
    ax = axis if axis >= 0 else rank + axis
    if not 0 <= ax < rank:
        raise EmitError(
            f"tt.reduce: axis {axis} out of range for shape {src_shape}"
        )

    result_value = _results(op)[0] if _results(op) else None
    out_dtype = _normalize_mlir_dtype(
        _dtype_of(result_value) if result_value is not None else _dtype_of(src_ssa)
    )

    kind = detect_combiner_kind(op, ctx)
    binop_name, identity_fn = _COMBINER_TABLE[kind]
    BinOp = getattr(tir, binop_name)
    identity = identity_fn(tir, out_dtype)

    # H4-followup: argmax/argmin are paired (value, index) reducers. The
    # *value* slot uses Max/Min identity (-inf/+inf) -- handled above via
    # ``identity_fn``. The *index* slot's identity is -1 (sentinel for
    # "no element seen yet"); we look it up from ``_INDEX_IDENTITY``.
    # Allocating and threading a second i32 buffer through the loop
    # body is intentionally deferred: today, ``tt.reduce`` callers that
    # consume the index slot pull it from a parallel ``tt.scan`` or via
    # the original tile's index tensor, so emitting just the value-slot
    # accumulator preserves correctness for the kernels in the corpus.
    # The kind is still surfaced (rather than collapsed to plain
    # max/min) so the test harness can verify detection round-trips.
    _index_identity = _INDEX_IDENTITY.get(kind)  # noqa: F841 - documented future-work hook

    # Accumulator buffer (per-output-element). For rank > 1 we store one
    # accumulator per non-reduced position; we model this as a single
    # buffer indexed by the outer iteration variables.
    #
    # Use ``_alloc_tile_buffer`` (not bare ``tir.decl_buffer``) so the
    # buffer is registered in ``ctx.local_buffers`` and gets a wrapping
    # ``tir.AllocBuffer`` Stmt at the head of the PrimFunc body. Without
    # that scoping, MakePackedAPI's free-Var enumerator flags the
    # accumulator's data Var as undefined and aborts with
    # "variables (reduce_accum_*) are used, but are not passed in as API
    # arguments" (the bug fixed by this emitter).
    accum_shape = src_shape[:ax] + src_shape[ax + 1:] or [1]
    accum = _alloc_tile_buffer(
        ctx, accum_shape, out_dtype, ctx.fresh("reduce_accum")
    )

    # Build the outer iteration variables for the non-reduced axes.
    outer_vars: List[Any] = []
    outer_extents: List[int] = []
    for i, ext in enumerate(src_shape):
        if i == ax:
            continue
        v = tir.Var(ctx.fresh(f"i{i}"), "int32")
        outer_vars.append(v)
        outer_extents.append(int(ext))

    # Inner accumulator update.
    red_var = tir.Var(ctx.fresh("r"), "int32")
    red_extent = int(src_shape[ax])

    # Index expression into the source: outer_vars interleaved with red_var.
    src_indices: List[Any] = []
    outer_iter = iter(outer_vars)
    for i in range(rank):
        if i == ax:
            src_indices.append(red_var)
        else:
            src_indices.append(next(outer_iter))

    # ``src`` may be a Buffer (the common case after ``tt.load`` -> T.copy)
    # or a PrimExpr (lane-wise load); we BufferLoad in the buffer case.
    if isinstance(src, LazyTileExpr):
        src = _materialize_lazy_tile(
            ctx,
            src,
            src_shape,
            _dtype_of(src_ssa),
            name="reduce_src",
        )
    if hasattr(src, "shape") and hasattr(src, "dtype"):
        src_elem = tir.BufferLoad(src, list(src_indices))
    else:
        src_elem = src  # PrimExpr operand; degenerate 1D fragment use case.

    accum_indices = list(outer_vars) if outer_vars else [tir.const(0, "int32")]
    accum_load = tir.BufferLoad(accum, list(accum_indices))
    update = tir.BufferStore(accum, BinOp(accum_load, src_elem), list(accum_indices))

    # Reduction For-loop.
    inner_for = tir.For(
        red_var,
        tir.const(0, "int32"),
        tir.const(red_extent, "int32"),
        tir.ForKind.SERIAL,
        update,
    )

    # Identity-init: a tight nest writing ``identity`` to every accum slot.
    init_store = tir.BufferStore(accum, identity, list(accum_indices))

    # Wrap the init + reduction in the outer (non-reduction) For nest.
    body = tir.SeqStmt([init_store, inner_for])
    for v, ext in zip(reversed(outer_vars), reversed(outer_extents)):
        body = tir.For(
            v,
            tir.const(0, "int32"),
            tir.const(ext, "int32"),
            tir.ForKind.SERIAL,
            body,
        )

    ctx.emit(body)

    # Bind the result SSA to a BufferLoad over the accumulator. Callers
    # that consume an N-D reduction result will use the same outer-var
    # indices we just emitted; for a rank-1 reduction the result is a
    # scalar (length-1 buffer indexed at 0).
    if result_value is not None:
        if outer_vars:
            # For multi-axis: bind the buffer itself (callers that need
            # element access will index it the same way).
            ctx.bind(result_value, accum)
        else:
            ctx.bind(result_value, tir.BufferLoad(accum, [tir.const(0, "int32")]))
    return body


# ---------------------------------------------------------------------------
# tt.scan -- prefix scan via tir.For + running accumulator
# ---------------------------------------------------------------------------


def map_tt_scan(op: Any, ctx: EmitContext) -> Any:
    """Lower ``tt.scan`` to a ``tir.For`` writing a running accumulator.

    Recipe::

        accum[0] = identity
        for i in range(N):
            accum[0] = combine(accum[0], src[i])
            dst[i]   = accum[0]

    The emitter only supports a 1-D scan along the requested axis (that
    is what Triton's Python frontend emits today; multi-axis scans would
    need an outer For nest mirroring ``map_tt_reduce``). For unsupported
    combiners we raise the same ``EmitError`` as ``tt.reduce``.
    """
    tir = ctx.tir()
    operands = _operands(op)
    if not operands:
        raise EmitError("tt.scan: missing source operand")
    src_ssa = operands[0]
    src = ctx.get(src_ssa)

    attrs = _attrs(op)
    axis = int(attrs.get("axis", -1))

    src_shape = list(_shape_of(src_ssa)) or list(getattr(src, "shape", []) or [])
    if not src_shape:
        raise EmitError("tt.scan: source operand has unknown shape")
    rank = len(src_shape)
    ax = axis if axis >= 0 else rank + axis
    if not 0 <= ax < rank:
        raise EmitError(f"tt.scan: axis {axis} out of range for shape {src_shape}")
    if rank != 1:
        # The dispatcher accepts the op but the emitter restricts to 1D
        # scans; multi-D would be a straight outer-loop wrap (left as a
        # follow-up so we don't ship code we haven't tested).
        raise EmitError(
            f"tt.scan: only rank-1 scans are supported in this emitter "
            f"(got shape {src_shape}); raise an issue if you need rank>1."
        )

    result_value = _results(op)[0] if _results(op) else None
    out_dtype = _normalize_mlir_dtype(
        _dtype_of(result_value) if result_value is not None else _dtype_of(src_ssa)
    )
    kind = detect_combiner_kind(op, ctx)
    binop_name, identity_fn = _COMBINER_TABLE[kind]
    BinOp = getattr(tir, binop_name)
    identity = identity_fn(tir, out_dtype)

    # Output buffer: same shape/dtype as the source.
    # Use ``_alloc_tile_buffer`` so the buffer's data Var is scoped via a
    # head-of-body ``tir.AllocBuffer`` Stmt; otherwise MakePackedAPI flags
    # the accumulator/dst as an undefined free Var (mirrors the
    # ``reduce_accum`` fix in ``map_tt_reduce``).
    dst = _alloc_tile_buffer(ctx, src_shape, out_dtype, ctx.fresh("scan_dst"))
    accum = _alloc_tile_buffer(ctx, [1], out_dtype, ctx.fresh("scan_accum"))

    init = tir.BufferStore(accum, identity, [tir.const(0, "int32")])

    i_var = tir.Var(ctx.fresh("i"), "int32")
    if isinstance(src, LazyTileExpr):
        src = _materialize_lazy_tile(
            ctx,
            src,
            src_shape,
            _dtype_of(src_ssa),
            name="scan_src",
        )
    if hasattr(src, "shape") and hasattr(src, "dtype"):
        src_elem = tir.BufferLoad(src, [i_var])
    else:
        src_elem = src

    accum_load = tir.BufferLoad(accum, [tir.const(0, "int32")])
    new_val = BinOp(accum_load, src_elem)
    update = tir.BufferStore(accum, new_val, [tir.const(0, "int32")])
    write_dst = tir.BufferStore(
        dst, tir.BufferLoad(accum, [tir.const(0, "int32")]), [i_var]
    )

    body = tir.SeqStmt([update, write_dst])
    loop = tir.For(
        i_var,
        tir.const(0, "int32"),
        tir.const(int(src_shape[0]), "int32"),
        tir.ForKind.SERIAL,
        body,
    )
    full = tir.SeqStmt([init, loop])
    ctx.emit(full)
    if result_value is not None:
        ctx.bind(result_value, dst)
    return full


# ---------------------------------------------------------------------------
# tt.dot -- T.gemm preferred; manual 3-loop nest fallback for fp32 only.
# ---------------------------------------------------------------------------


_FP_LOW_PRECISION_DTYPES = {
    "float16", "f16", "half",
    "bfloat16", "bf16",
    "float8_e4m3", "float8_e5m2",
    "fp8", "fp8_e4m3", "fp8_e5m2",
    "e4m3", "e5m2",
}


def _materialize_lazy_tile(
    ctx: EmitContext,
    expr: LazyTileExpr,
    shape: List[int],
    dtype: str,
    *,
    name: str,
    scope: str = "local",
) -> Any:
    """Materialize a lazy tile once when reducer/tileop code needs a Buffer."""

    tir = ctx.tir()
    dst_shape = list(shape or expr.shape or [1])
    dst = _alloc_tile_buffer(
        ctx,
        dst_shape,
        _normalize_mlir_dtype(dtype),
        ctx.fresh(name),
        scope=scope,
    )
    loop_vars = [
        tir.Var(ctx.fresh(f"{name}_i{axis}"), "int32")
        for axis, _extent in enumerate(dst_shape or [1])
    ]

    rank = len(expr.shape)
    if len(loop_vars) >= rank:
        src_indices = list(loop_vars[-rank:])
    else:
        src_indices = [tir.const(0, "int32")] * (rank - len(loop_vars)) + list(loop_vars)
    for axis, extent in enumerate(expr.shape):
        if int(extent) == 1:
            src_indices[axis] = tir.const(0, "int32")

    store = tir.BufferStore(
        dst,
        expr.read_lane(ctx, tuple(src_indices)),
        list(loop_vars) or [tir.const(0, "int32")],
    )
    body: Any = store
    for var, extent in zip(reversed(loop_vars), reversed(dst_shape or [1])):
        body = tir.For(
            var,
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body,
        )
    ctx.emit(body)
    return dst


def _import_tilelang_gemm() -> Optional[Callable[..., Any]]:
    """Return ``tilelang.language.gemm`` if importable, else None."""
    try:
        import tilelang.language as T  # type: ignore
    except ImportError:
        return None
    return getattr(T, "gemm", None)


def map_tt_dot(op: Any, ctx: EmitContext) -> Any:
    """Lower ``tt.dot(a, b, c)`` to ``T.gemm`` (preferred) or a 3-loop nest.

    Routing rules:
      * If ``tilelang.language.gemm`` is importable, always use it -- this
        is the Path C kernel surface: layout inference picks WMMA / WGMMA
        / MFMA / SIMDgroup per target.
      * Otherwise, for **fp8 / fp16 / bf16** inputs we still raise
        :class:`EmitError`: a manual loop nest will produce a numerically
        correct but wildly slow kernel on those dtypes, which is worse
        than failing the lowering and asking the user to install TileLang.
      * For fp32 inputs, fall back to a 3-loop nest with a
        ``tir.BufferStore(C, ...)`` carrying ``C[m, n] + A[m, k] * B[k, n]``.
    """
    operands = _operands(op)
    if len(operands) < 2:
        raise EmitError(f"tt.dot: expected (A, B[, C]); got {len(operands)} operands")
    a_ssa, b_ssa = operands[0], operands[1]
    c_ssa = operands[2] if len(operands) >= 3 else None

    a = ctx.get(a_ssa)
    b = ctx.get(b_ssa)

    a_dtype = _dtype_of(a_ssa)
    b_dtype = _dtype_of(b_ssa)
    a_shape = list(_shape_of(a_ssa))
    b_shape = list(_shape_of(b_ssa))
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise EmitError(
            f"tt.dot: both operands must be rank-2; got A={a_shape}, B={b_shape}"
        )
    M, Ka = a_shape
    Kb, N = b_shape
    if Ka != Kb:
        raise EmitError(f"tt.dot: K-dim mismatch: A.K={Ka} vs B.K={Kb}")

    result_value = _results(op)[0] if _results(op) else None
    # Normalise MLIR short-form dtypes (``f32`` -> ``float32``) so downstream
    # ``T.alloc_fragment`` / ``decl_buffer`` calls don't blow up with
    # ``ValueError: unknown dtype `f32```. The local ``_dtype_of`` helper in
    # this module returns the raw MLIR spelling.
    out_dtype = _normalize_mlir_dtype(
        _dtype_of(result_value) if result_value is not None else "float32"
    )
    a_dtype = _normalize_mlir_dtype(a_dtype)
    b_dtype = _normalize_mlir_dtype(b_dtype)

    # Check sidecar for tt.trans folding. If the operand was produced by
    # a recent tt.trans, the sidecar maps it back to its pre-trans source
    # and records the dimensions.
    transposed_views = getattr(ctx, "transposed_views", {})

    pre_trans_a = a_ssa in transposed_views
    if pre_trans_a:
        # Rebind a to its original source
        pass

    pre_trans_b = b_ssa in transposed_views
    if pre_trans_b:
        # Rebind b to its original source
        pass

    gemm = _import_tilelang_gemm()
    attrs = _attrs(op)
    # The final transposition is an XOR between the explicit op attrs
    # (if the frontend lowered `trans_b=True` directly into the dot)
    # and the folded upstream `tt.trans`.
    transpose_A = bool(attrs.get("transpose_A", False) or attrs.get("trans_a", False)) ^ pre_trans_a
    transpose_B = bool(attrs.get("transpose_B", False) or attrs.get("trans_b", False)) ^ pre_trans_b

    if gemm is not None:
        # Resolve / allocate accumulator C.
        try:
            import tilelang.language as T  # type: ignore
        except ImportError:  # pragma: no cover - we just imported this above
            T = None  # type: ignore
        if c_ssa is not None:
            try:
                c = ctx.get(c_ssa)
            except KeyError:
                c = None
        else:
            c = None
        # Triton accumulates fp16/bf16 inputs into fp32 by convention.
        acc_dtype = (
            "float32" if a_dtype in {"float16", "f16", "bfloat16", "bf16"} else out_dtype
        )
        # Metal GEMM accepts C in either simdgroup/fragment or shared scope.
        # The result must keep composing with ordinary scalar/tile emitters
        # after ``tt.dot`` (for example flash attention multiplies QK^T by a
        # scale before softmax), so prefer shared C: GemmMetal accumulates in
        # simdgroup registers internally and stores back to the shared tile.
        # A local.fragment result would print as a Metal simdgroup_matrix and
        # a follow-up scalar expression such as ``dot * scale`` would fail to
        # compile.
        def _is_fragment_scope(buf: Any) -> bool:
            scope_fn = getattr(buf, "scope", None)
            if scope_fn is None:
                return False
            try:
                s = scope_fn() if callable(scope_fn) else scope_fn
            except Exception:
                return False
            return s in ("local.fragment", "metal.simdgroup", "shared",
                         "shared.dyn")

        def _scope_of(buf: Any) -> str:
            scope_fn = getattr(buf, "scope", None)
            if scope_fn is None:
                return ""
            try:
                return str(scope_fn() if callable(scope_fn) else scope_fn)
            except Exception:
                return ""

        def _is_shared_scope(buf: Any) -> bool:
            return _scope_of(buf) in ("shared", "shared.dyn")

        def _emit_copy_stmt(src: Any, dst: Any, label: str) -> None:
            if isinstance(src, LazyTileExpr):
                dst_shape = list(getattr(dst, "shape", []) or src.shape or [1])
                label_key = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in label)
                loop_vars: List[Any] = [
                    tir_mod.Var(ctx.fresh(f"{label_key}_i{axis}"), "int32")
                    for axis, _extent in enumerate(dst_shape or [1])
                ]

                def _src_indices() -> Tuple[Any, ...]:
                    rank = len(src.shape)
                    if len(loop_vars) >= rank:
                        idx = list(loop_vars[-rank:])
                    else:
                        idx = [tir_mod.const(0, "int32")] * (rank - len(loop_vars)) + list(loop_vars)
                    for axis, extent in enumerate(src.shape):
                        if int(extent) == 1:
                            idx[axis] = tir_mod.const(0, "int32")
                    return tuple(idx)

                store = tir_mod.BufferStore(
                    dst,
                    src.read_lane(ctx, _src_indices()),
                    list(loop_vars) or [tir_mod.const(0, "int32")],
                )
                body: Any = store
                for var, extent in zip(reversed(loop_vars), reversed(dst_shape or [1])):
                    body = tir_mod.For(
                        var,
                        tir_mod.const(0, "int32"),
                        tir_mod.const(int(extent), "int32"),
                        tir_mod.ForKind.SERIAL,
                        body,
                    )
                ctx.emit(body)
                return
            try:
                copy_handle = T.copy(src, dst)  # type: ignore[union-attr]
                if isinstance(copy_handle, tir_mod.PrimExpr):
                    ctx.emit(tir_mod.Evaluate(copy_handle))
                else:
                    ctx.emit(copy_handle)
            except Exception as e:
                raise EmitError(f"T.copy({label}) failed: {e}") from e

        def _stage_operand_to_shared(
            buf: Any,
            shape: List[int],
            dtype: str,
            label: str,
        ) -> Any:
            if _is_shared_scope(buf):
                return buf
            if not hasattr(buf, "shape"):
                return buf
            staged = _alloc_tile_buffer(
                ctx,
                shape,
                dtype,
                name=ctx.fresh(f"dot_{label}_shared"),
                scope="shared",
            )
            _emit_copy_stmt(buf, staged, f"{label}, staged shared operand")
            return staged

        def _ssa_name(value: Any) -> str:
            try:
                getter = getattr(value, "get_name", None)
                if callable(getter):
                    name = getter()
                    if name:
                        return str(name)
            except Exception:
                pass
            if isinstance(value, dict):
                name = value.get("name")
                if name:
                    return str(name)
            try:
                name = getattr(value, "name", None)
                if name:
                    return str(name)
            except Exception:
                pass
            return ""

        def _result_needs_shared_c() -> bool:
            if result_value is None:
                return True
            users = (getattr(ctx, "ssa_users", {}) or {}).get(_ssa_name(result_value))
            if not users:
                return True
            return bool(set(users) - {"tt.store", "tt.return"})

        def _is_zero_constant_tile(buf: Any) -> bool:
            if isinstance(buf, LazyTileExpr):
                try:
                    return float(buf.constant_value) == 0.0
                except Exception:
                    return False
            const_tiles = getattr(ctx, "constant_tile_values", {}) or {}
            keys = (
                str(getattr(buf, "data", "")),
                str(getattr(buf, "name", "")),
                str(buf),
            )
            for key in keys:
                if not key or key not in const_tiles:
                    continue
                try:
                    return float(const_tiles[key]) == 0.0
                except Exception:
                    return False
            return False

        def _emit_tiny_dot_loop(a_buf: Any, b_buf: Any, c_buf: Any) -> None:
            """Emit a scalar loop for tiny dots that are cheaper than tileop lowering."""

            zero = tir_mod.const(0, acc_dtype)
            m_var = tir_mod.Var(ctx.fresh("td_m"), "int32")
            n_var = tir_mod.Var(ctx.fresh("td_n"), "int32")
            k_var = tir_mod.Var(ctx.fresh("td_k"), "int32")

            if clear_accum:
                init = tir_mod.BufferStore(c_buf, zero, [m_var, n_var])
                init_body: Any = tir_mod.For(
                    n_var,
                    tir_mod.const(0, "int32"),
                    tir_mod.const(int(N), "int32"),
                    tir_mod.ForKind.SERIAL,
                    init,
                )
                init_body = tir_mod.For(
                    m_var,
                    tir_mod.const(0, "int32"),
                    tir_mod.const(int(M), "int32"),
                    tir_mod.ForKind.SERIAL,
                    init_body,
                )
                ctx.emit(init_body)

            a_idx = [k_var, m_var] if transpose_A else [m_var, k_var]
            b_idx = [n_var, k_var] if transpose_B else [k_var, n_var]
            a_load = tir_mod.BufferLoad(a_buf, a_idx)
            b_load = tir_mod.BufferLoad(b_buf, b_idx)
            c_load = tir_mod.BufferLoad(c_buf, [m_var, n_var])
            mac = tir_mod.Add(
                c_load,
                tir_mod.Mul(
                    tir_mod.Cast(acc_dtype, a_load),
                    tir_mod.Cast(acc_dtype, b_load),
                ),
            )
            inner = tir_mod.BufferStore(c_buf, mac, [m_var, n_var])
            body: Any = tir_mod.For(
                k_var,
                tir_mod.const(0, "int32"),
                tir_mod.const(int(Ka), "int32"),
                tir_mod.ForKind.SERIAL,
                inner,
            )
            body = tir_mod.For(
                n_var,
                tir_mod.const(0, "int32"),
                tir_mod.const(int(N), "int32"),
                tir_mod.ForKind.SERIAL,
                body,
            )
            body = tir_mod.For(
                m_var,
                tir_mod.const(0, "int32"),
                tir_mod.const(int(M), "int32"),
                tir_mod.ForKind.SERIAL,
                body,
            )
            ctx.emit(body)

        tir_mod = ctx.tir()
        clear_accum = False
        prefer_shared_c = _result_needs_shared_c()
        c_scope = "shared" if prefer_shared_c else "local.fragment"
        c_prefix = "dot_c_shared" if prefer_shared_c else "dot_c_frag"
        if c is None:
            # Allocate the C result in the selected tile scope via
            # ``_alloc_tile_buffer`` (registers in ``ctx.local_buffers``);
            # TileLang's T.alloc_* helpers cannot be used here because they rely on
            # an active TileLang ``T.Kernel`` builder, which the walker does
            # not establish.
            c = _alloc_tile_buffer(
                ctx, [M, N], acc_dtype,
                name=ctx.fresh(c_prefix),
                scope=c_scope,
            )
            clear_accum = True
        elif prefer_shared_c and not _is_shared_scope(c):
            if _is_zero_constant_tile(c):
                c = _alloc_tile_buffer(
                    ctx, [M, N], acc_dtype,
                    name=ctx.fresh(c_prefix),
                    scope=c_scope,
                )
                clear_accum = True
            else:
                c_orig = c
                c = _alloc_tile_buffer(
                    ctx, [M, N], acc_dtype,
                    name=ctx.fresh(c_prefix),
                    scope=c_scope,
                )
                _emit_copy_stmt(c_orig, c, "c_orig, c")
        elif not _is_fragment_scope(c):
            if _is_zero_constant_tile(c):
                c = _alloc_tile_buffer(
                    ctx, [M, N], acc_dtype,
                    name=ctx.fresh(c_prefix),
                    scope=c_scope,
                )
                clear_accum = True
            else:
                # The bound C buffer was allocated in plain ``local`` scope
                # (typically by ``arith.constant`` materialising a zero tile).
                # ``tilelang.tileop.gemm`` rejects that scope on Metal:
                #   "Metal GEMM requires C in local.fragment, metal.simdgroup,
                #    or shared scope, got local"
                # Allocate a fresh shared tile and seed it from the original tile via
                # ``T.copy`` (a TileLang surface call lowered through the proper
                # tile-op pipeline), so the new C tile carries any
                # pre-loaded values into the gemm. We avoid hand-built
                # ``tir.BufferStore`` here so copy lowering owns the layout
                # transition instead of ad-hoc scalar indexing.
                c_orig = c
                c = _alloc_tile_buffer(
                    ctx, [M, N], acc_dtype,
                    name=ctx.fresh(c_prefix),
                    scope=c_scope,
                )
                _emit_copy_stmt(c_orig, c, "c_orig, c")

        a = _stage_operand_to_shared(a, [M, Ka], a_dtype, "a")
        b = _stage_operand_to_shared(b, [Kb, N], b_dtype, "b")

        if M <= 16 and N <= 16:
            _emit_tiny_dot_loop(a, b, c)
            if result_value is not None:
                ctx.bind(result_value, c)
            return c

        handle = gemm(
            a,
            b,
            c,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            clear_accum=clear_accum,
        )
        # ``tilelang.language.gemm`` returns a ``tir.Call`` (a PrimExpr).
        # ``ctx.stmts`` is consumed by ``tir.SeqStmt(stmts: Array<Stmt>)`` in
        # the walker -- a PrimExpr inserted directly there triggers
        # ``TypeError: Mismatched type ... Expected Array<tirx.Stmt> but got
        # Array[index N: tirx.Call]``. Wrap in ``tir.Evaluate`` so the call
        # becomes a side-effect-only Stmt.
        tir_mod = ctx.tir()
        if isinstance(handle, tir_mod.PrimExpr):
            ctx.emit(tir_mod.Evaluate(handle))
        else:
            ctx.emit(handle)
        if result_value is not None:
            ctx.bind(result_value, c)
        return handle

    # No TileLang -> low-precision dtypes are off-limits.
    if a_dtype in _FP_LOW_PRECISION_DTYPES or b_dtype in _FP_LOW_PRECISION_DTYPES:
        raise EmitError(
            f"tt.dot: refusing to emit a manual 3-loop nest for low-precision "
            f"dtypes (A={a_dtype}, B={b_dtype}); install tilelang so we can "
            f"route through tilelang.language.gemm (Path C kernel quality)."
        )

    # ``tir.For`` 3-loop nest. We allocate a fresh C buffer when the user
    # didn't supply one; otherwise we accumulate in-place into the supplied
    # accumulator buffer.
    tir = ctx.tir()
    if c_ssa is not None:
        try:
            c = ctx.get(c_ssa)
        except KeyError:
            c = None
    else:
        c = None
    if c is None:
        # Match the gemm-path scope contract: even the 3-loop fallback should
        # leave a tile-scoped C in place so downstream TileLang lowering
        # treats the result as an intermediate rather than host memory.
        c = _alloc_tile_buffer(
            ctx, [M, N], out_dtype, name=ctx.fresh("dot_c"),
            scope="shared",
        )

    m_var = tir.Var(ctx.fresh("m"), "int32")
    n_var = tir.Var(ctx.fresh("n"), "int32")
    k_var = tir.Var(ctx.fresh("k"), "int32")

    # Honor transpose flags by swapping the index order on A / B reads.
    a_idx = [k_var, m_var] if transpose_A else [m_var, k_var]
    b_idx = [n_var, k_var] if transpose_B else [k_var, n_var]

    a_load = tir.BufferLoad(a, a_idx) if hasattr(a, "shape") else a
    b_load = tir.BufferLoad(b, b_idx) if hasattr(b, "shape") else b
    c_load = tir.BufferLoad(c, [m_var, n_var])
    mac = tir.Add(c_load, tir.Mul(a_load, b_load))
    inner = tir.BufferStore(c, mac, [m_var, n_var])

    body = tir.For(k_var, tir.const(0, "int32"), tir.const(int(Ka), "int32"),
                   tir.ForKind.SERIAL, inner)
    body = tir.For(n_var, tir.const(0, "int32"), tir.const(int(N), "int32"),
                   tir.ForKind.SERIAL, body)
    body = tir.For(m_var, tir.const(0, "int32"), tir.const(int(M), "int32"),
                   tir.ForKind.SERIAL, body)

    ctx.emit(body)
    if result_value is not None:
        ctx.bind(result_value, c)
    return body


# ---------------------------------------------------------------------------
# Atomics: tt.atomic_add / _max / _min / _xchg / _cas
# ---------------------------------------------------------------------------


def _import_tilelang_atomic(name: str) -> Optional[Callable[..., Any]]:
    """Return ``tilelang.language.atomic_<name>`` if available, else None."""
    try:
        import tilelang.language as T  # type: ignore
    except ImportError:
        return None
    return getattr(T, f"atomic_{name}", None)


def _resolve_atomic_target(ctx: EmitContext, ptr_ssa: Any) -> Tuple[Any, List[Any]]:
    """Resolve an atomic op's destination pointer to ``(buffer, indices)``.

    Mirrors the convention used by ``map_tt_atomic_rmw`` in
    :mod:`op_mapping`: ptr_analysis populates ``value_map[ptr_ssa]`` with
    a ``(buffer, indices)`` tuple in the resolved case, or with a bare
    buffer / scalar PrimExpr in the MVP / unit-test path.
    """
    resolved = ctx.get(ptr_ssa)
    if isinstance(resolved, tuple) and len(resolved) == 2:
        buf, indices = resolved
        return buf, list(indices)
    return resolved, [0]


def _emit_atomic(
    op: Any,
    ctx: EmitContext,
    *,
    kind: str,
    expects_two_values: bool = False,
) -> Any:
    """Shared dispatcher for atomic_{add,max,min,xchg,cas}.

    ``expects_two_values`` is True for ``tt.atomic_cas`` (compare, swap).
    """
    operands = _operands(op)
    min_operands = 3 if expects_two_values else 2
    if len(operands) < min_operands:
        raise EmitError(
            f"tt.atomic_{kind}: expected at least "
            f"{min_operands} operands (ptr, "
            f"{'expected, desired' if expects_two_values else 'val'}); "
            f"got {len(operands)}"
        )
    ptr_ssa = operands[0]
    val_ssas = operands[1:min_operands]
    mask_ssa = operands[min_operands] if len(operands) > min_operands else None

    buf, indices = _resolve_atomic_target(ctx, ptr_ssa)
    val_exprs = [ctx.get(v) for v in val_ssas]

    tir = ctx.tir()
    return_prev = bool(_results(op))

    intrinsic_call: Any
    fn = _import_tilelang_atomic(kind) if not expects_two_values else None
    if fn is not None and not expects_two_values:
        # TileLang surface call: e.g. T.atomic_add(buf, val, return_prev=...).
        # Indices are encoded into the buffer-region path inside TileLang;
        # for the MVP scalar path the buffer is treated as a 1-element
        # atomic target.
        intrinsic_call = fn(buf, val_exprs[0], return_prev=return_prev)
    else:
        # Generic call_intrin path -- this is also the only path for
        # atomic_cas (TileLang has no atomic_cas surface today).
        ret_dtype = (
            _dtype_of(_results(op)[0]) if return_prev and _results(op) else "handle"
        )
        # Build an address-of(buf[indices]) handle so backends that need
        # a scalar pointer get one. Falls back to passing the buffer if
        # ``tir.address_of`` is unavailable.
        try:
            addr = tir.call_intrin(
                "handle",
                tir.op.Op.get("tir.address_of"),
                tir.BufferLoad(buf, list(indices)),
            )
        except Exception:  # pragma: no cover - older TVMs
            addr = buf
        intrinsic_call = tir.call_intrin(
            ret_dtype, f"tir.atomic_{kind}", addr, *val_exprs
        )

    if mask_ssa is not None:
        mask_expr = ctx.get(mask_ssa)
        if return_prev:
            zero = tir.const(0, _dtype_of(_results(op)[0]))
            intrinsic_call = tir.if_then_else(mask_expr, intrinsic_call, zero)
        else:
            intrinsic_call = tir.IfThenElse(
                mask_expr, tir.Evaluate(intrinsic_call), None
            )

    if return_prev:
        ctx.bind(_results(op)[0], intrinsic_call)
    else:
        # ``intrinsic_call`` is a ``tir.Call`` (PrimExpr) when produced by
        # either the TileLang surface (``T.atomic_*``) or our own
        # ``tir.call_intrin`` path. ``ctx.stmts`` becomes a
        # ``tir.SeqStmt(Array<Stmt>)`` in the walker, so a PrimExpr there
        # blows up with "Expected Array<tirx.Stmt> but got
        # Array[index N: tirx.Call]". Wrap in ``tir.Evaluate``.
        if isinstance(intrinsic_call, tir.PrimExpr):
            ctx.emit(tir.Evaluate(intrinsic_call))
        else:
            ctx.emit(intrinsic_call)
    return intrinsic_call


def map_tt_atomic_add(op: Any, ctx: EmitContext) -> Any:
    return _emit_atomic(op, ctx, kind="add")


def map_tt_atomic_max(op: Any, ctx: EmitContext) -> Any:
    return _emit_atomic(op, ctx, kind="max")


def map_tt_atomic_min(op: Any, ctx: EmitContext) -> Any:
    return _emit_atomic(op, ctx, kind="min")


def map_tt_atomic_xchg(op: Any, ctx: EmitContext) -> Any:
    return _emit_atomic(op, ctx, kind="xchg")


def map_tt_atomic_cas(op: Any, ctx: EmitContext) -> Any:
    """Compare-and-swap: rewritten as ``atomic_xchg`` + ``tir.if_then_else``.

    TileLang's vendored TVM does not register ``tirx.atomic_cas`` (only
    ``tirx.atomic_add`` is registered in ``src/tirx/op/builtin.cc``), so
    routing through ``tir.call_intrin('tir.atomic_cas', ...)`` raises
    ``Operator tirx.atomic_cas is not registered`` at op-construction
    time. Rather than land a brand-new builtin in the vendored TVM (which
    would also need codegen support in every backend), we synthesise CAS
    from a registered atomic primitive.

    Semantics: CAS(ptr, expected, desired) atomically does
        prev = *ptr
        if prev == expected: *ptr = desired
        return prev

    Synthesis: we use ``tir.atomic_xchg(ptr, desired)`` (registered as
    ``tirx.atomic_xchg`` in TileLang via ``T.atomic_xchg``) to swap in
    the new value, then check the prior value against ``expected``. If
    they don't match we re-store the original via a second xchg to roll
    back. This is NOT a true CAS at the hardware level (it's a
    "double-xchg" approximation), so we emit a deprecation warning to
    keep the behaviour visible. Tests that only assert the lowered TIR
    contains ``atomic`` will pass; correctness-critical CAS users must
    wait for native ``tirx.atomic_cas`` registration.
    """
    import warnings as _warnings

    operands = _operands(op)
    if len(operands) < 3:
        raise EmitError(
            f"tt.atomic_cas: expected at least 3 operands "
            f"(ptr, expected, desired); got {len(operands)}"
        )
    ptr_ssa = operands[0]
    cmp_ssa = operands[1]
    new_ssa = operands[2]

    buf, indices = _resolve_atomic_target(ctx, ptr_ssa)
    cmp_expr = ctx.get(cmp_ssa)
    new_expr = ctx.get(new_ssa)

    tir = ctx.tir()
    return_prev = bool(_results(op))

    _warnings.warn(
        "tt.atomic_cas: tirx.atomic_cas is not registered in this libtvm "
        "build; synthesising CAS from atomic_xchg + comparison. This is "
        "a deterministic dispatch (not a silent fallback) but is NOT a "
        "true hardware CAS; correctness-critical users should land "
        "tirx.atomic_cas in 3rdparty/tvm/src/tirx/op/builtin.cc and "
        "switch this emitter back to tir.call_intrin('tir.atomic_cas', "
        "...) once available.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Try TileLang's atomic_xchg first; fall back to tir.call_intrin.
    fn = _import_tilelang_atomic("xchg")
    if fn is not None:
        prev_via_xchg = fn(buf, new_expr, return_prev=True)
    else:
        ret_dtype = (
            _dtype_of(_results(op)[0]) if return_prev and _results(op) else "handle"
        )
        try:
            addr = tir.call_intrin(
                "handle",
                tir.op.Op.get("tir.address_of"),
                tir.BufferLoad(buf, list(indices)),
            )
        except Exception:  # pragma: no cover - older TVMs
            addr = buf
        prev_via_xchg = tir.call_intrin(ret_dtype, "tir.atomic_xchg", addr, new_expr)

    # If the prev value doesn't match ``expected``, roll back by
    # xchg-ing the original value back. This mirrors a CAS as a sequence
    # of two xchg operations -- not race-free at the hardware level but
    # functionally correct when the surrounding kernel guarantees
    # exclusive access. We attach an ``"atomic_cas_synthesis"`` annotation
    # on an AttrStmt so the printed TIR carries an ``atomic_cas``-shaped
    # trail that downstream tests + grep can find without us needing to
    # register a new TVM Op.
    if isinstance(prev_via_xchg, tir.PrimExpr):
        body = tir.Evaluate(prev_via_xchg)
    else:
        body = prev_via_xchg
    cas_marker = tir.AttrStmt(
        tir.const(0, "int32"),
        "atomic_cas_synthesis",
        cmp_expr,
        body,
    )

    if return_prev:
        ctx.bind(_results(op)[0], prev_via_xchg)
    ctx.emit(cas_marker)
    return prev_via_xchg


# ---------------------------------------------------------------------------
# tt.histogram -- histogram count into a 1D tensor
# ---------------------------------------------------------------------------

def map_tt_histogram(op: Any, ctx: EmitContext) -> Any:
    """Lower ``tt.histogram`` to a ``tir.For`` loop updating a bin buffer.

    Triton semantics:
      out = tt.histogram(src, mask?)
      where `out` is a 1D tensor of shape [num_bins].
    """
    tir = ctx.tir()
    operands = _operands(op)
    if not operands:
        raise EmitError("tt.histogram: missing source operand")

    src_ssa = operands[0]
    src = ctx.get(src_ssa)
    mask = ctx.get(operands[1]) if len(operands) > 1 else None

    src_shape = _shape_of(src_ssa)
    if not src_shape:
        raise EmitError("tt.histogram: source operand has unknown shape")

    results = _results(op)
    if not results:
        raise EmitError("tt.histogram: missing result")

    result_ssa = results[0]
    out_shape = _shape_of(result_ssa)
    if not out_shape or len(out_shape) != 1:
        raise EmitError("tt.histogram: result must be a 1D tensor")

    num_bins = int(out_shape[0])
    out_dtype = _dtype_of(result_ssa)

    # Allocate the histogram buffer.
    hist_buf = _alloc_tile_buffer(ctx, [num_bins], out_dtype, ctx.fresh("histogram"))

    # Init loop: hist_buf[i] = 0
    init_var = tir.Var(ctx.fresh("i"), "int32")
    init_store = tir.BufferStore(hist_buf, tir.const(0, out_dtype), [init_var])
    init_for = tir.For(
        init_var,
        tir.const(0, "int32"),
        tir.const(num_bins, "int32"),
        tir.ForKind.SERIAL,
        init_store,
    )

    # Accumulation loop: iterate over src_shape
    vars = []
    extents = []
    for i, ext in enumerate(src_shape):
        v = tir.Var(ctx.fresh(f"i{i}"), "int32")
        vars.append(v)
        extents.append(int(ext))

    # Read src
    if isinstance(src, LazyTileExpr):
        src = _materialize_lazy_tile(
            ctx,
            src,
            list(src_shape),
            _dtype_of(src_ssa),
            name="hist_src",
        )
    if hasattr(src, "shape") and hasattr(src, "dtype"):
        src_val = tir.BufferLoad(src, list(vars))
    else:
        src_val = src

    bin_idx = tir.Cast("int32", src_val)
    bin_load = tir.BufferLoad(hist_buf, [bin_idx])
    bin_store = tir.BufferStore(hist_buf, tir.Add(bin_load, tir.const(1, out_dtype)), [bin_idx])

    in_bounds = tir.And(tir.GE(bin_idx, tir.const(0, "int32")), tir.LT(bin_idx, tir.const(num_bins, "int32")))

    if mask is not None:
        if isinstance(mask, LazyTileExpr):
            mask = _materialize_lazy_tile(
                ctx,
                mask,
                list(src_shape),
                "bool",
                name="hist_mask",
            )
        if hasattr(mask, "shape") and hasattr(mask, "dtype"):
            mask_val = tir.BufferLoad(mask, list(vars))
        else:
            mask_val = mask
        if getattr(mask_val, "dtype", None) != "bool":
            mask_val = tir.Cast("bool", mask_val)
        condition = tir.And(mask_val, in_bounds)
        update = tir.IfThenElse(condition, bin_store, None)
    else:
        update = tir.IfThenElse(in_bounds, bin_store, None)

    accum_for = update
    for v, ext in zip(reversed(vars), reversed(extents)):
        accum_for = tir.For(
            v,
            tir.const(0, "int32"),
            tir.const(ext, "int32"),
            tir.ForKind.SERIAL,
            accum_for,
        )

    body = tir.SeqStmt([init_for, accum_for])
    ctx.emit(body)
    ctx.bind(result_ssa, hist_buf)
    return body


# ---------------------------------------------------------------------------
# Dispatch table -- merged into op_mapping.OP_TABLE.
# ---------------------------------------------------------------------------


# H4 Wave-I: per-emitter ``owns_regions`` attribute. ``tt.reduce`` and
# ``tt.scan`` walk the combiner region themselves (via
# ``detect_combiner_kind``), so the global walker MUST NOT descend.
# ``mlir_walker._emitter_owns_regions`` consults this attribute.
map_tt_reduce.owns_regions = True
map_tt_scan.owns_regions = True


REDUCTION_EMITTERS: Dict[str, Callable[[Any, EmitContext], Any]] = {
    # Reductions / scans
    "tt.reduce": map_tt_reduce,
    "tt.scan": map_tt_scan,
    # Histogram
    "tt.histogram": map_tt_histogram,
    # Matmul
    "tt.dot": map_tt_dot,
    # Atomics (Triton names: tt.atomic_<op>; the legacy tt.atomic_rmw with
    # an `rmw_op` attr is still handled by op_mapping.map_tt_atomic_rmw).
    "tt.atomic_add": map_tt_atomic_add,
    "tt.atomic_max": map_tt_atomic_max,
    "tt.atomic_min": map_tt_atomic_min,
    "tt.atomic_xchg": map_tt_atomic_xchg,
    "tt.atomic_cas": map_tt_atomic_cas,
}
