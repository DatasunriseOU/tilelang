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
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

# We import op_mapping lazily inside emitters when we need to reach into
# WalkerCtx machinery; the type alias below is just for static readers.
EmitContext = Any  # poc.triton_frontend.op_mapping.WalkerCtx


__all__ = [
    "ARITH_EMITTERS",
    "EmitError",
]


class EmitError(RuntimeError):
    """Raised when an emitter cannot lower an op (precise, never silent)."""


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
# Float arithmetic (arith.*f)
# ---------------------------------------------------------------------------


def _emit_addf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.addf on non-float type {dt!r}; use arith.addi")
    return _bind_result(op, ctx, ctx.tir().Add(a, b))


def _emit_subf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.subf on non-float type {dt!r}; use arith.subi")
    return _bind_result(op, ctx, ctx.tir().Sub(a, b))


def _emit_mulf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.mulf on non-float type {dt!r}; use arith.muli")
    return _bind_result(op, ctx, ctx.tir().Mul(a, b))


def _emit_divf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.divf on non-float type {dt!r}; use arith.divsi/divui")
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
    dt = _dtype_of(a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.addi on float type {dt!r}; use arith.addf")
    return _bind_result(op, ctx, ctx.tir().Add(a, b))


def _emit_subi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.subi on float type {dt!r}; use arith.subf")
    return _bind_result(op, ctx, ctx.tir().Sub(a, b))


def _emit_muli(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.muli on float type {dt!r}; use arith.mulf")
    return _bind_result(op, ctx, ctx.tir().Mul(a, b))


def _emit_divsi(op: Any, ctx: EmitContext) -> Any:
    """Signed integer division -- TIR ``truncdiv`` matches MLIR ``arith.divsi``."""
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.divsi on float type {dt!r}; use arith.divf")
    tir = ctx.tir()
    return _bind_result(op, ctx, tir.truncdiv(a, b))


def _emit_divui(op: Any, ctx: EmitContext) -> Any:
    """Unsigned integer division -- ``floordiv`` for non-negative operands.

    MLIR ``arith.divui`` is well-defined only for non-negative operands; the
    backend dtype (``uint*``) carries that contract. ``floordiv`` lowers to
    the same hardware divide on every backend we target.
    """
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.divui on float type {dt!r}; use arith.divf")
    return _bind_result(op, ctx, ctx.tir().floordiv(a, b))


def _emit_remsi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.remsi on float type {dt!r}")
    return _bind_result(op, ctx, ctx.tir().truncmod(a, b))


def _emit_remui(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.remui on float type {dt!r}")
    return _bind_result(op, ctx, ctx.tir().floormod(a, b))


# ---------------------------------------------------------------------------
# Min / max
# ---------------------------------------------------------------------------


def _emit_minimumf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.minimumf on non-float type {dt!r}; use arith.minsi")
    return _bind_result(op, ctx, ctx.tir().Min(a, b))


def _emit_maximumf(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.maximumf on non-float type {dt!r}; use arith.maxsi")
    return _bind_result(op, ctx, ctx.tir().Max(a, b))


def _emit_minsi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.minsi on float type {dt!r}; use arith.minimumf")
    return _bind_result(op, ctx, ctx.tir().Min(a, b))


def _emit_maxsi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.maxsi on float type {dt!r}; use arith.maximumf")
    return _bind_result(op, ctx, ctx.tir().Max(a, b))


# ---------------------------------------------------------------------------
# Math intrinsics (math.*)
# ---------------------------------------------------------------------------


def _math_unary(name: str, tir_fn_name: str):
    """Build an emitter that maps math.<name> to ``tvm.tir.<tir_fn_name>``."""

    def _emit(op: Any, ctx: EmitContext) -> Any:
        x = _resolve_one(op, ctx)
        dt = _dtype_of(x)
        if not _is_float_dtype(dt):
            raise EmitError(
                f"math.{name} on non-float type {dt!r}; "
                f"Triton frontends emit math.* only on float operands"
            )
        tir = ctx.tir()
        fn = getattr(tir, tir_fn_name, None)
        if fn is None:
            # Fall back to a call_intrin so we still get a typed Call PrimExpr.
            result = tir.call_intrin(dt, f"tir.{tir_fn_name}", x)
        else:
            result = fn(x)
        return _bind_result(op, ctx, result)

    _emit.__name__ = f"_emit_math_{name}"
    return _emit


_emit_math_sqrt = _math_unary("sqrt", "sqrt")
_emit_math_exp = _math_unary("exp", "exp")
_emit_math_log = _math_unary("log", "log")
_emit_math_sin = _math_unary("sin", "sin")
_emit_math_cos = _math_unary("cos", "cos")
_emit_math_tanh = _math_unary("tanh", "tanh")


def _emit_math_absf(op: Any, ctx: EmitContext) -> Any:
    x = _resolve_one(op, ctx)
    dt = _dtype_of(x)
    if not _is_float_dtype(dt):
        raise EmitError(f"math.absf on non-float type {dt!r}; use math.absi")
    tir = ctx.tir()
    abs_fn = getattr(tir, "abs", None)
    if abs_fn is None:
        result = tir.call_intrin(dt, "tir.fabs", x)
    else:
        result = abs_fn(x)
    return _bind_result(op, ctx, result)


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
    dt = _dtype_of(a)
    if not _is_float_dtype(dt):
        raise EmitError(f"arith.cmpf on non-float type {dt!r}; use arith.cmpi")
    attrs = _attrs(op)
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
    return _bind_result(op, ctx, cls(a, b))


def _emit_cmpi(op: Any, ctx: EmitContext) -> Any:
    a, b = _resolve_two(op, ctx)
    dt = _dtype_of(a)
    if _is_float_dtype(dt):
        raise EmitError(f"arith.cmpi on float type {dt!r}; use arith.cmpf")
    attrs = _attrs(op)
    raw = attrs.get("predicate", attrs.get("kind"))
    if raw is None:
        raise EmitError("arith.cmpi: missing 'predicate' attribute")
    pred = _normalize_predicate(raw, _CMPI_PREDICATES, _CMPI_NUMERIC)
    cls_name = _CMPI_PREDICATES[pred]
    tir = ctx.tir()
    cls = getattr(tir, cls_name)
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
    # Min / max
    "arith.minimumf": _emit_minimumf,
    "arith.maximumf": _emit_maximumf,
    "arith.minsi": _emit_minsi,
    "arith.maxsi": _emit_maxsi,
    # Math intrinsics
    "math.sqrt": _emit_math_sqrt,
    "math.exp": _emit_math_exp,
    "math.log": _emit_math_log,
    "math.sin": _emit_math_sin,
    "math.cos": _emit_math_cos,
    "math.tanh": _emit_math_tanh,
    "math.absf": _emit_math_absf,
    # Comparisons
    "arith.cmpf": _emit_cmpf,
    "arith.cmpi": _emit_cmpi,
    # FMA
    "tt.fma": _emit_fma,
}
