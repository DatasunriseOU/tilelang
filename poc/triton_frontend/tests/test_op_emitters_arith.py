"""Unit tests for ``poc.triton_frontend.op_emitters.arith``.

Each test builds a dict-shaped fake TTIR op (the same pattern used in
``test_dot_reduce_atomic.py``), seeds the WalkerCtx ``value_map`` with
real ``tvm.tir.Var`` operands, calls the emitter, and asserts the
resulting PrimExpr matches the expected ``tir.<Node>(x, y)`` via
``tvm.ir.structural_equal``.

We also cover:
* dtype guards (``arith.addi`` on float, ``math.exp`` on int, etc.) raise
  ``EmitError``;
* registry merge (``OP_TABLE`` contains every arith/math/tt.fma name we
  exported);
* comparison predicate normalisation (string and int forms both work).
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

tvm = pytest.importorskip("tvm")

from poc.triton_frontend.op_emitters.arith import (  # noqa: E402
    ARITH_EMITTERS,
    EmitError,
)
from poc.triton_frontend.op_mapping import OP_TABLE, WalkerCtx  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ssa(name: str, dtype: str = "float32") -> Dict[str, Any]:
    """Build a dict-shaped SSA stand-in carrying just the dtype."""
    return {"name": name, "dtype": dtype}


def _op(name: str, operands: List[Any], results: List[Any], **attrs: Any) -> Dict[str, Any]:
    return {
        "name": name,
        "operands": operands,
        "results": results,
        "attrs": dict(attrs),
    }


def _make_binary_ctx(dtype: str = "float32"):
    """Return (ctx, lhs_ssa, rhs_ssa, lhs_var, rhs_var) for a binary-op test."""
    ctx = WalkerCtx()
    lhs_ssa = _ssa("a", dtype)
    rhs_ssa = _ssa("b", dtype)
    lhs_var = tvm.tir.Var("a", dtype)
    rhs_var = tvm.tir.Var("b", dtype)
    ctx.bind(lhs_ssa, lhs_var)
    ctx.bind(rhs_ssa, rhs_var)
    return ctx, lhs_ssa, rhs_ssa, lhs_var, rhs_var


def _make_unary_ctx(dtype: str = "float32"):
    ctx = WalkerCtx()
    ssa = _ssa("x", dtype)
    var = tvm.tir.Var("x", dtype)
    ctx.bind(ssa, var)
    return ctx, ssa, var


def _seq(*items):
    """Wrap items so structural_equal sees a deterministic order."""
    return list(items)


# ---------------------------------------------------------------------------
# Float arithmetic
# ---------------------------------------------------------------------------


def test_addf_lowers_to_tir_add():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("float32")
    out = _ssa("o", "float32")
    op = _op("arith.addf", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.addf"](op, ctx)
    expected = tvm.tir.Add(a_var, b_var)
    assert tvm.ir.structural_equal(expr, expected)
    assert tvm.ir.structural_equal(ctx.value_map[out], expected)


def test_subf_lowers_to_tir_sub():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("float32")
    out = _ssa("o", "float32")
    op = _op("arith.subf", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.subf"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.Sub(a_var, b_var))


def test_mulf_lowers_to_tir_mul():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("float16")
    out = _ssa("o", "float16")
    op = _op("arith.mulf", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.mulf"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.Mul(a_var, b_var))


def test_divf_lowers_to_tir_div():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("float32")
    out = _ssa("o", "float32")
    op = _op("arith.divf", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.divf"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.Div(a_var, b_var))


def test_negf_lowers_to_zero_minus_x():
    ctx, x_ssa, x_var = _make_unary_ctx("float32")
    out = _ssa("o", "float32")
    op = _op("arith.negf", [x_ssa], [out])
    expr = ARITH_EMITTERS["arith.negf"](op, ctx)
    expected = tvm.tir.Sub(tvm.tir.const(0, "float32"), x_var)
    assert tvm.ir.structural_equal(expr, expected)


# ---------------------------------------------------------------------------
# Integer arithmetic
# ---------------------------------------------------------------------------


def test_addi_lowers_to_tir_add():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("int32")
    out = _ssa("o", "int32")
    op = _op("arith.addi", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.addi"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.Add(a_var, b_var))


def test_subi_lowers_to_tir_sub():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("int32")
    out = _ssa("o", "int32")
    op = _op("arith.subi", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.subi"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.Sub(a_var, b_var))


def test_muli_lowers_to_tir_mul():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("int32")
    out = _ssa("o", "int32")
    op = _op("arith.muli", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.muli"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.Mul(a_var, b_var))


def test_divsi_lowers_to_truncdiv():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("int32")
    out = _ssa("o", "int32")
    op = _op("arith.divsi", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.divsi"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.truncdiv(a_var, b_var))


def test_divui_lowers_to_floordiv():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("uint32")
    out = _ssa("o", "uint32")
    op = _op("arith.divui", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.divui"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.floordiv(a_var, b_var))


def test_remsi_lowers_to_truncmod():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("int32")
    out = _ssa("o", "int32")
    op = _op("arith.remsi", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.remsi"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.truncmod(a_var, b_var))


def test_remui_lowers_to_floormod():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("uint32")
    out = _ssa("o", "uint32")
    op = _op("arith.remui", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.remui"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.floormod(a_var, b_var))


# ---------------------------------------------------------------------------
# Min / max
# ---------------------------------------------------------------------------


def test_minimumf_lowers_to_tir_min():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("float32")
    out = _ssa("o", "float32")
    op = _op("arith.minimumf", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.minimumf"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.Min(a_var, b_var))


def test_maximumf_lowers_to_tir_max():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("float32")
    out = _ssa("o", "float32")
    op = _op("arith.maximumf", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.maximumf"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.Max(a_var, b_var))


def test_minsi_lowers_to_tir_min():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("int32")
    out = _ssa("o", "int32")
    op = _op("arith.minsi", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.minsi"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.Min(a_var, b_var))


def test_maxsi_lowers_to_tir_max():
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("int32")
    out = _ssa("o", "int32")
    op = _op("arith.maxsi", [a_ssa, b_ssa], [out])
    expr = ARITH_EMITTERS["arith.maxsi"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.Max(a_var, b_var))


# ---------------------------------------------------------------------------
# Math intrinsics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_name,tir_attr",
    [
        ("math.sqrt", "sqrt"),
        ("math.exp",  "exp"),
        ("math.log",  "log"),
        ("math.sin",  "sin"),
        ("math.cos",  "cos"),
        ("math.tanh", "tanh"),
    ],
)
def test_math_unary_lowers_to_tir_intrinsic(op_name, tir_attr):
    ctx, x_ssa, x_var = _make_unary_ctx("float32")
    out = _ssa("o", "float32")
    op = _op(op_name, [x_ssa], [out])
    expr = ARITH_EMITTERS[op_name](op, ctx)
    expected = getattr(tvm.tir, tir_attr)(x_var)
    assert tvm.ir.structural_equal(expr, expected)


def test_math_absf_lowers_to_tir_abs():
    ctx, x_ssa, x_var = _make_unary_ctx("float32")
    out = _ssa("o", "float32")
    op = _op("math.absf", [x_ssa], [out])
    expr = ARITH_EMITTERS["math.absf"](op, ctx)
    expected = tvm.tir.abs(x_var)
    assert tvm.ir.structural_equal(expr, expected)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "predicate,tir_cls",
    [
        ("oeq", tvm.tir.EQ),
        ("ogt", tvm.tir.GT),
        ("oge", tvm.tir.GE),
        ("olt", tvm.tir.LT),
        ("ole", tvm.tir.LE),
        ("one", tvm.tir.NE),
        ("ueq", tvm.tir.EQ),
        ("ult", tvm.tir.LT),
    ],
)
def test_cmpf_predicate_dispatch(predicate, tir_cls):
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("float32")
    out = _ssa("o", "uint1")
    op = _op("arith.cmpf", [a_ssa, b_ssa], [out], predicate=predicate)
    expr = ARITH_EMITTERS["arith.cmpf"](op, ctx)
    assert tvm.ir.structural_equal(expr, tir_cls(a_var, b_var))


def test_cmpf_numeric_predicate_resolves():
    """Predicate enum integer (e.g. 4 == 'olt') resolves to tir.LT."""
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("float32")
    out = _ssa("o", "uint1")
    op = _op("arith.cmpf", [a_ssa, b_ssa], [out], predicate=4)  # olt
    expr = ARITH_EMITTERS["arith.cmpf"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.LT(a_var, b_var))


@pytest.mark.parametrize(
    "predicate,tir_cls",
    [
        ("eq",  tvm.tir.EQ),
        ("ne",  tvm.tir.NE),
        ("slt", tvm.tir.LT),
        ("sle", tvm.tir.LE),
        ("sgt", tvm.tir.GT),
        ("sge", tvm.tir.GE),
        ("ult", tvm.tir.LT),
        ("ule", tvm.tir.LE),
    ],
)
def test_cmpi_predicate_dispatch(predicate, tir_cls):
    ctx, a_ssa, b_ssa, a_var, b_var = _make_binary_ctx("int32")
    out = _ssa("o", "uint1")
    op = _op("arith.cmpi", [a_ssa, b_ssa], [out], predicate=predicate)
    expr = ARITH_EMITTERS["arith.cmpi"](op, ctx)
    assert tvm.ir.structural_equal(expr, tir_cls(a_var, b_var))


def test_cmpf_unsupported_predicate_raises():
    ctx, a_ssa, b_ssa, *_ = _make_binary_ctx("float32")
    out = _ssa("o", "uint1")
    op = _op("arith.cmpf", [a_ssa, b_ssa], [out], predicate="ord")
    with pytest.raises(EmitError):
        ARITH_EMITTERS["arith.cmpf"](op, ctx)


def test_cmpf_missing_predicate_raises():
    ctx, a_ssa, b_ssa, *_ = _make_binary_ctx("float32")
    out = _ssa("o", "uint1")
    op = _op("arith.cmpf", [a_ssa, b_ssa], [out])
    with pytest.raises(EmitError, match="missing 'predicate'"):
        ARITH_EMITTERS["arith.cmpf"](op, ctx)


# ---------------------------------------------------------------------------
# FMA
# ---------------------------------------------------------------------------


def test_fma_lowers_to_mul_then_add_or_intrinsic():
    """tt.fma either calls tir.fma (when registered) or expands to (a*b)+c."""
    ctx = WalkerCtx()
    a_ssa, b_ssa, c_ssa = _ssa("a"), _ssa("b"), _ssa("c")
    a, b, c = (tvm.tir.Var(n, "float32") for n in ("a", "b", "c"))
    ctx.bind(a_ssa, a)
    ctx.bind(b_ssa, b)
    ctx.bind(c_ssa, c)
    out = _ssa("o", "float32")
    op = _op("tt.fma", [a_ssa, b_ssa, c_ssa], [out])
    expr = ARITH_EMITTERS["tt.fma"](op, ctx)
    # Accept either Call("tir.fma", ...) or Add(Mul(a,b), c).
    expected_fallback = tvm.tir.Add(tvm.tir.Mul(a, b), c)
    if isinstance(expr, tvm.tir.Call):
        # Intrinsic path: assert the operands appear in the call args.
        args = list(expr.args)
        assert tvm.ir.structural_equal(args[0], a)
        assert tvm.ir.structural_equal(args[1], b)
        assert tvm.ir.structural_equal(args[2], c)
    else:
        assert tvm.ir.structural_equal(expr, expected_fallback)


# ---------------------------------------------------------------------------
# Dtype guards (no silent fallback)
# ---------------------------------------------------------------------------


def test_addi_on_float_raises():
    ctx, a_ssa, b_ssa, *_ = _make_binary_ctx("float32")
    out = _ssa("o", "float32")
    op = _op("arith.addi", [a_ssa, b_ssa], [out])
    with pytest.raises(EmitError, match="float type"):
        ARITH_EMITTERS["arith.addi"](op, ctx)


def test_addf_on_int_raises():
    ctx, a_ssa, b_ssa, *_ = _make_binary_ctx("int32")
    out = _ssa("o", "int32")
    op = _op("arith.addf", [a_ssa, b_ssa], [out])
    with pytest.raises(EmitError, match="non-float"):
        ARITH_EMITTERS["arith.addf"](op, ctx)


def test_math_exp_on_int_raises():
    ctx, x_ssa, _x_var = _make_unary_ctx("int32")
    out = _ssa("o", "int32")
    op = _op("math.exp", [x_ssa], [out])
    with pytest.raises(EmitError, match="non-float"):
        ARITH_EMITTERS["math.exp"](op, ctx)


def test_cmpi_on_float_raises():
    ctx, a_ssa, b_ssa, *_ = _make_binary_ctx("float32")
    out = _ssa("o", "uint1")
    op = _op("arith.cmpi", [a_ssa, b_ssa], [out], predicate="eq")
    with pytest.raises(EmitError, match="float type"):
        ARITH_EMITTERS["arith.cmpi"](op, ctx)


def test_negf_on_int_raises():
    ctx, x_ssa, _x_var = _make_unary_ctx("int32")
    out = _ssa("o", "int32")
    op = _op("arith.negf", [x_ssa], [out])
    with pytest.raises(EmitError, match="non-float"):
        ARITH_EMITTERS["arith.negf"](op, ctx)


# ---------------------------------------------------------------------------
# Registry merge
# ---------------------------------------------------------------------------


def test_registry_merged_into_op_table():
    """Every key from ARITH_EMITTERS appears in the global OP_TABLE."""
    for name, fn in ARITH_EMITTERS.items():
        assert name in OP_TABLE, f"OP_TABLE missing {name}"
        assert OP_TABLE[name] is fn, f"OP_TABLE[{name}] != ARITH_EMITTERS[{name}]"


def test_registry_covers_required_ops():
    """Sanity-check the spec list of ops is present."""
    required = {
        "arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf",
        "arith.addi", "arith.subi", "arith.muli",
        "arith.divsi", "arith.divui", "arith.remsi", "arith.remui",
        "arith.minimumf", "arith.maximumf", "arith.minsi", "arith.maxsi",
        "math.sqrt", "math.exp", "math.log", "math.sin", "math.cos",
        "math.tanh", "math.absf",
        "arith.cmpf", "arith.cmpi",
        "tt.fma",
    }
    missing = required - set(ARITH_EMITTERS.keys())
    assert not missing, f"emitters missing for: {missing}"


# ---------------------------------------------------------------------------
# Regression: jaxlib mlir.ir Properties fallback (Wave D2)
# ---------------------------------------------------------------------------


class _FakeMlirOp:
    """Minimal stand-in for a jaxlib ``mlir.ir.Operation`` whose inherent
    attrs live in MLIR Properties storage.

    Under ``allow_unregistered_dialects=True`` jaxlib hides the dialect's
    Properties from ``op.attributes`` (it stays empty), but the printed
    op text still includes the ``<{predicate = N : i64}>`` block.  This
    fake mirrors that exact shape so the emitter is forced through the
    ``_attrs_with_properties_shared`` fallback in op_mapping.py.
    """

    def __init__(self, name: str, operands, results, printed: str) -> None:
        self.name = name
        self.operands = list(operands)
        self.results = list(results)
        self.attributes = []  # empty -- jaxlib Properties path
        self._printed = printed

    def __str__(self) -> str:  # what _parse_generic_properties_shared reads
        return self._printed


class _HashableSSA:
    """Hashable SSA stand-in carrying a dtype.

    The existing dict-shaped ``_ssa`` helper in this file isn't hashable
    so it can't be used as a key in ``WalkerCtx.value_map``. The arith
    emitter only reads ``.dtype`` off operands, so a tiny class with that
    one attribute is enough to drive the regression test end-to-end.
    """

    __slots__ = ("name", "dtype")

    def __init__(self, name: str, dtype: str) -> None:
        self.name = name
        self.dtype = dtype


def test_cmpi_predicate_from_properties_block():
    """jaxlib-shape arith.cmpi with ``<{predicate = 2 : i64}>`` -> tir.LT.

    Regression for Wave D2: prior to lifting ``_attrs_with_properties``
    into op_mapping, the arith emitter called the bare ``_attrs`` helper
    which returned ``{}`` for property-only ops, producing
    ``EmitError: arith.cmpi: missing 'predicate' attribute`` on every
    softmax / layer_norm kernel. Predicate enum 2 is ``slt`` -> tir.LT.
    """
    ctx = WalkerCtx()
    a_ssa = _HashableSSA("a", "int32")
    b_ssa = _HashableSSA("b", "int32")
    a_var = tvm.tir.Var("a", "int32")
    b_var = tvm.tir.Var("b", "int32")
    ctx.bind(a_ssa, a_var)
    ctx.bind(b_ssa, b_var)
    out = _HashableSSA("o", "uint1")
    printed = (
        '%2 = "arith.cmpi"(%0, %1) <{predicate = 2 : i64}>'
        ' : (i32, i32) -> i1'
    )
    op = _FakeMlirOp(
        name="arith.cmpi",
        operands=[a_ssa, b_ssa],
        results=[out],
        printed=printed,
    )
    expr = ARITH_EMITTERS["arith.cmpi"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.LT(a_var, b_var))


def test_cmpf_predicate_from_properties_block():
    """jaxlib-shape arith.cmpf with ``<{predicate = 4 : i64}>`` -> tir.LT.

    Predicate enum 4 in MLIR ``CmpFPredicate`` is ``olt`` -> tir.LT.
    """
    ctx = WalkerCtx()
    a_ssa = _HashableSSA("a", "float32")
    b_ssa = _HashableSSA("b", "float32")
    a_var = tvm.tir.Var("a", "float32")
    b_var = tvm.tir.Var("b", "float32")
    ctx.bind(a_ssa, a_var)
    ctx.bind(b_ssa, b_var)
    out = _HashableSSA("o", "uint1")
    printed = (
        '%2 = "arith.cmpf"(%0, %1) <{predicate = 4 : i64}>'
        ' : (f32, f32) -> i1'
    )
    op = _FakeMlirOp(
        name="arith.cmpf",
        operands=[a_ssa, b_ssa],
        results=[out],
        printed=printed,
    )
    expr = ARITH_EMITTERS["arith.cmpf"](op, ctx)
    assert tvm.ir.structural_equal(expr, tvm.tir.LT(a_var, b_var))
