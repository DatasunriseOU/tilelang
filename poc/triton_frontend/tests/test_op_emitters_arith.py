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

pytest.importorskip("tilelang")
tvm = pytest.importorskip("tvm")

from poc.triton_frontend.op_emitters.arith import (  # noqa: E402
    ARITH_EMITTERS,
    EmitError,
)
from poc.triton_frontend.op_mapping import OP_TABLE, WalkerCtx  # noqa: E402

from ._fixtures import FakeSSA  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ssa(name: str, dtype: str = "float32") -> FakeSSA:
    """Build a hashable SSA stand-in carrying just the dtype.

    Delegates to the shared :class:`FakeSSA` fixture (a ``dict`` subclass)
    so all op-emitter tests share the same hashable-dict semantics.
    """
    return FakeSSA(name, dtype=dtype)


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


def test_arith_addf_handles_buffer_descriptor_operand():
    """Regression: ``arith.addf(buf_c, prod)`` where ``buf_c`` is a Buffer.

    After Wave I3 the matmul accumulator surfaces at the arith layer as a
    plain ``tir.Buffer`` (an scf.for / while iter_arg bound directly to
    the descriptor). Before this fix, ``_emit_addf`` called
    ``ctx.tir().Add(buf_c, prod_expr)`` and TVM raised ``_OpAdd: Expected
    PrimExpr but got tirx.Buffer``. The tile path in
    :func:`poc.triton_frontend.op_emitters.arith._emit_tile_binop` must
    detect the Buffer operand, allocate a fresh result Buffer, and emit a
    per-lane ``tir.For`` nest with ``BufferLoad`` -> ``Add`` ->
    ``BufferStore``.

    We assert:
      * the emitter binds a fresh ``tir.Buffer`` (not a PrimExpr) as the
        SSA result;
      * the ``ctx.stmts`` queue gained exactly one ``tir.For`` whose body
        is a ``BufferStore`` of the correct dtype/shape;
      * the stored value is ``Add(BufferLoad(buf_c, [j]),
        BufferLoad(buf_prod, [j]))`` -- i.e. both Buffer operands are
        materialised per-lane rather than passed as descriptors.
    """
    ctx = WalkerCtx()
    # 1D Buffer accumulator (the per-lane case; 2D shapes go through the
    # same ``_read_lane`` machinery so the rank-1 test pins the contract).
    buf_c = tvm.tir.decl_buffer((128,), "float32", name="buf_c", scope="local")
    buf_prod = tvm.tir.decl_buffer((128,), "float32", name="buf_prod", scope="local")
    c_ssa = _ssa("c", "float32")
    prod_ssa = _ssa("prod", "float32")
    ctx.bind(c_ssa, buf_c)
    ctx.bind(prod_ssa, buf_prod)
    out_ssa = _ssa("o", "float32")
    op = _op("arith.addf", [c_ssa, prod_ssa], [out_ssa])

    result = ARITH_EMITTERS["arith.addf"](op, ctx)

    # Result must be a freshly-allocated Buffer (not a PrimExpr).
    assert isinstance(result, tvm.tir.Buffer), (
        f"expected tir.Buffer, got {type(result).__name__}: {result!r}"
    )
    assert tuple(int(s) for s in result.shape) == (128,)
    assert str(result.dtype) == "float32"
    # Same Buffer is bound to the SSA result.
    assert ctx.value_map[out_ssa] is result

    # The tile-path emits exactly one For wrapping a BufferStore body.
    fors = [s for s in ctx.stmts if isinstance(s, tvm.tir.For)]
    assert len(fors) == 1, f"expected one For, got stmts={ctx.stmts!r}"
    loop = fors[0]
    assert int(loop.extent) == 128, f"loop extent {loop.extent!r} != 128"
    body = loop.body
    assert isinstance(body, tvm.tir.BufferStore), (
        f"loop body should be BufferStore, got {type(body).__name__}"
    )
    assert body.buffer.same_as(result), "BufferStore target must be result buffer"

    # Stored value must be Add(BufferLoad(buf_c, [j]), BufferLoad(buf_prod, [j])).
    val = body.value
    assert isinstance(val, tvm.tir.Add), (
        f"per-lane combine must be tir.Add, got {type(val).__name__}: {val!r}"
    )
    lhs, rhs = val.a, val.b
    assert isinstance(lhs, tvm.tir.BufferLoad), (
        f"lhs must be BufferLoad of buf_c, got {type(lhs).__name__}"
    )
    assert isinstance(rhs, tvm.tir.BufferLoad), (
        f"rhs must be BufferLoad of buf_prod, got {type(rhs).__name__}"
    )
    assert lhs.buffer.same_as(buf_c), "lhs BufferLoad must read buf_c"
    assert rhs.buffer.same_as(buf_prod), "rhs BufferLoad must read buf_prod"
    # Both loads index with the same loop var (single-axis tile).
    assert len(lhs.indices) == 1 and len(rhs.indices) == 1
    assert tvm.ir.structural_equal(lhs.indices[0], rhs.indices[0])
    assert tvm.ir.structural_equal(lhs.indices[0], loop.loop_var)


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


def test_math_exp_handles_buffer_descriptor_operand():
    """Regression: ``math.exp(buf)`` where ``buf`` is a tile ``tir.Buffer``.

    Surfaced by the softmax e2e harness: ``math.exp`` on the centred logits
    tile (``buf`` materialised by a prior ``arith.subf``) tripped
    ``tirx.convert: Expected PrimExpr but got tirx.Buffer`` because the
    scalar ``tir.exp`` constructor cannot accept a Buffer descriptor. The
    emitter must detect the tile operand, allocate a fresh result Buffer,
    and emit a per-lane ``tir.For`` whose body is
    ``BufferStore(out, exp(BufferLoad(buf, [j])), [j])``.
    """
    ctx = WalkerCtx()
    buf_x = tvm.tir.decl_buffer((128,), "float32", name="buf_x", scope="local")
    x_ssa = _ssa("x", "float32")
    ctx.bind(x_ssa, buf_x)
    out_ssa = _ssa("o", "float32")
    op = _op("math.exp", [x_ssa], [out_ssa])

    result = ARITH_EMITTERS["math.exp"](op, ctx)

    assert isinstance(result, tvm.tir.Buffer), (
        f"expected tir.Buffer, got {type(result).__name__}: {result!r}"
    )
    assert tuple(int(s) for s in result.shape) == (128,)
    assert str(result.dtype) == "float32"
    assert ctx.value_map[out_ssa] is result

    fors = [s for s in ctx.stmts if isinstance(s, tvm.tir.For)]
    assert len(fors) == 1, f"expected one For, got stmts={ctx.stmts!r}"
    loop = fors[0]
    assert int(loop.extent) == 128
    body = loop.body
    assert isinstance(body, tvm.tir.BufferStore)
    assert body.buffer.same_as(result)
    val = body.value
    # The stored value should be exp(BufferLoad(buf_x, [j])).
    assert isinstance(val, tvm.tir.Call), (
        f"per-lane apply must be tir.Call (exp intrinsic), got "
        f"{type(val).__name__}: {val!r}"
    )
    inner = val.args[0]
    assert isinstance(inner, tvm.tir.BufferLoad), (
        f"exp argument must be BufferLoad, got {type(inner).__name__}"
    )
    assert inner.buffer.same_as(buf_x)
    assert len(inner.indices) == 1
    assert tvm.ir.structural_equal(inner.indices[0], loop.loop_var)


def test_math_exp_scalar_path_unchanged():
    """Scalar ``math.exp`` still produces a bare PrimExpr (no tile lowering)."""
    ctx, x_ssa, x_var = _make_unary_ctx("float32")
    out = _ssa("o", "float32")
    op = _op("math.exp", [x_ssa], [out])
    expr = ARITH_EMITTERS["math.exp"](op, ctx)
    # Scalar path: no For nest queued, returned expr is the bare exp Call.
    assert not [s for s in ctx.stmts if isinstance(s, tvm.tir.For)]
    assert tvm.ir.structural_equal(expr, tvm.tir.exp(x_var))


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


# ``_FakeMlirOp`` and ``_HashableSSA`` previously lived inline; they now
# come from the shared fixtures module so all op-emitter tests use the
# same hashing / Properties-shape behaviour.
from ._fixtures import FakeMlirOp as _FakeMlirOp, FakeSSA as _HashableSSA


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
