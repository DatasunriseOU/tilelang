"""Unit tests for the control-flow / cast emitters in ``op_emitters.control``.

These follow the dict-shaped fake-op pattern documented on
:class:`poc.triton_frontend.op_mapping.WalkerCtx`. ``tvm`` is required (the
emitters lazy-import it) so the tests skip cleanly on environments without
a TVM build.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

tvm = pytest.importorskip("tvm")

from poc.triton_frontend.op_mapping import WalkerCtx  # noqa: E402
from poc.triton_frontend.op_emitters.control import (  # noqa: E402
    CONTROL_EMITTERS,
    EmitError,
    map_arith_bitcast,
    map_arith_extf,
    map_arith_select,
    map_arith_extsi,
    map_arith_fptosi,
    map_scf_for,
    map_scf_if,
    map_scf_yield,
    map_tt_advance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _val(name: str, *, shape: List[int] = (), dtype: str = "float32") -> Dict[str, Any]:
    """Build a dict-shaped fake SSA value."""
    return {"name": name, "shape": tuple(shape), "dtype": dtype}


def _stringify(node: Any) -> str:
    return str(node)


def _bind_const(ctx: WalkerCtx, ssa: Any, value: int, dtype: str = "int32") -> None:
    ctx.bind(ssa, tvm.tir.const(value, dtype))


# ---------------------------------------------------------------------------
# arith.select  -> tir.if_then_else
# ---------------------------------------------------------------------------


def test_arith_select_scalar_emits_if_then_else() -> None:
    ctx = WalkerCtx()
    cond = _val("c", shape=[], dtype="bool")
    t = _val("t", shape=[], dtype="float32")
    f = _val("f", shape=[], dtype="float32")
    out = _val("o", shape=[], dtype="float32")

    ctx.bind(cond, tvm.tir.const(1, "bool"))
    ctx.bind(t, tvm.tir.const(1.0, "float32"))
    ctx.bind(f, tvm.tir.const(2.0, "float32"))

    op = {
        "name": "arith.select",
        "operands": [cond, t, f],
        "results": [out],
        "attrs": {},
    }
    expr = map_arith_select(op, ctx)
    text = _stringify(expr).lower()
    # tir.if_then_else lowers to a Call to tir.if_then_else; the printed
    # form contains "if_then_else" in TVM's standard repr.
    assert "if_then_else" in text or "tir.if_then_else" in text or "select" in text, \
        f"expected if_then_else-like node, got: {text!r}"
    # Result must be bound.
    assert ctx.value_map[out] is expr


def test_arith_select_wrong_arity_raises() -> None:
    ctx = WalkerCtx()
    op = {"name": "arith.select", "operands": [], "results": [], "attrs": {}}
    with pytest.raises(EmitError):
        map_arith_select(op, ctx)


# ---------------------------------------------------------------------------
# arith.extf  fp16 -> fp32
# ---------------------------------------------------------------------------


def test_arith_extf_emits_cast_to_float32() -> None:
    ctx = WalkerCtx()
    src = _val("x_f16", shape=[16], dtype="float16")
    dst = _val("x_f32", shape=[16], dtype="float32")
    # Bind src to a Var of dtype float16 so Cast has something to consume.
    ctx.bind(src, tvm.tir.Var("x_f16", "float16"))

    op = {
        "name": "arith.extf",
        "operands": [src],
        "results": [dst],
        "attrs": {},
    }
    cast = map_arith_extf(op, ctx)
    assert isinstance(cast, tvm.tir.Cast), f"expected tir.Cast, got {type(cast)}"
    assert str(cast.dtype) == "float32"
    # Source dtype preserved on operand.
    assert str(cast.value.dtype) == "float16"


def test_arith_extf_rejects_int_src() -> None:
    ctx = WalkerCtx()
    src = _val("x_i32", shape=[], dtype="int32")
    dst = _val("y_f32", shape=[], dtype="float32")
    ctx.bind(src, tvm.tir.Var("x_i32", "int32"))
    op = {
        "name": "arith.extf",
        "operands": [src],
        "results": [dst],
        "attrs": {},
    }
    with pytest.raises(EmitError):
        map_arith_extf(op, ctx)


# ---------------------------------------------------------------------------
# arith.bitcast -> tir.reinterpret
# ---------------------------------------------------------------------------


def test_arith_bitcast_emits_reinterpret() -> None:
    ctx = WalkerCtx()
    src = _val("x_f32", shape=[], dtype="float32")
    dst = _val("x_i32", shape=[], dtype="int32")
    ctx.bind(src, tvm.tir.Var("x_f32", "float32"))

    op = {
        "name": "arith.bitcast",
        "operands": [src],
        "results": [dst],
        "attrs": {},
    }
    expr = map_arith_bitcast(op, ctx)
    text = _stringify(expr).lower()
    assert "reinterpret" in text, f"expected reinterpret in repr: {text!r}"
    assert str(expr.dtype) == "int32"


def test_arith_bitcast_width_mismatch_raises() -> None:
    ctx = WalkerCtx()
    src = _val("x_f32", shape=[], dtype="float32")
    dst = _val("x_i64", shape=[], dtype="int64")
    ctx.bind(src, tvm.tir.Var("x_f32", "float32"))
    op = {
        "name": "arith.bitcast",
        "operands": [src],
        "results": [dst],
        "attrs": {},
    }
    with pytest.raises(EmitError):
        map_arith_bitcast(op, ctx)


# ---------------------------------------------------------------------------
# arith.fptosi  -- exotic bf16->int8 must be rejected
# ---------------------------------------------------------------------------


def test_arith_fptosi_rejects_bf16_to_int8() -> None:
    ctx = WalkerCtx()
    src = _val("x_bf16", shape=[], dtype="bfloat16")
    dst = _val("x_i8", shape=[], dtype="int8")
    ctx.bind(src, tvm.tir.Var("x_bf16", "bfloat16"))
    op = {
        "name": "arith.fptosi",
        "operands": [src],
        "results": [dst],
        "attrs": {},
    }
    with pytest.raises(EmitError):
        map_arith_fptosi(op, ctx)


def test_arith_extsi_widens_int() -> None:
    ctx = WalkerCtx()
    src = _val("x_i32", shape=[], dtype="int32")
    dst = _val("x_i64", shape=[], dtype="int64")
    ctx.bind(src, tvm.tir.Var("x_i32", "int32"))
    op = {
        "name": "arith.extsi",
        "operands": [src],
        "results": [dst],
        "attrs": {},
    }
    expr = map_arith_extsi(op, ctx)
    assert isinstance(expr, tvm.tir.Cast)
    assert str(expr.dtype) == "int64"


# ---------------------------------------------------------------------------
# tt.advance  -- block-pointer offset bump
# ---------------------------------------------------------------------------


def test_tt_advance_bumps_offsets_on_ptrstate() -> None:
    ctx = WalkerCtx()
    base_ssa = _val("base_ptr", shape=[16, 32], dtype="float16")
    out_ssa = _val("adv_ptr", shape=[16, 32], dtype="float16")
    delta0 = _val("d0", shape=[], dtype="int32")
    delta1 = _val("d1", shape=[], dtype="int32")

    base_state = {
        "_ptrstate": "tile",
        "source": "X",
        "offsets": [0, 0],
        "sizes": [16, 32],
    }
    ctx.bind(base_ssa, base_state)
    ctx.bind(delta0, 4)
    ctx.bind(delta1, 8)

    op = {
        "name": "tt.advance",
        "operands": [base_ssa, delta0, delta1],
        "results": [out_ssa],
        "attrs": {},
    }
    new_state = map_tt_advance(op, ctx)
    assert isinstance(new_state, dict)
    assert new_state.get("source") == "X"
    assert new_state.get("offsets") == [4, 8]
    # Original state must be untouched.
    assert base_state["offsets"] == [0, 0]
    assert ctx.value_map[out_ssa] is new_state


# ---------------------------------------------------------------------------
# scf.for  -- iter_arg materialisation
# ---------------------------------------------------------------------------


def test_scf_for_emits_tir_for_with_iter_arg() -> None:
    ctx = WalkerCtx()
    lb = _val("lb", shape=[], dtype="int32")
    ub = _val("ub", shape=[], dtype="int32")
    step = _val("step", shape=[], dtype="int32")
    init = _val("init", shape=[], dtype="float32")
    out = _val("final", shape=[], dtype="float32")

    # Block args: the induction var SSA + one iter_arg SSA.
    ind_arg = _val("i", shape=[], dtype="int32")
    iter_arg = _val("acc", shape=[], dtype="float32")

    _bind_const(ctx, lb, 0)
    _bind_const(ctx, ub, 32)
    _bind_const(ctx, step, 1)
    ctx.bind(init, tvm.tir.const(0.0, "float32"))

    op = {
        "name": "scf.for",
        "operands": [lb, ub, step, init],
        "results": [out],
        "attrs": {},
        "block_args": [ind_arg, iter_arg],
        "regions": [
            {
                "ops": [
                    # Just yield the iter_arg unchanged for the smoke test.
                    {
                        "name": "scf.yield",
                        "operands": [iter_arg],
                        "results": [],
                        "attrs": {},
                    },
                ],
            },
        ],
    }
    for_stmt = map_scf_for(op, ctx)
    assert isinstance(for_stmt, tvm.tir.For), f"expected tir.For, got {type(for_stmt)}"
    # Loop var is an int32.
    assert str(for_stmt.loop_var.dtype) == "int32"
    # Extent matches ub - lb = 32 (folded constant).
    assert int(for_stmt.extent) == 32
    # The body must contain a LetStmt that materialises the iter_arg as a
    # fresh tir.Var (assigned from ``init`` on entry).
    body_text = _stringify(for_stmt.body)
    assert "let" in body_text.lower() or "Let" in body_text, \
        f"expected LetStmt for iter_arg in body, got: {body_text!r}"


def test_scf_for_too_many_iter_args_raises() -> None:
    ctx = WalkerCtx()
    lb = _val("lb"); ub = _val("ub"); step = _val("step")
    _bind_const(ctx, lb, 0); _bind_const(ctx, ub, 8); _bind_const(ctx, step, 1)
    iter_args = []
    for n in range(5):
        v = _val(f"a{n}", shape=[], dtype="float32")
        ctx.bind(v, tvm.tir.const(0.0, "float32"))
        iter_args.append(v)
    op = {
        "name": "scf.for",
        "operands": [lb, ub, step, *iter_args],
        "results": [_val("o")],
        "attrs": {},
        "block_args": [_val("i", shape=[], dtype="int32"),
                       *[_val(f"b{n}", shape=[], dtype="float32") for n in range(5)]],
        "regions": [{"ops": [{"name": "scf.yield", "operands": iter_args, "results": [], "attrs": {}}]}],
    }
    with pytest.raises(EmitError):
        map_scf_for(op, ctx)


# ---------------------------------------------------------------------------
# scf.if  -> tir.IfThenElse
# ---------------------------------------------------------------------------


def test_scf_if_emits_if_then_else() -> None:
    ctx = WalkerCtx()
    cond = _val("c", shape=[], dtype="bool")
    ctx.bind(cond, tvm.tir.const(1, "bool"))

    op = {
        "name": "scf.if",
        "operands": [cond],
        "results": [],
        "attrs": {},
        "regions": [
            # then
            {"ops": [{"name": "scf.yield", "operands": [], "results": [], "attrs": {}}]},
            # else
            {"ops": [{"name": "scf.yield", "operands": [], "results": [], "attrs": {}}]},
        ],
    }
    if_stmt = map_scf_if(op, ctx)
    assert isinstance(if_stmt, tvm.tir.IfThenElse)


def test_scf_yield_is_noop() -> None:
    ctx = WalkerCtx()
    op = {"name": "scf.yield", "operands": [], "results": [], "attrs": {}}
    assert map_scf_yield(op, ctx) is None
    assert not ctx.stmts


# ---------------------------------------------------------------------------
# Dispatch table coverage
# ---------------------------------------------------------------------------


def test_control_emitters_table_has_expected_keys() -> None:
    expected = {
        "arith.select",
        "arith.extf", "arith.truncf",
        "arith.fptosi", "arith.sitofp", "arith.uitofp", "arith.fptoui",
        "arith.bitcast",
        "arith.extsi", "arith.extui", "arith.trunci",
        "tt.advance",
        "scf.for", "scf.if", "scf.yield",
    }
    missing = expected - set(CONTROL_EMITTERS.keys())
    assert not missing, f"CONTROL_EMITTERS missing keys: {missing}"
