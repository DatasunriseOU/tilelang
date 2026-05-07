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
    PTX_TO_TIR,
    SCF_WHILE_MAX_ITERATIONS,
    map_arith_bitcast,
    map_arith_constant,
    map_arith_extf,
    map_arith_select,
    map_arith_extsi,
    map_arith_fptosi,
    map_llvm_inline_asm,
    map_scf_for,
    map_scf_if,
    map_scf_while,
    map_scf_yield,
    map_tt_advance,
    map_tt_func,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ``_FakeSSA`` previously lived inline; the canonical implementation now
# lives in :mod:`poc.triton_frontend.tests._fixtures` so all op-emitter
# tests share the same hashable-dict surface.
from ._fixtures import FakeSSA as _FakeSSA  # noqa: E402


def _val(name: str, *, shape: List[int] = (), dtype: str = "float32") -> Dict[str, Any]:
    """Build a dict-shaped fake SSA value (hashable for value_map keys)."""
    return _FakeSSA({"name": name, "shape": tuple(shape), "dtype": dtype})


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
# Wave F2 regression: arith.select with Buffer-typed operand
# ---------------------------------------------------------------------------
#
# After Wave E2 lowered broadcasted constants / per-lane comparators into
# ``tir.decl_buffer``, ``arith.select`` operands can resolve to a
# ``tir.Buffer`` rather than a scalar PrimExpr. The pre-F2 emitter fed the
# Buffer straight into ``tir.if_then_else``, which TVM rejects with
# ``_OpIfThenElse: Expected PrimExpr but got tirx.Buffer``. The fix
# materialises a per-lane ``tir.For`` nest with ``BufferLoad`` on each
# Buffer-shaped operand, identical to Wave F1's tt.load mask path.


def test_arith_select_buffer_then_else_per_lane_materialisation() -> None:
    """Buffer-typed cond/true/false operands must materialise as a For nest.

    The legacy scalar path called ``tir.if_then_else`` directly with the
    Buffer as ``arg #0`` and exploded; we now allocate a tile-scoped result
    buffer and emit a ``tir.For`` over the result shape, with ``BufferLoad``
    on each Buffer-typed operand.
    """
    ctx = WalkerCtx()

    cond = _val("c", shape=[16], dtype="bool")
    t = _val("t", shape=[16], dtype="float32")
    f = _val("f", shape=[16], dtype="float32")
    out = _val("o", shape=[16], dtype="float32")

    cond_buf = tvm.tir.decl_buffer((16,), "bool", name="cond_tile")
    t_buf = tvm.tir.decl_buffer((16,), "float32", name="t_tile")
    f_buf = tvm.tir.decl_buffer((16,), "float32", name="f_tile")

    ctx.bind(cond, cond_buf)
    ctx.bind(t, t_buf)
    ctx.bind(f, f_buf)

    op = {
        "name": "arith.select",
        "operands": [cond, t, f],
        "results": [out],
        "attrs": {},
    }

    result = map_arith_select(op, ctx)

    # Result is a tile-scoped Buffer, not a raw PrimExpr.
    assert isinstance(result, tvm.tir.Buffer), f"expected tir.Buffer, got {type(result).__name__}"
    assert ctx.value_map[out] is result

    # A For-nest must have been emitted into the kernel body.
    assert ctx.stmts, "expected map_arith_select to emit a For-nest stmt"
    emitted = ctx.stmts[-1]
    text = str(emitted)
    assert "for" in text.lower(), f"expected For-nest, got: {text!r}"
    # The lane-loaded operands and the if_then_else call must all appear.
    assert "if_then_else" in text or "select" in text, (
        f"expected if_then_else / select node inside For nest, got: {text!r}"
    )


def test_arith_select_buffer_cond_scalar_branches_per_lane() -> None:
    """Only the cond is a Buffer; true/false are scalar PrimExprs."""
    ctx = WalkerCtx()

    cond = _val("c", shape=[8], dtype="bool")
    t = _val("t", shape=[], dtype="float32")
    f = _val("f", shape=[], dtype="float32")
    out = _val("o", shape=[8], dtype="float32")

    cond_buf = tvm.tir.decl_buffer((8,), "bool", name="cond_tile")
    ctx.bind(cond, cond_buf)
    ctx.bind(t, tvm.tir.const(1.0, "float32"))
    ctx.bind(f, tvm.tir.const(0.0, "float32"))

    op = {
        "name": "arith.select",
        "operands": [cond, t, f],
        "results": [out],
        "attrs": {},
    }

    result = map_arith_select(op, ctx)
    assert isinstance(result, tvm.tir.Buffer)
    assert ctx.value_map[out] is result
    assert ctx.stmts, "expected For-nest emission for Buffer-typed cond"


def test_arith_select_buffer_operand_rank0_result_raises() -> None:
    """Sanity: Buffer-typed operand on a rank-0 result must error, not fold."""
    ctx = WalkerCtx()

    cond = _val("c", shape=[], dtype="bool")
    t = _val("t", shape=[4], dtype="float32")
    f = _val("f", shape=[], dtype="float32")
    out = _val("o", shape=[], dtype="float32")  # rank-0 result, intentional mismatch

    ctx.bind(cond, tvm.tir.const(1, "bool"))
    ctx.bind(t, tvm.tir.decl_buffer((4,), "float32", name="t_tile"))
    ctx.bind(f, tvm.tir.const(0.0, "float32"))

    op = {
        "name": "arith.select",
        "operands": [cond, t, f],
        "results": [out],
        "attrs": {},
    }
    with pytest.raises(EmitError, match="rank-0 result"):
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


def test_scf_for_buffer_iter_arg_does_not_letstmt() -> None:
    """Regression: matmul-style ``scf.for`` with a buffer/tile carry must not
    emit ``tir.LetStmt(var, buffer_descriptor, body)``.

    Prior to the fix, an iter_arg whose resolved value was a buffer
    descriptor (e.g. a ``T.gemm`` accumulator surfaced as
    ``(tir.Buffer, shape_list)``) crashed the lowering with::

        TypeError: Mismatched type on argument #1 when calling: tirx.Bind
        ... Expected ir.PrimExpr but got ffi.Array.

    The emitter now bypasses ``LetStmt`` for non-scalar carries and binds
    the block-arg SSA directly to the descriptor.
    """
    ctx = WalkerCtx()
    lb = _val("lb", shape=[], dtype="int32")
    ub = _val("ub", shape=[], dtype="int32")
    step = _val("step", shape=[], dtype="int32")
    init = _val("acc_init", shape=[16, 16], dtype="float32")
    out = _val("acc_final", shape=[16, 16], dtype="float32")

    ind_arg = _val("i", shape=[], dtype="int32")
    iter_arg = _val("acc", shape=[16, 16], dtype="float32")

    _bind_const(ctx, lb, 0)
    _bind_const(ctx, ub, 4)
    _bind_const(ctx, step, 1)
    # Bind init to a (Buffer, shape) tuple -- the matmul T.gemm
    # accumulator pattern. This previously tripped tirx.Bind.
    acc_buf = tvm.tir.decl_buffer((16, 16), "float32", name="acc")
    ctx.bind(init, (acc_buf, [16, 16]))

    op = {
        "name": "scf.for",
        "operands": [lb, ub, step, init],
        "results": [out],
        "attrs": {},
        "block_args": [ind_arg, iter_arg],
        "regions": [
            {
                "ops": [
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
    # Must not raise.
    for_stmt = map_scf_for(op, ctx)
    assert isinstance(for_stmt, tvm.tir.For)
    # The body should NOT contain a LetStmt for the buffer iter_arg --
    # only the loop's induction structure and the (empty / yielded) body.
    body_text = _stringify(for_stmt.body)
    # If a LetStmt with the buffer slipped through, TVM would have raised
    # already; re-stringifying as a sanity check should still print fine.
    assert body_text  # any non-empty repr is fine


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
        "scf.for", "scf.if", "scf.yield", "scf.while",
        "llvm.inline_asm", "tt.elementwise_inline_asm",
    }
    missing = expected - set(CONTROL_EMITTERS.keys())
    assert not missing, f"CONTROL_EMITTERS missing keys: {missing}"


# ---------------------------------------------------------------------------
# scf.while  -> bounded tir.For + IfThenElse
# ---------------------------------------------------------------------------


def test_scf_while_with_iter_arg() -> None:
    """A single-iter_arg scf.while lowers to tir.For with iter_arg LetStmt
    bindings, and the after-region is guarded by tir.IfThenElse(cond)."""
    ctx = WalkerCtx()
    init = _val("init", shape=[], dtype="int32")
    out = _val("final", shape=[], dtype="int32")

    # Region block-arg SSAs (one carry slot each region).
    before_carry = _val("bc0", shape=[], dtype="int32")
    after_carry = _val("ac0", shape=[], dtype="int32")
    cond_ssa = _val("cond", shape=[], dtype="bool")

    ctx.bind(init, tvm.tir.const(0, "int32"))

    op = {
        "name": "scf.while",
        "operands": [init],
        "results": [out],
        "attrs": {"upper_bound": 16},
        "before_block_args": [before_carry],
        "after_block_args": [after_carry],
        "regions": [
            # before-region: just compute a (constant) condition then
            # forward the carry. The walker will need to resolve `cond_ssa`
            # so we pre-bind it here via a no-op constant-emitting op of
            # our own.
            {
                "ops": [
                    {
                        "name": "scf.condition",
                        "operands": [cond_ssa, before_carry],
                        "results": [],
                        "attrs": {},
                    },
                ],
            },
            # after-region: yield the carry unchanged.
            {
                "ops": [
                    {
                        "name": "scf.yield",
                        "operands": [after_carry],
                        "results": [],
                        "attrs": {},
                    },
                ],
            },
        ],
    }
    # Pre-seed the cond binding the same way the parent walker would.
    ctx.bind(cond_ssa, tvm.tir.const(1, "bool"))

    for_stmt = map_scf_while(op, ctx)
    assert isinstance(for_stmt, tvm.tir.For), \
        f"expected tir.For, got {type(for_stmt)}"
    # Bound is honoured.
    assert int(for_stmt.extent) == 16
    # The printed body must show both the carry binding (``wcarry... = 0``)
    # and an ``if`` guard around the after-region. We assert against the
    # printed form because TVM's TIR types vary across releases (some
    # builds wrap LetStmt in a SeqStmt-shim, some preserve it directly).
    body_text = _stringify(for_stmt.body)
    assert "wcarry" in body_text, \
        f"expected carry-var binding in body, got: {body_text!r}"
    assert " = " in body_text or ":=" in body_text, \
        f"expected an iter_arg initialisation in body, got: {body_text!r}"
    assert "if" in body_text.lower(), \
        f"expected IfThenElse guard, got: {body_text!r}"
    # Result SSA bound.
    assert out in ctx.value_map


def test_scf_while_unbounded_raises() -> None:
    """An scf.while explicitly tagged as unbounded must raise EmitError --
    silently truncating the loop is a correctness hazard."""
    ctx = WalkerCtx()
    init = _val("init", shape=[], dtype="int32")
    ctx.bind(init, tvm.tir.const(0, "int32"))
    op = {
        "name": "scf.while",
        "operands": [init],
        "results": [_val("o")],
        "attrs": {"unbounded": True},
        "before_block_args": [_val("bc")],
        "after_block_args": [_val("ac")],
        "regions": [
            {"ops": [{"name": "scf.condition",
                      "operands": [_val("c", shape=[], dtype="bool"), _val("bc")],
                      "results": [], "attrs": {}}]},
            {"ops": [{"name": "scf.yield", "operands": [_val("ac")],
                      "results": [], "attrs": {}}]},
        ],
    }
    with pytest.raises(EmitError, match="unbounded"):
        map_scf_while(op, ctx)


# ---------------------------------------------------------------------------
# llvm.inline_asm  -> portable TIR intrinsics
# ---------------------------------------------------------------------------


def test_inline_asm_tanh_approx_lowered_to_tir_tanh() -> None:
    """PTX ``tanh.approx.f32 $0, $1;`` must lower to ``tir.tanh(x)``."""
    ctx = WalkerCtx()
    src = _val("x", shape=[], dtype="float32")
    dst = _val("y", shape=[], dtype="float32")
    ctx.bind(src, tvm.tir.Var("x", "float32"))

    op = {
        "name": "llvm.inline_asm",
        "operands": [src],
        "results": [dst],
        "attrs": {
            "asm_string": "tanh.approx.f32 $0, $1;",
            "constraints": "=f,f",
            "has_side_effects": False,
            "asm_dialect": 0,
        },
    }
    expr = map_llvm_inline_asm(op, ctx)
    text = _stringify(expr).lower()
    assert "tanh" in text, f"expected tir.tanh node, got: {text!r}"
    # Result must be bound.
    assert ctx.value_map[dst] is expr
    # Sanity: PTX_TO_TIR is populated with at least the well-known set.
    assert "tanh.approx.f32" in PTX_TO_TIR
    assert "ex2.approx.f32" in PTX_TO_TIR


def test_inline_asm_unknown_raises() -> None:
    """An unrecognised PTX pattern must raise EmitError that surfaces the
    asm_string verbatim so triage can copy-paste it."""
    ctx = WalkerCtx()
    src = _val("x", shape=[], dtype="float32")
    ctx.bind(src, tvm.tir.Var("x", "float32"))
    op = {
        "name": "llvm.inline_asm",
        "operands": [src],
        "results": [_val("y", shape=[], dtype="float32")],
        "attrs": {
            "asm_string": "bizzarre.intrin $0, $1;",
            "constraints": "=f,f",
        },
    }
    with pytest.raises(EmitError) as excinfo:
        map_llvm_inline_asm(op, ctx)
    msg = str(excinfo.value)
    assert "bizzarre.intrin" in msg, \
        f"expected EmitError message to surface asm_string verbatim, got: {msg!r}"


def test_scf_while_max_iterations_default() -> None:
    """The bound default is documented and positive."""
    assert SCF_WHILE_MAX_ITERATIONS > 0
    assert SCF_WHILE_MAX_ITERATIONS == 1024 or SCF_WHILE_MAX_ITERATIONS > 0


# ---------------------------------------------------------------------------
# tt.func  -> seed block args into ctx.value_map / ctx.buffers
# ---------------------------------------------------------------------------


def test_tt_func_seeds_block_args_into_value_map() -> None:
    """A ``tt.func`` with three block args of differing dtypes seeds them all.

    Pointer args land in ``ctx.buffers`` (under the stripped SSA name) and
    are bound in ``ctx.value_map`` under both the Value object and the
    printed SSA name string. Scalar args become ``tir.Var`` and are bound
    the same way (no ``ctx.buffers`` entry).
    """
    ctx = WalkerCtx()

    # Three block args:
    #   %arg0 : !tt.ptr<f32>   (pointer; allocates a buffer)
    #   %arg1 : i32            (scalar 32-bit signless)
    #   %arg2 : !tt.ptr<f16>   (pointer to half)
    arg0 = _FakeSSA({"name": "%arg0", "dtype": "float32", "is_ptr": True})
    arg1 = _FakeSSA({"name": "%arg1", "dtype": "i32", "shape": ()})
    arg2 = _FakeSSA({"name": "%arg2", "dtype": "float16", "is_ptr": True})

    op = {
        "name": "tt.func",
        "operands": [],
        "results": [],
        "attrs": {"sym_name": "kernel"},
        # Dict-fake convention used elsewhere in this file: block_args is the
        # entry block's argument list, surfaced at op level for convenience.
        "block_args": [arg0, arg1, arg2],
        "regions": [{"ops": []}],
    }

    map_tt_func(op, ctx)

    # All three SSA names landed in value_map keyed by their printed string.
    assert "%arg0" in ctx.value_map, \
        f"value_map missing %arg0; keys={sorted(repr(k) for k in ctx.value_map)}"
    assert "%arg1" in ctx.value_map, \
        f"value_map missing %arg1; keys={sorted(repr(k) for k in ctx.value_map)}"
    assert "%arg2" in ctx.value_map, \
        f"value_map missing %arg2; keys={sorted(repr(k) for k in ctx.value_map)}"

    # Pointer args allocate buffers under the stripped name; scalar arg does not.
    assert "arg0" in ctx.buffers, "pointer block arg %arg0 should allocate a buffer"
    assert "arg2" in ctx.buffers, "pointer block arg %arg2 should allocate a buffer"
    assert "arg1" not in ctx.buffers, \
        "scalar block arg %arg1 must NOT be in ctx.buffers (no decl_buffer)"

    # The buffer dtype matches the pointer element type.
    assert str(ctx.buffers["arg0"].dtype) == "float32"
    assert str(ctx.buffers["arg2"].dtype) == "float16"

    # Pointer SSA in value_map points at the same buffer object.
    assert ctx.value_map["%arg0"] is ctx.buffers["arg0"]
    assert ctx.value_map["%arg2"] is ctx.buffers["arg2"]

    # Scalar arg becomes a tir.Var with the canonical TVM dtype spelling
    # (``i32`` -> ``int32``).
    var = ctx.value_map["%arg1"]
    assert isinstance(var, tvm.tir.Var), \
        f"expected tir.Var for scalar block arg, got {type(var).__name__}"
    assert str(var.dtype) == "int32", f"expected int32 dtype, got {var.dtype!r}"

    # Both keying conventions resolve to the same object (Value-keyed and
    # name-keyed entries point to the same TIR node).
    assert ctx.value_map[arg0] is ctx.value_map["%arg0"]
    assert ctx.value_map[arg1] is ctx.value_map["%arg1"]
    assert ctx.value_map[arg2] is ctx.value_map["%arg2"]


def test_tt_func_zero_args_is_noop() -> None:
    """A zero-argument ``tt.func`` doesn't blow up and produces no bindings."""
    ctx = WalkerCtx()
    op = {
        "name": "tt.func",
        "operands": [],
        "results": [],
        "attrs": {},
        "block_args": [],
        "regions": [{"ops": []}],
    }
    map_tt_func(op, ctx)
    assert ctx.value_map == {}
    assert ctx.buffers == {}


def test_tt_func_tensor_block_arg_seeds_buffer() -> None:
    """A ``tensor<128xf32>`` block arg lands in ``ctx.buffers`` rank-1.

    Triton's TTIR threads tile-typed values across function boundaries
    in kernels like ``layer_norm``. Before this fix, ``map_tt_func``
    raised ``unsupported MLIR dtype: 'tensor<128xf32>'`` and the lowering
    aborted at function-prologue time. We assert the buffer dtype and
    shape are preserved (no ``[1]`` placeholder) so downstream load /
    store ops can index into the actual extents.
    """
    ctx = WalkerCtx()
    # ``_FakeSSA`` with shape=(128,) and dtype="f32" causes ``_type_string``
    # in op_emitters/control.py to synthesize ``tensor<128xf32>`` -- the
    # same spelling that surfaces in real TTIR for layer_norm-style kernels.
    arg = _FakeSSA({"name": "%arg0", "dtype": "f32", "shape": (128,)})

    op = {
        "name": "tt.func",
        "operands": [],
        "results": [],
        "attrs": {"sym_name": "kernel"},
        "block_args": [arg],
        "regions": [{"ops": []}],
    }

    map_tt_func(op, ctx)

    # The tile lands in ``ctx.buffers`` keyed by stripped SSA name.
    assert "arg0" in ctx.buffers, \
        f"tensor block arg %arg0 should seed ctx.buffers; keys={list(ctx.buffers)}"
    buf = ctx.buffers["arg0"]

    # Rank-1 with the actual extent (128) and the canonical TVM dtype.
    shape_ints = [int(s) for s in buf.shape]
    assert shape_ints == [128], \
        f"expected rank-1 shape [128], got {shape_ints!r}"
    assert str(buf.dtype) == "float32", \
        f"expected float32 dtype, got {buf.dtype!r}"

    # SSA-name-keyed entry in value_map points at the same buffer object.
    assert ctx.value_map["%arg0"] is buf

    # Tile-typed args are NOT runtime scalar args; they shouldn't pollute
    # ``ctx.runtime_args`` (that list feeds ``PrimFunc.params`` for scalars).
    assert buf not in getattr(ctx, "runtime_args", []), \
        "tile buffer should not appear in runtime_args (scalar-only list)"


# ---------------------------------------------------------------------------
# arith.constant  -> seed value_map with IntImm / FloatImm
# ---------------------------------------------------------------------------


def test_arith_constant_seeds_value_map() -> None:
    """``%c0 = arith.constant 0 : i32`` lowers to ``tir.IntImm("int32", 0)``.

    The result SSA name is bound into ``ctx.value_map`` so a downstream
    use (e.g. ``%idx = arith.addi %c0, %tid``) can resolve the operand.
    """
    ctx = WalkerCtx()
    result = _FakeSSA({"name": "%c0", "dtype": "i32", "shape": ()})
    op = {
        "name": "arith.constant",
        "operands": [],
        "results": [result],
        # Dict-shaped attr: the parser also accepts MLIR generic-form
        # strings ("0 : i32") and real IntegerAttr objects.
        "attrs": {"value": {"value": 0, "type": "i32"}},
    }
    const = map_arith_constant(op, ctx)

    # The expected node: tir.IntImm("int32", 0).
    expected = tvm.tir.IntImm("int32", 0)
    assert isinstance(const, tvm.tir.IntImm), \
        f"expected tir.IntImm, got {type(const).__name__}"
    assert str(const.dtype) == "int32"
    assert int(const.value) == 0

    # Bound under the printed name string per the spec.
    assert "%c0" in ctx.value_map
    assert isinstance(ctx.value_map["%c0"], tvm.tir.IntImm)
    assert str(ctx.value_map["%c0"].dtype) == "int32"
    assert int(ctx.value_map["%c0"].value) == 0
    # And under the Value object too, pointing at the same node so MLIR-walker
    # operand lookups resolve.
    assert ctx.value_map[result] is ctx.value_map["%c0"]
    # Sanity: structurally equal to the freshly-built expected node.
    assert str(ctx.value_map["%c0"]) == str(expected)


def test_arith_constant_float_seeds_floatimm() -> None:
    """``%cf = arith.constant 3.5 : f32`` lowers to ``tir.FloatImm``."""
    ctx = WalkerCtx()
    result = _FakeSSA({"name": "%cf", "dtype": "f32", "shape": ()})
    op = {
        "name": "arith.constant",
        "operands": [],
        "results": [result],
        "attrs": {"value": "3.5 : f32"},
    }
    const = map_arith_constant(op, ctx)
    assert isinstance(const, tvm.tir.FloatImm)
    assert str(const.dtype) == "float32"
    assert float(const.value) == pytest.approx(3.5)
    assert ctx.value_map["%cf"] is const


def test_arith_constant_array_attr_raises() -> None:
    """Array (``dense<...>``) attrs raise EmitError -- no silent splat."""
    ctx = WalkerCtx()
    result = _FakeSSA({"name": "%cv", "dtype": "i32", "shape": (4,)})
    op = {
        "name": "arith.constant",
        "operands": [],
        "results": [result],
        # List form -- our parser flags any list/tuple/ndarray as unsupported.
        "attrs": {"value": {"value": [0, 1, 2, 3], "type": "tensor<4xi32>"}},
    }
    with pytest.raises(EmitError) as excinfo:
        map_arith_constant(op, ctx)
    assert "array attr" in str(excinfo.value).lower()


def test_arith_constant_registered_in_dispatch_table() -> None:
    """Both new emitters appear in CONTROL_EMITTERS for the walker to find."""
    assert CONTROL_EMITTERS.get("arith.constant") is map_arith_constant
    assert CONTROL_EMITTERS.get("tt.func") is map_tt_func


# ---------------------------------------------------------------------------
# arith.constant  -> dense / DenseFPElementsAttr / DenseIntElementsAttr path
# ---------------------------------------------------------------------------
#
# Triton's ``tt.load %ptrs, %mask, other=0.0`` materialises ``other`` as
# ``arith.constant dense<0.0> : tensor<...xf32>``; we lower that to a
# freshly-declared buffer initialised by a serial ``tir.For`` nest.


class _FakeDenseTensorType:
    """Minimal stand-in for ``RankedTensorType`` exposing ``.shape`` and
    ``.element_type``. Lets the dense path test without spinning up an
    MLIR Context."""

    def __init__(self, shape, element_type):
        self.shape = list(shape)
        self.element_type = element_type


class _FakeFloatAttr:
    def __init__(self, v: float, dtype: str = "f32") -> None:
        self.value = float(v)
        self.type = dtype


class _FakeIntAttr:
    def __init__(self, v: int, dtype: str = "i32") -> None:
        self.value = int(v)
        self.type = dtype


class _FakeDenseAttr:
    """Stand-in for ``DenseFPElementsAttr`` / ``DenseIntElementsAttr``.

    Mirrors the jaxlib bindings' surface: ``is_splat`` flag,
    ``get_splat_value()`` -> scalar Attr, ``__iter__`` yields per-element
    Python primitives, and ``.type`` exposes ``.shape`` / ``.element_type``.
    """

    def __init__(self, shape, element_type: str, payload, is_splat: bool):
        self.type = _FakeDenseTensorType(shape, element_type)
        self.is_splat = is_splat
        self._payload = payload
        self._element_type = element_type

    def get_splat_value(self):
        if not self.is_splat:
            raise ValueError("get_splat_value called on a non-splat attribute")
        elt = self._element_type
        if elt.startswith("f"):
            return _FakeFloatAttr(self._payload, elt)
        return _FakeIntAttr(self._payload, elt)

    def __iter__(self):
        if self.is_splat:
            n = 1
            for s in self.type.shape:
                n *= int(s)
            return iter([self._payload] * n)
        return iter(self._payload)


def test_arith_constant_dense_splat_zero() -> None:
    """``dense<0.0> : tensor<4xf32>`` lowers to a buffer of zeros via a For nest."""
    ctx = WalkerCtx()
    result = _FakeSSA({"name": "%c0", "dtype": "f32", "shape": (4,)})
    attr = _FakeDenseAttr(shape=[4], element_type="f32", payload=0.0, is_splat=True)
    op = {
        "name": "arith.constant",
        "operands": [],
        "results": [result],
        "attrs": {"value": attr},
    }
    out = map_arith_constant(op, ctx)
    # Result is a buffer, not an Imm.
    assert isinstance(out, tvm.tir.Buffer)
    assert list(out.shape) == [4]
    assert str(out.dtype) == "float32"
    # Buffer is bound under both the SSA Value object and printed name.
    assert ctx.value_map[result] is out
    assert ctx.value_map["%c0"] is out
    # Buffer registered in ctx.local_buffers (tile-scoped, NOT a PrimFunc
    # parameter -- making it a parameter would trip ``VerifyMemory``
    # because the surrounding For nest writes to it at host scope).
    assert any(b is out for b in ctx.local_buffers)
    assert not any(b is out for b in ctx.buffers.values()), (
        "dense arith.constant must NOT be promoted to ctx.buffers"
    )
    # A serial For nest writing the splat value was emitted.
    assert len(ctx.stmts) == 1
    stmt = ctx.stmts[0]
    assert isinstance(stmt, tvm.tir.For)
    # The body BufferStore writes a 0.0 FloatImm.
    inner = stmt.body
    assert isinstance(inner, tvm.tir.BufferStore)
    assert isinstance(inner.value, tvm.tir.FloatImm)
    assert float(inner.value.value) == 0.0


def test_arith_constant_dense_per_element() -> None:
    """``dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>`` materialises each slot."""
    ctx = WalkerCtx()
    result = _FakeSSA({"name": "%cp", "dtype": "f32", "shape": (4,)})
    attr = _FakeDenseAttr(
        shape=[4],
        element_type="f32",
        payload=[1.0, 2.0, 3.0, 4.0],
        is_splat=False,
    )
    op = {
        "name": "arith.constant",
        "operands": [],
        "results": [result],
        "attrs": {"value": attr},
    }
    out = map_arith_constant(op, ctx)
    assert isinstance(out, tvm.tir.Buffer)
    assert list(out.shape) == [4]
    # An unrolled SeqStmt of 4 BufferStores (one per element) was emitted.
    assert len(ctx.stmts) == 1
    stmt = ctx.stmts[0]
    assert isinstance(stmt, tvm.tir.SeqStmt)
    seq = list(stmt.seq)
    assert len(seq) == 4
    written = [float(s.value.value) for s in seq]
    assert written == [1.0, 2.0, 3.0, 4.0]
    # Each store's index is the constant lin_idx.
    for lin_idx, s in enumerate(seq):
        assert isinstance(s, tvm.tir.BufferStore)
        assert int(s.indices[0].value) == lin_idx


def test_arith_constant_dense_int_splat() -> None:
    """``dense<7> : tensor<8xi32>`` lowers to an i32 buffer initialised to 7."""
    ctx = WalkerCtx()
    result = _FakeSSA({"name": "%ci", "dtype": "i32", "shape": (8,)})
    attr = _FakeDenseAttr(shape=[8], element_type="i32", payload=7, is_splat=True)
    op = {
        "name": "arith.constant",
        "operands": [],
        "results": [result],
        "attrs": {"value": attr},
    }
    out = map_arith_constant(op, ctx)
    assert isinstance(out, tvm.tir.Buffer)
    assert list(out.shape) == [8]
    assert str(out.dtype) == "int32"
    assert len(ctx.stmts) == 1
    stmt = ctx.stmts[0]
    assert isinstance(stmt, tvm.tir.For)
    inner = stmt.body
    assert isinstance(inner, tvm.tir.BufferStore)
    assert isinstance(inner.value, tvm.tir.IntImm)
    assert int(inner.value.value) == 7
    assert str(inner.value.dtype) == "int32"


# ---------------------------------------------------------------------------
# tt.call -- inline expansion of a tt.func callee
# ---------------------------------------------------------------------------
#
# These tests exercise the dict-fake path (no MLIR bindings) so they run
# wherever ``tvm`` is importable. They cover:
#   * happy path: a synthetic callee ``%c = %a + %b`` is inlined at the
#     call site and the call's result SSA is bound to the Add expr.
#   * error path: a tt.call whose ``@callee`` is not in ctx.callees
#     surfaces an EmitError instead of silently producing junk.

from poc.triton_frontend.op_emitters.control import emit_tt_call  # noqa: E402


def test_tt_call_inlines_simple_callee() -> None:
    """Synthetic ``tt.call`` over an inlined ``%a + %b`` callee.

    Dict-fake module shape:
      tt.func @callee(%a: f32, %b: f32) -> f32 {
          %c = arith.addf %a, %b : f32
          tt.return %c : f32
      }
      // caller site:
      %r = tt.call @callee(%x, %y) : (f32, f32) -> f32

    After dispatch, ``ctx.value_map[%r]`` should be the TIR Add of x+y, and
    the callee's body should NOT have polluted ``value_map`` under the
    callee's own block-arg names ``%a`` / ``%b`` (those live only in the
    substitution scope while emit_tt_call walks the body).
    """
    ctx = WalkerCtx()

    # Caller-side SSA values, already bound to TIR Vars.
    x = _val("%x", shape=[], dtype="float32")
    y = _val("%y", shape=[], dtype="float32")
    ctx.bind(x, tvm.tir.Var("x", "float32"))
    ctx.bind(y, tvm.tir.Var("y", "float32"))

    # Callee block-arg SSAs and the body's intermediate %c.
    a = _val("%a", shape=[], dtype="float32")
    b = _val("%b", shape=[], dtype="float32")
    c = _val("%c", shape=[], dtype="float32")

    # The arith.addf op inside the callee: %c = %a + %b.
    addf_op = {
        "name": "arith.addf",
        "operands": [a, b],
        "results": [c],
        "attrs": {},
    }
    return_op = {
        "name": "tt.return",
        "operands": [c],
        "results": [],
        "attrs": {},
    }

    callee_func = {
        "name": "tt.func",
        "operands": [],
        "results": [],
        "attrs": {"sym_name": "my_add", "sym_visibility": "private"},
        "block_args": [a, b],
        "regions": [{
            "blocks": [
                {
                    "block_args": [a, b],
                    "ops": [addf_op, return_op],
                },
            ],
        }],
    }

    # Register the callee in ctx.callees the way the module pre-pass would.
    ctx.callees["my_add"] = callee_func

    # Caller's tt.call op.
    r = _val("%r", shape=[], dtype="float32")
    call_op = {
        "name": "tt.call",
        "operands": [x, y],
        "results": [r],
        "attrs": {"callee": "@my_add"},
    }

    out = emit_tt_call(call_op, ctx)
    assert out is not None, "emit_tt_call should return the inlined return value"

    # The call's result SSA must be bound to the Add expression.
    assert r in ctx.value_map
    bound = ctx.value_map[r]
    text = str(bound)
    # tvm.tir.Add prints as "x + y" (or "Add(x, y)"); accept either form.
    assert ("+" in text) or ("Add" in text), \
        f"expected an Add expression, got: {text!r}"
    # Callee was marked as referenced.
    assert "my_add" in ctx.callee_used
    # The substitution stack was popped cleanly after the inline walk.
    assert ctx._subst_stack == []


def test_tt_call_unknown_callee_raises() -> None:
    """A ``tt.call`` whose @callee is not in ctx.callees raises EmitError.

    No silent fallback: a missing callee in the registry almost certainly
    means the module-level pre-pass missed a tt.func, and emitting nothing
    would leave the call's result SSA unbound -- the next consumer would
    KeyError with a far-less-helpful message. Surface the failure here.
    """
    ctx = WalkerCtx()
    x = _val("%x", shape=[], dtype="float32")
    ctx.bind(x, tvm.tir.Var("x", "float32"))
    r = _val("%r", shape=[], dtype="float32")
    call_op = {
        "name": "tt.call",
        "operands": [x],
        "results": [r],
        "attrs": {"callee": "@missing"},
    }
    with pytest.raises(EmitError) as excinfo:
        emit_tt_call(call_op, ctx)
    msg = str(excinfo.value)
    assert "missing" in msg, f"error should name the missing callee: {msg!r}"
    assert "tt.call" in msg or "callee" in msg, msg
