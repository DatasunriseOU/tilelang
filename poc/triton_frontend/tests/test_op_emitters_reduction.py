"""Unit tests for ``poc.triton_frontend.op_emitters.reduction``.

Each test builds a dict-shaped fake TTIR op (the same pattern used in
``test_dot_reduce_atomic.py`` and ``test_op_emitters_arith.py``) and
checks that:

* ``tt.reduce`` with an ``addf`` combiner emits a ``tir.For`` whose body
  performs ``accum[0] = accum[0] + buf[i]`` and that ``accum[0]`` is
  initialised to 0.
* ``tt.reduce`` with a ``maximumf`` combiner initialises the accumulator
  to ``tir.min_value(dtype)`` (which is -inf for float dtypes).
* ``tt.dot`` (M=N=K=32, fp32) emits either a ``T.gemm`` block (when
  ``tilelang.language.gemm`` is importable) or a 3-loop ``tir.For`` nest
  with a ``BufferStore(C, ...)`` whose RHS is a multiply-accumulate.
* ``tt.atomic_add`` emits a TileLang atomic call (preferred) or a raw
  ``tir.atomic_add`` ``call_intrin``.
* The combiner detector is precise: an unsupported combiner raises
  :class:`EmitError` rather than silently degrading.

Tests skip cleanly when ``tvm`` isn't importable -- the existing
``test_dot_reduce_atomic.py`` follows the same pattern.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

tvm = pytest.importorskip("tvm")

from poc.triton_frontend.op_emitters.reduction import (  # noqa: E402
    EmitError,
    REDUCTION_EMITTERS,
    detect_combiner_kind,
)
from poc.triton_frontend.op_mapping import OP_TABLE, WalkerCtx  # noqa: E402

from ._fixtures import FakeSSA  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ssa(name: str, *, shape: List[int] = (), dtype: str = "float32") -> FakeSSA:
    """Hashable SSA stand-in (delegates to :class:`FakeSSA`)."""
    return FakeSSA(name=name, shape=tuple(shape), dtype=dtype)


def _op(
    name: str,
    operands: List[Any],
    results: List[Any],
    *,
    combiner: str = None,  # type: ignore[assignment]
    **attrs: Any,
) -> Dict[str, Any]:
    op: Dict[str, Any] = {
        "name": name,
        "operands": operands,
        "results": results,
        "attrs": dict(attrs),
    }
    if combiner is not None:
        op["combiner"] = combiner
    return op


def _walk_for_loops(stmt: Any) -> List[Any]:
    """Yield every ``tvm.tir.For`` node in a TIR statement subtree."""
    found: List[Any] = []

    def _visit(node: Any) -> None:
        # Use functional post-order via tvm.tir.stmt_functor.post_order_visit.
        pass

    def _collect(node: Any) -> None:
        if isinstance(node, tvm.tir.For):
            found.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, _collect)
    return found


def _walk_buffer_stores(stmt: Any) -> List[Any]:
    found: List[Any] = []

    def _collect(node: Any) -> None:
        if isinstance(node, tvm.tir.BufferStore):
            found.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, _collect)
    return found


def _seq_or_single(stmts: List[Any]) -> Any:
    """Wrap a stmt list in a SeqStmt if needed; if singleton, return it."""
    if len(stmts) == 1:
        return stmts[0]
    return tvm.tir.SeqStmt(stmts)


# ---------------------------------------------------------------------------
# Combiner detection
# ---------------------------------------------------------------------------


def test_detect_combiner_addf_via_dict_field():
    op = _op("tt.reduce", [], [], combiner="addf")
    assert detect_combiner_kind(op) == "add"


def test_detect_combiner_maximumf_via_region_text():
    op = {
        "name": "tt.reduce",
        "operands": [],
        "results": [],
        "attrs": {},
        "region": "(%a, %b) -> %c { %c = arith.maximumf %a, %b }",
    }
    assert detect_combiner_kind(op) == "max"


def test_detect_combiner_unsupported_raises():
    op = _op("tt.reduce", [], [], combiner="divf")
    with pytest.raises(EmitError, match="unsupported"):
        detect_combiner_kind(op)


def test_detect_combiner_no_clue_raises():
    op = _op("tt.reduce", [], [])  # no combiner, no region
    with pytest.raises(EmitError, match="cannot determine"):
        detect_combiner_kind(op)


# ---------------------------------------------------------------------------
# tt.reduce -- addf and maximumf identities, For-loop shape
# ---------------------------------------------------------------------------


def test_tt_reduce_addf_emits_for_with_zero_init():
    """addf reducer: identity 0; body = accum + src[i]."""
    ctx = WalkerCtx()
    src_ssa = _ssa("buf", shape=[8], dtype="float32")
    out_ssa = _ssa("acc", shape=[], dtype="float32")
    src_buf = tvm.tir.decl_buffer([8], "float32", name="buf")
    ctx.bind(src_ssa, src_buf)

    op = _op("tt.reduce", [src_ssa], [out_ssa], combiner="addf", axis=0)
    REDUCTION_EMITTERS["tt.reduce"](op, ctx)

    body = _seq_or_single(ctx.stmts)
    fors = _walk_for_loops(body)
    assert fors, f"expected at least one tir.For; got {ctx.stmts!r}"

    stores = _walk_buffer_stores(body)
    # Two stores: the identity init and the accumulator update.
    assert len(stores) >= 2, f"expected init + update BufferStore; got {stores!r}"

    # Init store writes zero of the correct dtype.
    init_store = stores[0]
    assert init_store.value.dtype.startswith("float32")
    # Compare structurally to const(0, "float32").
    zero = tvm.tir.const(0, "float32")
    assert tvm.ir.structural_equal(init_store.value, zero), (
        f"init store value {init_store.value!r} != zero {zero!r}"
    )

    # Update store writes Add(accum_load, src_load).
    update_store = stores[-1]
    assert isinstance(update_store.value, tvm.tir.Add), (
        f"expected Add in accumulator update; got {type(update_store.value).__name__}"
    )


def test_tt_reduce_maximumf_emits_neg_inf_init():
    """maximumf reducer: identity = tir.min_value('float32') (-inf)."""
    ctx = WalkerCtx()
    src_ssa = _ssa("buf", shape=[16], dtype="float32")
    out_ssa = _ssa("max", shape=[], dtype="float32")
    src_buf = tvm.tir.decl_buffer([16], "float32", name="buf")
    ctx.bind(src_ssa, src_buf)

    op = _op("tt.reduce", [src_ssa], [out_ssa], combiner="maximumf", axis=0)
    REDUCTION_EMITTERS["tt.reduce"](op, ctx)

    stores = _walk_buffer_stores(_seq_or_single(ctx.stmts))
    init_store = stores[0]
    expected_init = tvm.tir.min_value("float32")
    assert tvm.ir.structural_equal(init_store.value, expected_init), (
        f"init store value {init_store.value!r} != min_value(float32)"
    )

    update_store = stores[-1]
    assert isinstance(update_store.value, tvm.tir.Max), (
        f"expected Max in accumulator update; got {type(update_store.value).__name__}"
    )


def test_tt_reduce_minimumf_emits_pos_inf_init():
    ctx = WalkerCtx()
    src_ssa = _ssa("buf", shape=[16], dtype="float32")
    out_ssa = _ssa("min", shape=[], dtype="float32")
    src_buf = tvm.tir.decl_buffer([16], "float32", name="buf")
    ctx.bind(src_ssa, src_buf)

    op = _op("tt.reduce", [src_ssa], [out_ssa], combiner="minimumf", axis=0)
    REDUCTION_EMITTERS["tt.reduce"](op, ctx)
    stores = _walk_buffer_stores(_seq_or_single(ctx.stmts))
    expected_init = tvm.tir.max_value("float32")
    assert tvm.ir.structural_equal(stores[0].value, expected_init)


def test_tt_reduce_mulf_init_one():
    ctx = WalkerCtx()
    src_ssa = _ssa("buf", shape=[4], dtype="float32")
    out_ssa = _ssa("o", shape=[], dtype="float32")
    src_buf = tvm.tir.decl_buffer([4], "float32", name="buf")
    ctx.bind(src_ssa, src_buf)

    op = _op("tt.reduce", [src_ssa], [out_ssa], combiner="mulf", axis=0)
    REDUCTION_EMITTERS["tt.reduce"](op, ctx)
    stores = _walk_buffer_stores(_seq_or_single(ctx.stmts))
    one = tvm.tir.const(1, "float32")
    assert tvm.ir.structural_equal(stores[0].value, one)


def test_tt_reduce_unsupported_combiner_raises():
    ctx = WalkerCtx()
    src_ssa = _ssa("buf", shape=[8], dtype="float32")
    out_ssa = _ssa("o", shape=[], dtype="float32")
    src_buf = tvm.tir.decl_buffer([8], "float32", name="buf")
    ctx.bind(src_ssa, src_buf)

    op = _op("tt.reduce", [src_ssa], [out_ssa], combiner="divf", axis=0)
    with pytest.raises(EmitError, match="unsupported"):
        REDUCTION_EMITTERS["tt.reduce"](op, ctx)


# ---------------------------------------------------------------------------
# tt.scan
# ---------------------------------------------------------------------------


def test_tt_scan_addf_emits_for_loop_with_running_accumulator():
    ctx = WalkerCtx()
    src_ssa = _ssa("buf", shape=[8], dtype="float32")
    out_ssa = _ssa("scan", shape=[8], dtype="float32")
    src_buf = tvm.tir.decl_buffer([8], "float32", name="buf")
    ctx.bind(src_ssa, src_buf)

    op = _op("tt.scan", [src_ssa], [out_ssa], combiner="addf", axis=0)
    REDUCTION_EMITTERS["tt.scan"](op, ctx)

    body = _seq_or_single(ctx.stmts)
    fors = _walk_for_loops(body)
    assert fors, "expected a tir.For for the scan loop"

    stores = _walk_buffer_stores(body)
    # init + (accum-update + dst-write per iter): >= 3 stores.
    assert len(stores) >= 3
    # First store should be the identity (0).
    zero = tvm.tir.const(0, "float32")
    assert tvm.ir.structural_equal(stores[0].value, zero)


# ---------------------------------------------------------------------------
# tt.dot -- T.gemm preferred; 3-loop nest fallback
# ---------------------------------------------------------------------------


def test_tt_dot_emits_gemm_or_3loop_with_mul_add():
    """tt.dot(M=32, K=32, N=32, fp32) -> T.gemm OR explicit 3-loop nest.

    When ``tilelang.language.gemm`` is importable in the environment we
    require the gemm path; otherwise we accept the manual fallback as
    long as the inner BufferStore on C is a multiply-accumulate.
    """
    ctx = WalkerCtx()
    a_ssa = _ssa("A", shape=[32, 32], dtype="float32")
    b_ssa = _ssa("B", shape=[32, 32], dtype="float32")
    c_ssa = _ssa("C", shape=[32, 32], dtype="float32")
    a_buf = tvm.tir.decl_buffer([32, 32], "float32", name="A")
    b_buf = tvm.tir.decl_buffer([32, 32], "float32", name="B")
    ctx.bind(a_ssa, a_buf)
    ctx.bind(b_ssa, b_buf)
    op = _op("tt.dot", [a_ssa, b_ssa], [c_ssa])

    handle = REDUCTION_EMITTERS["tt.dot"](op, ctx)

    # Path 1: tilelang.gemm path -- the handle string mentions gemm.
    try:
        import tilelang.language as T  # noqa: F401
        gemm_available = True
    except ImportError:
        gemm_available = False

    if gemm_available:
        text = str(handle) + " " + str(ctx.stmts)
        assert "gemm" in text.lower(), (
            f"expected gemm in emitted TIR when tilelang is importable; "
            f"got {text!r}"
        )
        return

    # Path 2: manual 3-loop nest. We expect:
    #  * three nested tir.For nodes,
    #  * a single BufferStore on C whose RHS is Add(BufferLoad(C), Mul(...)).
    body = _seq_or_single(ctx.stmts)
    fors = _walk_for_loops(body)
    assert len(fors) == 3, f"expected 3 nested tir.For; got {len(fors)}"
    stores = _walk_buffer_stores(body)
    assert len(stores) == 1
    store = stores[0]
    assert isinstance(store.value, tvm.tir.Add), (
        f"expected outer Add for accumulate; got {type(store.value).__name__}"
    )
    assert isinstance(store.value.b, tvm.tir.Mul), (
        f"expected inner Mul for A*B; got {type(store.value.b).__name__}"
    )


def test_tt_dot_fp16_requires_tilelang_or_raises():
    """fp16 inputs must route through T.gemm; the manual nest is forbidden."""
    try:
        import tilelang.language as T  # noqa: F401
        # If TileLang is importable, gemm path will succeed regardless.
        return
    except ImportError:
        pass

    ctx = WalkerCtx()
    a_ssa = _ssa("A", shape=[16, 16], dtype="float16")
    b_ssa = _ssa("B", shape=[16, 16], dtype="float16")
    c_ssa = _ssa("C", shape=[16, 16], dtype="float32")
    a_buf = tvm.tir.decl_buffer([16, 16], "float16", name="A")
    b_buf = tvm.tir.decl_buffer([16, 16], "float16", name="B")
    ctx.bind(a_ssa, a_buf)
    ctx.bind(b_ssa, b_buf)
    op = _op("tt.dot", [a_ssa, b_ssa], [c_ssa])

    with pytest.raises(EmitError, match="low-precision"):
        REDUCTION_EMITTERS["tt.dot"](op, ctx)


# ---------------------------------------------------------------------------
# Atomics
# ---------------------------------------------------------------------------


def test_tt_atomic_add_emits_intrinsic_or_tilelang_call():
    ctx = WalkerCtx()
    ptr_ssa = _ssa("ptr", dtype="float32")
    val_ssa = _ssa("val", dtype="float32")
    buf = tvm.tir.decl_buffer([1024], "float32", name="dst")
    val_var = tvm.tir.Var("val", "float32")
    ctx.bind(ptr_ssa, (buf, [0]))
    ctx.bind(val_ssa, val_var)

    op = _op("tt.atomic_add", [ptr_ssa, val_ssa], [])
    handle = REDUCTION_EMITTERS["tt.atomic_add"](op, ctx)
    text = str(handle) + " " + str(ctx.stmts)
    # Either TileLang's atomic_add lowering ("atomicadd"/"atomic_add") or
    # the raw tir.atomic_add intrinsic must appear in the emitted TIR.
    text_l = text.lower()
    assert "atomic_add" in text_l or "atomicadd" in text_l, (
        f"expected atomic_add in emitted TIR; got {text!r}"
    )


def test_tt_atomic_max_dispatches_to_max_emitter():
    ctx = WalkerCtx()
    ptr_ssa = _ssa("ptr", dtype="float32")
    val_ssa = _ssa("val", dtype="float32")
    buf = tvm.tir.decl_buffer([1024], "float32", name="dst")
    val_var = tvm.tir.Var("val", "float32")
    ctx.bind(ptr_ssa, (buf, [0]))
    ctx.bind(val_ssa, val_var)

    op = _op("tt.atomic_max", [ptr_ssa, val_ssa], [])
    handle = REDUCTION_EMITTERS["tt.atomic_max"](op, ctx)
    text_l = (str(handle) + " " + str(ctx.stmts)).lower()
    assert "atomic_max" in text_l or "atomicmax" in text_l


def test_tt_atomic_cas_uses_call_intrin_path():
    """CAS isn't on the TileLang surface; we always use call_intrin."""
    ctx = WalkerCtx()
    ptr_ssa = _ssa("ptr", dtype="int32")
    cmp_ssa = _ssa("cmp", dtype="int32")
    new_ssa = _ssa("new", dtype="int32")
    buf = tvm.tir.decl_buffer([1024], "int32", name="dst")
    cmp_var = tvm.tir.Var("cmp", "int32")
    new_var = tvm.tir.Var("new", "int32")
    ctx.bind(ptr_ssa, (buf, [0]))
    ctx.bind(cmp_ssa, cmp_var)
    ctx.bind(new_ssa, new_var)

    op = _op("tt.atomic_cas", [ptr_ssa, cmp_ssa, new_ssa], [])
    handle = REDUCTION_EMITTERS["tt.atomic_cas"](op, ctx)
    text_l = (str(handle) + " " + str(ctx.stmts)).lower()
    assert "atomic_cas" in text_l


# ---------------------------------------------------------------------------
# Registry merge
# ---------------------------------------------------------------------------


def test_op_table_contains_reduction_emitters():
    """OP_TABLE.update(REDUCTION_EMITTERS) registered every entry."""
    expected_names = {
        "tt.reduce",
        "tt.scan",
        "tt.dot",
        "tt.atomic_add",
        "tt.atomic_max",
        "tt.atomic_min",
        "tt.atomic_xchg",
        "tt.atomic_cas",
    }
    for name in expected_names:
        assert name in OP_TABLE, f"missing {name!r} in OP_TABLE"
        assert OP_TABLE[name] is REDUCTION_EMITTERS[name], (
            f"OP_TABLE[{name!r}] not pointing at reduction emitter"
        )
