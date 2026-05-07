"""Smoke tests for ``tt.sync_threads_partial`` lowering (Phase-1 migration).

Exercises the dict-shaped fake-op path so we don't need MLIR bindings.
The walker code in ``poc.triton_frontend.op_mapping`` accepts either a
real ``mlir.ir.Operation`` or a dict shape with ``operands`` / ``results``
/ ``attrs``. The cppmega.mlx topk_selector kernel emits
``tl.sync_threads_partial(mask, n)`` so radix histogram lanes can wait on
each other without a full block barrier.
"""
from __future__ import annotations

import pytest


def _om():
    try:
        from poc.triton_frontend import op_mapping
    except Exception as exc:
        pytest.skip(f"poc.triton_frontend.op_mapping unavailable: {exc!r}")
    return op_mapping


def test_op_table_registers_three_aliases():
    om = _om()
    expected = (
        "tt.sync_threads_partial",
        "tt.partial_barrier",
        "triton.language.partial_barrier",
    )
    for alias in expected:
        assert alias in om.OP_TABLE, f"OP_TABLE missing {alias!r}"
        assert om.OP_TABLE[alias] is om.map_tt_sync_threads_partial


def test_map_tt_sync_threads_partial_emits_one_handle():
    om = _om()
    try:
        import tilelang.language  # noqa: F401
    except Exception as exc:
        pytest.skip(f"tilelang unavailable: {exc!r}")

    ctx = om.WalkerCtx()
    ctx.bind("%mask", 0xFFFFFFFF)
    ctx.bind("%n", 32)

    op = {
        "operands": ["%mask", "%n"],
        "results": [],
        "attrs": {},
    }
    handle = om.map_tt_sync_threads_partial(op, ctx)

    assert handle is not None
    assert len(ctx.stmts) == 1
    # Either the high-level T.sync_threads_partial wrapper or the raw
    # call_intrin fallback — both produce a tir.Call carrying the
    # ``tl.sync_threads_partial`` Op.
    rendered = repr(ctx.stmts[0])
    assert "sync_threads_partial" in rendered.lower()


def test_map_tt_sync_threads_partial_rejects_short_operands():
    om = _om()
    ctx = om.WalkerCtx()
    op = {"operands": ["%mask"], "results": [], "attrs": {}}
    with pytest.raises(ValueError, match="expected"):
        om.map_tt_sync_threads_partial(op, ctx)
