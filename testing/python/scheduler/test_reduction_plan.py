from __future__ import annotations

import json

import pytest
from tvm import tir

from tilelang.analysis.reduction_plan import (
    ReductionPlanError,
    attach_reduction_plan_metadata,
    candidate_strategies_for_extent,
    extract_reduction_plans,
)


def _make_thread_allreduce_func(extent: int) -> tir.PrimFunc:
    src = tir.decl_buffer((extent,), "float32", name="src")
    dst = tir.decl_buffer((1,), "float32", name="dst")
    lane = tir.Var("lane", "int32")
    reduce_index = lane % tir.IntImm("int32", extent)
    reducer = tir.comm_reducer(
        lambda x, y: x + y,
        lambda dtype: tir.const(0, dtype=dtype),
        name="sum",
    )
    call = tir.call_intrin(
        "handle",
        "tir.tvm_thread_allreduce",
        tir.const(1, "uint32"),
        tir.BufferLoad(src, [lane]),
        tir.const(True, "bool"),
        tir.BufferLoad(dst, [0]),
        reduce_index,
    )
    body = tir.AttrStmt(
        reducer,
        "reduce_scope",
        tir.reinterpret("handle", tir.const(0, "uint64")),
        tir.Evaluate(call),
    )
    return tir.PrimFunc([], body)


def _make_tileop_reduce_func(
    kind: str = "sum",
    extent: int = 32,
    *,
    dim: int = 0,
) -> tir.PrimFunc:
    src = tir.decl_buffer((extent,), "float32", name="src")
    dst = tir.decl_buffer((1,), "float32", name="dst")
    src_region = tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.region"),
        tir.BufferLoad(src, [0]),
        tir.const(1, "int32"),
        tir.const(extent, "int32"),
    )
    dst_region = tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.region"),
        tir.BufferLoad(dst, [0]),
        tir.const(2, "int32"),
        tir.const(1, "int32"),
    )
    reduce_call = tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.reduce"),
        src_region,
        dst_region,
        tir.StringImm(kind),
        tir.const(dim, "int32"),
        tir.const(True, "bool"),
    )
    return tir.PrimFunc([], tir.Evaluate(reduce_call))


def test_reduction_plan_extracts_thread_allreduce_regions():
    plans = extract_reduction_plans(_make_thread_allreduce_func(64))
    assert len(plans) == 1
    plan = plans[0]
    assert plan.op == "sum"
    assert plan.predicate == "T.bool(True)"
    assert plan.accumulator_dtype == "float32"
    assert plan.input_regions[0].name == "src"
    assert plan.input_regions[0].indices == ("lane",)
    assert plan.output_region.name == "dst"
    assert plan.axes[0].extent == 64
    assert plan.axes[0].role == "lane"


def test_reduction_plan_extracts_tileop_reduce_sum_and_max():
    for kind in ("sum", "max"):
        plans = extract_reduction_plans(_make_tileop_reduce_func(kind, 32))
        assert len(plans) == 1
        plan = plans[0]
        assert plan.op == kind
        assert plan.predicate == "T.bool(True)"
        assert plan.accumulator_dtype == "float32"
        assert plan.input_regions[0].name == "src"
        assert plan.input_regions[0].indices == ("0",)
        assert plan.output_region.name == "dst"
        assert plan.axes[0].extent == 32
        assert plan.axes[0].role == "block"
        assert plan.candidate_strategies[0] == "same-simdgroup"


def test_reduction_plan_rejects_malformed_tileop_reduce_axis_before_codegen():
    with pytest.raises(
        ReductionPlanError,
        match="malformed tl\\.tileop\\.reduce axis dim=1",
    ):
        extract_reduction_plans(_make_tileop_reduce_func("sum", 32, dim=1))


def test_reduction_plan_strategy_selection_by_extent():
    assert candidate_strategies_for_extent(32) == (
        "same-simdgroup",
        "split-simdgroup",
        "threadgroup",
    )
    assert candidate_strategies_for_extent(64) == (
        "split-simdgroup",
        "threadgroup",
    )
    assert candidate_strategies_for_extent(96) == (
        "split-simdgroup",
        "threadgroup",
    )
    assert candidate_strategies_for_extent(128) == (
        "split-simdgroup",
        "threadgroup",
    )
    assert candidate_strategies_for_extent(256) == (
        "split-simdgroup",
        "threadgroup",
    )
    assert candidate_strategies_for_extent(512) == (
        "two-pass-global",
        "vectorized-cpu-fallback",
    )


def test_reduction_plan_metadata_serializes_stably():
    func = attach_reduction_plan_metadata(_make_thread_allreduce_func(128))
    raw = func.attrs["tl.reduction_plans"].value
    payload = json.loads(raw)
    assert payload == [
        {
            "accumulator_dtype": "float32",
            "aliasing_allowed": False,
            "axes": [
                {
                    "expr": "lane % 128",
                    "extent": 128,
                    "name": "lane",
                    "role": "lane",
                }
            ],
            "candidate_strategies": ["split-simdgroup", "threadgroup"],
            "in_place": False,
            "input_regions": [
                {
                    "dtype": "float32",
                    "indices": ["lane"],
                    "name": "src",
                    "role": "read",
                }
            ],
            "memory_visibility_scope": "threadgroup",
            "op": "sum",
            "output_region": {
                "dtype": "float32",
                "indices": ["0"],
                "name": "dst",
                "role": "write",
            },
            "predicate": "T.bool(True)",
        }
    ]
