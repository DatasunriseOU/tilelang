from __future__ import annotations

import json

from tvm import tir

from tilelang.analysis.reduction_plan import (
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


def test_reduction_plan_extracts_thread_allreduce_regions():
    plans = extract_reduction_plans(_make_thread_allreduce_func(64))
    assert len(plans) == 1
    plan = plans[0]
    assert plan.op == "sum"
    assert plan.accumulator_dtype == "float32"
    assert plan.input_regions[0].name == "src"
    assert plan.input_regions[0].indices == ("lane",)
    assert plan.output_region.name == "dst"
    assert plan.axes[0].extent == 64


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
        }
    ]
