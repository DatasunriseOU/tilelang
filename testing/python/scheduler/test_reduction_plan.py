from __future__ import annotations

import json

import pytest
from tvm import tir
from tvm.target import Target

from tilelang.analysis.backend_lowerer_selection import (
    attach_reduction_backend_lowerer_metadata,
    build_reduction_backend_lowerer_diagnostics,
)
from tilelang.analysis.reduction_plan import (
    ReductionPlanError,
    attach_reduction_plan_metadata,
    candidate_strategies_for_extent,
    extract_reduction_plans,
    selected_strategy_for_extent,
)
from tilelang.backend.reduction import select_reduction_lowerer


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
    assert plan.selected_strategy == "split-simdgroup"
    assert plan.thread_mapping.threads_per_threadgroup == 64
    assert plan.thread_mapping.blocks_per_output == 1
    assert plan.alias_constraints.constraint == "distinct_input_output_buffers"
    assert plan.memory_plan.scratch_scope == "threadgroup"


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
        "threadgroup-staging",
        "row-reduce",
        "vectorized-cpu-fallback",
    )
    assert candidate_strategies_for_extent(64) == (
        "split-simdgroup",
        "threadgroup-staging",
        "row-reduce",
        "vectorized-cpu-fallback",
    )
    assert candidate_strategies_for_extent(96) == (
        "split-simdgroup",
        "threadgroup-staging",
        "row-reduce",
        "vectorized-cpu-fallback",
    )
    assert candidate_strategies_for_extent(128) == (
        "split-simdgroup",
        "threadgroup-staging",
        "row-reduce",
        "vectorized-cpu-fallback",
    )
    assert candidate_strategies_for_extent(256) == (
        "split-simdgroup",
        "threadgroup-staging",
        "row-reduce",
        "vectorized-cpu-fallback",
    )
    assert candidate_strategies_for_extent(512) == (
        "two-pass-global",
        "vectorized-cpu-fallback",
    )
    assert selected_strategy_for_extent(32) == "same-simdgroup"
    assert selected_strategy_for_extent(64) == "split-simdgroup"
    assert selected_strategy_for_extent(256) == "split-simdgroup"
    assert selected_strategy_for_extent(512) == "two-pass-global"


def test_reduction_plan_metadata_serializes_stably():
    func = attach_reduction_plan_metadata(_make_thread_allreduce_func(128))
    raw = func.attrs["tl.reduction_plans"].value
    payload = json.loads(raw)
    assert payload == [
        {
            "accumulator_dtype": "float32",
            "alias_constraints": {
                "constraint": "distinct_input_output_buffers",
                "in_place_allowed": False,
                "input_buffer_names": ["src"],
                "may_alias": False,
                "output_buffer_name": "dst",
            },
            "aliasing_allowed": False,
            "axes": [
                {
                    "expr": "lane % 128",
                    "extent": 128,
                    "name": "lane",
                    "role": "lane",
                }
            ],
            "candidate_strategies": [
                "split-simdgroup",
                "threadgroup-staging",
                "row-reduce",
                "vectorized-cpu-fallback",
            ],
            "in_place": False,
            "input_regions": [
                {
                    "dtype": "float32",
                    "indices": ["lane"],
                    "name": "src",
                    "role": "read",
                }
            ],
            "memory_plan": {
                "external_materialization_required": False,
                "internal_scratch_required": True,
                "scratch_scope": "threadgroup",
                "visibility_scope": "threadgroup",
            },
            "memory_visibility_scope": "threadgroup",
            "op": "sum",
            "output_region": {
                "dtype": "float32",
                "indices": ["0"],
                "name": "dst",
                "role": "write",
            },
            "predicate": "T.bool(True)",
            "selected_strategy": "split-simdgroup",
            "thread_mapping": {
                "axis": "lane",
                "blocks_per_output": 1,
                "reduction_extent": 128,
                "selected_strategy": "split-simdgroup",
                "simdgroup_size": 32,
                "simdgroups_per_threadgroup": 4,
                "threads_per_threadgroup": 128,
            },
        }
    ]


def test_reduction_plan_large_extent_selects_internal_two_pass_metadata():
    plan = extract_reduction_plans(_make_thread_allreduce_func(512))[0]
    assert plan.selected_strategy == "two-pass-global"
    assert plan.thread_mapping.blocks_per_output == 2
    assert plan.thread_mapping.threads_per_threadgroup == 256
    assert plan.memory_plan.scratch_scope == "device"
    assert plan.memory_plan.internal_scratch_required is True
    assert plan.memory_plan.external_materialization_required is False


def test_backend_reduction_registry_selects_cached_metal_lowerers():
    first = select_reduction_lowerer(
        Target("metal"),
        op="sum",
        strategy="same-simdgroup",
        reduction_extent=32,
        accumulator_dtype="float32",
    )
    second = select_reduction_lowerer(
        Target("metal"),
        op="sum",
        strategy="same-simdgroup",
        reduction_extent=32,
        accumulator_dtype="float32",
    )
    assert first is second
    assert first.name == "metal.same-simdgroup.sum"
    assert first.lowerer == "tirx.metal.simd_sum"
    assert first.memory_visibility_scope == "simdgroup"
    assert first.scratch_scope is None

    split = select_reduction_lowerer(
        Target("metal"),
        op="sum",
        strategy="split-simdgroup",
        reduction_extent=64,
        accumulator_dtype="float32",
    )
    assert split.name == "metal.split-simdgroup"
    assert split.memory_visibility_scope == "threadgroup"
    assert split.scratch_scope == "threadgroup"


def test_backend_lowerer_metadata_names_selected_registry_entry():
    func = attach_reduction_backend_lowerer_metadata(
        _make_thread_allreduce_func(128),
        Target("metal"),
    )
    payload = json.loads(func.attrs["tl.reduction_backend_lowerers"].value)
    assert payload == [
        {
            "backend": "metal",
            "cache_key": "metal|sum|split-simdgroup|128|float32",
            "external_materialization_required": False,
            "internal_scratch_required": True,
            "lowerer": "metal.thread_allreduce.threadgroup_staging",
            "lowerer_name": "metal.split-simdgroup",
            "memory_visibility_scope": "threadgroup",
            "notes": "cross-simdgroup reductions stage partials in threadgroup memory",
            "op": "sum",
            "plan_selected_strategy": "split-simdgroup",
            "scratch_scope": "threadgroup",
            "selected_strategy": "split-simdgroup",
            "source": "reduction:0:sum",
            "target_kind": "metal",
        }
    ]


def test_backend_lowerer_metadata_falls_back_to_cpu_candidate():
    diagnostics = build_reduction_backend_lowerer_diagnostics(
        _make_thread_allreduce_func(32),
        Target("llvm"),
    )
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.plan_selected_strategy == "same-simdgroup"
    assert diagnostic.selected_strategy == "vectorized-cpu-fallback"
    assert diagnostic.backend == "cpu"
    assert diagnostic.lowerer_name == "cpu.vectorized-fallback"
