from __future__ import annotations

import json

from tvm import tir

from tilelang.analysis.cost_model import (
    attach_reduction_cost_metadata,
    build_reduction_cost_estimates,
    estimate_recurrence_scan_cost,
)
from tilelang.analysis.scan_plan import plan_recurrence_scan


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


def test_reduction_cost_metadata_explains_inline_simdgroup_choice():
    estimates = build_reduction_cost_estimates(_make_thread_allreduce_func(32))

    assert len(estimates) == 1
    estimate = estimates[0]
    assert estimate.selected_strategy == "same-simdgroup"
    assert estimate.reduction_extent == 32
    assert estimate.threadgroup_memory_bytes == 0
    assert estimate.device_memory_bytes_per_output == 0
    assert estimate.dispatch_count == 1
    assert estimate.sync_count == 0
    assert estimate.occupancy_limit == "not-limited"
    assert estimate.split_or_inline_decision == "inline-simdgroup"


def test_reduction_cost_metadata_explains_threadgroup_staging_costs():
    estimates = build_reduction_cost_estimates(_make_thread_allreduce_func(128))
    estimate = estimates[0]

    assert estimate.selected_strategy == "split-simdgroup"
    assert estimate.threadgroup_memory_bytes == 128 * 4
    assert estimate.materialization_cost == "threadgroup-scratch"
    assert estimate.dispatch_count == 1
    assert estimate.sync_count == 2
    assert estimate.split_or_inline_decision == "inline-single-dispatch"


def test_reduction_cost_metadata_explains_two_pass_split():
    estimates = build_reduction_cost_estimates(_make_thread_allreduce_func(512))
    estimate = estimates[0]

    assert estimate.selected_strategy == "two-pass-global"
    assert estimate.device_memory_bytes_per_output == 2 * 4
    assert estimate.dispatch_count == 2
    assert estimate.sync_count == 3
    assert estimate.materialization_cost == "internal-device-scratch"
    assert estimate.occupancy_limit == "dispatch-count"
    assert estimate.split_or_inline_decision == "split-two-pass"


def test_reduction_cost_metadata_serializes_stably():
    func = attach_reduction_cost_metadata(_make_thread_allreduce_func(128))
    payload = json.loads(func.attrs["tl.reduction_costs"].value)

    assert payload == [
        {
            "accumulator_dtype": "float32",
            "device_memory_bytes_per_output": 0,
            "dispatch_count": 1,
            "estimated_registers_per_thread": 8,
            "index_math_ops_per_output": 1,
            "local_memory_bytes_per_thread": 0,
            "materialization_cost": "threadgroup-scratch",
            "occupancy_limit": "not-limited",
            "op": "sum",
            "reason": "extent fits one threadgroup; stage partials internally",
            "reduction_extent": 128,
            "selected_strategy": "split-simdgroup",
            "source": "reduction:0:sum",
            "split_or_inline_decision": "inline-single-dispatch",
            "sync_count": 2,
            "threadgroup_memory_bytes": 512,
        }
    ]


def test_scan_cost_metadata_explains_snapshot_reuse():
    plan = plan_recurrence_scan(
        name="mamba3_bwd",
        direction="reverse",
        sequence_length=8,
        state_shape=(1, 2, 64, 32),
        state_dtype="float32",
        chunk_size=1,
        decay_may_underflow=True,
        fused_post_ops=("skip_D", "silu_gate"),
    )
    estimate = estimate_recurrence_scan_cost(plan, source="scan:0:mamba3_bwd")

    assert estimate.dispatch_count == 2
    assert estimate.sync_count == 0
    assert estimate.state_bytes == 1 * 2 * 64 * 32 * 4
    assert estimate.snapshot_bytes == 1 * 2 * 64 * 32 * 9 * 4
    assert estimate.estimated_registers_per_thread == 8
    assert estimate.materialization_cost == "state-boundary-cache"
    assert estimate.split_or_inline_decision == "split-snapshot-reuse"
