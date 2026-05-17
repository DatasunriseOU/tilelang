from __future__ import annotations

import json

from tvm import tir

from tilelang.analysis.sync_event_plan import (
    attach_sync_event_plan_metadata,
    build_reduction_sync_event_plan,
)


def _make_thread_allreduce_func(
    extent: int,
    *,
    alias_output: bool = False,
) -> tir.PrimFunc:
    src = tir.decl_buffer((max(extent, 1),), "float32", name="src")
    dst = tir.decl_buffer((1,), "float32", name="dst")
    lane = tir.Var("lane", "int32")
    input_load = tir.BufferLoad(
        dst if alias_output else src,
        [0 if alias_output else lane],
    )
    reducer = tir.comm_reducer(
        lambda x, y: x + y,
        lambda dtype: tir.const(0, dtype=dtype),
        name="sum",
    )
    call = tir.call_intrin(
        "handle",
        "tir.tvm_thread_allreduce",
        tir.const(1, "uint32"),
        input_load,
        tir.const(True, "bool"),
        tir.BufferLoad(dst, [0]),
        lane % tir.IntImm("int32", extent),
    )
    body = tir.AttrStmt(
        reducer,
        "reduce_scope",
        tir.reinterpret("handle", tir.const(0, "uint64")),
        tir.Evaluate(call),
    )
    return tir.PrimFunc([], body)


def _first_decision(extent: int):
    decisions = build_reduction_sync_event_plan(_make_thread_allreduce_func(extent))
    assert len(decisions) == 1
    return decisions[0]


def test_same_simdgroup_reduction_requires_no_sync_or_materialization():
    decision = _first_decision(32)
    assert decision.action == "none"
    assert decision.reason == "proved_no_sync"
    assert decision.host_sync_required is False
    assert decision.threadgroup_barrier_required is False
    assert decision.device_event_required is False
    assert decision.selected_strategy == "same-simdgroup"
    assert decision.memory_visibility_scope == "simdgroup"
    assert decision.scratch_scope is None
    assert decision.internal_scratch_required is False
    assert decision.external_materialization_required is False


def test_split_reduction_requires_threadgroup_barrier_only():
    decision = _first_decision(128)
    assert decision.action == "threadgroup_barrier"
    assert decision.threadgroup_barrier_required is True
    assert decision.device_event_required is False
    assert decision.host_sync_required is False
    assert decision.selected_strategy == "split-simdgroup"
    assert decision.memory_visibility_scope == "threadgroup"
    assert decision.scratch_scope == "threadgroup"
    assert decision.internal_scratch_required is True
    assert decision.external_materialization_required is False


def test_large_reduction_requires_internal_two_pass_device_event():
    decision = _first_decision(512)
    assert decision.action == "device_event"
    assert decision.device_event_required is True
    assert decision.two_pass_required is True
    assert decision.selected_strategy == "two-pass-global"
    assert decision.memory_visibility_scope == "device"
    assert decision.scratch_scope == "device"
    assert decision.internal_scratch_required is True
    assert decision.external_materialization_required is False


def test_illegal_alias_plan_is_rejected_before_codegen():
    decisions = build_reduction_sync_event_plan(
        _make_thread_allreduce_func(32, alias_output=True)
    )
    assert decisions[0].action == "reject"
    assert decisions[0].reason == "input_output_alias_without_in_place_plan"
    assert decisions[0].host_sync_required is False


def test_sync_event_plan_metadata_is_inspectable_json():
    func = attach_sync_event_plan_metadata(_make_thread_allreduce_func(128))
    payload = json.loads(func.attrs["tl.sync_event_plan"].value)
    assert payload[0]["source"] == "reduction:0:sum"
    assert payload[0]["action"] == "threadgroup_barrier"
    assert payload[0]["selected_strategy"] == "split-simdgroup"
    assert payload[0]["memory_visibility_scope"] == "threadgroup"
    assert payload[0]["scratch_scope"] == "threadgroup"
    assert payload[0]["threadgroup_barrier_required"] is True
    assert payload[0]["internal_scratch_required"] is True
    assert payload[0]["external_materialization_required"] is False
