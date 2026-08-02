from __future__ import annotations

import json
import os

from tvm import tir

from tilelang.analysis.reduction_legality import (
    _Z3_AVAILABLE,
    attach_reduction_legality_metadata,
    prove_reduction_plan_legality,
)
from tilelang.analysis.reduction_plan import extract_reduction_plans


def _z3_legality_enabled() -> bool:
    if not _Z3_AVAILABLE:
        return False
    for var in ("TILELANG_DISABLE_Z3", "TILELANG_DISABLE_Z3_REDUCTION_LEGALITY"):
        value = os.environ.get(var, "")
        if value and value != "0":
            return False
    return True


def _make_thread_allreduce_func(
    extent: int,
    *,
    static_extent: bool = True,
    alias_output: bool = False,
    extent_dtype: str = "int32",
) -> tir.PrimFunc:
    src = tir.decl_buffer((max(extent, 1),), "float32", name="src")
    dst = tir.decl_buffer((1,), "float32", name="dst")
    lane = tir.Var("lane", "int32")
    reduce_index = lane % tir.IntImm(extent_dtype, extent) if static_extent else lane
    input_load = tir.BufferLoad(dst if alias_output else src, [0 if alias_output else lane])
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
        reduce_index,
    )
    body = tir.AttrStmt(
        reducer,
        "reduce_scope",
        tir.reinterpret("handle", tir.const(0, "uint64")),
        tir.Evaluate(call),
    )
    return tir.PrimFunc([], body)


def _first_plan(func: tir.PrimFunc):
    plans = extract_reduction_plans(func)
    assert len(plans) == 1
    return plans[0]


def test_same_simdgroup_reduction_proves_no_sync():
    proof = prove_reduction_plan_legality(_first_plan(_make_thread_allreduce_func(32)))
    assert proof.proved_exact_coverage is True
    assert proof.proved_no_oob is True
    assert proof.proved_no_sync is True
    assert proof.requires_threadgroup_barrier is False
    assert proof.requires_device_event is False
    assert proof.requires_two_pass is False
    assert proof.cannot_parallelize_reason is None
    assert proof.z3_proved is _z3_legality_enabled()


def test_split_simdgroup_reduction_requires_threadgroup_barrier_only():
    proof = prove_reduction_plan_legality(_first_plan(_make_thread_allreduce_func(128)))
    assert proof.proved_exact_coverage is True
    assert proof.proved_tail_broadcast_legal is True
    assert proof.proved_no_sync is False
    assert proof.requires_threadgroup_barrier is True
    assert proof.requires_device_event is False
    assert proof.requires_two_pass is False


def test_tail_extent_between_simdgroups_proves_bounds_with_barrier():
    proof = prove_reduction_plan_legality(_first_plan(_make_thread_allreduce_func(33)))
    assert proof.proved_exact_coverage is True
    assert proof.proved_no_oob is True
    assert proof.proved_tail_broadcast_legal is True
    assert proof.proved_index_width_safe is True
    assert proof.requires_threadgroup_barrier is True
    assert proof.requires_device_event is False
    assert proof.cannot_parallelize_reason is None


def test_large_reduction_requires_two_pass_device_edge():
    proof = prove_reduction_plan_legality(_first_plan(_make_thread_allreduce_func(512)))
    assert proof.proved_exact_coverage is True
    assert proof.proved_tail_broadcast_legal is True
    assert proof.requires_threadgroup_barrier is False
    assert proof.requires_device_event is True
    assert proof.requires_two_pass is True


def test_extent_over_int32_blocks_index_width_proof():
    proof = prove_reduction_plan_legality(_first_plan(_make_thread_allreduce_func(1 << 31, extent_dtype="int64")))
    assert proof.proved_exact_coverage is False
    assert proof.proved_no_oob is False
    assert proof.proved_index_width_safe is False
    assert proof.proved_tail_broadcast_legal is False
    assert proof.cannot_parallelize_reason == "extent_legality_unproved"
    assert "exceeds int32 index limit" in proof.query


def test_missing_static_extent_blocks_parallel_proof():
    proof = prove_reduction_plan_legality(_first_plan(_make_thread_allreduce_func(64, static_extent=False)))
    assert proof.proved_exact_coverage is False
    assert proof.proved_no_sync is False
    assert proof.cannot_parallelize_reason == "missing_static_axis_extent"


def test_input_output_alias_requires_explicit_in_place_plan():
    proof = prove_reduction_plan_legality(_first_plan(_make_thread_allreduce_func(32, alias_output=True)))
    assert proof.proved_exact_coverage is False
    assert proof.proved_no_read_after_write_hazard is False
    assert proof.proved_in_place_legal is False
    assert proof.cannot_parallelize_reason == "input_output_alias_without_in_place_plan"


def test_reduction_legality_metadata_serializes_decisions():
    func = attach_reduction_legality_metadata(_make_thread_allreduce_func(128))
    payload = json.loads(func.attrs["tl.reduction_legality"].value)
    assert payload[0]["proved_exact_coverage"] is True
    assert payload[0]["proved_no_sync"] is False
    assert payload[0]["requires_threadgroup_barrier"] is True
    assert payload[0]["requires_device_event"] is False
    assert payload[0]["requires_two_pass"] is False
    assert payload[0]["cannot_parallelize_reason"] is None
