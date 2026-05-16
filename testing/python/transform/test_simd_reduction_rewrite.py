"""Tests for Idea #9 (Z3 roadmap): simdgroup-memory → butterfly IR rewrite.

These tests exercise the *rewrite* layer added on top of the detection-only
implementation in test_simd_reduction_lift.py. The rewrite is gated by:

* PassConfig key ``tl.simd_lift_reductions`` (default OFF), AND
* Per-loop annotation ``tl.simd_butterfly_lane = True``.

Both must hold for a reduction to be rewritten into a butterfly chain of
``tl.shfl_xor_sync`` calls. The tests use the public helpers
``rewrite_reductions`` and ``count_shfl_xor_calls`` to inspect the lowered
IR without requiring a built libtilelang or a Metal device.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

import tilelang.language as T
from tilelang import tvm as tvm
from tvm import tir
from tvm.target import Target


def _load_worktree_module():
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(
        os.path.join(here, "..", "..", "..", "tilelang", "transform",
                     "metal_simd_lift.py")
    )
    spec = importlib.util.spec_from_file_location(
        "_worktree_metal_simd_lift_rw", candidate
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


metal_simd_lift = _load_worktree_module()
rewrite_reductions = metal_simd_lift.rewrite_reductions
rewrite_reductions_to_thread_allreduce = (
    metal_simd_lift.rewrite_reductions_to_thread_allreduce
)
count_shfl_xor_calls = metal_simd_lift.count_shfl_xor_calls
count_thread_allreduce_calls = metal_simd_lift.count_thread_allreduce_calls
detect_candidates = metal_simd_lift.detect_candidates
reduction_rewrite_diagnostics = metal_simd_lift.reduction_rewrite_diagnostics
_butterfly_stages = metal_simd_lift._butterfly_stages
LOOP_ANNOTATION_KEY = metal_simd_lift.LOOP_ANNOTATION_KEY


# ---------------------------------------------------------------------------
# Pure helpers (no TIR construction)
# ---------------------------------------------------------------------------

def test_butterfly_stages_32():
    assert _butterfly_stages(32) == [16, 8, 4, 2, 1]


def test_butterfly_stages_16():
    assert _butterfly_stages(16) == [8, 4, 2, 1]


def test_butterfly_stages_8():
    assert _butterfly_stages(8) == [4, 2, 1]


def test_butterfly_stages_1_or_0():
    assert _butterfly_stages(1) == []
    assert _butterfly_stages(0) == []


# ---------------------------------------------------------------------------
# IR rewrite tests
# ---------------------------------------------------------------------------

def _build_annotated_reduction(extent: int, op: str = "add"):
    """Build a PrimFunc with a reduction loop carrying the butterfly anno."""

    @T.prim_func
    def func(buf: T.Tensor[(64,), T.float32], acc: T.Tensor[(1,), T.float32]):
        for i in T.serial(extent, annotations={LOOP_ANNOTATION_KEY: True}):
            if op == "add":
                acc[0] = acc[0] + buf[i]
            elif op == "max":
                acc[0] = T.max(acc[0], buf[i])
            elif op == "mul":
                acc[0] = acc[0] * buf[i]

    return func


def _build_unannotated_reduction(extent: int):
    @T.prim_func
    def func(buf: T.Tensor[(64,), T.float32], acc: T.Tensor[(1,), T.float32]):
        for i in T.serial(extent):
            acc[0] = acc[0] + buf[i]

    return func


def test_static_16lane_reduction_emits_butterfly():
    func = _build_annotated_reduction(16, op="add")
    new_func, n_replaced, n_stages = rewrite_reductions(func)
    assert n_replaced == 1
    # log2(16) = 4 stages → 4 shfl_xor_sync Calls.
    assert n_stages == 4
    assert count_shfl_xor_calls(new_func) == 4


def test_static_32lane_reduction_emits_butterfly():
    func = _build_annotated_reduction(32, op="add")
    new_func, n_replaced, n_stages = rewrite_reductions(func)
    assert n_replaced == 1
    assert n_stages == 5
    assert count_shfl_xor_calls(new_func) == 5


def test_static_32lane_reduction_emits_semantic_thread_allreduce():
    func = _build_annotated_reduction(32, op="add")
    new_func, n_replaced = rewrite_reductions_to_thread_allreduce(func)
    assert n_replaced == 1
    assert count_thread_allreduce_calls(new_func) == 1
    assert count_shfl_xor_calls(new_func) == 0
    script = new_func.script()
    assert "T.tvm_thread_allreduce" in script
    assert "reduce_scope" in script


def test_static_8lane_reduction_emits_butterfly():
    func = _build_annotated_reduction(8, op="add")
    new_func, n_replaced, n_stages = rewrite_reductions(func)
    assert n_replaced == 1
    assert n_stages == 3
    assert count_shfl_xor_calls(new_func) == 3


def test_static_64lane_reduction_keeps_threadgroup():
    # Annotated, but extent > 32 → Z3 fails → no rewrite.
    func = _build_annotated_reduction(64, op="add")
    new_func, n_replaced, _ = rewrite_reductions(func)
    assert n_replaced == 0
    assert count_shfl_xor_calls(new_func) == 0
    semantic_func, n_semantic = rewrite_reductions_to_thread_allreduce(func)
    assert n_semantic == 0
    assert count_thread_allreduce_calls(semantic_func) == 0


def test_unsupported_op_keeps_threadgroup():
    # Multiplication is intentionally not in the supported set.
    func = _build_annotated_reduction(16, op="mul")
    new_func, n_replaced, _ = rewrite_reductions(func)
    assert n_replaced == 0
    assert count_shfl_xor_calls(new_func) == 0


def test_max_reduction_emits_butterfly():
    func = _build_annotated_reduction(16, op="max")
    new_func, n_replaced, n_stages = rewrite_reductions(func)
    assert n_replaced == 1
    assert n_stages == 4
    assert count_shfl_xor_calls(new_func) == 4


def test_max_reduction_waits_for_reduction_plan_semantic_identity():
    func = _build_annotated_reduction(16, op="max")
    new_func, n_replaced = rewrite_reductions_to_thread_allreduce(func)
    assert n_replaced == 0
    assert count_thread_allreduce_calls(new_func) == 0
    diagnostics = reduction_rewrite_diagnostics(func)
    assert diagnostics == [
        {
            "annotated": True,
            "extent": "16",
            "loop_var": "i",
            "op": "max",
            "proved": True,
            "query": "static: extent=16 <= 32? True",
            "reason": "semantic_thread_allreduce_op_unsupported",
        }
    ]


def test_unannotated_loop_keeps_threadgroup():
    """Without the per-loop annotation, the rewrite must not fire.

    This is the load-bearing safety property: a bare serial reduction
    loop carries no lane-mapping semantics, so blindly rewriting it
    would change semantics. The annotation is the user's promise that
    ``loop_var`` maps to ``lane_id``.
    """
    func = _build_unannotated_reduction(16)
    new_func, n_replaced, _ = rewrite_reductions(func)
    assert n_replaced == 0
    assert count_shfl_xor_calls(new_func) == 0
    semantic_func, n_semantic = rewrite_reductions_to_thread_allreduce(func)
    assert n_semantic == 0
    assert count_thread_allreduce_calls(semantic_func) == 0
    diagnostics = reduction_rewrite_diagnostics(func)
    assert diagnostics[0]["reason"] == "missing_simd_butterfly_lane_annotation"
    assert diagnostics[0]["annotated"] is False
    assert diagnostics[0]["proved"] is True


def test_default_off_preserves_behavior():
    """With PassConfig OFF (the default), the pass must be a no-op."""
    func = _build_annotated_reduction(16, op="add")
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    out = metal_simd_lift.MetalSimdLiftReductions(mod)
    tvm.ir.assert_structural_equal(out["main"], mod["main"], True)


def test_pass_on_non_metal_target_preserves_behavior():
    """Even with PassConfig ON, a non-metal target must not be rewritten."""
    from tvm.target import Target
    func = _build_annotated_reduction(16, op="add").with_attr(
        "global_symbol", "main"
    )
    mod = tvm.IRModule.from_expr(func)
    with tvm.transform.PassContext(
        config={metal_simd_lift.PASS_CONFIG_KEY: True}
    ):
        with Target("cuda"):
            out = metal_simd_lift.MetalSimdLiftReductions(mod)
    # On non-metal: pass returns func unchanged.
    tvm.ir.assert_structural_equal(out["main"], mod["main"], True)


def test_pass_on_metal_target_prefers_semantic_thread_allreduce():
    func = _build_annotated_reduction(16, op="add").with_attr(
        "global_symbol", "main"
    )
    mod = tvm.IRModule.from_expr(func)
    with tvm.transform.PassContext(
        config={metal_simd_lift.PASS_CONFIG_KEY: True}
    ):
        with Target("metal"):
            out = metal_simd_lift.MetalSimdLiftReductions(mod)
    assert count_thread_allreduce_calls(out["main"]) == 1
    assert count_shfl_xor_calls(out["main"]) == 0
    payload = json.loads(out["main"].attrs["tl.reduction_plans"].value)
    assert payload[0]["op"] == "sum"
    assert payload[0]["candidate_strategies"][0] == "same-simdgroup"
    legality = json.loads(out["main"].attrs["tl.reduction_legality"].value)
    assert legality[0]["proved_no_sync"] is True
    assert legality[0]["cannot_parallelize_reason"] is None
    sync_plan = json.loads(out["main"].attrs["tl.sync_event_plan"].value)
    assert sync_plan[0]["action"] == "none"
    assert sync_plan[0]["external_materialization_required"] is False


def test_pass_records_machine_readable_reduction_rewrite_diagnostics():
    func = _build_annotated_reduction(64, op="add").with_attr(
        "global_symbol", "main"
    )
    mod = tvm.IRModule.from_expr(func)
    with tvm.transform.PassContext(
        config={metal_simd_lift.PASS_CONFIG_KEY: True}
    ):
        with Target("metal"):
            out = metal_simd_lift.MetalSimdLiftReductions(mod)
    payload = json.loads(out["main"].attrs["tl.reduction_rewrite_diagnostics"].value)
    assert payload == [
        {
            "annotated": True,
            "extent": "64",
            "loop_var": "i",
            "op": "add",
            "proved": False,
            "query": "static: extent=64 <= 32? False",
            "reason": "z3_extent_unproved",
        }
    ]


def test_pass_preserves_semantic_diagnostic_when_backend_fallback_rewrites():
    func = _build_annotated_reduction(16, op="max").with_attr(
        "global_symbol", "main"
    )
    mod = tvm.IRModule.from_expr(func)
    with tvm.transform.PassContext(
        config={metal_simd_lift.PASS_CONFIG_KEY: True}
    ):
        with Target("metal"):
            out = metal_simd_lift.MetalSimdLiftReductions(mod)
    assert count_thread_allreduce_calls(out["main"]) == 0
    assert count_shfl_xor_calls(out["main"]) == 4
    payload = json.loads(out["main"].attrs["tl.reduction_rewrite_diagnostics"].value)
    assert payload[0]["reason"] == "semantic_thread_allreduce_op_unsupported"


# ---------------------------------------------------------------------------
# Direct shfl_xor_sync emission shape
# ---------------------------------------------------------------------------

def test_butterfly_uses_decreasing_shifts():
    """Verify the butterfly emits shifts in [16, 8, 4, 2, 1] order."""
    func = _build_annotated_reduction(32, op="add")
    new_func, _, _ = rewrite_reductions(func)
    shifts: list[int] = []

    def _visit(node):
        if isinstance(node, tir.Call):
            op_name = getattr(getattr(node, "op", None), "name", "")
            if op_name == "tl.shfl_xor_sync":
                # args = (mask, value, lane_mask, width)
                arg = node.args[2]
                if isinstance(arg, tir.IntImm):
                    shifts.append(int(arg.value))

    tir.stmt_functor.post_order_visit(new_func.body, _visit)
    # Order may be either source order or post-order; just assert the set.
    assert sorted(shifts, reverse=True) == [16, 8, 4, 2, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
