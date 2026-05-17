"""Explainable cost metadata for scheduler-selected plans."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from tvm import tir

from tilelang.analysis.reduction_plan import ReductionPlan, extract_reduction_plans
from tilelang.analysis.scan_plan import RecurrenceScanPlan


def _dtype_size_bytes(dtype: str | None) -> int:
    if dtype is None:
        return 4
    normalized = str(dtype).lower()
    if "64" in normalized:
        return 8
    if "16" in normalized or "bfloat16" in normalized:
        return 2
    if "8" in normalized and "int" in normalized:
        return 1
    return 4


def _index_math_ops(expr: str) -> int:
    explicit_ops = sum(expr.count(op) for op in ("+", "-", "*", "/", "%"))
    if expr.strip().lstrip("-").isdigit():
        return explicit_ops
    return max(1, explicit_ops)


def _plan_index_math_ops(plan: ReductionPlan) -> int:
    regions = (*plan.input_regions, plan.output_region)
    return sum(_index_math_ops(index) for region in regions for index in region.indices)


def _strategy_decision(strategy: str) -> tuple[str, str]:
    if strategy == "same-simdgroup":
        return (
            "inline-simdgroup",
            "extent fits one simdgroup; no threadgroup scratch or split dispatch",
        )
    if strategy in {"split-simdgroup", "threadgroup-staging", "row-reduce"}:
        return (
            "inline-single-dispatch",
            "extent fits one threadgroup; stage partials internally",
        )
    if strategy == "two-pass-global":
        return (
            "split-two-pass",
            "extent exceeds one threadgroup; use internal device scratch",
        )
    return (
        "vectorized-fallback",
        "backend selected a non-specialized vectorized fallback",
    )


@dataclass(frozen=True)
class ReductionCostEstimate:
    """Cost estimate for one semantic reduction plan."""

    source: str
    op: str
    selected_strategy: str
    reduction_extent: int | None
    accumulator_dtype: str
    estimated_registers_per_thread: int
    local_memory_bytes_per_thread: int
    threadgroup_memory_bytes: int
    device_memory_bytes_per_output: int
    index_math_ops_per_output: int
    dispatch_count: int
    sync_count: int
    materialization_cost: str
    occupancy_limit: str
    split_or_inline_decision: str
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "op": self.op,
            "selected_strategy": self.selected_strategy,
            "reduction_extent": self.reduction_extent,
            "accumulator_dtype": self.accumulator_dtype,
            "estimated_registers_per_thread": self.estimated_registers_per_thread,
            "local_memory_bytes_per_thread": self.local_memory_bytes_per_thread,
            "threadgroup_memory_bytes": self.threadgroup_memory_bytes,
            "device_memory_bytes_per_output": self.device_memory_bytes_per_output,
            "index_math_ops_per_output": self.index_math_ops_per_output,
            "dispatch_count": self.dispatch_count,
            "sync_count": self.sync_count,
            "materialization_cost": self.materialization_cost,
            "occupancy_limit": self.occupancy_limit,
            "split_or_inline_decision": self.split_or_inline_decision,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RecurrenceScanCostEstimate:
    """Cost estimate for one recurrence scan plan."""

    source: str
    name: str
    direction: str
    sequence_length: int
    state_elements: int
    state_bytes: int
    snapshot_bytes: int
    estimated_registers_per_thread: int
    dispatch_count: int
    sync_count: int
    materialization_cost: str
    split_or_inline_decision: str
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "name": self.name,
            "direction": self.direction,
            "sequence_length": self.sequence_length,
            "state_elements": self.state_elements,
            "state_bytes": self.state_bytes,
            "snapshot_bytes": self.snapshot_bytes,
            "estimated_registers_per_thread": self.estimated_registers_per_thread,
            "dispatch_count": self.dispatch_count,
            "sync_count": self.sync_count,
            "materialization_cost": self.materialization_cost,
            "split_or_inline_decision": self.split_or_inline_decision,
            "reason": self.reason,
        }


def estimate_reduction_cost(
    plan: ReductionPlan,
    *,
    source: str = "reduction:0",
) -> ReductionCostEstimate:
    """Build an explainable static cost estimate for one reduction plan."""

    dtype_bytes = _dtype_size_bytes(plan.accumulator_dtype)
    threads = plan.thread_mapping.threads_per_threadgroup or 0
    simdgroups = plan.thread_mapping.simdgroups_per_threadgroup or 0
    blocks = plan.thread_mapping.blocks_per_output or 0
    strategy = plan.selected_strategy
    registers = 4 + len(plan.input_regions) + len(plan.axes)
    if plan.memory_plan.internal_scratch_required:
        registers += 2
    if strategy == "two-pass-global":
        registers += 1

    if plan.memory_plan.scratch_scope == "threadgroup":
        threadgroup_memory_bytes = max(threads, simdgroups + 1) * dtype_bytes
        materialization_cost = "threadgroup-scratch"
    elif plan.memory_plan.scratch_scope == "device":
        threadgroup_memory_bytes = 0
        materialization_cost = "internal-device-scratch"
    else:
        threadgroup_memory_bytes = 0
        materialization_cost = "none"

    device_memory_bytes = blocks * dtype_bytes if strategy == "two-pass-global" else 0
    dispatch_count = 2 if strategy == "two-pass-global" else 1
    sync_count = 0 if strategy == "same-simdgroup" else 2
    if strategy == "two-pass-global":
        sync_count += 1

    if dispatch_count > 1:
        occupancy_limit = "dispatch-count"
    elif threadgroup_memory_bytes > 8192:
        occupancy_limit = "threadgroup-memory"
    elif registers > 16:
        occupancy_limit = "registers"
    else:
        occupancy_limit = "not-limited"

    decision, reason = _strategy_decision(strategy)
    return ReductionCostEstimate(
        source=source,
        op=plan.op,
        selected_strategy=strategy,
        reduction_extent=plan.thread_mapping.reduction_extent,
        accumulator_dtype=plan.accumulator_dtype,
        estimated_registers_per_thread=registers,
        local_memory_bytes_per_thread=0,
        threadgroup_memory_bytes=threadgroup_memory_bytes,
        device_memory_bytes_per_output=device_memory_bytes,
        index_math_ops_per_output=_plan_index_math_ops(plan),
        dispatch_count=dispatch_count,
        sync_count=sync_count,
        materialization_cost=materialization_cost,
        occupancy_limit=occupancy_limit,
        split_or_inline_decision=decision,
        reason=reason,
    )


def estimate_recurrence_scan_cost(
    plan: RecurrenceScanPlan,
    *,
    source: str = "scan:0",
) -> RecurrenceScanCostEstimate:
    """Build an explainable static cost estimate for one recurrence scan plan."""

    dtype_bytes = _dtype_size_bytes(plan.state_dtype)
    state_bytes = plan.snapshot_plan.state_elements * dtype_bytes
    snapshot_bytes = plan.snapshot_plan.snapshot_elements * dtype_bytes
    uses_snapshots = plan.snapshot_plan.policy == "state-boundary-cache"
    if uses_snapshots:
        decision = "split-snapshot-reuse"
        reason = "long reverse recurrence reuses forward state-boundary snapshots"
        dispatch_count = 2
        materialization_cost = "state-boundary-cache"
    elif plan.direction == "reverse":
        decision = "inline-direct-recompute"
        reason = "short reverse recurrence can recompute without snapshots"
        dispatch_count = 1
        materialization_cost = "none"
    else:
        decision = "inline-forward-scan"
        reason = "forward recurrence does not need reverse rematerialization"
        dispatch_count = 1
        materialization_cost = "none"

    return RecurrenceScanCostEstimate(
        source=source,
        name=plan.name,
        direction=plan.direction,
        sequence_length=plan.sequence_length,
        state_elements=plan.snapshot_plan.state_elements,
        state_bytes=state_bytes,
        snapshot_bytes=snapshot_bytes,
        estimated_registers_per_thread=6 + len(plan.fused_post_ops),
        dispatch_count=dispatch_count,
        sync_count=int(plan.host_sync_required) + int(plan.device_event_required),
        materialization_cost=materialization_cost,
        split_or_inline_decision=decision,
        reason=reason,
    )


def build_reduction_cost_estimates(
    func: tir.PrimFunc,
) -> tuple[ReductionCostEstimate, ...]:
    """Build static cost estimates for all semantic reductions in a PrimFunc."""

    return tuple(
        estimate_reduction_cost(plan, source=f"reduction:{idx}:{plan.op}")
        for idx, plan in enumerate(extract_reduction_plans(func))
    )


def serialize_reduction_cost_estimates(
    estimates: tuple[ReductionCostEstimate, ...],
) -> str:
    return json.dumps([estimate.to_json() for estimate in estimates], sort_keys=True)


def serialize_recurrence_scan_cost_estimates(
    estimates: tuple[RecurrenceScanCostEstimate, ...],
) -> str:
    return json.dumps([estimate.to_json() for estimate in estimates], sort_keys=True)


def attach_reduction_cost_metadata(func: tir.PrimFunc) -> tir.PrimFunc:
    """Attach stable JSON cost metadata for semantic reduction plans."""

    estimates = build_reduction_cost_estimates(func)
    if not estimates:
        return func
    attrs = dict(func.attrs) if func.attrs is not None else {}
    attrs["tl.reduction_costs"] = tir.StringImm(
        serialize_reduction_cost_estimates(estimates)
    )
    return func.with_attrs(attrs)


__all__ = [
    "RecurrenceScanCostEstimate",
    "ReductionCostEstimate",
    "attach_reduction_cost_metadata",
    "build_reduction_cost_estimates",
    "estimate_recurrence_scan_cost",
    "estimate_reduction_cost",
    "serialize_recurrence_scan_cost_estimates",
    "serialize_reduction_cost_estimates",
]
