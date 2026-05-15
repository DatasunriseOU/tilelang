"""Scheduler sync/event decisions derived from legality proofs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal

from tvm import tir

from tilelang.analysis.reduction_legality import (
    ReductionLegalityProof,
    prove_reduction_plans,
)
from tilelang.analysis.reduction_plan import extract_reduction_plans


SyncEventAction = Literal[
    "none",
    "threadgroup_barrier",
    "device_event",
    "reject",
]


@dataclass(frozen=True)
class SyncEventDecision:
    """One scheduler-visible sync/materialization decision."""

    source: str
    action: SyncEventAction
    reason: str
    host_sync_required: bool
    threadgroup_barrier_required: bool
    device_event_required: bool
    two_pass_required: bool
    internal_scratch_required: bool
    external_materialization_required: bool
    z3_proved: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "action": self.action,
            "reason": self.reason,
            "host_sync_required": self.host_sync_required,
            "threadgroup_barrier_required": self.threadgroup_barrier_required,
            "device_event_required": self.device_event_required,
            "two_pass_required": self.two_pass_required,
            "internal_scratch_required": self.internal_scratch_required,
            "external_materialization_required": self.external_materialization_required,
            "z3_proved": self.z3_proved,
        }


def sync_event_decision_for_reduction(
    proof: ReductionLegalityProof,
    *,
    source: str,
) -> SyncEventDecision:
    """Map one reduction proof to the narrowest required sync action."""

    if proof.cannot_parallelize_reason is not None:
        return SyncEventDecision(
            source=source,
            action="reject",
            reason=proof.cannot_parallelize_reason,
            host_sync_required=False,
            threadgroup_barrier_required=False,
            device_event_required=False,
            two_pass_required=False,
            internal_scratch_required=False,
            external_materialization_required=False,
            z3_proved=proof.z3_proved,
        )
    if proof.proved_no_sync:
        return SyncEventDecision(
            source=source,
            action="none",
            reason="proved_no_sync",
            host_sync_required=False,
            threadgroup_barrier_required=False,
            device_event_required=False,
            two_pass_required=False,
            internal_scratch_required=False,
            external_materialization_required=False,
            z3_proved=proof.z3_proved,
        )
    if proof.requires_threadgroup_barrier:
        return SyncEventDecision(
            source=source,
            action="threadgroup_barrier",
            reason="threadgroup_shared_memory_visibility",
            host_sync_required=False,
            threadgroup_barrier_required=True,
            device_event_required=False,
            two_pass_required=False,
            internal_scratch_required=False,
            external_materialization_required=False,
            z3_proved=proof.z3_proved,
        )
    if proof.requires_device_event:
        return SyncEventDecision(
            source=source,
            action="device_event",
            reason="two_pass_or_cross_scope_device_visibility",
            host_sync_required=False,
            threadgroup_barrier_required=False,
            device_event_required=True,
            two_pass_required=proof.requires_two_pass,
            internal_scratch_required=proof.requires_two_pass,
            external_materialization_required=False,
            z3_proved=proof.z3_proved,
        )
    return SyncEventDecision(
        source=source,
        action="reject",
        reason="no_legal_sync_strategy",
        host_sync_required=False,
        threadgroup_barrier_required=False,
        device_event_required=False,
        two_pass_required=False,
        internal_scratch_required=False,
        external_materialization_required=False,
        z3_proved=proof.z3_proved,
    )


def build_reduction_sync_event_plan(
    func: tir.PrimFunc,
) -> tuple[SyncEventDecision, ...]:
    plans = extract_reduction_plans(func)
    if not plans:
        return ()
    proofs = prove_reduction_plans(plans)
    return tuple(
        sync_event_decision_for_reduction(
            proof,
            source=f"reduction:{idx}:{plan.op}",
        )
        for idx, (plan, proof) in enumerate(zip(plans, proofs, strict=True))
    )


def serialize_sync_event_plan(decisions: tuple[SyncEventDecision, ...]) -> str:
    return json.dumps([decision.to_json() for decision in decisions], sort_keys=True)


def attach_sync_event_plan_metadata(func: tir.PrimFunc) -> tir.PrimFunc:
    decisions = build_reduction_sync_event_plan(func)
    if not decisions:
        return func
    attrs = dict(func.attrs) if func.attrs is not None else {}
    attrs["tl.sync_event_plan"] = tir.StringImm(serialize_sync_event_plan(decisions))
    return func.with_attrs(attrs)


__all__ = [
    "SyncEventAction",
    "SyncEventDecision",
    "attach_sync_event_plan_metadata",
    "build_reduction_sync_event_plan",
    "serialize_sync_event_plan",
    "sync_event_decision_for_reduction",
]
