"""Scheduler-visible scan and recurrence planning metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal


ScanDirection = Literal["forward", "reverse"]
SnapshotPolicy = Literal["none", "state-boundary-cache"]
RematerializationPolicy = Literal[
    "not-needed",
    "reuse-forward-state-snapshots",
    "direct-recompute",
]


@dataclass(frozen=True)
class RecurrenceAliasPlan:
    """In-place and alias legality for a recurrence plan."""

    input_output_alias: bool
    in_place_requested: bool
    in_place_allowed: bool
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "input_output_alias": self.input_output_alias,
            "in_place_requested": self.in_place_requested,
            "in_place_allowed": self.in_place_allowed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RecurrenceSnapshotPlan:
    """State snapshot/cache policy for a scan or reverse recurrence."""

    policy: SnapshotPolicy
    chunk_size: int
    chunk_count: int
    snapshot_count: int
    state_elements: int
    snapshot_elements: int
    state_dtype: str

    def to_json(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "chunk_size": self.chunk_size,
            "chunk_count": self.chunk_count,
            "snapshot_count": self.snapshot_count,
            "state_elements": self.state_elements,
            "snapshot_elements": self.snapshot_elements,
            "state_dtype": self.state_dtype,
        }


@dataclass(frozen=True)
class RecurrenceScanPlan:
    """A backend-neutral plan for a scan/reverse recurrence."""

    name: str
    direction: ScanDirection
    sequence_length: int
    state_shape: tuple[int, ...]
    state_dtype: str
    state_dependency: str
    chunk_independence_proved: bool
    proof_method: str
    snapshot_plan: RecurrenceSnapshotPlan
    rematerialization_policy: RematerializationPolicy
    alias_plan: RecurrenceAliasPlan
    host_sync_required: bool
    device_event_required: bool
    fused_post_ops: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "sequence_length": self.sequence_length,
            "state_shape": list(self.state_shape),
            "state_dtype": self.state_dtype,
            "state_dependency": self.state_dependency,
            "chunk_independence_proved": self.chunk_independence_proved,
            "proof_method": self.proof_method,
            "snapshot_plan": self.snapshot_plan.to_json(),
            "rematerialization_policy": self.rematerialization_policy,
            "alias_plan": self.alias_plan.to_json(),
            "host_sync_required": self.host_sync_required,
            "device_event_required": self.device_event_required,
            "fused_post_ops": list(self.fused_post_ops),
        }


def _ceildiv(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _state_elements(shape: tuple[int, ...]) -> int:
    total = 1
    for extent in shape:
        total *= int(extent)
    return total


def _alias_plan(
    *,
    input_output_alias: bool,
    in_place_requested: bool,
    chunk_independence_proved: bool,
) -> RecurrenceAliasPlan:
    if not input_output_alias:
        return RecurrenceAliasPlan(
            input_output_alias=False,
            in_place_requested=in_place_requested,
            in_place_allowed=in_place_requested,
            reason="distinct_input_output_buffers",
        )
    if in_place_requested and chunk_independence_proved:
        return RecurrenceAliasPlan(
            input_output_alias=True,
            in_place_requested=True,
            in_place_allowed=True,
            reason="in_place_alias_proved_chunk_independent",
        )
    return RecurrenceAliasPlan(
        input_output_alias=True,
        in_place_requested=in_place_requested,
        in_place_allowed=False,
        reason="input_output_alias_without_in_place_proof",
    )


def plan_recurrence_scan(
    *,
    name: str,
    direction: ScanDirection,
    sequence_length: int,
    state_shape: tuple[int, ...],
    state_dtype: str,
    chunk_size: int,
    decay_may_underflow: bool = False,
    input_output_alias: bool = False,
    in_place_requested: bool = False,
    fused_post_ops: tuple[str, ...] = (),
) -> RecurrenceScanPlan:
    """Plan snapshot/rematerialization policy for one recurrence scan."""

    if sequence_length < 0:
        raise ValueError(f"sequence_length must be non-negative, got {sequence_length}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not state_shape or any(int(extent) <= 0 for extent in state_shape):
        raise ValueError(f"state_shape must contain positive extents, got {state_shape}")

    chunk_count = _ceildiv(sequence_length, chunk_size) if sequence_length else 0
    state_elements = _state_elements(state_shape)
    needs_snapshots = direction == "reverse" and sequence_length > chunk_size and decay_may_underflow
    snapshot_count = chunk_count + 1 if needs_snapshots else 0
    snapshot_plan = RecurrenceSnapshotPlan(
        policy="state-boundary-cache" if needs_snapshots else "none",
        chunk_size=chunk_size,
        chunk_count=chunk_count,
        snapshot_count=snapshot_count,
        state_elements=state_elements,
        snapshot_elements=snapshot_count * state_elements,
        state_dtype=state_dtype,
    )
    chunk_independence_proved = not input_output_alias
    alias_plan = _alias_plan(
        input_output_alias=input_output_alias,
        in_place_requested=in_place_requested,
        chunk_independence_proved=chunk_independence_proved,
    )
    if needs_snapshots:
        rematerialization_policy = "reuse-forward-state-snapshots"
    elif direction == "reverse":
        rematerialization_policy = "direct-recompute"
    else:
        rematerialization_policy = "not-needed"
    return RecurrenceScanPlan(
        name=name,
        direction=direction,
        sequence_length=sequence_length,
        state_shape=state_shape,
        state_dtype=state_dtype,
        state_dependency="loop-carried-state",
        chunk_independence_proved=chunk_independence_proved,
        proof_method="static-shape-and-alias",
        snapshot_plan=snapshot_plan,
        rematerialization_policy=rematerialization_policy,
        alias_plan=alias_plan,
        host_sync_required=False,
        device_event_required=False,
        fused_post_ops=fused_post_ops,
    )


def serialize_recurrence_scan_plans(plans: tuple[RecurrenceScanPlan, ...]) -> str:
    return json.dumps([plan.to_json() for plan in plans], sort_keys=True)


__all__ = [
    "RecurrenceAliasPlan",
    "RecurrenceScanPlan",
    "RecurrenceSnapshotPlan",
    "plan_recurrence_scan",
    "serialize_recurrence_scan_plans",
]
