from __future__ import annotations

import json

import pytest

from tilelang.analysis.scan_plan import (
    plan_recurrence_scan,
    serialize_recurrence_scan_plans,
)


def test_reverse_recurrence_scan_uses_state_snapshots_for_underflow_risk():
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

    assert plan.direction == "reverse"
    assert plan.snapshot_plan.policy == "state-boundary-cache"
    assert plan.snapshot_plan.chunk_count == 8
    assert plan.snapshot_plan.snapshot_count == 9
    assert plan.snapshot_plan.snapshot_elements == 1 * 2 * 64 * 32 * 9
    assert plan.rematerialization_policy == "reuse-forward-state-snapshots"
    assert plan.chunk_independence_proved is True
    assert plan.host_sync_required is False
    assert plan.device_event_required is False
    assert plan.fused_post_ops == ("skip_D", "silu_gate")


def test_short_reverse_recurrence_scan_can_direct_recompute():
    plan = plan_recurrence_scan(
        name="mamba3_bwd",
        direction="reverse",
        sequence_length=1,
        state_shape=(1, 1, 32, 16),
        state_dtype="float32",
        chunk_size=1,
        decay_may_underflow=True,
    )

    assert plan.snapshot_plan.policy == "none"
    assert plan.snapshot_plan.snapshot_count == 0
    assert plan.rematerialization_policy == "direct-recompute"


def test_scan_plan_rejects_in_place_alias_without_proof():
    plan = plan_recurrence_scan(
        name="aliased_scan",
        direction="forward",
        sequence_length=4,
        state_shape=(1, 1, 8, 8),
        state_dtype="float32",
        chunk_size=2,
        input_output_alias=True,
        in_place_requested=True,
    )

    assert plan.chunk_independence_proved is False
    assert plan.alias_plan.in_place_allowed is False
    assert plan.alias_plan.reason == "input_output_alias_without_in_place_proof"


def test_scan_plan_serializes_stably():
    plan = plan_recurrence_scan(
        name="mamba3_bwd",
        direction="reverse",
        sequence_length=2,
        state_shape=(1, 1, 4, 4),
        state_dtype="float32",
        chunk_size=1,
        decay_may_underflow=True,
    )
    payload = json.loads(serialize_recurrence_scan_plans((plan,)))

    assert payload[0]["name"] == "mamba3_bwd"
    assert payload[0]["snapshot_plan"]["policy"] == "state-boundary-cache"
    assert payload[0]["rematerialization_policy"] == "reuse-forward-state-snapshots"


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_scan_plan_rejects_invalid_chunk_size(chunk_size: int):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        plan_recurrence_scan(
            name="bad",
            direction="forward",
            sequence_length=1,
            state_shape=(1,),
            state_dtype="float32",
            chunk_size=chunk_size,
        )
