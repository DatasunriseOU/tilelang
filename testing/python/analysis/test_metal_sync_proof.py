from __future__ import annotations

import pytest

from tilelang.analysis.metal_sync_proof import _Z3_AVAILABLE, plan_metal_buffer_sync


@pytest.mark.skipif(not _Z3_AVAILABLE, reason="z3-solver not installed")
def test_same_command_buffer_path_c_needs_no_host_sync():
    plan = plan_metal_buffer_sync(
        may_alias=True,
        same_command_buffer=True,
        producer_before_consumer=True,
    )

    assert plan.action == "none"
    assert plan.where == "same_command_buffer_encode_order"
    assert plan.host_sync_required is False
    assert plan.device_event_required is False
    assert plan.z3_proved is True


@pytest.mark.skipif(not _Z3_AVAILABLE, reason="z3-solver not installed")
def test_same_command_buffer_opaque_external_encoder_uses_device_event():
    plan = plan_metal_buffer_sync(
        may_alias=True,
        same_command_buffer=True,
        producer_before_consumer=True,
        resource_tracked=False,
    )

    assert plan.action == "device_event"
    assert plan.where == "opaque_external_encoder_edge"
    assert plan.host_sync_required is False
    assert plan.device_event_required is True


@pytest.mark.skipif(not _Z3_AVAILABLE, reason="z3-solver not installed")
def test_cross_command_buffer_hazard_is_device_event_not_host_sync():
    plan = plan_metal_buffer_sync(
        may_alias=True,
        same_command_buffer=False,
        producer_before_consumer=None,
    )

    assert plan.action == "device_event"
    assert plan.host_sync_required is False
    assert plan.device_event_required is True
    assert plan.where == "producer_to_consumer_command_buffer_edge"


@pytest.mark.skipif(not _Z3_AVAILABLE, reason="z3-solver not installed")
def test_same_command_buffer_wrong_order_fails_scheduler_not_runtime_sync():
    plan = plan_metal_buffer_sync(
        may_alias=True,
        same_command_buffer=True,
        producer_before_consumer=False,
    )

    assert plan.action == "reorder_or_fail"
    assert plan.host_sync_required is False
    assert plan.device_event_required is False


@pytest.mark.skipif(not _Z3_AVAILABLE, reason="z3-solver not installed")
def test_host_observer_is_the_only_host_sync_boundary():
    plan = plan_metal_buffer_sync(
        may_alias=True,
        same_command_buffer=True,
        producer_before_consumer=True,
        host_observer=True,
    )

    assert plan.action == "host_sync"
    assert plan.where == "graph_output_host_boundary"
    assert plan.host_sync_required is True
    assert plan.device_event_required is False
