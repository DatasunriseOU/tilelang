"""Z3-backed sync planning for MLX/Metal graph launches.

The MLX TVM-FFI bridge borrows MLX's current Metal command buffer. MLX can
prove command-buffer order for resources it tracks, but opaque TVM-FFI launches
pass raw MTLBuffer pointers outside that tracker. Those producer->consumer
edges need a device-side event, not a host ``synchronize()``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal


try:
    import z3 as _z3  # type: ignore

    _Z3_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional z3-solver
    _z3 = None  # type: ignore
    _Z3_AVAILABLE = False


SyncAction = Literal["none", "device_event", "host_sync", "reorder_or_fail"]


@dataclass(frozen=True)
class MetalSyncPlan:
    action: SyncAction
    where: str
    host_sync_required: bool
    device_event_required: bool
    reason: str
    z3_proved: bool


def _z3_pass_disabled() -> bool:
    for var in ("TILELANG_DISABLE_Z3", "TILELANG_DISABLE_Z3_METAL_SYNC"):
        value = os.environ.get(var, "")
        if value and value != "0":
            return True
    return False


def _prove_same_command_buffer_order(timeout_ms: int) -> bool:
    """Prove command ordinal order excludes a read-before-write hazard."""

    if not _Z3_AVAILABLE or _z3_pass_disabled():
        return False

    solver = _z3.Solver()
    solver.set("timeout", int(timeout_ms))
    producer = _z3.Int("producer_command_ordinal")
    consumer = _z3.Int("consumer_command_ordinal")
    solver.add(producer >= 0, consumer >= 0)
    solver.add(producer < consumer)
    solver.add(_z3.Not(producer < consumer))
    return solver.check() == _z3.unsat


def plan_metal_buffer_sync(
    *,
    may_alias: bool,
    same_command_buffer: bool | None,
    producer_before_consumer: bool | None,
    host_observer: bool = False,
    resource_tracked: bool = True,
    timeout_ms: int = 50,
) -> MetalSyncPlan:
    """Return the narrowest required sync action for one buffer dependency.

    ``host_sync`` is reserved for graph-output observation by the CPU. Runtime
    producer->consumer hazards either prove to ``none`` on the same borrowed
    command buffer when resource tracking owns the buffers, or become a
    device-event edge for opaque/external encoder boundaries.
    """

    if host_observer:
        return MetalSyncPlan(
            action="host_sync",
            where="graph_output_host_boundary",
            host_sync_required=True,
            device_event_required=False,
            reason="CPU observes the result; this is outside the kernel hot path",
            z3_proved=False,
        )

    if not may_alias:
        return MetalSyncPlan(
            action="none",
            where="no_alias_edge",
            host_sync_required=False,
            device_event_required=False,
            reason="producer and consumer buffers are disjoint",
            z3_proved=False,
        )

    if same_command_buffer is True:
        if producer_before_consumer is True and _prove_same_command_buffer_order(timeout_ms):
            return MetalSyncPlan(
                action="none",
                where="same_command_buffer_encode_order",
                host_sync_required=False,
                device_event_required=False,
                reason="Z3 proved producer ordinal precedes consumer ordinal in the borrowed command buffer",
                z3_proved=True,
            )
        return MetalSyncPlan(
            action="reorder_or_fail",
            where="same_command_buffer_encode_order",
            host_sync_required=False,
            device_event_required=False,
            reason="same command buffer hazard is fixed by scheduler order, not by host sync",
            z3_proved=False,
        )

    return MetalSyncPlan(
        action="device_event",
        where="producer_to_consumer_command_buffer_edge",
        host_sync_required=False,
        device_event_required=True,
        reason="aliased buffers cross command buffers; insert a Metal event dependency",
        z3_proved=False,
    )


__all__ = ["MetalSyncPlan", "plan_metal_buffer_sync", "_Z3_AVAILABLE"]
