"""Graph-level Metal event planning for MLX TVM-FFI launches."""

from __future__ import annotations

from dataclasses import dataclass
import weakref
from typing import Any

from tilelang.analysis.metal_sync_proof import plan_metal_buffer_sync


_DEFAULT_COMMAND_BUFFER_DOMAIN = ("mlx_tvm_ffi", "default_metal_stream")
_PRODUCERS: dict[int, tuple[weakref.ref[Any], "MetalProducerRecord"]] = {}


@dataclass(frozen=True)
class MetalProducerRecord:
    sync_state: Any
    command_buffer_domain: Any


def _domain(command_buffer_domain: Any | None) -> Any:
    return _DEFAULT_COMMAND_BUFFER_DOMAIN if command_buffer_domain is None else command_buffer_domain


def clear_metal_graph_sync_state_for_tests() -> None:
    _PRODUCERS.clear()


def _lookup_producer(array: Any) -> MetalProducerRecord | None:
    key = id(array)
    entry = _PRODUCERS.get(key)
    if entry is None:
        return None
    ref, record = entry
    if ref() is array:
        return record
    _PRODUCERS.pop(key, None)
    return None


def _remember_producer(array: Any, record: MetalProducerRecord) -> None:
    key = id(array)

    def remove(_ref: weakref.ref[Any], *, stale_key: int = key) -> None:
        _PRODUCERS.pop(stale_key, None)

    _PRODUCERS[key] = (weakref.ref(array, remove), record)


def plan_mlx_tvm_ffi_launch(
    native: Any,
    inputs: list[Any],
    *,
    command_buffer_domain: Any | None = None,
) -> tuple[Any, list[Any]]:
    """Plan device-event edges for a new MLX TVM-FFI launch.

    MLX's resource tracker cannot see the raw MTLBuffer arguments passed
    through TVM-FFI, so every real TVM-FFI output->input dependency gets a
    device-side Metal shared event: producer signal, consumer wait, no host
    synchronization.
    """

    current_domain = _domain(command_buffer_domain)
    launch_sync_state = native.make_launch_sync_state()
    wait_edges: list[Any] = []
    seen_producers: set[int] = set()

    for array in inputs:
        producer = _lookup_producer(array)
        if producer is None:
            continue

        producer_key = id(producer.sync_state)
        if producer_key in seen_producers:
            continue
        seen_producers.add(producer_key)

        plan = plan_metal_buffer_sync(
            may_alias=True,
            same_command_buffer=producer.command_buffer_domain == current_domain,
            producer_before_consumer=True,
            resource_tracked=False,
        )
        if plan.action == "none":
            continue
        if plan.action == "device_event":
            edge = native.make_sync_edge()
            producer.sync_state.add_signal_edge(edge)
            wait_edges.append(edge)
            continue
        raise RuntimeError(f"unsupported Metal graph sync plan for TVM-FFI launch: {plan}")

    return launch_sync_state, wait_edges


def register_mlx_tvm_ffi_outputs(
    outputs: list[Any],
    launch_sync_state: Any,
    *,
    command_buffer_domain: Any | None = None,
) -> None:
    record = MetalProducerRecord(
        sync_state=launch_sync_state,
        command_buffer_domain=_domain(command_buffer_domain),
    )
    for output in outputs:
        _remember_producer(output, record)


__all__ = [
    "MetalProducerRecord",
    "clear_metal_graph_sync_state_for_tests",
    "plan_mlx_tvm_ffi_launch",
    "register_mlx_tvm_ffi_outputs",
]
