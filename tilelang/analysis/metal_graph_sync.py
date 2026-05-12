"""Graph-level Metal event planning for MLX TVM-FFI launches."""

from __future__ import annotations

from dataclasses import dataclass
import weakref
from typing import Any, Literal, Sequence

from tilelang.analysis.metal_sync_proof import plan_metal_buffer_sync


_DEFAULT_COMMAND_BUFFER_DOMAIN = ("mlx_tvm_ffi", "default_metal_stream")
_PRODUCERS: dict[int, tuple[weakref.ref[Any], "MetalProducerRecord"]] = {}


MetalBufferAccessMode = Literal["read", "write"]


@dataclass(frozen=True)
class MetalBufferDependency:
    """Scheduler-visible access metadata for one TVM-FFI Metal buffer param."""

    param_index: int
    name: str
    mode: MetalBufferAccessMode
    result_position: int | None = None
    resource_tracked: bool = False


@dataclass(frozen=True)
class MetalLaunchDependencyMetadata:
    """Explicit dependency metadata emitted by graph lowering for one launch."""

    kernel_symbol: str
    command_buffer_domain: Any
    input_accesses: tuple[MetalBufferDependency, ...]
    output_accesses: tuple[MetalBufferDependency, ...]
    scheduler_source: str = "tilelang.tvm_ffi.metal"
    producer_before_consumer: bool = True


@dataclass(frozen=True)
class MetalProducerRecord:
    sync_state: Any
    launch_metadata: MetalLaunchDependencyMetadata
    output_access: MetalBufferDependency

    @property
    def command_buffer_domain(self) -> Any:
        return self.launch_metadata.command_buffer_domain


def _domain(command_buffer_domain: Any | None) -> Any:
    return _DEFAULT_COMMAND_BUFFER_DOMAIN if command_buffer_domain is None else command_buffer_domain


def _param_name(param_names: Sequence[str] | None, param_index: int) -> str:
    if param_names is None:
        return f"param_{param_index}"
    if param_index < len(param_names):
        return str(param_names[param_index])
    return f"param_{param_index}"


def make_tvm_ffi_metal_dependency_metadata(
    *,
    kernel_symbol: str,
    input_param_indices: Sequence[int],
    output_param_indices: Sequence[int],
    param_names: Sequence[str] | None = None,
    command_buffer_domain: Any | None = None,
    resource_tracked: bool = False,
) -> MetalLaunchDependencyMetadata:
    """Build explicit scheduler metadata for a lowered TVM-FFI Metal launch."""

    return MetalLaunchDependencyMetadata(
        kernel_symbol=str(kernel_symbol),
        command_buffer_domain=_domain(command_buffer_domain),
        input_accesses=tuple(
            MetalBufferDependency(
                param_index=int(param_index),
                name=_param_name(param_names, int(param_index)),
                mode="read",
                resource_tracked=resource_tracked,
            )
            for param_index in input_param_indices
        ),
        output_accesses=tuple(
            MetalBufferDependency(
                param_index=int(param_index),
                name=_param_name(param_names, int(param_index)),
                mode="write",
                result_position=result_position,
                resource_tracked=resource_tracked,
            )
            for result_position, param_index in enumerate(output_param_indices)
        ),
    )


def with_command_buffer_domain(
    metadata: MetalLaunchDependencyMetadata,
    command_buffer_domain: Any | None,
) -> MetalLaunchDependencyMetadata:
    return MetalLaunchDependencyMetadata(
        kernel_symbol=metadata.kernel_symbol,
        command_buffer_domain=_domain(command_buffer_domain),
        input_accesses=metadata.input_accesses,
        output_accesses=metadata.output_accesses,
        scheduler_source=metadata.scheduler_source,
        producer_before_consumer=metadata.producer_before_consumer,
    )


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


def _fallback_metadata(
    *,
    input_count: int,
    output_count: int = 0,
    command_buffer_domain: Any | None = None,
) -> MetalLaunchDependencyMetadata:
    return make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="<unknown>",
        input_param_indices=tuple(range(input_count)),
        output_param_indices=tuple(range(input_count, input_count + output_count)),
        command_buffer_domain=command_buffer_domain,
    )


def _real_interbuffer_hazard(
    producer_access: MetalBufferDependency,
    consumer_access: MetalBufferDependency,
) -> bool:
    return producer_access.mode == "write" and consumer_access.mode == "read"


def plan_mlx_tvm_ffi_launch(
    native: Any,
    inputs: list[Any],
    *,
    dependency_metadata: MetalLaunchDependencyMetadata | None = None,
    command_buffer_domain: Any | None = None,
) -> tuple[Any, list[Any]]:
    """Plan device-event edges for a new MLX TVM-FFI launch.

    Graph lowering supplies explicit per-buffer read/write metadata. The
    runtime array registry only resolves which prior output object feeds which
    current input object; the hazard decision comes from the metadata.
    """

    metadata = dependency_metadata or _fallback_metadata(
        input_count=len(inputs),
        command_buffer_domain=command_buffer_domain,
    )
    if command_buffer_domain is not None and dependency_metadata is not None:
        metadata = with_command_buffer_domain(metadata, command_buffer_domain)
    if len(inputs) != len(metadata.input_accesses):
        raise ValueError(
            "TVM-FFI Metal dependency metadata/input count mismatch: "
            f"{len(metadata.input_accesses)} metadata inputs for {len(inputs)} runtime inputs"
        )

    launch_sync_state = native.make_launch_sync_state()
    wait_edges: list[Any] = []
    seen_producers: set[int] = set()

    for array, consumer_access in zip(inputs, metadata.input_accesses):
        producer = _lookup_producer(array)
        if producer is None:
            continue
        if not _real_interbuffer_hazard(producer.output_access, consumer_access):
            continue

        producer_key = id(producer.sync_state)
        if producer_key in seen_producers:
            continue
        seen_producers.add(producer_key)

        plan = plan_metal_buffer_sync(
            may_alias=True,
            same_command_buffer=producer.command_buffer_domain == metadata.command_buffer_domain,
            producer_before_consumer=metadata.producer_before_consumer,
            resource_tracked=producer.output_access.resource_tracked and consumer_access.resource_tracked,
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
    dependency_metadata: MetalLaunchDependencyMetadata | None = None,
    command_buffer_domain: Any | None = None,
) -> None:
    metadata = dependency_metadata or _fallback_metadata(
        input_count=0,
        output_count=len(outputs),
        command_buffer_domain=command_buffer_domain,
    )
    if command_buffer_domain is not None and dependency_metadata is not None:
        metadata = with_command_buffer_domain(metadata, command_buffer_domain)
    if len(outputs) != len(metadata.output_accesses):
        raise ValueError(
            "TVM-FFI Metal dependency metadata/output count mismatch: "
            f"{len(metadata.output_accesses)} metadata outputs for {len(outputs)} runtime outputs"
        )
    for output, output_access in zip(outputs, metadata.output_accesses):
        _remember_producer(
            output,
            MetalProducerRecord(
                sync_state=launch_sync_state,
                launch_metadata=metadata,
                output_access=output_access,
            ),
        )


__all__ = [
    "MetalBufferDependency",
    "MetalLaunchDependencyMetadata",
    "MetalProducerRecord",
    "clear_metal_graph_sync_state_for_tests",
    "make_tvm_ffi_metal_dependency_metadata",
    "plan_mlx_tvm_ffi_launch",
    "register_mlx_tvm_ffi_outputs",
    "with_command_buffer_domain",
]
