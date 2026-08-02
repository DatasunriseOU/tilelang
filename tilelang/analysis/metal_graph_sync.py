"""Graph-level Metal event planning for MLX TVM-FFI launches."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import weakref
from typing import Any, Literal
from collections.abc import Sequence

from tilelang.analysis.metal_sync_proof import plan_metal_buffer_sync


_DEFAULT_COMMAND_BUFFER_DOMAIN = ("mlx_tvm_ffi", "default_metal_stream")

# Producer registry keyed by ``id(array)``. ``id()`` is NOT a stable identity:
# CPython reuses the id of a freed object for the next allocation of the same
# type, so a later MLX array can land on the same key as a freed producer.
#
# The registry therefore stores, per key, a *generation token* (a process-wide
# monotonic counter) alongside the weakref + record. The generation token is the
# stable identity of one ``_remember_producer`` call. Both the weakref finalizer
# eviction and every lookup are generation-checked: a stale finalizer (for a
# producer whose array was freed and whose ``id()`` was reused by a newer
# producer) can only evict the entry it created -- never a newer id-reused
# record that now legitimately occupies the same key. Without this, the stale
# finalizer would blindly ``pop`` the key and delete the live producer's record,
# leaving its consumer with no device-event edge -> the consumer kernel reads the
# buffer before the producer kernel has finished -> NaN / wrong output.
_PRODUCER_GENERATION = itertools.count(1)
_PRODUCERS: dict[int, _ProducerEntry] = {}


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


@dataclass(frozen=True)
class _ProducerEntry:
    """One registry slot: a generation-stamped producer record.

    ``generation`` is the stable identity of the ``_remember_producer`` call
    that created this slot, so neither a stale weakref finalizer nor an
    id-reusing lookup can ever act on a slot that a newer call has replaced.
    """

    ref: weakref.ref[Any]
    record: MetalProducerRecord
    generation: int


@dataclass(frozen=True)
class MetalLaunchSyncDecision:
    """Inspectable sync decision for one producer->consumer graph edge."""

    consumer_kernel_symbol: str
    consumer_input_param_index: int
    consumer_input_name: str
    producer_kernel_symbol: str
    producer_output_param_index: int
    producer_result_position: int | None
    action: str
    where: str
    host_sync_required: bool
    device_event_required: bool
    external_materialization_required: bool
    reason: str
    z3_proved: bool


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
    # Identity guard: ``id()`` can collide with a freed producer whose slot has
    # not been evicted yet. Only honour the record when the stored weakref still
    # resolves to *this exact* array; otherwise the slot belongs to a dead (or
    # different) object and must not be attributed to ``array``.
    if entry.ref() is array:
        return entry.record
    # The slot is stale relative to ``array`` (the weakref is dead, or resolves
    # to a different object that happens to share this id). Evict only if the
    # weakref is actually dead; if it resolves to a live different object, that
    # object's own (correctly-keyed) slot must be left intact. Either way,
    # ``array`` has no producer.
    if entry.ref() is None:
        # Generation-checked eviction so we never drop a slot a newer
        # ``_remember_producer`` has installed at this key in the meantime.
        current = _PRODUCERS.get(key)
        if current is not None and current.generation == entry.generation:
            _PRODUCERS.pop(key, None)
    return None


def has_mlx_tvm_ffi_producer(array: Any) -> bool:
    """Return whether ``array`` is a registered native TVM-FFI graph output."""

    return _lookup_producer(array) is not None


def _remember_producer(array: Any, record: MetalProducerRecord) -> None:
    key = id(array)
    generation = next(_PRODUCER_GENERATION)

    # Bind the registry dict into the closure so the finalizer never touches the
    # module global ``_PRODUCERS`` -- during interpreter shutdown module globals
    # are reset to ``None`` while weakref callbacks can still fire, and a stale
    # ``_PRODUCERS.get`` on a torn-down global would raise. The closed-over
    # reference keeps the exact dict alive for the finalizer's lifetime.
    producers = _PRODUCERS

    def remove(_ref: weakref.ref[Any], *, stale_key: int = key, gen: int = generation) -> None:
        # Generation-checked eviction. When this producer's array is finalized
        # CPython may already have reused ``stale_key`` for a *newer* producer
        # array (the id()-reuse hazard). Popping unconditionally would delete
        # that live producer's record and leave its consumer with no
        # device-event edge -> NaN. Only evict when the slot still carries our
        # own generation, i.e. nothing newer has taken this key.
        entry = producers.get(stale_key)
        if entry is not None and entry.generation == gen:
            producers.pop(stale_key, None)

    _PRODUCERS[key] = _ProducerEntry(
        ref=weakref.ref(array, remove),
        record=record,
        generation=generation,
    )


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


def _launch_metadata(
    inputs: list[Any],
    *,
    dependency_metadata: MetalLaunchDependencyMetadata | None = None,
    command_buffer_domain: Any | None = None,
) -> MetalLaunchDependencyMetadata:
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
    return metadata


def _sync_decision_from_plan(
    *,
    producer: MetalProducerRecord,
    consumer_metadata: MetalLaunchDependencyMetadata,
    consumer_access: MetalBufferDependency,
) -> MetalLaunchSyncDecision:
    plan = plan_metal_buffer_sync(
        may_alias=True,
        same_command_buffer=(producer.command_buffer_domain == consumer_metadata.command_buffer_domain),
        producer_before_consumer=consumer_metadata.producer_before_consumer,
        resource_tracked=(producer.output_access.resource_tracked and consumer_access.resource_tracked),
    )
    return MetalLaunchSyncDecision(
        consumer_kernel_symbol=consumer_metadata.kernel_symbol,
        consumer_input_param_index=consumer_access.param_index,
        consumer_input_name=consumer_access.name,
        producer_kernel_symbol=producer.launch_metadata.kernel_symbol,
        producer_output_param_index=producer.output_access.param_index,
        producer_result_position=producer.output_access.result_position,
        action=plan.action,
        where=plan.where,
        host_sync_required=plan.host_sync_required,
        device_event_required=plan.device_event_required,
        external_materialization_required=False,
        reason=plan.reason,
        z3_proved=plan.z3_proved,
    )


def _planned_launch_dependencies(
    inputs: list[Any],
    metadata: MetalLaunchDependencyMetadata,
) -> tuple[tuple[MetalProducerRecord, MetalLaunchSyncDecision], ...]:
    planned: list[tuple[MetalProducerRecord, MetalLaunchSyncDecision]] = []
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
        planned.append(
            (
                producer,
                _sync_decision_from_plan(
                    producer=producer,
                    consumer_metadata=metadata,
                    consumer_access=consumer_access,
                ),
            )
        )
    return tuple(planned)


def inspect_mlx_tvm_ffi_launch_sync(
    inputs: list[Any],
    *,
    dependency_metadata: MetalLaunchDependencyMetadata | None = None,
    command_buffer_domain: Any | None = None,
) -> tuple[MetalLaunchSyncDecision, ...]:
    """Return scheduler sync decisions for a launch without wiring events."""

    metadata = _launch_metadata(
        inputs,
        dependency_metadata=dependency_metadata,
        command_buffer_domain=command_buffer_domain,
    )
    return tuple(decision for _producer, decision in _planned_launch_dependencies(inputs, metadata))


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

    metadata = _launch_metadata(
        inputs,
        dependency_metadata=dependency_metadata,
        command_buffer_domain=command_buffer_domain,
    )

    launch_sync_state = native.make_launch_sync_state()
    wait_edges: list[Any] = []

    for producer, decision in _planned_launch_dependencies(inputs, metadata):
        if decision.action == "none":
            continue
        if decision.action == "device_event":
            edge = native.make_sync_edge()
            producer.sync_state.add_signal_edge(edge)
            wait_edges.append(edge)
            continue
        raise RuntimeError(f"unsupported Metal graph sync plan for TVM-FFI launch: {decision}")

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


def register_mlx_tvm_ffi_output(
    output: Any,
    launch_sync_state: Any,
    *,
    dependency_metadata: MetalLaunchDependencyMetadata | None = None,
    command_buffer_domain: Any | None = None,
) -> None:
    metadata = dependency_metadata or _fallback_metadata(
        input_count=0,
        output_count=1,
        command_buffer_domain=command_buffer_domain,
    )
    if command_buffer_domain is not None and dependency_metadata is not None:
        metadata = with_command_buffer_domain(metadata, command_buffer_domain)
    if len(metadata.output_accesses) != 1:
        raise ValueError(
            "TVM-FFI Metal dependency metadata/output count mismatch: "
            f"{len(metadata.output_accesses)} metadata outputs for 1 runtime output"
        )
    _remember_producer(
        output,
        MetalProducerRecord(
            sync_state=launch_sync_state,
            launch_metadata=metadata,
            output_access=metadata.output_accesses[0],
        ),
    )


__all__ = [
    "MetalBufferDependency",
    "MetalLaunchDependencyMetadata",
    "MetalLaunchSyncDecision",
    "MetalProducerRecord",
    "clear_metal_graph_sync_state_for_tests",
    "has_mlx_tvm_ffi_producer",
    "inspect_mlx_tvm_ffi_launch_sync",
    "make_tvm_ffi_metal_dependency_metadata",
    "plan_mlx_tvm_ffi_launch",
    "register_mlx_tvm_ffi_output",
    "register_mlx_tvm_ffi_outputs",
    "with_command_buffer_domain",
]
