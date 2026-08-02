from __future__ import annotations

from tilelang.analysis.metal_graph_sync import (
    clear_metal_graph_sync_state_for_tests,
    has_mlx_tvm_ffi_producer,
    inspect_mlx_tvm_ffi_launch_sync,
    make_tvm_ffi_metal_dependency_metadata,
    plan_mlx_tvm_ffi_launch,
    register_mlx_tvm_ffi_outputs,
)


class _FakeArray:
    pass


class _FakeEdge:
    pass


class _FakeLaunchState:
    def __init__(self):
        self.signal_edges = []

    def add_signal_edge(self, edge):
        self.signal_edges.append(edge)


class _FakeNative:
    def __init__(self):
        self.edges = []
        self.states = []

    def make_launch_sync_state(self):
        state = _FakeLaunchState()
        self.states.append(state)
        return state

    def make_sync_edge(self):
        edge = _FakeEdge()
        self.edges.append(edge)
        return edge


def test_same_command_buffer_domain_for_opaque_ffi_uses_encode_order():
    clear_metal_graph_sync_state_for_tests()
    native = _FakeNative()
    output = _FakeArray()
    producer_metadata = make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="producer",
        input_param_indices=(),
        output_param_indices=(0,),
        param_names=("C",),
        command_buffer_domain="domain-a",
    )
    consumer_metadata = make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="consumer",
        input_param_indices=(0,),
        output_param_indices=(1,),
        param_names=("C", "D"),
        command_buffer_domain="domain-a",
    )

    producer_state, producer_waits = plan_mlx_tvm_ffi_launch(
        native,
        [],
        dependency_metadata=producer_metadata,
    )
    register_mlx_tvm_ffi_outputs(
        [output],
        producer_state,
        dependency_metadata=producer_metadata,
    )
    _, consumer_waits = plan_mlx_tvm_ffi_launch(
        native,
        [output],
        dependency_metadata=consumer_metadata,
    )

    assert producer_waits == []
    assert len(native.edges) == 0
    assert consumer_waits == []
    assert producer_state.signal_edges == []


def test_cross_command_buffer_domain_wires_signal_and_wait_event():
    clear_metal_graph_sync_state_for_tests()
    native = _FakeNative()
    output = _FakeArray()
    producer_metadata = make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="producer",
        input_param_indices=(),
        output_param_indices=(0,),
        command_buffer_domain="domain-a",
    )
    consumer_metadata = make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="consumer",
        input_param_indices=(0,),
        output_param_indices=(1,),
        command_buffer_domain="domain-b",
    )

    producer_state, _ = plan_mlx_tvm_ffi_launch(
        native,
        [],
        dependency_metadata=producer_metadata,
    )
    register_mlx_tvm_ffi_outputs(
        [output],
        producer_state,
        dependency_metadata=producer_metadata,
    )
    decisions = inspect_mlx_tvm_ffi_launch_sync(
        [output],
        dependency_metadata=consumer_metadata,
    )
    _, consumer_waits = plan_mlx_tvm_ffi_launch(
        native,
        [output],
        dependency_metadata=consumer_metadata,
    )

    assert len(decisions) == 1
    assert decisions[0].action == "device_event"
    assert decisions[0].device_event_required is True
    assert decisions[0].host_sync_required is False
    assert decisions[0].external_materialization_required is False
    assert decisions[0].producer_kernel_symbol == "producer"
    assert decisions[0].consumer_kernel_symbol == "consumer"
    assert len(native.edges) == 1
    assert consumer_waits == native.edges
    assert producer_state.signal_edges == native.edges


def test_explicit_metadata_only_wires_actual_output_to_input_hazard():
    clear_metal_graph_sync_state_for_tests()
    native = _FakeNative()
    source = _FakeArray()
    output = _FakeArray()
    producer_metadata = make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="producer",
        input_param_indices=(0,),
        output_param_indices=(1,),
        param_names=("A", "B"),
    )
    consumer_metadata = make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="consumer",
        input_param_indices=(0, 1),
        output_param_indices=(2,),
        param_names=("A", "B", "C"),
    )

    producer_state, producer_waits = plan_mlx_tvm_ffi_launch(
        native,
        [source],
        dependency_metadata=producer_metadata,
    )
    register_mlx_tvm_ffi_outputs(
        [output],
        producer_state,
        dependency_metadata=producer_metadata,
    )
    _, consumer_waits = plan_mlx_tvm_ffi_launch(
        native,
        [source, output],
        dependency_metadata=consumer_metadata,
    )

    assert producer_waits == []
    assert len(native.edges) == 0
    assert consumer_waits == []
    assert producer_state.signal_edges == []


def test_dependency_metadata_preserves_tir_param_names_and_result_positions():
    metadata = make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="main",
        input_param_indices=(0, 2),
        output_param_indices=(1, 3),
        param_names=("A", "C", "B", "D"),
        command_buffer_domain="domain-a",
    )

    assert metadata.kernel_symbol == "main"
    assert metadata.command_buffer_domain == "domain-a"
    assert [(access.param_index, access.name, access.mode) for access in metadata.input_accesses] == [
        (0, "A", "read"),
        (2, "B", "read"),
    ]
    assert [(access.param_index, access.name, access.mode, access.result_position) for access in metadata.output_accesses] == [
        (1, "C", "write", 0),
        (3, "D", "write", 1),
    ]


def test_registered_native_output_identity_is_preserved_before_compaction():
    from tilelang.jit.adapter._mlx_tvm_ffi import _contiguous_mlx_input

    clear_metal_graph_sync_state_for_tests()
    native = _FakeNative()
    output = _FakeArray()
    producer_metadata = make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="producer",
        input_param_indices=(),
        output_param_indices=(0,),
    )
    producer_state, _ = plan_mlx_tvm_ffi_launch(
        native,
        [],
        dependency_metadata=producer_metadata,
    )
    register_mlx_tvm_ffi_outputs(
        [output],
        producer_state,
        dependency_metadata=producer_metadata,
    )

    assert has_mlx_tvm_ffi_producer(output)
    assert _contiguous_mlx_input(output) is output


class _IdableArray:
    """Weakref-able stand-in array so the registry's finalizer can be driven."""

    __slots__ = ("tag", "__weakref__")

    def __init__(self, tag):
        self.tag = tag


def test_stale_finalizer_does_not_evict_id_reused_producer_record():
    """A freed producer's finalizer must not delete a newer record at the same key.

    CPython reuses the ``id()`` of a freed object for the next allocation, so a
    later MLX array can land on the same registry key as a freed producer. The
    freed producer's weakref finalizer must be generation-checked: it may only
    evict the slot it created, never a newer producer that reused the key. Prior
    to the fix the finalizer popped the key unconditionally, deleting the live
    producer's record and leaving its consumer with no device-event edge.
    """

    import weakref

    from tilelang.analysis import metal_graph_sync as mg
    from tilelang.analysis.metal_graph_sync import (
        MetalProducerRecord,
        _ProducerEntry,
        _PRODUCERS,
    )

    clear_metal_graph_sync_state_for_tests()

    stale = _IdableArray("stale")
    stale_metadata = make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="stale_producer",
        input_param_indices=(),
        output_param_indices=(0,),
    )
    register_mlx_tvm_ffi_outputs(
        [stale],
        _FakeLaunchState(),
        dependency_metadata=stale_metadata,
    )
    key = id(stale)
    stale_entry = _PRODUCERS[key]
    # Capture the stale producer's real (generation-checked) weakref finalizer.
    stale_finalizer = stale_entry.ref.__callback__

    # A newer producer reuses the same registry key. This is what id() reuse
    # produces: after ``stale`` is freed, a later array gets the same id and is
    # registered as a producer at the same dict key with a *newer* generation.
    # We install it at ``key`` directly because the test allocator cannot be
    # made to win the id-reuse lottery deterministically.
    fresh = _IdableArray("fresh")
    fresh_metadata = make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="fresh_producer",
        input_param_indices=(),
        output_param_indices=(0,),
    )
    fresh_generation = next(mg._PRODUCER_GENERATION)
    fresh_record = MetalProducerRecord(
        sync_state=_FakeLaunchState(),
        launch_metadata=fresh_metadata,
        output_access=fresh_metadata.output_accesses[0],
    )
    _PRODUCERS[key] = _ProducerEntry(
        ref=weakref.ref(fresh),
        record=fresh_record,
        generation=fresh_generation,
    )
    assert _PRODUCERS[key].record.launch_metadata.kernel_symbol == "fresh_producer"

    # Stale finalizer fires now (stale array long dead, key reused by ``fresh``).
    stale_finalizer(stale_entry.ref)

    # The newer record must survive: an unconditional pop (the bug) would have
    # deleted it, dropping ``fresh``'s producer->consumer device-event edge.
    assert key in _PRODUCERS
    assert _PRODUCERS[key].record.launch_metadata.kernel_symbol == "fresh_producer"


def test_finalizer_evicts_its_own_record_when_array_freed():
    import gc

    from tilelang.analysis.metal_graph_sync import _PRODUCERS

    clear_metal_graph_sync_state_for_tests()
    output = _IdableArray("solo")
    metadata = make_tvm_ffi_metal_dependency_metadata(
        kernel_symbol="solo",
        input_param_indices=(),
        output_param_indices=(0,),
    )
    register_mlx_tvm_ffi_outputs(
        [output],
        _FakeLaunchState(),
        dependency_metadata=metadata,
    )
    key = id(output)
    assert key in _PRODUCERS
    del output
    gc.collect()
    assert key not in _PRODUCERS
