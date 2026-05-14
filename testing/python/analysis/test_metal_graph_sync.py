from __future__ import annotations

from tilelang.analysis.metal_graph_sync import (
    clear_metal_graph_sync_state_for_tests,
    has_mlx_tvm_ffi_producer,
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
    _, consumer_waits = plan_mlx_tvm_ffi_launch(
        native,
        [output],
        dependency_metadata=consumer_metadata,
    )

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
    assert [
        (access.param_index, access.name, access.mode, access.result_position)
        for access in metadata.output_accesses
    ] == [
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
