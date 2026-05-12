from __future__ import annotations

from tilelang.analysis.metal_graph_sync import (
    clear_metal_graph_sync_state_for_tests,
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


def test_same_command_buffer_domain_for_opaque_ffi_wires_device_event():
    clear_metal_graph_sync_state_for_tests()
    native = _FakeNative()
    output = _FakeArray()

    producer_state, producer_waits = plan_mlx_tvm_ffi_launch(
        native,
        [],
        command_buffer_domain="domain-a",
    )
    register_mlx_tvm_ffi_outputs(
        [output],
        producer_state,
        command_buffer_domain="domain-a",
    )
    _, consumer_waits = plan_mlx_tvm_ffi_launch(
        native,
        [output],
        command_buffer_domain="domain-a",
    )

    assert producer_waits == []
    assert len(native.edges) == 1
    assert consumer_waits == native.edges
    assert producer_state.signal_edges == native.edges


def test_cross_command_buffer_domain_wires_signal_and_wait_event():
    clear_metal_graph_sync_state_for_tests()
    native = _FakeNative()
    output = _FakeArray()

    producer_state, _ = plan_mlx_tvm_ffi_launch(
        native,
        [],
        command_buffer_domain="domain-a",
    )
    register_mlx_tvm_ffi_outputs(
        [output],
        producer_state,
        command_buffer_domain="domain-a",
    )
    _, consumer_waits = plan_mlx_tvm_ffi_launch(
        native,
        [output],
        command_buffer_domain="domain-b",
    )

    assert len(native.edges) == 1
    assert consumer_waits == native.edges
    assert producer_state.signal_edges == native.edges
