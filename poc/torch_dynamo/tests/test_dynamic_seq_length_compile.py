from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _register_backend():
    from poc.torch_dynamo import register

    register()


def _exported_graph_module(exported):
    return exported.graph_module if hasattr(exported, "graph_module") else exported[0]


def test_fx_shape_spec_preserves_symbolic_sequence_dim():
    from poc.torch_dynamo.fx_to_tilelang import _spec_from_node

    def fn(x):
        return torch.nn.functional.gelu(x)

    x = torch.randn(2, 5, 8, dtype=torch.float32)
    torch._dynamo.mark_dynamic(x, 1)
    exported = torch._dynamo.export(
        fn,
        aten_graph=True,
        assume_static_by_default=False,
    )(x)
    gm = _exported_graph_module(exported)
    placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")

    spec = _spec_from_node(placeholder)

    assert not isinstance(spec.shape[1], int)
    assert "s" in repr(spec.shape[1])


def test_dynamic_seq_length_compile_runs_multiple_lengths():
    def fn(x):
        return torch.nn.functional.gelu(x)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang", dynamic=True, fullgraph=True)

    for seq_len in (5, 7):
        x = torch.randn(2, seq_len, 8, dtype=torch.float32)
        actual = compiled(x)
        expected = fn(x)
        torch.testing.assert_close(actual, expected)
