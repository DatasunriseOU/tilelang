from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _register_backend():
    from poc.torch_dynamo import register

    register()


def _t_gemm_chain(x, w1, w2):
    return torch.nn.functional.gelu(x @ w1) @ w2


def test_fx_to_tilelang_partitions_t_gemm_chain_into_real_regions():
    from torch.fx.passes.shape_prop import ShapeProp

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    x = torch.randn(8, 16, dtype=torch.float32)
    w1 = torch.randn(16, 16, dtype=torch.float32)
    w2 = torch.randn(16, 8, dtype=torch.float32)
    gm = torch.fx.symbolic_trace(_t_gemm_chain)
    ShapeProp(gm).propagate(x, w1, w2)

    artifact = FXToTileLang(gm, [x, w1, w2]).run()
    script = "\n".join(func.script() for func in artifact.prim_funcs)

    assert len(artifact.prim_funcs) == 2
    assert "using extern slot" not in artifact.source
    assert script.count("T.gemm") >= 2


def test_t_gemm_chain_autograd_parity_vs_eager():
    def loss_fn(x, w1, w2):
        return _t_gemm_chain(x, w1, w2).sum()

    torch.manual_seed(0)
    x_ref = torch.randn(8, 16, dtype=torch.float32, requires_grad=True)
    w1_ref = torch.randn(16, 16, dtype=torch.float32, requires_grad=True)
    w2_ref = torch.randn(16, 8, dtype=torch.float32, requires_grad=True)

    expected = loss_fn(x_ref, w1_ref, w2_ref)
    expected.backward()
    expected_grads = tuple(
        tensor.grad.detach().clone() for tensor in (x_ref, w1_ref, w2_ref)
    )

    x = x_ref.detach().clone().requires_grad_(True)
    w1 = w1_ref.detach().clone().requires_grad_(True)
    w2 = w2_ref.detach().clone().requires_grad_(True)

    compiled = torch.compile(loss_fn, backend="tilelang", fullgraph=False)
    actual = compiled(x, w1, w2)
    actual.backward()

    torch.testing.assert_close(actual.detach(), expected.detach())
    for actual_grad, expected_grad in zip((x.grad, w1.grad, w2.grad), expected_grads):
        assert actual_grad is not None
        torch.testing.assert_close(actual_grad, expected_grad)
