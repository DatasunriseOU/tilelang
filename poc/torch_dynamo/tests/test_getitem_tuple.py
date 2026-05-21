import torch
import torch._dynamo
import pytest

from poc.torch_dynamo import register

@pytest.fixture(autouse=True)
def _register_backend():
    register()

def test_topk_getitem():
    def fn(x):
        v, _ = torch.topk(x, 2, dim=-1)
        return v + 1.0

    x = torch.randn(4, 8, device="cpu")

    torch._dynamo.reset()

    opt_fn = torch.compile(backend="tilelang")(fn)

    expected = fn(x)
    actual = opt_fn(x)

    torch.testing.assert_close(actual, expected)


def test_topk_indices_getitem_preserves_int64_spec():
    def fn(x):
        _, idx = torch.topk(x, 2, dim=-1)
        return idx

    import torch.fx
    from torch.fx.passes.shape_prop import ShapeProp

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    x = torch.randn(4, 8, device="cpu")
    gm = torch.fx.symbolic_trace(fn)
    ShapeProp(gm).propagate(x)

    artifact = FXToTileLang(gm, [x]).run()

    assert len(artifact.output_specs) == 1
    assert artifact.output_specs[0].shape == (4, 2)
    assert artifact.output_specs[0].dtype == "int64"

if __name__ == "__main__":
    test_topk_getitem()
