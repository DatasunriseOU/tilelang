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

if __name__ == "__main__":
    test_topk_getitem()