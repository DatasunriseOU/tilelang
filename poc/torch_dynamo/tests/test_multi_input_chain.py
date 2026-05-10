import pytest
import torch
from poc.torch_dynamo import register

@pytest.fixture(autouse=True)
def _register_backend():
    register()

def test_multi_input_chain():
    class Model(torch.nn.Module):
        def forward(self, a, b, c, d):
            return (a + b) * (c + d)

    mod = Model().cpu()
    opt_mod = torch.compile(mod, backend="tilelang")

    a = torch.randn(16, 16, device="cpu")
    b = torch.randn(16, 16, device="cpu")
    c = torch.randn(16, 16, device="cpu")
    d = torch.randn(16, 16, device="cpu")

    expected = mod(a, b, c, d)
    actual = opt_mod(a, b, c, d)

    torch.testing.assert_close(actual, expected)

if __name__ == "__main__":
    pytest.main([__file__])
