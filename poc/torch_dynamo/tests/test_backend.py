import torch
import pytest

from poc.torch_dynamo import register

@pytest.fixture(autouse=True)
def _register_backend():
    register()

def test_tilelang_backend_smoke():
    def simple_add(x, y):
        return x + y

    compiled_add = torch.compile(simple_add, backend="tilelang")
    
    x = torch.randn(10, 10, device="cpu")
    y = torch.randn(10, 10, device="cpu")
    
    # First run triggers compilation
    out = compiled_add(x, y)
    
    torch.testing.assert_close(out, x + y)

def test_tilelang_backend_identity():
    def identity(x):
        return x

    compiled_id = torch.compile(identity, backend="tilelang")
    
    x = torch.randn(10, 10, device="cpu")
    
    # First run triggers compilation
    out = compiled_id(x)
    
    torch.testing.assert_close(out, x)

if __name__ == "__main__":
    pytest.main([__file__])
