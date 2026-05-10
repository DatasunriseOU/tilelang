import torch
import torch.nn.functional as F
import pytest

from poc.torch_dynamo import register

@pytest.fixture(autouse=True)
def _register_backend():
    register()

def test_matmul_mapping():
    def simple_matmul(x, y):
        return torch.matmul(x, y)

    compiled_matmul = torch.compile(simple_matmul, backend="tilelang")
    
    x = torch.randn(10, 10, device="cpu")
    y = torch.randn(10, 10, device="cpu")
    
    # First run triggers compilation
    out = compiled_matmul(x, y)
    
    torch.testing.assert_close(out, torch.matmul(x, y))

def test_layer_norm_mapping():
    def simple_layer_norm(x):
        return F.layer_norm(x, [10])

    compiled_ln = torch.compile(simple_layer_norm, backend="tilelang")
    
    x = torch.randn(10, 10, device="cpu")
    
    # First run triggers compilation
    out = compiled_ln(x)
    
    torch.testing.assert_close(out, F.layer_norm(x, [10]))

def test_softmax_mapping():
    def simple_softmax(x):
        return F.softmax(x, dim=-1)

    compiled_softmax = torch.compile(simple_softmax, backend="tilelang")
    
    x = torch.randn(10, 10, device="cpu")
    
    # First run triggers compilation
    out = compiled_softmax(x)
    
    torch.testing.assert_close(out, F.softmax(x, dim=-1))

def test_gelu_mapping():
    def simple_gelu(x):
        return F.gelu(x)

    compiled_gelu = torch.compile(simple_gelu, backend="tilelang")
    
    x = torch.randn(10, 10, device="cpu")
    
    # First run triggers compilation
    out = compiled_gelu(x)
    
    torch.testing.assert_close(out, F.gelu(x))

def test_sdpa_mapping():
    def simple_sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    compiled_sdpa = torch.compile(simple_sdpa, backend="tilelang")
    
    q = torch.randn(2, 4, 16, 32, device="cpu", requires_grad=False)
    k = torch.randn(2, 4, 16, 32, device="cpu", requires_grad=False)
    v = torch.randn(2, 4, 16, 32, device="cpu", requires_grad=False)
    
    # First run triggers compilation
    out = compiled_sdpa(q, k, v)
    
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v))

if __name__ == "__main__":
    pytest.main([__file__])
