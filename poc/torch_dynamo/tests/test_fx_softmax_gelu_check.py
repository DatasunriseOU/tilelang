import torch
import torch.nn.functional as F
from poc.torch_dynamo import register

def test_check_compilation():
    register()

    def simple_softmax(x):
        return F.softmax(x, dim=-1)

    compiled_softmax = torch.compile(simple_softmax, backend="tilelang")
    x = torch.randn(10, 10, device="cpu")
    out = compiled_softmax(x)
    print("SOFTMAX SOURCE:", out)

    def simple_gelu(x):
        return F.gelu(x)

    compiled_gelu = torch.compile(simple_gelu, backend="tilelang")
    x = torch.randn(10, 10, device="cpu")
    out = compiled_gelu(x)
    print("GELU SOURCE:", out)

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", "-s", __file__])
