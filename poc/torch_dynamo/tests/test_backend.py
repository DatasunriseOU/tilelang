import torch
import pytest

from poc.torch_dynamo import UnsupportedFXOpError, _validate_graph, register

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


def test_validate_graph_accepts_supported_tensor_method_view():
    class ViewModule(torch.nn.Module):
        def forward(self, x):
            return x.view(2, 3)

    gm = torch.fx.symbolic_trace(ViewModule().eval())

    _validate_graph(gm)


def test_validate_graph_rejects_unsupported_tensor_method():
    class SortModule(torch.nn.Module):
        def forward(self, x):
            return x.sort()

    gm = torch.fx.symbolic_trace(SortModule().eval())

    with pytest.raises(UnsupportedFXOpError, match="method='sort'"):
        _validate_graph(gm)

if __name__ == "__main__":
    pytest.main([__file__])
