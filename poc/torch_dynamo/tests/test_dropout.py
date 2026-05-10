import pytest
import torch
import torch.nn.functional as F

def test_dropout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    def f(x):
        return F.dropout(x, p=0.5, training=True)
        
    x = torch.randn(4, 4, device="cuda")
    # Tracing to see what ops it generates
    import torch._dynamo as dynamo
    
    @dynamo.optimize("tilelang")
    def g(x):
        return f(x)
        
    try:
        g(x)
    except Exception as e:
        print("EXCEPTION:", type(e), e)

if __name__ == "__main__":
    test_dropout()
