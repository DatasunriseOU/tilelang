import torch
import torch.nn.functional as F
from poc.torch_dynamo import register

register()

def simple_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)

compiled_sdpa = torch.compile(simple_sdpa, backend="tilelang")

q = torch.randn(2, 4, 16, 32, device="cpu", requires_grad=False)
k = torch.randn(2, 4, 16, 32, device="cpu", requires_grad=False)
v = torch.randn(2, 4, 16, 32, device="cpu", requires_grad=False)

out = compiled_sdpa(q, k, v)
print("SUCCESS")
