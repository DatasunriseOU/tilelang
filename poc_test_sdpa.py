import torch
from poc.torch_dynamo import register
import torch.nn.functional as F

register()

class Model(torch.nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

model = Model().eval()
q = torch.randn(1, 1, 64, 64, dtype=torch.float16)
k = torch.randn(1, 1, 64, 64, dtype=torch.float16)
v = torch.randn(1, 1, 64, 64, dtype=torch.float16)

compiled = torch.compile(model, backend="tilelang", fullgraph=True)
with torch.no_grad():
    y = compiled(q, k, v)
print("SUCCESS!")
