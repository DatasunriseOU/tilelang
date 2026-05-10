import torch
import torch.nn.functional as F
from poc.torch_dynamo import register

register()

@torch.compile(backend="tilelang")
def f(x):
    return F.dropout(x, p=0.5, training=True)

x = torch.randn(64, 64, device="cpu")
try:
    print("Testing dropout")
    f(x)
except Exception as e:
    print(e)
