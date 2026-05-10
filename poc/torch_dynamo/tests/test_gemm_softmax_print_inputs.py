import torch
import torch.fx as fx
from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

def fn(q, k):
    return torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)

q = torch.randn(128, 64)
k = torch.randn(128, 64) # n=128, k=64
gm = fx.symbolic_trace(fn)
lowerer = FXToTileLang(gm, [q, k])
for node in lowerer._linearised_nodes():
    handler = getattr(lowerer, f"on_{node.op}", None)
    if handler: handler(node)
regions = lowerer._partition_fusable_subgraphs()
region_io = lowerer._derive_region_io(regions)
print("REGION IO:", region_io)
