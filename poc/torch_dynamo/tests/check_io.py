import torch
import torch.fx as fx
from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

def fn(q, k):
    return torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)

q = torch.randn(128, 64)
k = torch.randn(128, 64)
gm = fx.symbolic_trace(fn)
lowerer = FXToTileLang(gm, [q, k])
lowerer.run()
regions = lowerer._partition_fusable_subgraphs()
print("3-op IO:", lowerer._derive_region_io(regions))

def fn2(q, k_t):
    return torch.softmax(torch.matmul(q, k_t), dim=-1)
gm2 = fx.symbolic_trace(fn2)
lowerer2 = FXToTileLang(gm2, [q, k])
lowerer2.run()
regions2 = lowerer2._partition_fusable_subgraphs()
print("2-op IO:", lowerer2._derive_region_io(regions2))
