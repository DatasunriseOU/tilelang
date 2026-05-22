import pytest
import torch
import torch.fx as fx
from poc.torch_dynamo.fx_to_tilelang import FXToTileLang
from poc.torch_dynamo._fusion_patterns import FUSION_PATTERNS, _FUSION_HITS
from torch.library import custom_op

# Register a mock custom op so fx can trace it
@custom_op("tilelang::qk_reduce", mutates_args=())
def qk_reduce(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.matmul(q, k.transpose(-1, -2))

@qk_reduce.register_fake
def _(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.matmul(q, k.transpose(-1, -2))

def test_qk_reduce_sm_scale():
    _FUSION_HITS.clear()

    def fn(q, k):
        # q: (m, k), k: (n, k)
        # qk_reduce(q, k) -> (m, n)
        # return qk_reduce * scale
        out = torch.ops.tilelang.qk_reduce(q, k)
        return out * 0.125

    q = torch.randn(128, 64)
    k = torch.randn(128, 64) # n=128, k=64
    gm = fx.symbolic_trace(fn)
    lowerer = FXToTileLang(gm, [q, k])
    artifact = lowerer.run()

    assert _FUSION_HITS.get("qk_reduce_sm_scale", 0) == 1
    # Run the launcher
    out_ref = fn(q, k)
    out_tl = artifact.launcher(q, k)
    torch.testing.assert_close(out_ref, out_tl, atol=1e-2, rtol=1e-2)


def test_qk_reduce_sm_scale_scalar_tensor_with_indices():
    _FUSION_HITS.clear()
    import torch._dynamo as dynamo

    def qk_reduce(q, k, indices):
        del indices
        return q @ k.transpose(-1, -2)

    dynamo.allow_in_graph(qk_reduce)

    def fn(q, k, indices, sm_scale):
        return qk_reduce(q, k, indices) * sm_scale

    q = torch.randn(16, 8)
    k = torch.randn(12, 8)
    indices = torch.arange(12)
    sm_scale = torch.tensor(0.125)
    try:
        exported = dynamo.export(fn)(q, k, indices, sm_scale)
    except TypeError:
        exported = dynamo.export(fn, q, k, indices, sm_scale)
    gm = (
        exported.graph_module
        if hasattr(exported, "graph_module")
        else exported[0]
    )
    lowerer = FXToTileLang(gm, [q, k, indices, sm_scale])
    artifact = lowerer.run()

    assert _FUSION_HITS.get("qk_reduce_sm_scale", 0) == 1
    assert getattr(artifact, "prim_funcs", ())
    assert "tilelang.compile ok" in str(getattr(artifact, "source", ""))
    out_ref = fn(q, k, indices, sm_scale)
    out_tl = artifact.launcher(q, k, indices, sm_scale)
    torch.testing.assert_close(out_ref, out_tl, atol=1e-2, rtol=1e-2)
