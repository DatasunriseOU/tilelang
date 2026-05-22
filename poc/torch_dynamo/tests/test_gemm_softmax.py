import pytest
import torch
import torch.fx as fx
from poc.torch_dynamo.fx_to_tilelang import FXToTileLang
from poc.torch_dynamo._fusion_patterns import FUSION_PATTERNS, _FUSION_HITS

def test_gemm_softmax_3op():
    _FUSION_HITS.clear()
    def fn(q, k):
        # q: (m, k), k: (n, k)
        # transpose: (k, n)
        # matmul: (m, n)
        # softmax: (m, n)
        return torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)

    q = torch.randn(128, 64)
    k = torch.randn(128, 64) # n=128, k=64
    gm = fx.symbolic_trace(fn)
    lowerer = FXToTileLang(gm, [q, k])
    artifact = lowerer.run()
    
    assert _FUSION_HITS.get("gemm_softmax", 0) == 1
    # Run the launcher
    out_ref = fn(q, k)
    out_tl = artifact.launcher(q, k)
    torch.testing.assert_close(out_ref, out_tl, atol=1e-2, rtol=1e-2)


def test_gemm_softmax_3op_compiles_without_extern_fallback():
    def fn(q, k):
        return torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)

    q = torch.randn(16, 8)
    k = torch.randn(16, 8)
    gm = fx.symbolic_trace(fn)
    lowerer = FXToTileLang(gm, [q, k])
    artifact = lowerer.run()

    source = getattr(artifact, "source", "")
    assert "tilelang.compile failed" not in source
    assert "extern slot" not in source
    torch.testing.assert_close(
        artifact.launcher(q, k),
        fn(q, k),
        rtol=1e-2,
        atol=1e-2,
    )


def test_batched_gemm_softmax_3op_compiles_without_extern_fallback():
    def fn(q, k):
        return torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)

    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    gm = fx.symbolic_trace(fn)
    lowerer = FXToTileLang(gm, [q, k])
    artifact = lowerer.run()

    source = getattr(artifact, "source", "")
    assert "tilelang.compile failed" not in source
    assert "extern slot" not in source
    torch.testing.assert_close(
        artifact.launcher(q, k),
        fn(q, k),
        rtol=1e-2,
        atol=1e-2,
    )


def test_gemm_softmax_2op():
    _FUSION_HITS.clear()
    def fn(q, k_t):
        # q: (m, k), k_t: (k, n)
        return torch.softmax(torch.matmul(q, k_t), dim=-1)

    q = torch.randn(128, 64)
    k_t = torch.randn(64, 128)
    gm = fx.symbolic_trace(fn)
    lowerer = FXToTileLang(gm, [q, k_t])
    artifact = lowerer.run()
    
    assert _FUSION_HITS.get("softmax_epilogue", 0) == 1
    # Run the launcher
    out_ref = fn(q, k_t)
    out_tl = artifact.launcher(q, k_t)
    torch.testing.assert_close(out_ref, out_tl, atol=1e-2, rtol=1e-2)
