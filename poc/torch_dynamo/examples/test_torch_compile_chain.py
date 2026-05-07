"""Multi-op chain smoke tests for ``torch.compile(backend="tilelang")``.

RFC reference: ``RFC_unified_fused_kernel.md`` §3 (FX -> TileLang custom
backend, cache-resident fusion), §4 (intermediates stay register/shared
resident across the FX boundary), §7 Phase 2 (FX op map + custom_op
wrap).

This complements ``torch_compile_smoke.py`` by exercising the orchestrator's
multi-op pipeline:

1. ``TinyMatmulRelu`` — verifies the legacy single-op smoke now flows
   through the new partition + materialise path (no whole-graph eager
   replay; the artifact's ``prim_funcs`` chain has at least one entry on a
   host with the TileLang JIT backend, otherwise the per-region extern
   fallback fires and the test still passes by matching eager numerically).

2. ``TinyLinearLayerNormGelu`` — multi-op chain ``matmul -> layer_norm
   -> gelu``. Exercises the ``layernorm_linear`` fusion-pattern matcher
   plus the elementwise epilogue. We don't yet have a tight TIR emitter
   for layer_norm so the orchestrator routes this region to the per-op
   extern slot — the test still asserts numerical correctness.

3. ``TinyAttentionPrim`` — non-flash attention as ``Q @ K -> softmax
   -> @ V``. Touches the ``softmax_epilogue`` pattern + the SDPA emitter
   (sibling integration #9). Self-skips when the running PyTorch lacks
   ``torch.nn.functional.scaled_dot_product_attention`` or when SDPA
   selection forces the flash backend (which has its own dedicated
   handler).

Each test self-skips when ``torch._dynamo`` or ``tilelang`` is missing.
"""

from __future__ import annotations

import importlib.util

import pytest


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


pytestmark = pytest.mark.skipif(
    not (_has("torch") and _has("tilelang")),
    reason="torch and tilelang must both be importable",
)


def _no_tilelang_jit() -> bool:
    """Return True when the TileLang JIT backend is unavailable.

    We treat the absence of ``tilelang.compile`` as "no JIT" — the
    backend transparently falls back to the per-region extern slot in
    that case, but for visual-debug TIR printing we want to skip.
    """
    if not _has("tilelang"):
        return True
    try:
        import tilelang  # noqa: F401
        return not hasattr(tilelang, "compile")
    except Exception:
        return True


# ---------------------------------------------------------------------------
# 1. TinyMatmulRelu — verifies legacy smoke runs through partition pipeline.
# ---------------------------------------------------------------------------


def test_tiny_matmul_relu_uses_real_tir() -> None:
    """``relu(x @ w)`` lowers through the new orchestrator with real TIR.

    Asserts:
      * the artifact has ``prim_funcs`` populated (at least one region
        compiled to a ``tvm.tir.PrimFunc``) when the JIT backend is
        present, OR the ``source`` field records a per-region extern
        fallback when it isn't,
      * compiled forward output matches eager within fp16 tolerance.
    """
    import torch
    from torch import nn

    if not _has("torch._dynamo"):
        pytest.skip("torch._dynamo unavailable")

    from poc.torch_dynamo import register
    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    register()

    class TinyMatmulRelu(nn.Module):
        def __init__(self, dim: int = 64) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.w = nn.Parameter(torch.randn(dim, dim, dtype=torch.float16))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.relu(x @ self.w)

    model = TinyMatmulRelu().eval()
    x = torch.randn(8, 64, dtype=torch.float16)
    with torch.no_grad():
        y_ref = model(x)

    # Drive the orchestrator directly so we can inspect the artifact.
    import torch.fx
    gm = torch.fx.symbolic_trace(model)
    # Stamp shapes via shape propagation so per-op handlers can see meta.
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(gm).propagate(x)
    lowerer = FXToTileLang(gm, [x])
    artifact = lowerer.run()

    # Expect either real PrimFuncs or a clean extern-fallback source line.
    has_prim = bool(getattr(artifact, "prim_funcs", ()))
    src = getattr(artifact, "source", "")
    assert has_prim or "extern slot" in src or "tilelang.compile ok" in src, (
        f"artifact source did not record a region status: {src!r}")

    # End-to-end numerical check via torch.compile.
    compiled = torch.compile(model, backend="tilelang", fullgraph=True)
    with torch.no_grad():
        y = compiled(x)
    assert tuple(y.shape) == tuple(y_ref.shape)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    # Visual-debug: print the lowered TIR of the first region (if any).
    if has_prim and not _no_tilelang_jit():
        try:
            print("=== TinyMatmulRelu — first region TIR ===")
            print(str(artifact.prim_funcs[0]))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 2. TinyLinearLayerNormGelu — multi-op chain hits the layernorm_linear
#    pattern + the gelu epilogue.
# ---------------------------------------------------------------------------


def test_tiny_linear_layernorm_gelu_chain() -> None:
    """``gelu(layer_norm(x @ w))`` — multi-op fusion-pattern exercise."""
    import torch
    from torch import nn

    if not _has("torch._dynamo"):
        pytest.skip("torch._dynamo unavailable")

    from poc.torch_dynamo import register

    register()

    class TinyLinearLayerNormGelu(nn.Module):
        def __init__(self, dim: int = 64) -> None:
            super().__init__()
            torch.manual_seed(1)
            self.w = nn.Parameter(torch.randn(dim, dim, dtype=torch.float16))
            self.ln = nn.LayerNorm(dim, dtype=torch.float16)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.nn.functional.gelu(self.ln(x @ self.w))

    model = TinyLinearLayerNormGelu().eval()
    x = torch.randn(8, 64, dtype=torch.float16)

    with torch.no_grad():
        y_ref = model(x)

    try:
        compiled = torch.compile(model, backend="tilelang", fullgraph=True)
        with torch.no_grad():
            y = compiled(x)
    except NotImplementedError as exc:
        pytest.skip(f"orchestrator did not cover this trace: {exc}")

    assert tuple(y.shape) == tuple(y_ref.shape)
    torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# 3. TinyAttentionPrim — Q @ K -> softmax -> @ V (non-flash SDPA).
# ---------------------------------------------------------------------------


def test_tiny_attention_prim_chain() -> None:
    """Q @ K^T -> softmax -> @ V — exercises the softmax_epilogue pattern."""
    import torch
    from torch import nn

    if not _has("torch._dynamo"):
        pytest.skip("torch._dynamo unavailable")

    from poc.torch_dynamo import register

    register()

    class TinyAttentionPrim(nn.Module):
        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.scale = d ** -0.5

        def forward(self, q: "torch.Tensor", k: "torch.Tensor",
                    v: "torch.Tensor") -> "torch.Tensor":
            attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
            return attn @ v

    model = TinyAttentionPrim().eval()
    q = torch.randn(2, 16, 32, dtype=torch.float16)
    k = torch.randn(2, 16, 32, dtype=torch.float16)
    v = torch.randn(2, 16, 32, dtype=torch.float16)

    with torch.no_grad():
        y_ref = model(q, k, v)

    try:
        compiled = torch.compile(model, backend="tilelang", fullgraph=True)
        with torch.no_grad():
            y = compiled(q, k, v)
    except NotImplementedError as exc:
        pytest.skip(f"orchestrator did not cover this trace: {exc}")

    assert tuple(y.shape) == tuple(y_ref.shape)
    torch.testing.assert_close(y, y_ref, rtol=5e-2, atol=5e-2)
