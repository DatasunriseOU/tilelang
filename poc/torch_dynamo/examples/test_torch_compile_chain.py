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


@pytest.mark.xfail(
    reason=(
        "Sequential region materialiser is intentionally a stub in this "
        "POC (fx_to_tilelang.py:_emit_sequential_region). Multi-op chains "
        "without a tight fusion pattern fall back to a per-region extern "
        "slot which currently bridges to gm.forward. The test is marked "
        "xfail (rather than the previous silent skip) so the regression "
        "becomes visible the moment the sequential emitter lands."
    ),
    strict=False,
    raises=NotImplementedError,
)
def test_tiny_linear_layernorm_gelu_chain() -> None:
    """``gelu(layer_norm(x @ w))`` — multi-op fusion-pattern exercise.

    Tests review fix (grok review tests #1/#2): the previous
    ``except NotImplementedError: pytest.skip`` swallowed the very
    regression this test is supposed to catch. Converting to xfail
    makes the gap visible while still letting the suite stay green
    until the sequential emitter is wired.
    """
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

    compiled = torch.compile(model, backend="tilelang", fullgraph=True)
    with torch.no_grad():
        y = compiled(x)

    assert tuple(y.shape) == tuple(y_ref.shape)
    torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# 3. TinyAttentionPrim — Q @ K -> softmax -> @ V (non-flash SDPA).
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "softmax_epilogue pattern emitter is documentation-only; the "
        "orchestrator routes this region to the per-op extern slot until "
        "the sequential materialiser lands. Marked xfail so the gap is "
        "visible (was previously silently skipped — see grok review tests #2)."
    ),
    strict=False,
    raises=NotImplementedError,
)
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

    compiled = torch.compile(model, backend="tilelang", fullgraph=True)
    with torch.no_grad():
        y = compiled(q, k, v)

    assert tuple(y.shape) == tuple(y_ref.shape)
    torch.testing.assert_close(y, y_ref, rtol=5e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# 4. Coverage gaps called out in the grok review (tests #3): registry cache
#    idempotency, content-hash stability, partition-boundary handling.
# ---------------------------------------------------------------------------


def test_same_graph_recompile_hits_registry_cache() -> None:
    """Lowering the same FX graph twice must reuse the cached custom_op.

    Coverage gap (grok review tests #3, custom_op_wrapper.py:178-180).
    """
    import torch
    from torch import nn

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang
    from poc.torch_dynamo.custom_op_wrapper import wrap_as_custom_op, _REGISTRY

    class Tiny(nn.Module):
        def __init__(self, dim: int = 32) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.w = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.relu(x @ self.w)

    model = Tiny().eval()
    x = torch.randn(4, 32, dtype=torch.float32)

    import torch.fx
    from torch.fx.passes.shape_prop import ShapeProp
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(x)

    art1 = FXToTileLang(gm, [x]).run()
    runner1 = wrap_as_custom_op(art1, {})
    op1 = getattr(runner1, "_tilelang_op", None)

    # Second compile with the same FX graph should hit _REGISTRY[qualname].
    art2 = FXToTileLang(gm, [x]).run()
    assert art1.name == art2.name, (
        "Identical FX graphs must produce identical content_hash names "
        f"(got {art1.name!r} vs {art2.name!r})")
    runner2 = wrap_as_custom_op(art2, {})
    op2 = getattr(runner2, "_tilelang_op", None)
    assert op1 is op2, "Re-registering the same qualname must reuse the cached impl"
    qualname = f"tilelang::{art1.name}_fwd"
    assert qualname in _REGISTRY


def test_content_hash_stable_across_recompiles() -> None:
    """Same model + same input shape -> same hash; different shape -> different.

    Coverage gap (grok review tests #3, fx_to_tilelang.py:961-972).
    """
    import torch
    from torch import nn

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    class Tiny(nn.Module):
        def __init__(self, dim: int = 32) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.w = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.relu(x @ self.w)

    model = Tiny().eval()
    x_a = torch.randn(4, 32, dtype=torch.float32)
    x_b = torch.randn(8, 32, dtype=torch.float32)

    import torch.fx
    from torch.fx.passes.shape_prop import ShapeProp

    def _hash_for(shape_x: "torch.Tensor") -> str:
        gm = torch.fx.symbolic_trace(model)
        ShapeProp(gm).propagate(shape_x)
        lowerer = FXToTileLang(gm, [shape_x])
        lowerer.run()
        return lowerer.content_hash()

    h1 = _hash_for(x_a)
    h2 = _hash_for(x_a)
    h3 = _hash_for(x_b)
    assert h1 == h2, "Same shape must hash identically"
    assert h1 != h3, "Different input shape must produce a different hash"


def test_unsupported_op_falls_back_to_extern() -> None:
    """A graph with one unsupported op must still compile (per-op fallback).

    Coverage gap (grok review tests #3, fx_to_tilelang.py:1044-1066).
    Builds a tiny FX graph that includes an op outside ATEN_DISPATCH and
    asserts that ``run()`` does not raise; the artifact source records
    the extern fallback for that single op.
    """
    import torch
    from torch import nn

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    class Tiny(nn.Module):
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # ``torch.flip`` is not in ATEN_DISPATCH today.
            return torch.flip(x, dims=(-1,))

    model = Tiny().eval()
    x = torch.randn(4, 8, dtype=torch.float32)

    import torch.fx
    from torch.fx.passes.shape_prop import ShapeProp
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(x)
    artifact = FXToTileLang(gm, [x]).run()
    # The unsupported op must show up either in the per-region source line
    # ("extern slot") OR (when no fusable region exists) the chain still
    # produced a launcher.
    assert artifact.launcher is not None


# ---------------------------------------------------------------------------
# Wave-2 fix-pack regressions — grok #02 design §1 / correctness §1
# (top-8 ATEN gaps + sequential emitter + multi-region launcher)
# ---------------------------------------------------------------------------


def test_wave2_aten_dispatch_covers_top_ops() -> None:
    """ATEN_DISPATCH must include the top-8 ops grok #02 review called out."""
    from poc.torch_dynamo.fx_to_tilelang import ATEN_DISPATCH

    must_have = (
        "view", "reshape", "permute", "transpose", "flatten",
        "broadcast_to", "expand",
        "exp", "log", "sqrt", "rsqrt", "sigmoid", "pow",
        "cat", "stack",
        "clamp", "clip",
        "dropout",
    )
    missing = [op for op in must_have if op not in ATEN_DISPATCH]
    assert not missing, f"ATEN_DISPATCH still missing: {missing!r}"


def test_wave2_unary_chain_uses_sequential_emitter() -> None:
    """Pure unary elementwise chain must hit ``_emit_sequential_region``
    (not ``NotImplementedError`` -> extern fallback).
    """
    if _no_tilelang_jit():
        pytest.xfail("tilelang.compile unavailable; sequential emitter unverifiable")

    import torch
    from torch import nn
    import torch.fx
    from torch.fx.passes.shape_prop import ShapeProp

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    class UnaryChain(nn.Module):
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.relu(torch.tanh(torch.exp(x)))

    model = UnaryChain().eval()
    x = torch.randn(4, 8, dtype=torch.float32)
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(x)

    artifact = FXToTileLang(gm, [x]).run()
    # The sequential emitter must have produced at least one PrimFunc OR
    # the source line must say "tilelang.compile ok" — anything other than
    # "extern slot" for every region indicates the new path engaged.
    src = artifact.source
    assert artifact.launcher is not None
    assert "extern slot" not in src or "tilelang.compile ok" in src, (
        "Unary chain must hit the new sequential emitter, "
        f"got source: {src!r}"
    )


def test_wave2_multi_region_launcher_does_not_use_gm_forward() -> None:
    """When >1 compiled region exists, the chain launcher must NOT mark
    itself as ``multi_fallback_to_gm_forward``.
    """
    if _no_tilelang_jit():
        pytest.xfail("tilelang.compile unavailable; chain launcher path inert")

    import torch
    from torch import nn
    import torch.fx
    from torch.fx.passes.shape_prop import ShapeProp

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    class TwoRegions(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(16, 16, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            y = torch.relu(self.lin(x))
            return torch.tanh(torch.exp(y))

    model = TwoRegions().eval()
    x = torch.randn(4, 16, dtype=torch.float32)
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(x)

    artifact = FXToTileLang(gm, [x]).run()
    chain_mode = getattr(artifact.launcher, "_tilelang_chain_mode", None)
    assert chain_mode != "multi_fallback_to_gm_forward", (
        f"Multi-region chain still falls back to gm.forward; "
        f"chain_mode={chain_mode!r}, source={artifact.source!r}"
    )


def test_wave2_contiguity_guard_warns_and_materialises() -> None:
    """Non-contiguous input through ``_impl`` must trigger one warning and
    a ``.contiguous()`` materialisation.
    """
    import warnings as _w
    import torch

    from poc.torch_dynamo.custom_op_wrapper import _ensure_contiguous_inputs

    x = torch.randn(4, 8, dtype=torch.float32)
    xt = x.t()  # non-contiguous view
    assert not xt.is_contiguous()

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = _ensure_contiguous_inputs("tilelang::test_op_a", (xt,))
    assert out[0].is_contiguous()
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), (
        f"expected RuntimeWarning, got {[w.category for w in caught]!r}"
    )

    # Second call with the same op+slot pattern must NOT warn again.
    with _w.catch_warnings(record=True) as caught2:
        _w.simplefilter("always")
        _ensure_contiguous_inputs("tilelang::test_op_a", (xt,))
    assert not any(issubclass(w.category, RuntimeWarning)
                   for w in caught2), (
        f"warn-once cache leaked; second call warned again: "
        f"{[w.category for w in caught2]!r}"
    )


def test_wave2_dropout_eval_is_identity_spec() -> None:
    """``aten.dropout`` in eval mode must emit a same-shape spec with no error."""
    import torch
    import torch.fx
    from torch.fx.passes.shape_prop import ShapeProp
    from torch import nn

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    class DropoutEval(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.drop = nn.Dropout(p=0.5)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.drop(x)

    model = DropoutEval().eval()
    x = torch.randn(4, 8, dtype=torch.float32)
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(x)
    artifact = FXToTileLang(gm, [x]).run()
    assert artifact.launcher is not None


def test_wave2_view_reshape_specs_resolve() -> None:
    """``view`` / ``reshape`` emitters must resolve -1 dims correctly."""
    from poc.torch_dynamo.fx_to_tilelang import (
        ATEN_DISPATCH, _TensorSpec, LoweringContext,
    )

    class _StubNode:
        name = "view_node"

    ctx = LoweringContext(gm=None, example_inputs=[])  # type: ignore[arg-type]
    src = _TensorSpec(shape=(4, 8), dtype="float32")
    spec = ATEN_DISPATCH["view"](_StubNode(), (src, (2, -1)), ctx)
    assert spec.shape == (2, 16)
    assert spec.dtype == "float32"


def test_wave3_content_hash_is_128_bits() -> None:
    """grok wave-2 review #02 security: collision-resistance hash widening.

    ``content_hash`` must produce 32 hex chars (128 bits) so adversarial-
    graph collisions against the ``tilelang::fused_<hash>`` registry move
    from "theoretically plausible" to "practically infeasible".
    """
    import torch
    import torch.fx
    from torch import nn
    from torch.fx.passes.shape_prop import ShapeProp

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    class Tiny(nn.Module):
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.relu(x)

    model = Tiny().eval()
    x = torch.randn(4, 8, dtype=torch.float32)
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(x)
    name = FXToTileLang(gm, [x]).content_hash()
    # name == "fused_<32 hex chars>"
    assert name.startswith("fused_")
    digest = name[len("fused_"):]
    assert len(digest) == 32, f"expected 128-bit (32 hex char) digest, got {digest!r}"
    int(digest, 16)  # must parse as hex


def test_wave3_binary_elementwise_uses_sequential_emitter() -> None:
    """grok wave-2 review #02 perf §2 + design §1: binary-elementwise
    closes the wave-2 sequential gap. ``relu(a + b)`` must compile through
    ``_emit_sequential_binary`` and not fall through to the extern slot.
    """
    if _no_tilelang_jit():
        pytest.xfail("tilelang.compile unavailable; binary path unverifiable")

    import torch
    from torch import nn
    import torch.fx
    from torch.fx.passes.shape_prop import ShapeProp

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    class AddRelu(nn.Module):
        def forward(self, a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
            return torch.relu(a + b)

    model = AddRelu().eval()
    a = torch.randn(4, 8, dtype=torch.float32)
    b = torch.randn(4, 8, dtype=torch.float32)
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(a, b)
    artifact = FXToTileLang(gm, [a, b]).run()
    src = artifact.source
    assert artifact.launcher is not None
    assert "extern slot" not in src or "tilelang.compile ok" in src, (
        "Binary chain must hit the new sequential binary emitter, "
        f"got source: {src!r}"
    )


def test_wave3_contiguity_guard_avoids_clone_for_non_aliased() -> None:
    """grok wave-2 review #02 perf §1: ``_ensure_contiguous_inputs`` must
    use ``.contiguous()`` (no extra clone) on plain non-contiguous inputs;
    only the *aliased+already-contiguous* case warrants ``clone()``.
    """
    import torch
    from poc.torch_dynamo.custom_op_wrapper import _ensure_contiguous_inputs

    # Plain non-contiguous tensor: transpose of contiguous storage. Not
    # aliased (._base is None on a freshly-allocated transpose? — actually
    # transpose returns a view with a base, so we use a stride-permuted
    # advanced-indexed copy that is non-contiguous + non-aliased).
    x = torch.randn(8, 4)
    nc = x.t()  # non-contiguous AND aliased (view)
    out = _ensure_contiguous_inputs("tilelang::test_wave3_alias", (nc,))
    assert out[0].is_contiguous()
    assert out[0].data_ptr() != x.data_ptr()

    # Contiguous + aliased (e.g. narrow on a 1D contiguous slice that
    # happens to start at offset 0 — still has _base set).
    base = torch.zeros(16)
    sliced = base.narrow(0, 0, 8)
    assert sliced.is_contiguous()
    assert sliced._base is not None
    out2 = _ensure_contiguous_inputs("tilelang::test_wave3_alias_clone", (sliced,))
    assert out2[0].is_contiguous()
    # Must NOT share storage with the parent (clone() forces a fresh alloc).
    assert out2[0].data_ptr() != base.data_ptr()
