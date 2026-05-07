"""Regression tests for the two B2 wave fixes in :mod:`poc.torch_dynamo.fx_to_tilelang`.

These tests pin the *materialisation* contract — i.e. they assert that the
``FXToTileLang.run()`` pipeline produces a real ``tvm.tir.PrimFunc`` for
the four canonical FX-graph shapes that were previously routed to the
extern-replay launcher by two cooperating bugs:

1. **Bug 1** — closure-capture NameError: the ``T.prim_func`` annotation
   evaluator at ``tilelang/language/eager/builder.py:888`` walks
   ``func.__code__.co_freevars`` to resolve names referenced in the
   ``T.Tensor(shape, dtype)`` annotation. Python only marks a name as a
   freevar if the function body **references** it; the pre-fix kernel
   body never used ``shape``, so ``_eval_type`` raised
   ``NameError: name 'shape' is not defined`` and the orchestrator
   silently routed every unary / binary chain to the extern launcher.

2. **Bug 2** — sequential emitter only handled unary chains + a single
   binary-elementwise + unary tail. Reductions (``aten.sum``) and
   matmul (``aten.matmul`` / ``aten.mm``) hit the
   ``contains non-unary-elementwise ops`` ``NotImplementedError`` and
   silently fell through to extern.

Each test below symbolic-traces a tiny ``torch.fx`` graph, runs it through
``FXToTileLang(gm, example_inputs).run()``, and asserts that
``len(artifact.prim_funcs) > 0``. We do NOT assert numerical correctness
here — that's covered by ``poc/fx_to_tilelang/tests/test_e2e_torch_to_mlx.py``;
the point of this file is to lock the materialisation contract so future
regressions in either bug surface as a hard test failure rather than a
silent degradation back to extern fallback.
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


def _lower(fn, example_inputs):
    """Symbolic-trace ``fn`` and return the ``FusedKernelArtifact``."""
    import torch.fx as fx

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    gm = fx.symbolic_trace(fn)
    return FXToTileLang(gm, list(example_inputs)).run()


# ---------------------------------------------------------------------------
# Bug 1 regression — closure capture in ``_emit_sequential_region`` /
# ``_emit_sequential_binary``. Both kernels declare
# ``X: T.Tensor(shape, dtype)`` but never reference ``shape`` in the body
# (only ``n_elem``, ``BLOCK``, ``dtype``, op-list closures). Pre-fix this
# raised ``NameError: name 'shape' is not defined`` from the eager-builder
# annotation evaluator and routed to extern.
# ---------------------------------------------------------------------------


def test_bug1_unary_chain_lowers_to_real_prim_func() -> None:
    """B2 wave Bug 1 fix: ``relu(x)`` materialises a ``tvm.tir.PrimFunc``.

    Pre-fix: ``_materialize_subgraph failed: name 'shape' is not defined``;
    artifact.prim_funcs == ().
    """
    import torch

    x = torch.randn(8, 16, dtype=torch.float32)
    artifact = _lower(lambda a: torch.relu(a), [x])
    assert len(artifact.prim_funcs) >= 1, (
        f"expected at least one PrimFunc, got source={artifact.source!r}")
    assert "name 'shape' is not defined" not in artifact.source, (
        "Bug 1 regression: closure-capture NameError surfaced again. "
        f"source={artifact.source!r}")
    assert "tilelang.compile ok" in artifact.source, (
        f"expected 'tilelang.compile ok' in source, got {artifact.source!r}")


def test_bug1_binary_lowers_to_real_prim_func() -> None:
    """B2 wave Bug 1 fix: ``a + b`` materialises a ``tvm.tir.PrimFunc``.

    The binary kernel sat behind the same NameError as the unary kernel.
    """
    import torch

    a = torch.randn(8, 16, dtype=torch.float32)
    b = torch.randn(8, 16, dtype=torch.float32)
    artifact = _lower(lambda x, y: x + y, [a, b])
    assert len(artifact.prim_funcs) >= 1, (
        f"expected at least one PrimFunc, got source={artifact.source!r}")
    assert "name 'shape' is not defined" not in artifact.source, (
        "Bug 1 regression: binary-kernel closure-capture NameError "
        f"surfaced again. source={artifact.source!r}")


# ---------------------------------------------------------------------------
# Bug 2 regression — sequential emitter coverage of reductions and matmul.
# ---------------------------------------------------------------------------


def test_bug2_sum_reduction_lowers_to_real_prim_func() -> None:
    """B2 wave Bug 2 fix: full ``aten.sum`` reduces to a real PrimFunc.

    Pre-fix: ``_emit_sequential_region`` only accepted unary chains + a
    single binary + unary-tail; ``sum`` hit
    ``contains non-unary-elementwise ops`` and routed to extern.
    """
    import torch

    x = torch.randn(8, 16, dtype=torch.float32)
    artifact = _lower(lambda a: a.sum(), [x])
    assert len(artifact.prim_funcs) >= 1, (
        f"expected at least one PrimFunc for sum reduction, "
        f"got source={artifact.source!r}")
    assert "contains non-unary-elementwise ops" not in artifact.source, (
        "Bug 2 regression: reduction routing fell back to the wave-3 "
        f"extern path. source={artifact.source!r}")


def test_bug2_matmul_lowers_to_real_prim_func() -> None:
    """B2 wave Bug 2 fix: ``a @ a.T`` reduces to a real ``T.gemm`` PrimFunc.

    Pre-fix: the op_trace ``['t', 'matmul']`` (where ``t`` was NOT in
    ``_SEQUENTIAL_VIEW_OPS``) hit the ``contains non-unary-elementwise``
    branch. Fix absorbs ``t`` as a view + adds ``_emit_sequential_matmul``.
    """
    import torch

    torch.manual_seed(0)
    x = torch.randn(8, 16, dtype=torch.float16)
    artifact = _lower(lambda a: torch.matmul(a, a.t()), [x])
    assert len(artifact.prim_funcs) >= 1, (
        f"expected at least one PrimFunc for matmul, "
        f"got source={artifact.source!r}")
    assert "contains non-unary-elementwise ops" not in artifact.source, (
        "Bug 2 regression: matmul routing fell back to the wave-3 extern "
        f"path. source={artifact.source!r}")
