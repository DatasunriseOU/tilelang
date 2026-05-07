"""End-to-end verification of the torch.fx -> TileLang lowering path.

Investigation notes (May 2026)
==============================

The user's task brief asked us to verify ``poc/fx_to_tilelang/`` and a top
level ``fx_to_tilelang.lower(graph_module) -> tvm.tir.PrimFunc`` entry point.
**That directory and that exact entry point do not exist in this tree.**

What actually exists is ``poc/torch_dynamo/fx_to_tilelang.py`` exposing the
class :class:`poc.torch_dynamo.fx_to_tilelang.FXToTileLang` whose public API
is::

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang
    artifact = FXToTileLang(gm, example_inputs).run()
    # artifact is a FusedKernelArtifact (defined in
    # poc.torch_dynamo.custom_op_wrapper.FusedKernelArtifact); the
    # callable launcher is artifact.launcher(*example_inputs)
    # artifact.prim_funcs is a tuple[tvm.tir.PrimFunc, ...] (possibly empty)

This file lives at the path requested in the brief
(``poc/fx_to_tilelang/tests/test_e2e_torch_to_mlx.py``); we keep it here so
the brief's directory layout is correct, while importing from the real
implementation site under ``poc.torch_dynamo``.

What this test verifies
-----------------------

Each case below symbolic-traces a tiny ``torch.fx`` graph, runs it through
``FXToTileLang(gm, example_inputs).run()``, calls
``artifact.launcher(*inputs)``, and compares numerically against the eager
``torch.*`` reference. We also record whether the lowering produced a real
``tvm.tir.PrimFunc`` (``len(artifact.prim_funcs) > 0``) or fell back to
the per-op extern launcher (``prim_funcs == ()``); both paths must
produce numerically correct output, but only the first exercises the
TileLang TIR codegen all the way to native target compilation.

Highest-level op reached
------------------------

* ``relu``       -> launcher OK numerically, falls back to extern slot
                   (``_materialize_subgraph`` raises ``NameError: shape``
                   inside ``T.prim_func`` body — see ``_emit_sequential_region``
                   line 1877). NUMERIC_PASS via extern.
* ``add``        -> launcher OK numerically, extern fallback (same NameError).
                   NUMERIC_PASS via extern.
* ``sum``        -> reduction not handled by sequential emitter (only
                   unary-elementwise + single-binary chains). Extern.
                   NUMERIC_PASS via extern.
* ``matmul``     -> ``_emit_matmul_relu_primfunc`` IS the legacy real path
                   and DOES compile when paired with another op (e.g.
                   ``relu(x @ w)``); standalone ``matmul`` falls back to
                   extern. NUMERIC_PASS via extern.

Conclusion: the FX -> TileLang path is **structurally complete end-to-end**
(walker, dispatch, content_hash, custom_op_wrapper.launcher, eager-replay
fallback) and is numerically correct. The TileLang TIR materialisation
half currently only fires for the legacy ``matmul + relu`` fusion shape and
a subset of attention / RMS-norm patterns triggered by Dynamo's aten
graphs (see ``poc/torch_dynamo/examples/torch_compile_smoke.py``). For
plain ``torch.fx.symbolic_trace`` graphs of single ops, every region we
tried hits the extern fallback. The failure mode is consistent: the
``T.prim_func`` decorator at ``fx_to_tilelang.py:1877`` references
``shape`` / ``dtype`` as free names inside the kernel signature
(``X: T.Tensor(shape, dtype)``) but the captured-variable resolution in
the TileLang frontend rejects it, so ``_materialize_subgraph`` raises
``NameError: name 'shape' is not defined`` and the orchestrator routes
to the extern slot.

This file numerically verifies the launcher path (the contract every
caller of ``FXToTileLang.run().launcher`` actually depends on) and
records ``HAS_PRIM_FUNC`` per case so a future fix to the ``T.prim_func``
shape-capture bug surfaces as a green ``HAS_PRIM_FUNC`` rather than a
silent regression.

MLX involvement
---------------

The brief asked for an MLX bridge. The current launcher is a pure-torch
callable (eager-replay or compiled TileLang artifact, both producing
``torch.Tensor`` outputs). There is no torch->MLX zero-copy path wired
into ``FusedKernelArtifact.launcher``. We therefore verify against the
torch reference directly; an MLX-side comparison would require a separate
``mlx.core.array(t.numpy())`` conversion, which is a no-op fidelity check
(``mlx.allclose(mx.array(a.numpy()), mx.array(b.numpy()))``). We include
that conversion as a smoke-only assertion to exercise the MLX import on
Metal hosts.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lower_and_launch(fn, example_inputs):
    """Symbolic-trace ``fn``, run it through ``FXToTileLang``, call launcher.

    Returns ``(out, artifact)``.
    """
    import torch.fx as fx

    from poc.torch_dynamo.fx_to_tilelang import FXToTileLang

    gm = fx.symbolic_trace(fn)
    artifact = FXToTileLang(gm, list(example_inputs)).run()
    out = artifact.launcher(*example_inputs)
    return out, artifact


def _maybe_mlx_check(t_a, t_b):
    """Best-effort MLX-side allclose smoke. No-op if MLX unimportable."""
    if not _has("mlx"):
        return
    import mlx.core as mx  # type: ignore[import-not-found]

    a = mx.array(t_a.detach().to("cpu").float().numpy())
    b = mx.array(t_b.detach().to("cpu").float().numpy())
    assert bool(mx.allclose(a, b, rtol=1e-2, atol=1e-2).item()), (
        "MLX side comparison disagrees with torch reference")


# ---------------------------------------------------------------------------
# Tests — escalating op complexity
# ---------------------------------------------------------------------------


def test_relu_unary_lowers_and_runs():
    """Smallest possible: ``torch.relu(x)`` end-to-end.

    B2 wave fix-pack: ``HAS_PRIM_FUNC`` must now be True — the
    ``_emit_sequential_region`` NameError on ``shape`` (closure-capture
    bug) and the ``T.cast(dtype, value)`` argument-order bug that
    previously routed every unary chain to the extern fallback have been
    fixed. Numerics still match because the runtime layer falls back to
    the extern launcher when the JIT kernel rejects CPU inputs (the
    common case on a Mac host).
    """
    import torch

    x = torch.randn(8, 16, dtype=torch.float32)

    def fn(a):
        return torch.relu(a)

    out, artifact = _lower_and_launch(fn, [x])
    y_ref = torch.relu(x)
    assert tuple(out.shape) == tuple(y_ref.shape)
    torch.testing.assert_close(out, y_ref, rtol=1e-5, atol=1e-5)
    _maybe_mlx_check(out, y_ref)
    # B2 wave fix-pack regression: real TIR materialisation MUST happen.
    has_prim_func = bool(getattr(artifact, "prim_funcs", ()))
    print(f"[relu] HAS_PRIM_FUNC={has_prim_func} source={artifact.source!r}")
    assert has_prim_func, (
        "relu unary chain must lower to a real T.prim_func — extern "
        f"fallback would mask the closure-capture bug. source={artifact.source!r}")
    assert "tilelang.compile ok" in artifact.source, (
        f"expected 'tilelang.compile ok' in source, got {artifact.source!r}")


def test_add_binary_lowers_and_runs():
    """Binary elementwise: ``a + b``.

    B2 wave fix-pack: ``HAS_PRIM_FUNC`` must now be True for the same
    reason as :func:`test_relu_unary_lowers_and_runs` — the binary kernel
    sat behind the same closure-capture bug.
    """
    import torch

    a = torch.randn(8, 16, dtype=torch.float32)
    b = torch.randn(8, 16, dtype=torch.float32)

    def fn(x, y):
        return x + y

    out, artifact = _lower_and_launch(fn, [a, b])
    y_ref = a + b
    assert tuple(out.shape) == tuple(y_ref.shape)
    torch.testing.assert_close(out, y_ref, rtol=1e-5, atol=1e-5)
    _maybe_mlx_check(out, y_ref)
    has_prim_func = bool(getattr(artifact, "prim_funcs", ()))
    print(f"[add] HAS_PRIM_FUNC={has_prim_func} source={artifact.source!r}")
    assert has_prim_func, (
        "binary add must lower to a real T.prim_func — extern fallback "
        f"would mask the closure-capture bug. source={artifact.source!r}")
    assert "tilelang.compile ok" in artifact.source, (
        f"expected 'tilelang.compile ok' in source, got {artifact.source!r}")


def test_sum_reduction_lowers_and_runs():
    """Reduction: ``x.sum()``.

    B2 wave fix-pack: ``HAS_PRIM_FUNC`` must now be True — the new
    :meth:`FXToTileLang._emit_sequential_reduction` builder lowers a sole
    ``sum`` op to a serial-accumulator PrimFunc instead of routing to the
    extern launcher with the
    ``contains non-unary-elementwise ops`` ``NotImplementedError``.
    """
    import torch

    x = torch.randn(8, 16, dtype=torch.float32)

    def fn(a):
        return a.sum()

    out, artifact = _lower_and_launch(fn, [x])
    y_ref = x.sum()
    assert tuple(out.shape) == tuple(y_ref.shape)
    torch.testing.assert_close(out, y_ref, rtol=1e-3, atol=1e-3)
    has_prim_func = bool(getattr(artifact, "prim_funcs", ()))
    print(f"[sum] HAS_PRIM_FUNC={has_prim_func} source={artifact.source!r}")
    assert has_prim_func, (
        "sum reduction must lower to a real T.prim_func — extern "
        f"fallback would mask the missing reduction emitter. "
        f"source={artifact.source!r}")
    assert "tilelang.compile ok" in artifact.source, (
        f"expected 'tilelang.compile ok' in source, got {artifact.source!r}")


def test_matmul_lowers_and_runs():
    """Matmul: ``torch.matmul(x, x.T)``.

    B2 wave fix-pack: ``HAS_PRIM_FUNC`` must now be True — the new
    :meth:`FXToTileLang._emit_sequential_matmul` builder + the absorption
    of ``aten.t`` as a view op cut the previous op-trace ``['t', 'matmul']``
    down to a sole-matmul region the dedicated ``T.gemm`` emitter handles.
    """
    import torch

    torch.manual_seed(0)
    x = torch.randn(8, 16, dtype=torch.float16)

    def fn(a):
        return torch.matmul(a, a.t())

    out, artifact = _lower_and_launch(fn, [x])
    y_ref = torch.matmul(x, x.t())
    assert tuple(out.shape) == tuple(y_ref.shape)
    torch.testing.assert_close(out, y_ref, rtol=1e-2, atol=1e-2)
    has_prim_func = bool(getattr(artifact, "prim_funcs", ()))
    print(f"[matmul] HAS_PRIM_FUNC={has_prim_func} source={artifact.source!r}")
    assert has_prim_func, (
        "matmul must lower to a real T.prim_func — extern fallback "
        f"would mask the missing matmul emitter. source={artifact.source!r}")
    assert "tilelang.compile ok" in artifact.source, (
        f"expected 'tilelang.compile ok' in source, got {artifact.source!r}")


def test_artifact_invariants():
    """Sanity-check the ``FusedKernelArtifact`` contract used downstream."""
    import torch

    from poc.torch_dynamo.custom_op_wrapper import FusedKernelArtifact

    def fn(x):
        return torch.relu(x)

    x = torch.randn(4, 4)
    _, artifact = _lower_and_launch(fn, [x])
    assert isinstance(artifact, FusedKernelArtifact)
    assert artifact.name.startswith("fused_"), (
        "content_hash should produce a 'fused_<hex>' qualname suffix")
    assert callable(artifact.launcher)
    assert len(artifact.input_specs) == 1
    assert len(artifact.output_specs) == 1
    # ``prim_funcs`` may be empty (extern-fallback path) but the attribute
    # must exist as the wrapper consults it.
    assert hasattr(artifact, "prim_funcs")
