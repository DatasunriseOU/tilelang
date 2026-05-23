"""CUDA-only conformance tests for the Triton TTIR frontend.

These tests exercise the **real** hardware paths that the Metal /
fallback CI hosts can only validate structurally:

* FA-v3 with TMA descriptor loads → ``cp.async.bulk`` (Hopper) +
  WGMMA-shaped ``tl.dot`` → ``wgmma.mma_async``.
* ``tt.descriptor_store`` end-to-end through the production TMA path
  rather than the pointer-arith fallback.

The whole module is skipped when ``torch.cuda.is_available()`` returns
False, so non-CUDA runners (e.g. our Apple Metal dev box) keep working.
On a CUDA Hopper host the tests assert NUMERIC_PASS against the
production numeric harness and additionally check that the TTIR
captured upstream carries the expected descriptor / dot ops.
"""
from __future__ import annotations

import importlib

import pytest


_HAS_TORCH = importlib.util.find_spec("torch") is not None
_CUDA_AVAILABLE = False
if _HAS_TORCH:
    try:
        import torch

        _CUDA_AVAILABLE = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:  # pragma: no cover -- defensive: torch import may fail
        _CUDA_AVAILABLE = False


pytestmark = [
    pytest.mark.skipif(
        not _CUDA_AVAILABLE,
        reason="CUDA hardware required: this test exercises Hopper TMA + WGMMA paths",
    ),
    pytest.mark.cuda_hardware,
]


def _run_numeric_one(name: str):
    """Run one kernel through the production numeric smoke harness."""
    numeric_smoke = importlib.import_module(
        "poc.triton_frontend._test_harness.numeric_smoke"
    )
    deps = numeric_smoke._probe_deps()
    if deps.get("triton") is not None:
        pytest.skip(f"triton unavailable: {deps['triton']}")
    if deps.get("tilelang") is not None:
        pytest.skip(f"tilelang unavailable: {deps['tilelang']}")
    return numeric_smoke.run_one(name, deps)


def test_fa_v3_real_tma_path_numeric_pass_on_cuda() -> None:
    """FA-v3 with TMA descriptors must NUMERIC_PASS on CUDA Hopper.

    The kernel uses ``tl.make_tensor_descriptor`` + ``desc.load`` plus a
    WGMMA-shaped ``tl.dot(..., out_dtype=tl.float32)`` accumulator. On
    Hopper this lowers to ``cp.async.bulk`` + ``wgmma.mma_async``; the
    test asserts numerics match the pure-numpy reference within the
    kernel's declared tolerances.
    """
    result = _run_numeric_one("fa_v3")
    assert result.verdict == "NUMERIC_PASS", (
        f"fa_v3 on CUDA must NUMERIC_PASS; got verdict={result.verdict} "
        f"detail={result.detail!r} max_abs={result.max_abs_err}"
    )


def test_descriptor_store_end_to_end_numeric_pass_on_cuda() -> None:
    """tt.descriptor_store round-trip must NUMERIC_PASS on CUDA.

    The kernel pairs ``descriptor.load`` and ``descriptor.store`` so the
    frontend's ``tt.descriptor_store`` emitter is exercised end-to-end
    through real TMA on NVIDIA hardware (rather than only the pointer-
    arith fallback).
    """
    result = _run_numeric_one("tma_descriptor_store")
    assert result.verdict == "NUMERIC_PASS", (
        f"tma_descriptor_store on CUDA must NUMERIC_PASS; got "
        f"verdict={result.verdict} detail={result.detail!r}"
    )


def test_fa_v3_ttir_carries_hopper_tma_and_wgmma_markers() -> None:
    """Structural check: the live FA-v3 TTIR must contain TMA + WGMMA hints.

    This is a CUDA-only test because we want to confirm the same
    captured TTIR that exercises real TMA on Hopper carries the
    descriptor/WGMMA markers. The check is text-level so it runs on any
    CUDA host even without launching the kernel.
    """
    jit_to_ttir = importlib.import_module(
        "poc.triton_frontend._test_harness.jit_to_ttir"
    )
    fa_v3 = importlib.import_module(
        "poc.triton_frontend._test_harness.numeric_kernels.fa_v3"
    )
    text = jit_to_ttir.triton_jit_to_ttir(
        fa_v3.TRITON_KERNEL,
        constexprs=fa_v3.META_ARGS,
        signature=fa_v3.TTIR_SIGNATURE,
    )
    assert "tt.make_tensor_descriptor" in text, (
        "FA-v3 TTIR is missing tt.make_tensor_descriptor; the rewrite to "
        "the real Hopper kernel did not survive"
    )
    assert "tt.descriptor_load" in text, (
        "FA-v3 TTIR is missing tt.descriptor_load; the rewrite to the "
        "real Hopper kernel did not survive"
    )
    assert "tt.dot" in text, "FA-v3 TTIR is missing tt.dot"
