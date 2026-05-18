"""Regression tests for BaseKernelAdapter.get_current_device_functor.

The functor must be safe to construct even when the underlying CUDA runtime
probe fails at call time (e.g. CUDA driver missing on Mac/ROCm hosts even
though ``torch.cuda.is_available()`` initially returned True). The actual
CUDA call lives inside the returned closure so callers can catch the
exception via the closure's own fallback path.
"""
from __future__ import annotations

import pytest
import torch

from tilelang.jit.adapter.base import BaseKernelAdapter


def test_device_functor_construction_does_not_invoke_cuda_get_device(monkeypatch):
    """Constructing the functor should never raise even if _cuda_getDevice errors."""

    def _raise(*_args, **_kwargs):  # noqa: D401 - simple stub
        raise RuntimeError("simulated CUDA driver failure")

    # Patch the raw probe so any unguarded call would explode.
    if hasattr(torch._C, "_cuda_getDevice"):
        monkeypatch.setattr(torch._C, "_cuda_getDevice", _raise, raising=False)

    # Should not raise at construction time.
    functor = BaseKernelAdapter.get_current_device_functor()
    assert callable(functor)


def test_device_functor_runtime_failure_falls_back(monkeypatch):
    """If CUDA is reported available but call-time probe fails, functor degrades gracefully."""

    if not torch.cuda.is_available():
        pytest.skip("Test requires torch.cuda.is_available() == True at import time")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated CUDA driver failure at call time")

    functor = BaseKernelAdapter.get_current_device_functor()
    # Patch *after* construction so the deferred path is exercised.
    monkeypatch.setattr(torch._C, "_cuda_getDevice", _raise, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", _raise, raising=False)

    # Must not propagate; falls back to CPU per documented contract.
    dev = functor()
    assert dev.type in {"cuda", "cpu"}


def test_device_functor_returns_callable_on_cpu_host():
    """Smoke test: functor must always return a callable that yields a torch.device."""
    functor = BaseKernelAdapter.get_current_device_functor()
    dev = functor()
    assert isinstance(dev, torch.device)
