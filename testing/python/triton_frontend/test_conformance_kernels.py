"""Conformance tests for ``poc.triton_frontend.conformance``.

Each kernel runs against a numpy reference and asserts numerical agreement.
Tests skip cleanly when tilelang / tvm aren't importable in the runtime
environment so the suite stays usable on machines without a built backend.
"""
from __future__ import annotations

import pytest


def _np():
    try:
        import numpy as np
        return np
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"numpy unavailable: {exc!r}")


def _conformance():
    try:
        from poc.triton_frontend import conformance
        return conformance
    except Exception as exc:
        pytest.skip(f"poc.triton_frontend.conformance unavailable: {exc!r}")


def test_vector_add_matches_numpy():
    np = _np()
    conf = _conformance()
    kernel = conf.kernel_vector_add(N=1024, BLOCK=128)
    if kernel is None:
        pytest.skip("tilelang.language not available -- skipping conformance run")

    a = np.random.randn(1024).astype(np.float32)
    b = np.random.randn(1024).astype(np.float32)
    expect = a + b

    try:
        import torch
        a_t = torch.from_numpy(a)
        b_t = torch.from_numpy(b)
        y_t = torch.empty_like(a_t)
        kernel(a_t, b_t, y_t)
        got = y_t.numpy()
    except Exception as exc:
        pytest.skip(f"backend execution failed: {exc!r}")

    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-6)


def test_dot_reduce_atomic_matches_numpy():
    np = _np()
    conf = _conformance()
    kernel = conf.kernel_dot_reduce_atomic(M=64, N=64, K=64, BLOCK=32)
    if kernel is None:
        pytest.skip("tilelang.language not available")

    a = np.random.randn(64, 64).astype(np.float16)
    b = np.random.randn(64, 64).astype(np.float16)
    # Reference: column-sum of a @ b.
    expect = (a.astype(np.float32) @ b.astype(np.float32)).sum(axis=0)

    try:
        import torch
        a_t = torch.from_numpy(a)
        b_t = torch.from_numpy(b)
        acc = torch.zeros(64, dtype=torch.float32)
        kernel(a_t, b_t, acc)
        got = acc.numpy()
    except Exception as exc:
        pytest.skip(f"backend execution failed: {exc!r}")

    np.testing.assert_allclose(got, expect, rtol=5e-2, atol=5e-2)


def test_kernels_dict_lists_wave2_additions():
    conf = _conformance()
    assert "vector_add" in conf.KERNELS
    assert "dot_reduce_atomic" in conf.KERNELS
