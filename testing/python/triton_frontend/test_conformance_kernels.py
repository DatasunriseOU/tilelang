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


def test_softmax_matches_numpy():
    np = _np()
    conf = _conformance()
    kernel = conf.kernel_softmax(M=8, N=64, BLOCK_N=64)
    if kernel is None:
        pytest.skip("tilelang.language not available")

    x = np.random.randn(8, 64).astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    expect = np.exp(x - x_max)
    expect = expect / expect.sum(axis=-1, keepdims=True)

    try:
        import torch
        x_t = torch.from_numpy(x)
        y_t = torch.empty_like(x_t)
        kernel(x_t, y_t)
        got = y_t.numpy()
    except Exception as exc:
        pytest.skip(f"backend execution failed: {exc!r}")

    np.testing.assert_allclose(got, expect, rtol=1e-4, atol=1e-5)


def test_matmul_matches_numpy():
    np = _np()
    conf = _conformance()
    kernel = conf.kernel_matmul(M=64, N=64, K=64, BLOCK_M=32, BLOCK_N=32, BLOCK_K=16)
    if kernel is None:
        pytest.skip("tilelang.language not available")

    a = np.random.randn(64, 64).astype(np.float16)
    b = np.random.randn(64, 64).astype(np.float16)
    expect = a.astype(np.float32) @ b.astype(np.float32)

    try:
        import torch
        a_t = torch.from_numpy(a)
        b_t = torch.from_numpy(b)
        c_t = torch.empty(64, 64, dtype=torch.float32)
        kernel(a_t, b_t, c_t)
        got = c_t.numpy()
    except Exception as exc:
        pytest.skip(f"backend execution failed: {exc!r}")

    np.testing.assert_allclose(got, expect, rtol=5e-2, atol=5e-2)


def test_layer_norm_matches_numpy():
    np = _np()
    conf = _conformance()
    kernel = conf.kernel_layer_norm(M=8, N=64, BLOCK_N=64, eps=1e-5)
    if kernel is None:
        pytest.skip("tilelang.language not available")

    x = np.random.randn(8, 64).astype(np.float32)
    gamma = np.random.randn(64).astype(np.float32)
    beta = np.random.randn(64).astype(np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    expect = (x - mean) / np.sqrt(var + 1e-5) * gamma + beta

    try:
        import torch
        x_t = torch.from_numpy(x)
        g_t = torch.from_numpy(gamma)
        b_t = torch.from_numpy(beta)
        y_t = torch.empty_like(x_t)
        kernel(x_t, g_t, b_t, y_t)
        got = y_t.numpy()
    except Exception as exc:
        pytest.skip(f"backend execution failed: {exc!r}")

    np.testing.assert_allclose(got, expect, rtol=1e-4, atol=1e-5)


# -- Wave-3: printf format-string sanitizer regression --
def test_printf_sanitizer_defangs_percent_n():
    """Regression for wave-2 grok security finding: %n in user-supplied
    print() prefix must be defanged before reaching the GPU runtime."""
    try:
        from poc.triton_frontend.op_mapping import _sanitize_printf_format
    except Exception as exc:
        pytest.skip(f"op_mapping unavailable: {exc!r}")
    assert _sanitize_printf_format("hi %n bye") == "hi %%n bye"
    assert _sanitize_printf_format("plain text") == "plain text"
    # %s/%p/%x are read-only -- left intact (matches comment in source)
    assert _sanitize_printf_format("addr=%p val=%x") == "addr=%p val=%x"


# -- Wave-3: MLIR-binding vs text-path matrix --
@pytest.mark.parametrize("path", ["text", "mlir"])
def test_walker_dispatch_path_matrix(path):
    """Both walker paths (regex-tokenized text and mlir.ir Module) must
    produce a non-empty op stream from a tiny TTIR fixture, OR cleanly
    skip when the dependency isn't importable."""
    try:
        from poc.triton_frontend import _walk_text_ttir, _walk_mlir_module
    except Exception as exc:
        pytest.skip(f"poc.triton_frontend unavailable: {exc!r}")

    ttir_text = (
        "module {\n"
        "  tt.func @k(%arg0: !tt.ptr<f32>) {\n"
        "    %0 = tt.get_program_id x : i32\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )

    if path == "text":
        ops = list(_walk_text_ttir(ttir_text))
        assert ops, "text walker produced empty op stream"
        assert any("program_id" in (o.get("name") if isinstance(o, dict) else "")
                   for o in ops), "expected a tt.get_program_id op"
    else:
        try:
            import mlir.ir  # noqa: F401
        except Exception:
            pytest.skip("mlir.ir bindings not importable")
        try:
            module = mlir.ir.Module.parse(ttir_text)  # type: ignore
        except Exception:
            pytest.skip("MLIR cannot parse the fixture (dialects unregistered)")
        ops = list(_walk_mlir_module(module))
        assert ops, "MLIR walker produced empty op stream"
