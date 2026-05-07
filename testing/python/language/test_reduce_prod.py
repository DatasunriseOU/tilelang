"""Smoke test for the Wave-2 ``reduce_prod`` primitive.

The primitive lowers to the ``"mul"`` reduction kind. Some backends do not
yet implement multiplicative all-reduce; this test only verifies that the
primitive is importable and that the high-level call constructs valid TIR.
A full numerical check requires a backend that supports the ``mul`` kind.
"""
from __future__ import annotations

import pytest


def test_reduce_prod_is_exported():
    try:
        import tilelang.language as T
    except Exception as exc:
        pytest.skip(f"tilelang.language unavailable: {exc!r}")
    assert hasattr(T, "reduce_prod"), "reduce_prod should be exported from tilelang.language"


def test_reduce_prod_constructs_call():
    try:
        import tilelang
        import tilelang.language as T
    except Exception as exc:
        pytest.skip(f"tilelang unavailable: {exc!r}")

    @T.prim_func
    def kernel(
        A: T.Tensor((4, 8), "float32"),
        Out: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=32):
            A_f = T.alloc_fragment((4, 8), "float32")
            O_f = T.alloc_fragment((4,), "float32")
            T.copy(A, A_f)
            T.reduce_prod(A_f, O_f, dim=1, clear=True)
            T.copy(O_f, Out)

    # Constructing the prim_func is enough to verify the surface; we don't
    # require a backend that supports "mul" all-reduce.
    assert kernel is not None
