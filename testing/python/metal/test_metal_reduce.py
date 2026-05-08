"""Focused Metal codegen/runtime tests for TileLang reductions."""

import re

import pytest
import torch

import tilelang
from tilelang import tvm as tvm
import tilelang.language as T
import tilelang.testing


_FORBIDDEN_CUDA_REDUCE_TOKENS = (
    "__syncthreads",
    "__shfl",
    "__threadfence",
    "cuda_runtime",
    "cuda_fp16",
    "cuda_bf16",
)


def _lower_source(func) -> str:
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")
    assert artifact.kernel_source is not None
    return artifact.kernel_source


def _assert_no_cuda_reduce_leakage(src: str) -> None:
    for token in _FORBIDDEN_CUDA_REDUCE_TOKENS:
        assert token not in src, f"unexpected CUDA reduce token {token!r} in Metal source:\n{src}"

    assert "blockDim" not in src, src


def _assert_metal_reduce_tokens(src: str, *, cross_simdgroup: bool = False) -> None:
    assert "kernel void" in src
    assert "namespace tl" in src
    assert "struct AllReduce" in src
    assert "simd_shuffle_xor" in src or re.search(r"\bsimd_(sum|max|min)\(", src), src
    assert "[[thread_position_in_threadgroup]]" in src
    if cross_simdgroup:
        assert "threadgroup_barrier" in src or "[[threadgroup" in src, src


def _make_reduce_kernel(op, *, length=32, dtype=T.float32, threads=32):
    @T.prim_func
    def reduce_kernel(A: T.Tensor((length,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((length,), dtype)
            dst = T.alloc_fragment((1,), dtype)
            T.copy(A, src)
            if op == "sum":
                T.reduce_sum(src, dst)
            elif op == "max":
                T.reduce_max(src, dst)
            elif op == "min":
                T.reduce_min(src, dst)
            elif op == "bitand":
                T.reduce_bitand(src, dst)
            elif op == "bitor":
                T.reduce_bitor(src, dst)
            elif op == "bitxor":
                T.reduce_bitxor(src, dst)
            else:
                raise ValueError(op)
            T.copy(dst, B)

    return reduce_kernel


@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_metal_reduce_codegen_uses_metal_simd_reduction(op):
    src = _lower_source(_make_reduce_kernel(op, length=32, threads=32))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src)


@pytest.mark.parametrize("op", ["bitand", "bitor", "bitxor"])
def test_metal_reduce_codegen_for_additional_ops_has_no_cuda_template_leakage(op):
    src = _lower_source(_make_reduce_kernel(op, length=32, dtype=T.int32, threads=32))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src)


def test_metal_reduce_cross_simdgroup_codegen_uses_metal_barrier_path():
    src = _lower_source(_make_reduce_kernel("sum", length=64, threads=64))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src, cross_simdgroup=True)


def test_metal_reduce_nan_propagate_does_not_emit_cuda_nan_intrinsics():
    src = _lower_source(_make_nan_reduce_kernel())

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src)
    assert "__hmax_nan" not in src
    assert "MaxOpNan" not in src


def _make_nan_reduce_kernel():
    @T.prim_func
    def reduce_nan_kernel(A: T.Tensor((32,), T.float16), B: T.Tensor((1,), T.float16)):
        with T.Kernel(1, threads=32):
            src = T.alloc_fragment((32,), T.float16)
            dst = T.alloc_fragment((1,), T.float16)
            T.copy(A, src)
            T.reduce_max(src, dst, nan_propagate=True)
            T.copy(dst, B)

    return reduce_nan_kernel


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    ("op", "values", "expected"),
    [
        ("sum", torch.arange(32, dtype=torch.float32), 496.0),
        ("max", torch.arange(32, dtype=torch.float32) - 7, 24.0),
        ("min", torch.arange(32, dtype=torch.float32) - 7, -7.0),
    ],
)
def test_metal_reduce_runtime_mps_small(op, values, expected):
    kernel = tilelang.compile(_make_reduce_kernel(op), target="metal")
    out = torch.empty(1, dtype=torch.float32, device="mps")

    kernel(values.to("mps"), out)
    torch.mps.synchronize()

    torch.testing.assert_close(out.cpu(), torch.tensor([expected], dtype=torch.float32))


if __name__ == "__main__":
    tilelang.testing.main()
