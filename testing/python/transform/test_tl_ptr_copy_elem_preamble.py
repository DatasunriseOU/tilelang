"""Smoke test for the ``__tl_ptr_copy_elem`` runtime helper preamble.

The non-NV TMA fallback pass (``src/transform/lower_tma_to_ptr_arith.cc``)
emits ``tl::call_extern("__tl_ptr_copy_elem", dst, src, bytes)`` for each
per-element memcpy in the rewritten pointer-arith For-nest. This test
verifies that the per-target codegen preambles (Metal, HIP, CUDA) inject
a matching definition so the resulting source compiles end-to-end.

Tests that invoke a real toolchain (``metallib``, ``hipcc``, ``nvcc``)
remain in their per-target test files; this file only inspects the
emitted source string and is therefore safe to run on Linux/x86 with
no GPU SDK installed.
"""

import pytest

pytest.importorskip("tvm")  # skip if TVM/TileLang stack is unavailable

import tilelang  # noqa: E402
from tilelang import tvm as tvm  # noqa: E402
import tilelang.language as T  # noqa: E402


def _trivial_kernel():
    """A minimal kernel — no TMA needed. The preamble is emitted
    unconditionally by each target codegen, so a no-op kernel is
    sufficient to surface the helper definition."""

    @T.prim_func
    def main(A: T.Tensor((16,), T.float32), B: T.Tensor((16,), T.float32)):
        with T.Kernel(1, threads=16) as bx:  # noqa: F841
            for i in T.Parallel(16):
                B[i] = A[i]

    return main


def _lower_source(target_str: str) -> str:
    func = _trivial_kernel()
    with tvm.transform.PassContext(), tvm.target.Target(target_str):
        artifact = tilelang.lower(func, target=target_str)
    src = artifact.kernel_source
    assert src is not None
    return src


def test_metal_preamble_has_tl_ptr_copy_elem():
    """Metal preamble must declare ``__tl_ptr_copy_elem`` overloads."""
    src = _lower_source("metal")
    assert "__tl_ptr_copy_elem" in src, (
        "Expected __tl_ptr_copy_elem helper in Metal preamble — "
        "see src/target/codegen_metal.cc CodeGenTileLangMetal ctor."
    )
    # Sanity: the device-only and threadgroup overloads should both exist.
    assert "device void* dst" in src
    assert "threadgroup void* dst" in src


def test_hip_preamble_has_tl_ptr_copy_elem():
    """HIP preamble (decl_stream in Finish()) must define the helper."""
    src = _lower_source("hip")
    assert "__tl_ptr_copy_elem" in src, (
        "Expected __tl_ptr_copy_elem helper in HIP preamble — "
        "see src/target/codegen_hip.cc CodeGenTileLangHIP::Finish."
    )
    assert "__device__ inline void __tl_ptr_copy_elem" in src


def test_cuda_preamble_has_tl_ptr_copy_elem():
    """Pre-Hopper CUDA also takes the pointer-arith fallback, so the
    helper must be present in the CUDA preamble too. (On sm_90+ it's
    dead code but harmless.)"""
    src = _lower_source("cuda")
    assert "__tl_ptr_copy_elem" in src, (
        "Expected __tl_ptr_copy_elem helper in CUDA preamble — "
        "see src/target/codegen_cuda.cc CodeGenTileLangCUDA::Finish."
    )
    assert "__device__ inline void __tl_ptr_copy_elem" in src


if __name__ == "__main__":
    tilelang.testing.main()
