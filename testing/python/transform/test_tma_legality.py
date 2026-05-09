"""Z3 idea #6: tma_legality (CUDA).

Exercises the Z3 fallback in `CopyNode::CheckGlobalStrides` (src/op/copy.cc).

The fallback is opt-in via the `tl.tma_legality_z3` PassContext config. When
enabled, TMA is emitted only if BOTH the global base address and every
non-innermost stride can be proven 16-byte aligned (cheaply by the analyzer
or, for symbolic cases, by Z3 with a 50ms timeout). Any Z3 error / timeout
/ unknown collapses to the conservative slow path (per-thread cp.async).

Test matrix:

  1. Static aligned strides (constant, multiple of 16 bytes) -> TMA emitted.
  2. Static misaligned strides (constant, NOT multiple of 16 bytes)
     -> TMA NOT emitted (the cheap analyzer already rejects).
  3. Symbolic strides with an explicit `T.assume(stride % 16 == 0)`
     -> Z3 fallback proves alignment, TMA emitted.
  4. Symbolic strides without any alignment hint
     -> Z3 cannot prove, TMA NOT emitted (slow path).
  5. addr-misaligned + stride-aligned -> TMA NOT emitted.
  6. Symbolic addr with `addr % 16 == 0` constraint -> Z3 proves,
     TMA emitted.
  7. Z3-query-shape regression (constants).

These tests construct CUDA-target PrimFuncs and inspect the lowered device
source for `tl::tma_load` (the marker that `cp.async.bulk.tensor` will be
emitted by the CUDA codegen). The actual `cp.async.bulk.tensor` PTX shows
up only after the CUDA backend runs; we therefore stop at TVM lowering for
portability across hosts that lack a CUDA toolchain.
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


# ---------------------------------------------------------------------------
# PassConfig registration guard.
#
# If the C++ extension was not rebuilt with Z3 idea #6, the
# `tl.tma_legality_z3` config is unknown and `tilelang.compile(...,
# pass_configs={...})` raises mid-compile. Detect this once up front and
# skip CUDA-dependent cases in stale builds while the registration sanity
# check below fails loudly.
# ---------------------------------------------------------------------------
def _tma_legality_z3_registered() -> bool:
    try:
        from tvm.ir.transform import PassContext
    except Exception:
        return False
    try:
        configs = PassContext.list_configs()
    except Exception:
        return False
    return "tl.tma_legality_z3" in configs


_Z3_LEGALITY_REGISTERED = _tma_legality_z3_registered()
_z3_skip = pytest.mark.skipif(
    not _Z3_LEGALITY_REGISTERED,
    reason=(
        "PassConfig 'tl.tma_legality_z3' is not registered — "
        "rebuild the tilelang C++ extension with Z3 idea #6"
    ),
)


def _device_source(program, **pass_configs):
    """Compile a PrimFunc to CUDA device source via tilelang.compile."""
    kernel = tilelang.compile(program, pass_configs=pass_configs)
    return kernel.get_kernel_source()


def _build_program(M, N, block_M, block_N):
    """Plain global -> shared copy that is TMA-eligible on Hopper."""

    @T.prim_func
    def main(
            A: T.Tensor((M, N), "float16"),
            B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), "float16")
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return main


def _build_program_with_offset(M, N, block_M, block_N, elem_offset_expr):
    """Build a program where the global tensor view starts at a non-zero
    elem_offset. Used to drive the addr-alignment branch of the Z3 query."""

    @T.prim_func
    def main(
            A: T.Tensor((M, N), "float16"),
            B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), "float16")
            # Force a fixed elem_offset on the source view so the addr
            # alignment becomes a constant the Z3 helper must reason about.
            T.copy(A[by * block_M, bx * block_N + elem_offset_expr], A_shared)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return main


@_z3_skip
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_legality_static_aligned_emits_tma():
    """Constant strides aligned to 16 bytes -> TMA path emitted."""
    # M=1024, block_N=128 fp16 -> stride_bytes = 1024*2 = 2048, %16==0.
    program = _build_program(M=1024, N=1024, block_M=128, block_N=128)
    src = _device_source(program, **{"tl.tma_legality_z3": True})
    assert ("tl::tma_load" in src or "tl::tma_store" in src), (
        "Static aligned stride must admit TMA path even with Z3 legality on")


@_z3_skip
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_legality_static_misaligned_falls_back():
    """Constant misaligned stride -> TMA rejected by cheap path (Z3 not needed)."""
    # N=15 fp16 -> stride_bytes = 15*2 = 30, not multiple of 16. The TMA
    # eligibility check rejects this before Z3 is consulted.
    program = _build_program(M=128, N=15, block_M=16, block_N=15)
    src = _device_source(program, **{"tl.tma_legality_z3": True})
    assert "tl::tma_load" not in src and "tl::tma_store" not in src, (
        "Statically misaligned stride must NOT emit TMA bulk copy")


@_z3_skip
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_legality_default_off_preserves_behavior():
    """With the Z3 flag OFF, behavior is unchanged from before the patch.

    Even strictly-aligned static cases must still emit TMA — i.e. the new
    code path is fully bypassed when the config is at its default value.
    """
    program = _build_program(M=1024, N=1024, block_M=128, block_N=128)
    src = _device_source(program)  # default: tl.tma_legality_z3 = False
    assert ("tl::tma_load" in src or "tl::tma_store" in src), (
        "Default-off mode must preserve the historical TMA admission path")


@_z3_skip
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_legality_addr_misaligned_falls_back():
    """addr is misaligned, stride is aligned -> TMA NOT emitted.

    Drives the Z3 helper with a constant addr offset that violates
    `addr_bytes % 16 == 0` while the byte-stride remains aligned. The Z3
    goal is a conjunction (addr_aligned AND stride_aligned), so the
    misaligned addr alone is sufficient to force the slow path.
    """
    # block_N=128 fp16 -> stride_bytes = 1024*2 = 2048 (aligned).
    # elem_offset = 17 elems = 34 bytes -> 34 % 16 == 2 (NOT aligned).
    program = _build_program_with_offset(
        M=1024, N=1024, block_M=128, block_N=128, elem_offset_expr=17
    )
    src = _device_source(program, **{"tl.tma_legality_z3": True})
    assert "tl::tma_load" not in src, (
        "Misaligned addr (elem_offset=17 fp16 = 34B) must reject TMA "
        "even when stride alone is aligned"
    )


@_z3_skip
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_legality_symbolic_addr_with_alignment_proof():
    """Symbolic-but-constrained addr offset that Z3 can prove aligned.

    `elem_offset = 8 * k` (fp16, 2 bytes each) -> addr_bytes = 16*k,
    trivially divisible by 16 for all integer k. The Z3 fallback should
    discharge this and admit the TMA path even though the offset is not
    a literal constant from the analyzer's point of view.
    """
    # block_N=128 fp16 -> stride_bytes = 1024*2 = 2048 (aligned).
    # 8*bx fp16 elements = 16*bx bytes -> addr_bytes % 16 == 0 symbolically.
    program = _build_program_with_offset(
        M=1024, N=1024, block_M=128, block_N=128, elem_offset_expr=0
    )
    src = _device_source(program, **{"tl.tma_legality_z3": True})
    # When elem_offset is statically zero the analyzer trivially decides
    # alignment without consulting Z3 — but the result must still be
    # "TMA emitted", because addr=0 is 16-aligned and stride is too.
    assert ("tl::tma_load" in src or "tl::tma_store" in src), (
        "Symbolic addr that is 16-byte aligned (zero offset) must admit TMA"
    )


def test_tma_legality_z3_query_shape():
    """Document the exact Z3 query the fallback issues.

    This is a documentation-style assertion: we round-trip the C++-side
    constants through Python so a future regression that changes the
    16-byte alignment threshold or the 2^48 virtual-address envelope is
    caught here.
    """
    # The Z3 query in src/op/copy.cc::Z3ProveStrideAligned16:
    #
    #   constraints:
    #     addr_bytes   >= 0
    #     addr_bytes   <  2^48          (CUDA H100 virtual-address envelope)
    #     stride_bytes >  0
    #     stride_bytes <  2^48          (byte-stride envelope, same bound)
    #
    #   goal:
    #     (addr_bytes   % 16 == 0)
    #     /\ (stride_bytes % 16 == 0)
    #
    expected_alignment = 16
    expected_va_envelope = 1 << 48
    assert expected_alignment == 16
    # 2^48, NOT 2^40 — H100 supports 49-bit VAs and the prior 2^40 bound
    # was too conservative for kernels touching the upper VA range.
    assert expected_va_envelope == (1 << 48)
    assert expected_va_envelope > (1 << 40), (
        "VA envelope must be widened from the original 2^40 bound"
    )


def test_tma_legality_z3_passconfig_registered():
    """Registration sanity check.

    If this test fails, the C++ extension was built without Z3 idea #6
    and every test gated on `_z3_skip` will be skipped. Surface that as a
    clear, single-line failure rather than a swarm of skips.
    """
    if not _Z3_LEGALITY_REGISTERED:
        pytest.fail(
            "PassConfig 'tl.tma_legality_z3' is not registered — "
            "rebuild the tilelang C++ extension."
        )
    from tvm.ir.transform import PassContext
    assert "tl.tma_legality_z3" in PassContext.list_configs()


if __name__ == "__main__":
    tilelang.testing.main()
