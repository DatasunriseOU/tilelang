"""Reference example: register an Apple Metal SIMDgroup 8x8 MMA op as a
``tl.extern_intrinsic`` (RFC §6).

This is the "tile-typed contract" pattern — the body is raw MSL using
``simdgroup_matrix<...>`` types, and we expose it to TileLang fusion as a
regular TIR block. Because the contract declares scope=``simdgroup`` and
layout=``simdgroup_a/b/c``, ``layout_inference.cc`` and
``thread_storage_sync.cc`` will treat this op exactly like a TileLang-native
``T.gemm`` lowered to SIMDgroup — i.e. it can be pipelined and double-buffered
by ``inject_pipeline.cc`` / ``auto_double_buffer.cc`` without an HBM bounce.

Usage shape (does NOT compile — illustrates how the decorator is used inside
a kernel definition; full lowering requires the full TileLang stack)::

    import tilelang.language as T

    @T.prim_func
    def kernel(...):
        with T.Kernel(...) as ...:
            A_frag = T.alloc_fragment((8, 8), "float16")
            B_frag = T.alloc_fragment((8, 8), "float16")
            C_frag = T.alloc_fragment((8, 8), "float32")
            ...
            simdgroup_mma_8x8(A_frag, B_frag, C_frag)

The decorator emits ``tir.call_extern("handle", "tl.extern_intrinsic.simdgroup_mma_8x8",
T.access_ptr(A,"r"), T.access_ptr(B,"r"), T.access_ptr(C,"rw"))``. The Metal
codegen materialises the MSL body keyed by ``"metal"`` from the registry.
"""

from __future__ import annotations

from tilelang.language.extern import (
    extern_intrinsic,
    simdgroup_a,
    simdgroup_b,
    simdgroup_c,
)

# Apple MSL body. Note:
#   - No threadgroup_barrier — barriers are inserted by thread_storage_sync.cc.
#   - All memory access is via the declared simdgroup_matrix<> args.
#   - No global pointer arithmetic.
SIMDGROUP_MMA_8x8_MSL: str = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline void simdgroup_mma_8x8(
    thread simdgroup_half8x8 a,
    thread simdgroup_half8x8 b,
    thread simdgroup_float8x8 c
) {
    // Fused multiply-accumulate: c += a * b.
    // SIMDgroup matrix MMA is a single-instruction op on Apple GPUs.
    simdgroup_multiply_accumulate(c, a, b, c);
}
"""

# Register the intrinsic. The ``simdgroup_a/b/c`` factories pin the canonical
# scope/dtype/alignment for each operand role; we only override the
# ``pipeline_stage`` hint since stages are use-site policy, not layout policy.
simdgroup_mma_8x8 = extern_intrinsic(
    name="simdgroup_mma_8x8",
    signature=lambda: (
        simdgroup_a("a", pipeline_stage=0),
        simdgroup_b("b", pipeline_stage=0),
        simdgroup_c("c", pipeline_stage=1),
    ),
    bodies={"metal": SIMDGROUP_MMA_8x8_MSL},
)


# ---------------------------------------------------------------------------
# FP8 forward-compat variant. Apple has not (as of 2026-05) shipped
# ``simdgroup_matrix<float8_e4m3>`` — the MSL body below is therefore
# ILLUSTRATIVE ONLY and references types Metal does not yet declare. It
# documents the expected call shape so cppmega.mlx kernels that already
# carry FP8 fragments (``fp8_msl_kernels.py``, ``sparse_mla_fp8.py``,
# ``fp8_vecmat_path_c.py``) can swap their inline raw-MSL into this
# extern_intrinsic registration the moment Apple ships FP8 simdgroup MMA.
# ---------------------------------------------------------------------------

# pragma: no cover — FP8 factories forward-compat
from tilelang.language.extern import simdgroup_a_fp8, simdgroup_b_fp8

# pragma: no cover — FP8 factories forward-compat
SIMDGROUP_MMA_8x8_FP8_MSL: str = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// FORWARD-COMPATIBLE STUB — Apple has not shipped float8_e4m3 simdgroup_matrix
// types as of 2026-05. The body below names types that Metal does not yet
// declare; it serves only to document the expected lowering shape.
inline void simdgroup_mma_8x8_fp8(
    thread simdgroup_float8_e4m3_8x8 a,    // not yet a Metal type
    thread simdgroup_float8_e4m3_8x8 b,    // not yet a Metal type
    thread simdgroup_float8x8         c    // fp32 accumulator (already exists)
) {
    simdgroup_multiply_accumulate(c, a, b, c);
}
"""

# pragma: no cover — FP8 factories forward-compat
simdgroup_mma_8x8_fp8 = extern_intrinsic(
    name="simdgroup_mma_8x8_fp8",
    signature=lambda: (
        simdgroup_a_fp8("a", pipeline_stage=0),
        simdgroup_b_fp8("b", pipeline_stage=0),
        simdgroup_c("c", pipeline_stage=1),
    ),
    bodies={"metal": SIMDGROUP_MMA_8x8_FP8_MSL},
)


if __name__ == "__main__":
    # Sanity check at import time — registry sees the entry and the contract is
    # well-formed.
    from tilelang.language import extern_registry

    entry = extern_registry.lookup("simdgroup_mma_8x8")
    assert entry is not None, "simdgroup_mma_8x8 not registered"
    assert entry.has_target("metal"), "metal body missing"
    frags = entry.signature()
    print(f"registered {entry.name} with {len(frags)} frags:")
    for f in frags:
        print(f"  - {f.name}: {f.shape} {f.dtype} scope={f.scope} layout={f.layout} "
              f"stage={f.pipeline_stage} out={f.is_output}")

    fp8_entry = extern_registry.lookup("simdgroup_mma_8x8_fp8")
    assert fp8_entry is not None, "simdgroup_mma_8x8_fp8 not registered"
    print(f"registered {fp8_entry.name} (forward-compat — Apple FP8 silicon pending)")
