"""Tests for the LowerTMAToPtrArith transform (RFC §5.4).

On NV Hopper+ targets the pass is a no-op (the existing
``LowerHopperIntrin`` pipeline owns the native lowering). On Metal /
HIP / pre-Hopper CUDA / CPU it rewrites ``tl::tma_load`` /
``tl::tma_store`` calls into a pointer-arith For-nest that lowers to
plain element copies between the global pointer and the staging shared
buffer.
"""

import pytest

pytest.importorskip("tvm")  # skip if TVM/TileLang stack is unavailable

from tilelang import tvm as tvm  # noqa: E402
import tilelang as tl  # noqa: E402
from tvm.script import tir as T  # noqa: E402


_CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6
_CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7


def _build_tma_kernel(data_type_code: int = _CU_TENSOR_MAP_DATA_TYPE_FLOAT16):
    """Build a tiny PrimFunc that contains a `tl::tma_load` Call wrapped in
    its `create_tma_descriptor` plumbing — i.e. exactly the IR shape that
    `LowerTileOp` emits on the Hopper TMA path.

    The ``data_type_code`` is the inverse-mapped ``CUtensorMapDataType``
    value (see ``src/op/utils.cc::to_CUtensorMapDataType``). Selecting a
    non-fp16 code exercises the dtype-recovery path the fallback uses to
    compute per-element byte strides.
    """

    create_tma = tvm.tir.op.Op.get("tl.create_tma_descriptor")
    tma_load = tvm.tir.op.Op.get("tl.tma_load")

    @T.prim_func
    def kernel(A_handle: T.handle, smem_handle: T.handle):
        # 2D descriptor: rank=2, shape=(64,64), stride=(2,128), box=(16,16)
        # Layout matches `TMADesc::EncodeCallArgs` in src/op/copy.cc:
        #   data_type, rank, global_addr, shape×R, stride×R, box×R,
        #   smem_stride×R, interleave, swizzle, l2_promotion, oob_fill
        desc = T.call_intrin(
            "handle",
            create_tma,
            T.int32(data_type_code),  # CUtensorMapDataType code
            T.int32(2),  # rank
            A_handle,    # global_addr
            T.int32(64), T.int32(64),       # shape
            T.int32(2), T.int32(128),       # stride (bytes)
            T.int32(16), T.int32(16),       # smem_box
            T.int32(1), T.int32(1),         # smem_stride
            T.int32(0),                     # interleave
            T.int32(0),                     # swizzle
            T.int32(0),                     # l2_promotion
            T.int32(0),                     # oob_fill
        )
        T.evaluate(
            T.call_intrin(
                "handle", tma_load,
                desc, T.int32(0), smem_handle,
                T.int32(0), T.int32(0),  # coord_0, coord_1
                T.int32(0),              # eviction policy
            ))

    return kernel


def _lower(func, target_str: str):
    target = tvm.target.Target(target_str)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)
    with tvm.transform.PassContext():
        with target:
            return tl.transform.LowerTMAToPtrArith()(mod)


def _has_tma_call(mod) -> bool:
    s = str(mod.script() if hasattr(mod, "script") else mod)
    return ("tl.tma_load" in s) or ("tl.tma_store" in s) or \
           ("tl.tma_load_im2col" in s)


def test_cuda_hopper_passes_through():
    """On sm_90 (Hopper) the pass is a no-op: TMA call must survive."""
    func = _build_tma_kernel()
    mod = _lower(func, "cuda -arch=sm_90")
    assert _has_tma_call(mod), \
        "LowerTMAToPtrArith on Hopper should leave TMA calls intact"


def test_metal_decomposes():
    """On Metal the pass must remove TMA calls in favor of pointer-arith."""
    func = _build_tma_kernel()
    mod = _lower(func, "metal")
    print("[metal lowered IR]\n", mod)
    assert not _has_tma_call(mod), \
        "LowerTMAToPtrArith on Metal must rewrite TMA calls"


def test_hip_decomposes():
    """On HIP the pass must remove TMA calls in favor of pointer-arith."""
    func = _build_tma_kernel()
    mod = _lower(func, "hip")
    print("[hip lowered IR]\n", mod)
    assert not _has_tma_call(mod), \
        "LowerTMAToPtrArith on HIP must rewrite TMA calls"


def test_metal_fp32_uses_4_byte_stride():
    """Regression: the fallback used to default to fp16 (2 bytes/elem)
    regardless of the descriptor's data_type code, silently corrupting
    every non-fp16 TMA copy on non-NV targets. Verify that an fp32
    descriptor emits a 4-byte stride in the rewritten IR (the
    ``__tl_ptr_copy_elem`` byte-count argument)."""
    func = _build_tma_kernel(data_type_code=_CU_TENSOR_MAP_DATA_TYPE_FLOAT32)
    mod = _lower(func, "metal")
    s = str(mod.script() if hasattr(mod, "script") else mod)
    assert "__tl_ptr_copy_elem" in s, \
        "expected pointer-arith copy helper in lowered IR"
    # The third arg to __tl_ptr_copy_elem is the element byte count; for
    # fp32 it must be 4 (not the legacy 2-byte fp16 default).
    assert ", 4)" in s or ", 4i64)" in s or ", T.int64(4)" in s, \
        f"expected 4-byte element stride in fp32 fallback, got:\n{s}"


def test_metal_fp16_uses_2_byte_stride():
    """Companion to the fp32 test: the legacy default should now only
    appear when the descriptor actually says fp16."""
    func = _build_tma_kernel(data_type_code=_CU_TENSOR_MAP_DATA_TYPE_FLOAT16)
    mod = _lower(func, "metal")
    s = str(mod.script() if hasattr(mod, "script") else mod)
    assert "__tl_ptr_copy_elem" in s
    assert ", 2)" in s or ", 2i64)" in s or ", T.int64(2)" in s, \
        f"expected 2-byte element stride in fp16 fallback, got:\n{s}"


def test_unknown_dtype_code_is_refused():
    """An unrecognized CUtensorMapDataType code must NOT silently lower
    to a wrong byte stride — the pass logs a warning and leaves the TMA
    call in place so codegen surfaces the bug instead of corrupting
    memory."""
    func = _build_tma_kernel(data_type_code=999)  # outside enum range
    mod = _lower(func, "metal")
    assert _has_tma_call(mod), \
        "Unknown dtype must NOT be silently lowered (would corrupt mem)."
