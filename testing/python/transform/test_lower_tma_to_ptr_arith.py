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


def _build_tma_kernel():
    """Build a tiny PrimFunc that contains a `tl::tma_load` Call wrapped in
    its `create_tma_descriptor` plumbing — i.e. exactly the IR shape that
    `LowerTileOp` emits on the Hopper TMA path."""

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
            T.int32(2),  # data_type code (placeholder)
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
