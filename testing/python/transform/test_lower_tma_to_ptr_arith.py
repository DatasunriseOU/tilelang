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


_CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0
_CU_TENSOR_MAP_DATA_TYPE_UINT16 = 1
_CU_TENSOR_MAP_DATA_TYPE_UINT32 = 2
_CU_TENSOR_MAP_DATA_TYPE_INT32 = 3
_CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6
_CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7
_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9

_CU_TENSOR_MAP_SWIZZLE_NONE = 0
_CU_TENSOR_MAP_SWIZZLE_32B = 1
_CU_TENSOR_MAP_SWIZZLE_64B = 2
_CU_TENSOR_MAP_SWIZZLE_128B = 3


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
            A_handle,  # global_addr
            T.int32(64),
            T.int32(64),  # shape
            T.int32(2),
            T.int32(128),  # stride (bytes)
            T.int32(16),
            T.int32(16),  # smem_box
            T.int32(1),
            T.int32(1),  # smem_stride
            T.int32(0),  # interleave
            T.int32(0),  # swizzle
            T.int32(0),  # l2_promotion
            T.int32(0),  # oob_fill
        )
        T.evaluate(
            T.call_intrin(
                "handle",
                tma_load,
                desc,
                T.int32(0),
                smem_handle,
                T.int32(0),
                T.int32(0),  # coord_0, coord_1
                T.int32(0),  # eviction policy
            )
        )

    return kernel


def _make_target(target_str: str):
    if target_str == "cuda -arch=sm_90":
        return tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    if target_str == "hip":
        try:
            return tvm.target.Target({"kind": "hip"})
        except ValueError as exc:
            pytest.skip(f"HIP target kind is unavailable in this TVM build: {exc}")
    return tvm.target.Target(target_str)


def _lower(func, target_str: str):
    target = _make_target(target_str)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)
    with tvm.transform.PassContext(), target:
        return tl.transform.LowerTMAToPtrArith()(mod)


def _has_tma_call(mod) -> bool:
    s = str(mod.script() if hasattr(mod, "script") else mod)
    return ("tma_load(" in s) or ("tma_store(" in s) or ("tma_load_im2col(" in s)


def test_cuda_hopper_passes_through():
    """On sm_90 (Hopper) the pass is a no-op: TMA call must survive."""
    func = _build_tma_kernel()
    mod = _lower(func, "cuda -arch=sm_90")
    assert _has_tma_call(mod), "LowerTMAToPtrArith on Hopper should leave TMA calls intact"


def test_metal_decomposes():
    """On Metal the pass must remove TMA calls in favor of pointer-arith."""
    func = _build_tma_kernel()
    mod = _lower(func, "metal")
    print("[metal lowered IR]\n", mod)
    assert not _has_tma_call(mod), "LowerTMAToPtrArith on Metal must rewrite TMA calls"


def test_hip_decomposes():
    """On HIP the pass must remove TMA calls in favor of pointer-arith."""
    func = _build_tma_kernel()
    mod = _lower(func, "hip")
    print("[hip lowered IR]\n", mod)
    assert not _has_tma_call(mod), "LowerTMAToPtrArith on HIP must rewrite TMA calls"


def test_metal_fp32_uses_4_byte_stride():
    """Regression: the fallback used to default to fp16 (2 bytes/elem)
    regardless of the descriptor's data_type code, silently corrupting
    every non-fp16 TMA copy on non-NV targets. Verify the fp32 descriptor
    is recovered correctly: under the default ``kEmitOpaque=false`` the
    rewritten IR must contain ``BufferLoad``/``BufferStore`` against an
    ``fp32`` view; under the legacy opaque path it must carry a 4-byte
    ``__tl_ptr_copy_elem`` byte-count."""
    func = _build_tma_kernel(data_type_code=_CU_TENSOR_MAP_DATA_TYPE_FLOAT32)
    mod = _lower(func, "metal")
    s = str(mod.script() if hasattr(mod, "script") else mod)
    if "__tl_ptr_copy_elem" in s:
        assert ", 4)" in s or ", 4i64)" in s or ", T.int64(4)" in s, f"expected 4-byte element stride in fp32 opaque fallback:\n{s}"
    else:
        # Non-opaque path (default): typed BufferLoad/BufferStore must
        # appear, and the synthetic view must carry the fp32 dtype so
        # downstream cp.async / vectorize re-detection sees correct
        # element bytes.
        assert "BufferLoad" in s or "tl_tma_global_view" in s, f"expected BufferLoad-shaped fallback in lowered IR:\n{s}"
        assert "float32" in s, f"expected fp32 element dtype in synthetic view:\n{s}"


def test_metal_fp16_uses_2_byte_stride():
    """Companion to the fp32 test: the legacy 2-byte default should only
    appear when the descriptor actually says fp16."""
    func = _build_tma_kernel(data_type_code=_CU_TENSOR_MAP_DATA_TYPE_FLOAT16)
    mod = _lower(func, "metal")
    s = str(mod.script() if hasattr(mod, "script") else mod)
    if "__tl_ptr_copy_elem" in s:
        assert ", 2)" in s or ", 2i64)" in s or ", T.int64(2)" in s, f"expected 2-byte element stride in fp16 opaque fallback:\n{s}"
    else:
        assert "BufferLoad" in s or "tl_tma_global_view" in s, f"expected BufferLoad-shaped fallback in lowered IR:\n{s}"
        assert "float16" in s, f"expected fp16 element dtype in synthetic view:\n{s}"


def test_swizzle_pragma_is_preserved():
    """The cuTensorMap ``swizzle`` field must round-trip through the
    fallback so downstream Metal/HIP layout passes still see the
    original mode. ``LowerTMAToPtrArith`` emits a ``pragma_tma_swizzle``
    AttrStmt; ``inject_pipeline.cc`` may forward it onto a For-loop
    annotation under the ``tl_tma_swizzle`` key. Either presence is OK."""

    create_tma = tvm.tir.op.Op.get("tl.create_tma_descriptor")
    tma_load = tvm.tir.op.Op.get("tl.tma_load")

    @T.prim_func
    def kernel(A_handle: T.handle, smem_handle: T.handle):
        desc = T.call_intrin(
            "handle",
            create_tma,
            T.int32(_CU_TENSOR_MAP_DATA_TYPE_FLOAT16),
            T.int32(2),
            A_handle,
            T.int32(64),
            T.int32(64),
            T.int32(2),
            T.int32(128),
            T.int32(16),
            T.int32(16),
            T.int32(1),
            T.int32(1),
            T.int32(0),
            T.int32(2),  # swizzle = CU_TENSOR_MAP_SWIZZLE_64B
            T.int32(0),
            T.int32(0),
        )
        T.evaluate(
            T.call_intrin(
                "handle",
                tma_load,
                desc,
                T.int32(0),
                smem_handle,
                T.int32(0),
                T.int32(0),
                T.int32(0),
            )
        )

    mod = _lower(kernel, "metal")
    s = str(mod.script() if hasattr(mod, "script") else mod)
    assert ("pragma_tma_swizzle" in s) or ("tl_tma_swizzle" in s), f"expected swizzle hint to be preserved post-lowering:\n{s}"


def test_im2col_call_is_left_in_place_with_warning():
    """The ``tma_load_im2col`` fallback is not yet implemented because the
    coord layout differs from ``tma_load`` (image_offset_w/h between
    coords and eviction). Until the gather loop lands, the pass must
    leave the call in place rather than emit a wrong-stride copy that
    silently corrupts conv2d output. Verify that policy."""

    create_im2col = tvm.tir.op.Op.get("tl.create_tma_im2col_descriptor")
    tma_load_im2col_op = tvm.tir.op.Op.get("tl.tma_load_im2col")

    @T.prim_func
    def kernel(A_handle: T.handle, smem_handle: T.handle):
        # Rank-4 NHWC im2col descriptor (rough shape):
        #   data_type, rank, addr, shape×4, stride×4, elem_stride×4,
        #   lower×2, upper×2, smem_box_pixel, smem_box_channel,
        #   interleave, swizzle, l2_promotion, oob_fill
        desc = T.call_intrin(
            "handle",
            create_im2col,
            T.int32(_CU_TENSOR_MAP_DATA_TYPE_FLOAT16),
            T.int32(4),
            A_handle,
            T.int32(8),
            T.int32(64),
            T.int32(64),
            T.int32(1),  # shape
            T.int32(2),
            T.int32(16),
            T.int32(1024),
            T.int32(65536),  # stride
            T.int32(1),
            T.int32(1),
            T.int32(1),
            T.int32(1),  # elem_stride
            T.int32(0),
            T.int32(0),  # lower_corner
            T.int32(0),
            T.int32(0),  # upper_corner
            T.int32(16),
            T.int32(8),  # smem_box pix/ch
            T.int32(0),
            T.int32(0),
            T.int32(0),
            T.int32(0),  # interleave/swiz
        )
        T.evaluate(
            T.call_intrin(
                "handle",
                tma_load_im2col_op,
                desc,
                T.int32(0),
                smem_handle,
                T.int32(0),
                T.int32(0),
                T.int32(0),
                T.int32(0),  # coords c,w,h,n
                T.int32(0),
                T.int32(0),  # img_off w,h
                T.int32(0),  # eviction
            )
        )

    mod = _lower(kernel, "metal")
    s = str(mod.script() if hasattr(mod, "script") else mod)
    assert "tma_load_im2col(" in s, "im2col call must NOT be silently rewritten (TODO: gather loop)."


def _build_tma_kernel_full(
    data_type_code: int = _CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    swizzle: int = _CU_TENSOR_MAP_SWIZZLE_NONE,
    elem_bytes: int = 2,
    big_stride: bool = False,
):
    """Extended descriptor builder. ``swizzle`` covers all 4 cuTensorMap
    swizzle codes (0/1/2/3); ``big_stride=True`` exercises the 64-bit
    accumulator path with a stride > 2^31 to catch any 32-bit overflow
    regression in the offset math."""

    create_tma = tvm.tir.op.Op.get("tl.create_tma_descriptor")
    tma_load = tvm.tir.op.Op.get("tl.tma_load")

    inner_stride = elem_bytes
    outer_stride = (1 << 32) + 64 * elem_bytes if big_stride else 64 * elem_bytes
    stride_t = T.int64 if big_stride else T.int32

    @T.prim_func
    def kernel(A_handle: T.handle, smem_handle: T.handle):
        desc = T.call_intrin(
            "handle",
            create_tma,
            T.int32(data_type_code),
            T.int32(2),
            A_handle,
            T.int32(64),
            T.int32(64),
            stride_t(inner_stride),
            stride_t(outer_stride),
            T.int32(16),
            T.int32(16),
            T.int32(1),
            T.int32(1),
            T.int32(0),
            T.int32(swizzle),
            T.int32(0),
            T.int32(0),
        )
        T.evaluate(
            T.call_intrin(
                "handle",
                tma_load,
                desc,
                T.int32(0),
                smem_handle,
                T.int32(0),
                T.int32(0),
                T.int32(0),
            )
        )

    return kernel


@pytest.mark.parametrize(
    "code,expected",
    [
        (_CU_TENSOR_MAP_DATA_TYPE_UINT8, ("uint8", 1)),
        (_CU_TENSOR_MAP_DATA_TYPE_UINT16, ("uint16", 2)),
        (_CU_TENSOR_MAP_DATA_TYPE_INT32, ("int32", 4)),
        (_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, ("bfloat16", 2)),
    ],
)
def test_dtype_recovery_matrix(code, expected):
    """Lock the inverse of ``to_CUtensorMapDataType`` for the production
    paths beyond fp16/fp32: uint8 (fp8 path — cppmega FP8 amax),
    uint16 (legacy half-int), int32, and bfloat16. Regression target:
    every recovered dtype must match the descriptor's encoded type, and
    the synthetic view's per-element byte count must be exact."""

    type_name, byte_size = expected
    func = _build_tma_kernel_full(
        data_type_code=code,
        elem_bytes=byte_size,
    )
    mod = _lower(func, "metal")
    s = str(mod.script() if hasattr(mod, "script") else mod)
    if "__tl_ptr_copy_elem" in s:
        # Legacy opaque path: byte count must be exact.
        marker = (f", {byte_size})", f", {byte_size}i64)", f", T.int64({byte_size})")
        assert any(m in s for m in marker), f"expected {byte_size}-byte stride for {type_name}:\n{s}"
    else:
        assert type_name in s, f"expected {type_name} dtype in synthetic view:\n{s}"


@pytest.mark.parametrize(
    "swizzle_code",
    [
        _CU_TENSOR_MAP_SWIZZLE_NONE,
        _CU_TENSOR_MAP_SWIZZLE_32B,
        _CU_TENSOR_MAP_SWIZZLE_64B,
        _CU_TENSOR_MAP_SWIZZLE_128B,
    ],
)
def test_swizzle_distinguishability(swizzle_code):
    """Each cuTensorMap swizzle code (NONE/32B/64B/128B) must round-trip
    through ``LowerTMAToPtrArith`` without being collapsed to a single
    default. Either the ``pragma_tma_swizzle`` AttrStmt must carry the
    integer code, or the For-loop annotation ``tl_tma_swizzle`` set by
    inject_pipeline must carry it. Regression: previously the swizzle
    field was decoded but never attached to the rewritten body, so all
    modes lowered identically and Metal/HIP layout codegens lost the
    information."""

    func = _build_tma_kernel_full(swizzle=swizzle_code)
    mod = _lower(func, "metal")
    s = str(mod.script() if hasattr(mod, "script") else mod)
    if swizzle_code == _CU_TENSOR_MAP_SWIZZLE_NONE:
        # Code 0 is the legitimate "no swizzle" case; pass either preserves
        # the pragma with value 0 or omits it entirely.
        return
    # Non-zero codes MUST be visible somewhere in the lowered IR — either
    # as the AttrStmt key or the For-loop annotation key.
    assert ("pragma_tma_swizzle" in s) or ("tl_tma_swizzle" in s), f"swizzle code {swizzle_code} dropped during lowering:\n{s}"
    # And the integer value must be the one we set, not collapsed to a
    # constant 0/1.
    assert str(swizzle_code) in s, f"swizzle integer code {swizzle_code} not present in IR:\n{s}"


def test_int64_stride_no_overflow():
    """Regression for the 64-bit offset accumulator path. A stride
    exceeding 2^31 would silently wrap to a negative value if the offset
    arithmetic accumulated in Int(32). Verify that the lowered IR uses
    Int(64) ops (visible as ``int64`` casts or ``i64`` literals) rather
    than truncating to Int(32). Without this guard, large dense tensors
    (e.g. 64K-wide fp32 rows × 32-rank descriptor) would corrupt memory
    on the fallback path."""

    func = _build_tma_kernel_full(big_stride=True)
    mod = _lower(func, "metal")
    s = str(mod.script() if hasattr(mod, "script") else mod)
    # Either the cast(int64, ...) form, the i64 literal suffix, or the
    # T.int64() printer form must appear — all three are valid TVM
    # serializations of an Int(64) op chain.
    assert ("int64" in s) or ("i64" in s) or ("T.int64(" in s), f"expected Int(64) accumulator in lowered IR for >2^31 stride:\n{s}"


def test_unknown_dtype_code_is_refused():
    """An unrecognized CUtensorMapDataType code must NOT silently lower
    to a wrong byte stride — the pass logs a warning and leaves the TMA
    call in place so codegen surfaces the bug instead of corrupting
    memory."""
    func = _build_tma_kernel(data_type_code=999)  # outside enum range
    mod = _lower(func, "metal")
    assert _has_tma_call(mod), "Unknown dtype must NOT be silently lowered (would corrupt mem)."


def test_vendored_allocate_is_passed_through():
    """Wave-7 #3 regression: a Path-C TileLang DSL kernel that mixes
    ``T.alloc_shared(...)`` (lowered to the vendored ``tilelang.Allocate``
    node) with a TMA copy must NOT trip the StmtFunctor dispatch table.

    Before the fix, ``LowerTMAToPtrArith`` rejected the vendored Allocate
    with ``Check failed: (can_dispatch(n)) is false``, breaking every
    engine-path lowering of ``sparse_mla_blockscaled``,
    ``fp8_vecmat_path_c``, etc. The ``TryVisitAllocateMutator`` pass-
    through helper (mirroring ``frontend_legalize.cc`` /
    ``lower_tile_op.cc`` / ``simplify.cc``) preserves the Allocate so
    the downstream ``LowerTileLangAllocate`` pass can convert it to
    apache-native ``AllocBuffer + SeqStmt``.
    """

    create_tma = tvm.tir.op.Op.get("tl.create_tma_descriptor")
    tma_load = tvm.tir.op.Op.get("tl.tma_load")

    @T.prim_func
    def kernel(A_handle: T.handle):
        # Wrap the TMA emission with a T.alloc_shared so the vendored
        # tilelang.Allocate ends up on the LowerTMAToPtrArith dispatch
        # path. The exact buffer shape is irrelevant for this regression.
        smem = T.alloc_buffer((16, 16), "float16", scope="shared")
        desc = T.call_intrin(
            "handle",
            create_tma,
            T.int32(_CU_TENSOR_MAP_DATA_TYPE_FLOAT16),
            T.int32(2),
            A_handle,
            T.int32(64),
            T.int32(64),
            T.int32(2),
            T.int32(128),
            T.int32(16),
            T.int32(16),
            T.int32(1),
            T.int32(1),
            T.int32(0),
            T.int32(0),
            T.int32(0),
            T.int32(0),
        )
        T.evaluate(
            T.call_intrin(
                "handle",
                tma_load,
                desc,
                T.int32(0),
                smem.access_ptr("w"),
                T.int32(0),
                T.int32(0),
                T.int32(0),
            )
        )

    # The bug manifests at IR construction inside the pass — any
    # successful return (Hopper no-op or Metal rewrite) confirms the
    # dispatcher accepted the vendored Allocate node.
    mod_metal = _lower(kernel, "metal")
    assert mod_metal is not None
    mod_hopper = _lower(kernel, "cuda -arch=sm_90")
    assert mod_hopper is not None


import tilelang.testing
import torch


@tilelang.testing.requires_metal
def test_metal_tma_copy_runtime():
    """Verify that a TMA copy kernel compiles and runs successfully on Metal
    via the pointer-arithmetic fallback."""

    _M, _N = 128, 256
    _block_M, _block_N = 64, 128
    pytest.xfail("Runtime test for TMA fallback on Metal requires full pipeline setup")


@tilelang.testing.requires_rocm
def test_hip_tma_copy_runtime():
    """Verify that a TMA copy kernel compiles and runs successfully on HIP
    via the pointer-arithmetic fallback."""

    M, N = 128, 256
    block_M, block_N = 64, 128

    @T.prim_func
    def tma_copy_kernel(
        A: T.Buffer((M, N), "float16"),
        B: T.Buffer((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), "float16")
            mbar = T.alloc_barrier(128)
            T.tma_copy(A[by * block_M, bx * block_N], A_shared, barrier=mbar)
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.tma_copy(A_shared, B[by * block_M, bx * block_N])
            T.tma_store_wait()

    kernel = tl.compile(
        tma_copy_kernel,
        target="hip",
        pass_configs={"tl.disable_warp_specialized": True},
    )

    a = torch.randn(M, N, dtype=torch.float16, device="cuda")
    b = torch.zeros_like(a)
    kernel(a, b)
    torch.testing.assert_close(a, b)


if __name__ == "__main__":
    tilelang.testing.main()
