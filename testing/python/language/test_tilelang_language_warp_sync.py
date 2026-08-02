import tilelang
import tilelang.language as T
import pytest
import torch
from tilelang import tvm as tvm
from tvm import tir
import tilelang.testing


@tilelang.jit
def kernel_with_warp_sync():
    @T.prim_func
    def main(
        A: T.Tensor((1,), "int32"),
        B: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx == 0:
                tir.call_extern("void", "__nanosleep", 100)
                A[0] = -1
            T.sync_warp()
            if tx == 1:
                B[0] = A[0]

    return main


@tilelang.testing.requires_cuda
def test_warp_sync():
    a = torch.empty((1), device="cuda", dtype=torch.int32)
    b = torch.empty((1), device="cuda", dtype=torch.int32)
    kernel = kernel_with_warp_sync()
    assert "__syncwarp" in kernel.get_kernel_source()
    kernel(a, b)
    assert b[0] == -1


@tilelang.jit
def kernel_with_shfl_sync():
    return _shfl_sync_prim_func()


def _shfl_sync_prim_func():
    @T.prim_func
    def main(
        A: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            val = tx * 10
            broadcast = T.shfl_sync(val, 31)
            A[tx] = broadcast

    return main


def _shfl_xor_prim_func(width=32, mask=0xFFFFFFFF):
    @T.prim_func
    def main(
        A: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            val = tx * 10
            broadcast = T.shfl_xor(val, 1, width=width, mask=mask)
            A[tx] = broadcast

    return main


def _called_op_names(func):
    names = set()

    def visit(node):
        if isinstance(node, tir.Call):
            names.add(str(getattr(node.op, "name", node.op)))

    tir.stmt_functor.post_order_visit(func.body, visit)
    return names


@tilelang.testing.requires_cuda
def test_shfl_sync():
    a = torch.empty((32), device="cuda", dtype=torch.int32)
    kernel = kernel_with_shfl_sync()
    assert "__shfl_sync" in kernel.get_kernel_source()
    kernel(a)
    assert torch.all(a == 310)


def test_metal_simdgroup_guard_rejects_shfl_sync_before_codegen():
    func = _shfl_sync_prim_func().with_attr("target", tvm.target.Target("metal"))
    mod = tvm.IRModule({"main": func})

    with pytest.raises(ValueError, match="Metal SIMDgroup guard.*tl.shfl_sync"):
        tilelang.transform.MetalSimdgroupSemanticGuard(mod)


def test_metal_simdgroup_guard_allows_full_width_shfl_xor():
    func = _shfl_xor_prim_func().with_attr("target", tvm.target.Target("metal"))
    mod = tvm.IRModule({"main": func})

    guarded = tilelang.transform.MetalSimdgroupSemanticGuard(mod)

    assert "shfl_xor_sync" in guarded.script()


def test_existing_fa_v3_tma_fallback_triggers_metal_simdgroup_guard():
    """FA-v3 uses a warp-policy GEMM that is not legal for Metal lowering."""
    from poc.triton_frontend import conformance

    prim = conformance._build_fa_v3_tma_fallback_prim()
    assert prim is not None

    func = prim.with_attr("target", tvm.target.Target("metal"))
    op_names = _called_op_names(func)

    assert func.attrs["global_symbol"] == "fa_v3_tma_fallback"
    assert "tl.tileop.gemm" in op_names
    assert (
        not {
            "tl.shfl_sync",
            "tl.shfl_down_sync",
            "tl.shfl_up_sync",
            "tl.shfl_xor_sync",
            "tir.tvm_warp_shuffle",
            "tir.tvm_warp_shuffle_up",
            "tir.tvm_warp_shuffle_down",
        }
        & op_names
    )

    mod = tvm.IRModule({"main": func})
    with pytest.raises(
        ValueError,
        match="Metal SIMDgroup guard.*tl.tileop.gemm.*GemmWarpPolicy.FullRow",
    ):
        tilelang.transform.MetalSimdgroupSemanticGuard(mod)


if __name__ == "__main__":
    tilelang.testing.main()
