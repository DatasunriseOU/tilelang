from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T
import tilelang.testing

auto_target = tvm.target.Target(determine_target("auto"))
_CLUSTER_ATTR_KEYS = {"clusterIdx.x", "clusterIdx.y", "clusterIdx.z"}


def _lower(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    return tvm.tir.transform.LowerOpaqueBlock()(mod)


def _int_attr(value):
    if isinstance(value, tvm.tir.IntImm):
        return int(value.value)
    return int(value)


def _check(original):
    lowered = _lower(original)
    planned = tl.transform.ClusterPlanning()(lowered)

    tvm.ir.assert_structural_equal(planned["main"].body, lowered["main"].body, True)
    cluster_attrs = {
        str(key): _int_attr(value)
        for key, value in planned["main"].attrs.items()
        if str(key) in _CLUSTER_ATTR_KEYS
    }
    assert len(cluster_attrs) == 1
    assert next(iter(cluster_attrs.values())) == 2


def test_cluster_planning():
    @T.prim_func
    def before(A: T.Tensor((1024, 32), T.float16), B: T.Tensor((32, 1024), T.float16), C: T.Tensor((1024, 1024), T.float16)):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), T.float16)
            B_shared = T.alloc_shared((32, 128), T.float16)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    _check(before)


if __name__ == "__main__":
    tilelang.testing.main()
