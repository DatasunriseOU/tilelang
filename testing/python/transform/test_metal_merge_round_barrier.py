import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.transform import PassConfigKey
from tilelang.transform.metal_merge_round import FUSION_ATTR


def _count_shared_sync(mod: tvm.IRModule) -> int:
    return str(mod.script()).count('T.tvm_storage_sync("shared")')


def _run_cleanup(func, *, enabled: bool) -> tvm.IRModule:
    config = {}
    if enabled:
        config[PassConfigKey.TL_Z3_PROOF_BARRIER_MINIMIZATION.value] = True
    with tvm.transform.PassContext(config=config), tvm.target.Target("metal"):
        return tilelang.transform.MetalMergeRoundBarrierCleanup()(tvm.IRModule({"main": func}))


def _merge_round_func():
    @T.prim_func
    def func():
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding()
            pair = T.alloc_buffer((256,), "float32", scope="shared")
            merged = T.alloc_buffer((8,), "float32", scope="local")
            stride = T.alloc_var("int32")
            other = T.alloc_var("int32")

            for round_id in T.serial(5):
                stride = T.shift_left(1, round_id)
                if lane % (stride * 2) == 0:
                    other = lane + stride
                    if other < 32:
                        merged[0] = pair[other * 8]
                T.tvm_storage_sync("shared")
                if lane % (stride * 2) == 0:  # noqa: SIM102 - keep canonical emitted shape
                    if other < 32:
                        pair[lane * 8] = merged[0]
                T.tvm_storage_sync("shared")

    return func


def _unsafe_different_shared_writeback_func():
    @T.prim_func
    def func():
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding()
            pair = T.alloc_buffer((256,), "float32", scope="shared")
            other_pair = T.alloc_buffer((256,), "float32", scope="shared")
            merged = T.alloc_buffer((8,), "float32", scope="local")
            stride = T.alloc_var("int32")
            other = T.alloc_var("int32")

            for round_id in T.serial(5):
                stride = T.shift_left(1, round_id)
                if lane % (stride * 2) == 0:
                    other = lane + stride
                    if other < 32:
                        merged[0] = pair[other * 8]
                T.tvm_storage_sync("shared")
                if lane % (stride * 2) == 0:  # noqa: SIM102 - keep canonical emitted shape
                    if other < 32:
                        other_pair[lane * 8] = merged[0]
                T.tvm_storage_sync("shared")

    return func


@tilelang.testing.requires_metal
def test_merge_round_cleanup_fuses_writeback_when_barrier_proof_enabled():
    mod = _run_cleanup(_merge_round_func(), enabled=True)

    assert _count_shared_sync(mod) == 1, mod.script()
    assert FUSION_ATTR in str(mod.script())


@tilelang.testing.requires_metal
def test_merge_round_cleanup_disabled_by_default_keeps_barrier():
    mod = _run_cleanup(_merge_round_func(), enabled=False)

    assert _count_shared_sync(mod) == 2, mod.script()
    assert FUSION_ATTR not in str(mod.script())


@tilelang.testing.requires_metal
def test_merge_round_cleanup_keeps_unmatched_shared_writeback():
    mod = _run_cleanup(_unsafe_different_shared_writeback_func(), enabled=True)

    assert _count_shared_sync(mod) == 2, mod.script()
    assert FUSION_ATTR not in str(mod.script())


if __name__ == "__main__":
    tilelang.testing.main()
