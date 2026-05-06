# ruff: noqa
"""Regression + correctness tests for Z3 roadmap idea #11
   (Apple intra-warp barrier elision in TileLang's ThreadSync pass).

Three behaviours are pinned:

1. ``test_2d_launch_intra_simdgroup_elides`` — covers the bug fix.
   With a 2-D thread launch (`threadIdx.y`, `threadIdx.x`), the
   pre-fix `prev.threads.size() + idx - 3` indexing left `tx_w`
   undefined and caused early-return false, so the barrier was
   *never* dropped on Apple regardless of the access pattern.
   After the tag-based axis fix, the intra-simdgroup case correctly
   elides the barrier.

2. ``test_1d_launch_straddling_simdgroup_keeps_barrier`` — Z3 must
   *not* elide when the writer / reader threads can land in
   different simdgroups (e.g. writer in 0..31, reader in 32..63).

3. ``test_1d_launch_within_single_simdgroup_elides`` — small launch
   (<= 32 threads) provably stays in one simdgroup and the barrier
   is elided in the lowered IR.

All tests target the Metal kind explicitly; on non-Metal targets the
elision pass is a no-op by design (`is_metal_` guard in
`ProveIntraWarpRAW`).
"""

from tilelang import tvm as tvm
import tilelang
import tilelang.testing
from tvm.script import tir as T


def _run_thread_sync_metal(func: tvm.tir.PrimFunc) -> tvm.IRModule:
    """Apply ThreadSync("shared") under a Metal target so the
    Apple intra-warp elision path is enabled."""
    mod = tvm.IRModule.from_expr(func)
    metal_target = tvm.target.Target("metal", host="llvm")
    mod = tvm.tir.transform.Apply(
        lambda f: f.with_attr({
            "global_symbol": "test",
            "target": metal_target,
        }))(mod)
    mod = tvm.tir.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tir.transform.SplitHostDevice()(mod)
    return tilelang.transform.ThreadSync("shared")(mod)


def _count_storage_sync(mod: tvm.IRModule) -> int:
    return str(mod.script()).count('T.tvm_storage_sync("shared")')


@tilelang.testing.requires_metal
def test_2d_launch_intra_simdgroup_elides():
    """2-D thread launch with RAW conflict that is provably
    intra-simdgroup. Before the bug fix, the positional-indexing
    path left tx_w undefined and the function bailed out → barrier
    kept. After the fix, the barrier should be elided.

    Launch shape: (ty=1, tx=16). Writer & reader both have
    threadIdx.x in [0, 16) → tx_w / 32 == tx_r / 32 == 0.
    """

    @T.prim_func(private=True)
    def func():
        A_shared = T.alloc_buffer((16,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        # Writer: every thread writes one element of A_shared.
        A_shared[tx] = T.float32(1)
        # Reader: every thread reads its neighbour. Cross-thread RAW.
        # Without the elision, this triggers a tvm_storage_sync.
        if tx > 0:
            _ = A_shared[tx - 1]

    mod = _run_thread_sync_metal(func)
    n_sync = _count_storage_sync(mod)
    assert n_sync == 0, (
        "Expected ProveIntraWarpRAW to elide the barrier on a 2-D "
        "launch with tx in [0, 16) (entirely inside one simdgroup); "
        f"found {n_sync} sync(s):\n{mod.script()}")


@tilelang.testing.requires_metal
def test_1d_launch_straddling_simdgroup_keeps_barrier():
    """1-D launch with 64 threads → tx in [0, 64). The conflicting
    reader at index `tx ^ 32` lands in a *different* simdgroup
    (e.g. writer tx=0 → reader tx=32). Z3 must NOT prove
    intra-warp; barrier must be kept.
    """

    @T.prim_func(private=True)
    def func():
        A_shared = T.alloc_buffer((64,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        A_shared[tx] = T.float32(1)
        # Read from a position that can be in another simdgroup.
        # tx ^ 32 maps {0..31} <-> {32..63}, straddling the boundary.
        _ = A_shared[T.bitwise_xor(tx, T.int32(32))]

    mod = _run_thread_sync_metal(func)
    n_sync = _count_storage_sync(mod)
    assert n_sync >= 1, (
        "Expected ThreadSync to keep the barrier when reader/writer "
        "can fall in different simdgroups (tx XOR 32 straddles the "
        f"boundary); got {n_sync} sync(s):\n{mod.script()}")


@tilelang.testing.requires_metal
def test_1d_launch_within_single_simdgroup_elides():
    """1-D launch with 16 threads — entirely within a single
    32-lane Apple simdgroup. RAW conflict is provably
    intra-simdgroup; barrier must be elided.
    """

    @T.prim_func(private=True)
    def func():
        A_shared = T.alloc_buffer((16,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        A_shared[tx] = T.float32(1)
        if tx > 0:
            _ = A_shared[tx - 1]

    mod = _run_thread_sync_metal(func)
    n_sync = _count_storage_sync(mod)
    assert n_sync == 0, (
        "Expected ProveIntraWarpRAW to elide the barrier on a "
        "1-D launch with tx in [0, 16) (single simdgroup); "
        f"found {n_sync} sync(s):\n{mod.script()}")


if __name__ == "__main__":
    tilelang.testing.main()
