from __future__ import annotations

from tilelang.backend.reduction import register_reduction_lowerer


_ALL_REDUCE_OPS = ("*",)


register_reduction_lowerer(
    name="cuda.warp-shuffle",
    backend="cuda",
    target_kinds="cuda",
    strategies="same-simdgroup",
    ops=_ALL_REDUCE_OPS,
    lowerer="tir.tvm_warp_shuffle",
    memory_visibility_scope="warp",
    scratch_scope=None,
    internal_scratch_required=False,
    external_materialization_required=False,
    notes="warp-local reductions lower through CUDA shuffle intrinsics",
)

register_reduction_lowerer(
    name="cuda.block-staging",
    backend="cuda",
    target_kinds="cuda",
    strategies=("split-simdgroup", "threadgroup-staging", "row-reduce"),
    ops=_ALL_REDUCE_OPS,
    lowerer="cuda.thread_allreduce.shared_staging",
    memory_visibility_scope="threadblock",
    scratch_scope="shared",
    internal_scratch_required=True,
    external_materialization_required=False,
    notes="cross-warp reductions stage partials in shared memory",
)

register_reduction_lowerer(
    name="cuda.two-pass-global",
    backend="cuda",
    target_kinds="cuda",
    strategies="two-pass-global",
    ops=_ALL_REDUCE_OPS,
    lowerer="cuda.thread_allreduce.two_pass",
    memory_visibility_scope="device",
    scratch_scope="device",
    internal_scratch_required=True,
    external_materialization_required=False,
    notes="large reductions require an internal device-visible second pass",
)
