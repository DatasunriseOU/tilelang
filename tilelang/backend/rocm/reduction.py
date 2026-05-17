from __future__ import annotations

from tilelang.backend.reduction import register_reduction_lowerer


_ALL_REDUCE_OPS = ("*",)


register_reduction_lowerer(
    name="rocm.wavefront-shuffle",
    backend="rocm",
    target_kinds=("rocm", "hip"),
    strategies="same-simdgroup",
    ops=_ALL_REDUCE_OPS,
    lowerer="tir.tvm_warp_shuffle",
    memory_visibility_scope="wavefront",
    scratch_scope=None,
    internal_scratch_required=False,
    external_materialization_required=False,
    notes="wavefront-local reductions lower through ROCm shuffle intrinsics",
)

register_reduction_lowerer(
    name="rocm.block-staging",
    backend="rocm",
    target_kinds=("rocm", "hip"),
    strategies=("split-simdgroup", "threadgroup-staging", "row-reduce"),
    ops=_ALL_REDUCE_OPS,
    lowerer="rocm.thread_allreduce.shared_staging",
    memory_visibility_scope="workgroup",
    scratch_scope="shared",
    internal_scratch_required=True,
    external_materialization_required=False,
    notes="cross-wavefront reductions stage partials in shared memory",
)

register_reduction_lowerer(
    name="rocm.two-pass-global",
    backend="rocm",
    target_kinds=("rocm", "hip"),
    strategies="two-pass-global",
    ops=_ALL_REDUCE_OPS,
    lowerer="rocm.thread_allreduce.two_pass",
    memory_visibility_scope="device",
    scratch_scope="device",
    internal_scratch_required=True,
    external_materialization_required=False,
    notes="large reductions require an internal device-visible second pass",
)
