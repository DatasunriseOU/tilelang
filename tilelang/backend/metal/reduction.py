from __future__ import annotations

from tilelang.backend.reduction import register_reduction_lowerer


_GENERIC_REDUCE_OPS = (
    "max",
    "min",
    "abssum",
    "absmax",
    "bitand",
    "bitor",
    "bitxor",
    "and",
    "or",
    "xor",
    "mul",
)
_ALL_REDUCE_OPS = ("*",)


register_reduction_lowerer(
    name="metal.same-simdgroup.sum",
    backend="metal",
    target_kinds="metal",
    strategies="same-simdgroup",
    ops="sum",
    lowerer="tirx.metal.simd_sum",
    memory_visibility_scope="simdgroup",
    scratch_scope=None,
    internal_scratch_required=False,
    external_materialization_required=False,
    notes="single-simdgroup sum maps to Metal simd_sum",
)

register_reduction_lowerer(
    name="metal.same-simdgroup.generic",
    backend="metal",
    target_kinds="metal",
    strategies="same-simdgroup",
    ops=_GENERIC_REDUCE_OPS,
    lowerer="tirx.metal.simd_shuffle_xor",
    memory_visibility_scope="simdgroup",
    scratch_scope=None,
    internal_scratch_required=False,
    external_materialization_required=False,
    notes="single-simdgroup non-sum reductions use shuffle lowering",
)

register_reduction_lowerer(
    name="metal.split-simdgroup",
    backend="metal",
    target_kinds="metal",
    strategies=("split-simdgroup", "threadgroup-staging", "row-reduce"),
    ops=_ALL_REDUCE_OPS,
    lowerer="metal.thread_allreduce.threadgroup_staging",
    memory_visibility_scope="threadgroup",
    scratch_scope="threadgroup",
    internal_scratch_required=True,
    external_materialization_required=False,
    notes="cross-simdgroup reductions stage partials in threadgroup memory",
)

register_reduction_lowerer(
    name="metal.two-pass-global",
    backend="metal",
    target_kinds="metal",
    strategies="two-pass-global",
    ops=_ALL_REDUCE_OPS,
    lowerer="metal.thread_allreduce.two_pass",
    memory_visibility_scope="device",
    scratch_scope="device",
    internal_scratch_required=True,
    external_materialization_required=False,
    notes="large reductions require an internal device-visible second pass",
)
