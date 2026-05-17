from __future__ import annotations

from tilelang.backend.reduction import register_reduction_lowerer


register_reduction_lowerer(
    name="cpu.vectorized-fallback",
    backend="cpu",
    target_kinds=("cpu", "llvm", "c"),
    strategies="vectorized-cpu-fallback",
    ops="*",
    lowerer="cpu.vectorized_reduce",
    memory_visibility_scope="thread",
    scratch_scope=None,
    internal_scratch_required=False,
    external_materialization_required=False,
    notes="CPU reductions use the backend vectorized fallback",
)
