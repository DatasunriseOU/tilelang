"""Diagnostics for backend-owned reduction lowerer selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from tvm import tir
from tvm.target import Target

from tilelang.analysis.reduction_plan import ReductionPlan, extract_reduction_plans


@dataclass(frozen=True)
class ReductionBackendLowererDiagnostic:
    """Scheduler-visible backend lowerer selected for one reduction plan."""

    source: str
    backend: str
    target_kind: str
    lowerer_name: str
    lowerer: str
    op: str
    plan_selected_strategy: str
    selected_strategy: str
    memory_visibility_scope: str
    scratch_scope: str | None
    internal_scratch_required: bool
    external_materialization_required: bool
    cache_key: str
    notes: str

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "backend": self.backend,
            "target_kind": self.target_kind,
            "lowerer_name": self.lowerer_name,
            "lowerer": self.lowerer,
            "op": self.op,
            "plan_selected_strategy": self.plan_selected_strategy,
            "selected_strategy": self.selected_strategy,
            "memory_visibility_scope": self.memory_visibility_scope,
            "scratch_scope": self.scratch_scope,
            "internal_scratch_required": self.internal_scratch_required,
            "external_materialization_required": self.external_materialization_required,
            "cache_key": self.cache_key,
            "notes": self.notes,
        }


def _target_from_func(func: tir.PrimFunc) -> Target | None:
    if func.attrs is not None:
        target = func.attrs.get("target", None)
        if isinstance(target, Target):
            return target
    return Target.current(allow_none=True)


def _diagnostic_for_plan(
    plan: ReductionPlan,
    *,
    source: str,
    target: str | Target,
) -> ReductionBackendLowererDiagnostic:
    from tilelang.backend.reduction import resolve_reduction_lowerer

    axis = plan.axes[0] if plan.axes else None
    selection = resolve_reduction_lowerer(
        target,
        op=plan.op,
        candidate_strategies=plan.candidate_strategies,
        reduction_extent=None if axis is None else axis.extent,
        accumulator_dtype=plan.accumulator_dtype,
    )
    return ReductionBackendLowererDiagnostic(
        source=source,
        backend=selection.backend,
        target_kind=selection.target_kind,
        lowerer_name=selection.name,
        lowerer=selection.lowerer,
        op=plan.op,
        plan_selected_strategy=plan.selected_strategy,
        selected_strategy=selection.strategy,
        memory_visibility_scope=selection.memory_visibility_scope,
        scratch_scope=selection.scratch_scope,
        internal_scratch_required=selection.internal_scratch_required,
        external_materialization_required=selection.external_materialization_required,
        cache_key=selection.cache_key,
        notes=selection.notes,
    )


def build_reduction_backend_lowerer_diagnostics(
    func: tir.PrimFunc,
    target: str | Target | None = None,
) -> tuple[ReductionBackendLowererDiagnostic, ...]:
    """Build backend-lowerer diagnostics for semantic reduction plans."""

    resolved_target = target if target is not None else _target_from_func(func)
    if resolved_target is None:
        raise ValueError("Cannot select reduction backend lowerers without a target")
    plans = extract_reduction_plans(func)
    return tuple(
        _diagnostic_for_plan(
            plan,
            source=f"reduction:{idx}:{plan.op}",
            target=resolved_target,
        )
        for idx, plan in enumerate(plans)
    )


def serialize_reduction_backend_lowerer_diagnostics(
    diagnostics: tuple[ReductionBackendLowererDiagnostic, ...],
) -> str:
    return json.dumps([diagnostic.to_json() for diagnostic in diagnostics], sort_keys=True)


def attach_reduction_backend_lowerer_metadata(
    func: tir.PrimFunc,
    target: str | Target | None = None,
) -> tir.PrimFunc:
    """Attach stable JSON naming the selected backend lowerers."""

    diagnostics = build_reduction_backend_lowerer_diagnostics(func, target)
    if not diagnostics:
        return func
    attrs = dict(func.attrs) if func.attrs is not None else {}
    attrs["tl.reduction_backend_lowerers"] = tir.StringImm(
        serialize_reduction_backend_lowerer_diagnostics(diagnostics)
    )
    return func.with_attrs(attrs)


__all__ = [
    "ReductionBackendLowererDiagnostic",
    "attach_reduction_backend_lowerer_metadata",
    "build_reduction_backend_lowerer_diagnostics",
    "serialize_reduction_backend_lowerer_diagnostics",
]
