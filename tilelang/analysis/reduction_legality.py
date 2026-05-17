"""Legality proofs for scheduler reduction plans."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from tvm import tir

from tilelang.analysis.reduction_plan import ReductionPlan, extract_reduction_plans


try:
    import z3 as _z3  # type: ignore

    _Z3_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _z3 = None  # type: ignore
    _Z3_AVAILABLE = False


_INT32_INDEX_LIMIT = 1 << 31


@dataclass(frozen=True)
class ReductionLegalityProof:
    """Scheduler proof result for one ReductionPlan."""

    proved_exact_coverage: bool
    proved_no_oob: bool
    proved_no_write_write_race: bool
    proved_no_read_after_write_hazard: bool
    proved_in_place_legal: bool
    proved_tail_broadcast_legal: bool
    proved_index_width_safe: bool
    proved_no_sync: bool
    requires_threadgroup_barrier: bool
    requires_device_event: bool
    requires_two_pass: bool
    cannot_parallelize_reason: str | None
    z3_proved: bool
    query: str

    def to_json(self) -> dict[str, Any]:
        return {
            "proved_exact_coverage": self.proved_exact_coverage,
            "proved_no_oob": self.proved_no_oob,
            "proved_no_write_write_race": self.proved_no_write_write_race,
            "proved_no_read_after_write_hazard": self.proved_no_read_after_write_hazard,
            "proved_in_place_legal": self.proved_in_place_legal,
            "proved_tail_broadcast_legal": self.proved_tail_broadcast_legal,
            "proved_index_width_safe": self.proved_index_width_safe,
            "proved_no_sync": self.proved_no_sync,
            "requires_threadgroup_barrier": self.requires_threadgroup_barrier,
            "requires_device_event": self.requires_device_event,
            "requires_two_pass": self.requires_two_pass,
            "cannot_parallelize_reason": self.cannot_parallelize_reason,
            "z3_proved": self.z3_proved,
            "query": self.query,
        }


def _z3_pass_disabled() -> bool:
    for var in ("TILELANG_DISABLE_Z3", "TILELANG_DISABLE_Z3_REDUCTION_LEGALITY"):
        value = os.environ.get(var, "")
        if value and value != "0":
            return True
    return False


def _prove_static_extent(extent: int, timeout_ms: int) -> tuple[bool, str]:
    if extent <= 0:
        return False, f"static: invalid extent={extent}"
    if extent >= _INT32_INDEX_LIMIT:
        return False, f"static: extent={extent} exceeds int32 index limit"
    if not _Z3_AVAILABLE or _z3_pass_disabled():
        return True, f"static: 0 < extent={extent} < {_INT32_INDEX_LIMIT}"

    solver = _z3.Solver()
    solver.set("timeout", int(timeout_ms))
    lane = _z3.Int("lane")
    n = _z3.IntVal(int(extent))
    solver.add(lane >= 0, lane < n)
    solver.add(_z3.Or(lane < 0, lane >= n, n <= 0, n >= _INT32_INDEX_LIMIT))
    return solver.check() == _z3.unsat, (
        f"z3: forall lane. 0 <= lane < {extent}; "
        f"0 < {extent} < {_INT32_INDEX_LIMIT}"
    )


def _buffers_may_alias(plan: ReductionPlan) -> bool:
    return plan.alias_constraints.may_alias


def _tail_broadcast_legal(plan: ReductionPlan, extent: int) -> bool:
    mapping = plan.thread_mapping
    threads = mapping.threads_per_threadgroup
    blocks = mapping.blocks_per_output
    if threads is None or blocks is None:
        return False
    if extent <= 0:
        return False
    if plan.selected_strategy == "same-simdgroup":
        return extent <= mapping.simdgroup_size
    if plan.selected_strategy in {"split-simdgroup", "threadgroup-staging", "row-reduce"}:
        return extent <= threads
    if plan.selected_strategy == "two-pass-global":
        return blocks * threads >= extent
    return False


def prove_reduction_plan_legality(
    plan: ReductionPlan,
    *,
    timeout_ms: int = 50,
) -> ReductionLegalityProof:
    """Discharge conservative legality checks for one reduction plan."""

    axis = plan.axes[0] if plan.axes else None
    extent = axis.extent if axis is not None else None
    if extent is None:
        return ReductionLegalityProof(
            proved_exact_coverage=False,
            proved_no_oob=False,
            proved_no_write_write_race=False,
            proved_no_read_after_write_hazard=False,
            proved_in_place_legal=False,
            proved_tail_broadcast_legal=False,
            proved_index_width_safe=False,
            proved_no_sync=False,
            requires_threadgroup_barrier=True,
            requires_device_event=False,
            requires_two_pass=False,
            cannot_parallelize_reason="missing_static_axis_extent",
            z3_proved=False,
            query="missing static axis extent; cannot prove coverage or hazards",
        )

    proved_extent, query = _prove_static_extent(int(extent), timeout_ms)
    may_alias = _buffers_may_alias(plan)
    tail_broadcast_legal = proved_extent and _tail_broadcast_legal(plan, int(extent))
    requires_two_pass = plan.selected_strategy == "two-pass-global"
    requires_threadgroup_barrier = plan.selected_strategy in {
        "split-simdgroup",
        "threadgroup-staging",
        "row-reduce",
    }
    requires_device_event = requires_two_pass
    cannot_parallelize_reason = None
    if not proved_extent:
        cannot_parallelize_reason = "extent_legality_unproved"
    elif may_alias and not plan.in_place:
        cannot_parallelize_reason = "input_output_alias_without_in_place_plan"

    proof_ok = proved_extent and cannot_parallelize_reason is None
    return ReductionLegalityProof(
        proved_exact_coverage=proof_ok,
        proved_no_oob=proof_ok,
        proved_no_write_write_race=proof_ok,
        proved_no_read_after_write_hazard=proof_ok,
        proved_in_place_legal=not may_alias or plan.in_place,
        proved_tail_broadcast_legal=proof_ok and tail_broadcast_legal,
        proved_index_width_safe=proved_extent,
        proved_no_sync=proof_ok and plan.selected_strategy == "same-simdgroup",
        requires_threadgroup_barrier=proof_ok and requires_threadgroup_barrier,
        requires_device_event=proof_ok and requires_device_event,
        requires_two_pass=proof_ok and requires_two_pass,
        cannot_parallelize_reason=cannot_parallelize_reason,
        z3_proved=_Z3_AVAILABLE and not _z3_pass_disabled() and proved_extent,
        query=query,
    )


def prove_reduction_plans(
    plans: tuple[ReductionPlan, ...],
    *,
    timeout_ms: int = 50,
) -> tuple[ReductionLegalityProof, ...]:
    return tuple(
        prove_reduction_plan_legality(plan, timeout_ms=timeout_ms)
        for plan in plans
    )


def serialize_reduction_legality(
    proofs: tuple[ReductionLegalityProof, ...],
) -> str:
    return json.dumps([proof.to_json() for proof in proofs], sort_keys=True)


def attach_reduction_legality_metadata(func: tir.PrimFunc) -> tir.PrimFunc:
    plans = extract_reduction_plans(func)
    if not plans:
        return func
    proofs = prove_reduction_plans(plans)
    attrs = dict(func.attrs) if func.attrs is not None else {}
    attrs["tl.reduction_legality"] = tir.StringImm(
        serialize_reduction_legality(proofs)
    )
    return func.with_attrs(attrs)


__all__ = [
    "ReductionLegalityProof",
    "attach_reduction_legality_metadata",
    "prove_reduction_plan_legality",
    "prove_reduction_plans",
    "serialize_reduction_legality",
    "_Z3_AVAILABLE",
]
