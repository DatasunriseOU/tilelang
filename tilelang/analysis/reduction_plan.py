"""Scheduler-visible metadata for semantic TileLang reductions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from tvm import tir


SAME_SIMDGROUP_MAX = 32
SINGLE_KERNEL_THREADGROUP_MAX = 256
METAL_SIMDGROUP_SIZE = 32


class ReductionPlanError(ValueError):
    """Raised when semantic reduction IR is malformed before codegen."""


@dataclass(frozen=True)
class BufferRegion:
    """A symbolic buffer region touched by a reduction value or output."""

    name: str
    dtype: str
    indices: tuple[str, ...]
    role: str

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "indices": list(self.indices),
            "role": self.role,
        }


@dataclass(frozen=True)
class ReductionAxisPlan:
    """Reduction axis metadata recovered from semantic reduction IR."""

    name: str
    expr: str
    extent: int | None
    role: str = "lane"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expr": self.expr,
            "extent": self.extent,
            "role": self.role,
        }


@dataclass(frozen=True)
class ReductionThreadMapping:
    """Backend-neutral launch-shape summary for one reduction output."""

    axis: str
    reduction_extent: int | None
    simdgroup_size: int
    threads_per_threadgroup: int | None
    simdgroups_per_threadgroup: int | None
    blocks_per_output: int | None
    selected_strategy: str

    def to_json(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "reduction_extent": self.reduction_extent,
            "simdgroup_size": self.simdgroup_size,
            "threads_per_threadgroup": self.threads_per_threadgroup,
            "simdgroups_per_threadgroup": self.simdgroups_per_threadgroup,
            "blocks_per_output": self.blocks_per_output,
            "selected_strategy": self.selected_strategy,
        }


@dataclass(frozen=True)
class ReductionAliasConstraints:
    """Input/output aliasing contract attached before codegen."""

    input_buffer_names: tuple[str, ...]
    output_buffer_name: str
    may_alias: bool
    in_place_allowed: bool
    constraint: str

    def to_json(self) -> dict[str, Any]:
        return {
            "input_buffer_names": list(self.input_buffer_names),
            "output_buffer_name": self.output_buffer_name,
            "may_alias": self.may_alias,
            "in_place_allowed": self.in_place_allowed,
            "constraint": self.constraint,
        }


@dataclass(frozen=True)
class ReductionMemoryPlan:
    """Scratch/materialization requirements implied by the selected strategy."""

    visibility_scope: str
    scratch_scope: str | None
    internal_scratch_required: bool
    external_materialization_required: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "visibility_scope": self.visibility_scope,
            "scratch_scope": self.scratch_scope,
            "internal_scratch_required": self.internal_scratch_required,
            "external_materialization_required": self.external_materialization_required,
        }


@dataclass(frozen=True)
class ReductionPlan:
    """Scheduler-level reduction plan candidates before backend lowering."""

    op: str
    input_regions: tuple[BufferRegion, ...]
    output_region: BufferRegion
    axes: tuple[ReductionAxisPlan, ...]
    predicate: str
    accumulator_dtype: str
    candidate_strategies: tuple[str, ...]
    selected_strategy: str
    thread_mapping: ReductionThreadMapping
    alias_constraints: ReductionAliasConstraints
    memory_plan: ReductionMemoryPlan
    memory_visibility_scope: str
    aliasing_allowed: bool = False
    in_place: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "input_regions": [region.to_json() for region in self.input_regions],
            "output_region": self.output_region.to_json(),
            "axes": [axis.to_json() for axis in self.axes],
            "predicate": self.predicate,
            "accumulator_dtype": self.accumulator_dtype,
            "candidate_strategies": list(self.candidate_strategies),
            "selected_strategy": self.selected_strategy,
            "thread_mapping": self.thread_mapping.to_json(),
            "alias_constraints": self.alias_constraints.to_json(),
            "memory_plan": self.memory_plan.to_json(),
            "memory_visibility_scope": self.memory_visibility_scope,
            "aliasing_allowed": self.aliasing_allowed,
            "in_place": self.in_place,
        }


def _op_name(node: tir.Call) -> str:
    return str(getattr(getattr(node, "op", None), "name", ""))


def _buffer_region(load: tir.BufferLoad, role: str) -> BufferRegion:
    return BufferRegion(
        name=str(load.buffer.name),
        dtype=str(load.dtype),
        indices=tuple(str(index) for index in load.indices),
        role=role,
    )


def _collect_buffer_loads(expr: tir.PrimExpr) -> tuple[BufferRegion, ...]:
    loads: list[BufferRegion] = []

    def _visit(node):
        if isinstance(node, tir.BufferLoad):
            loads.append(_buffer_region(node, "read"))

    tir.stmt_functor.post_order_visit(expr, _visit)
    return tuple(loads)


def _string_value(expr: tir.PrimExpr) -> str:
    if isinstance(expr, tir.StringImm):
        return str(expr.value)
    return str(expr).strip('"')


def _int_imm_value(expr: tir.PrimExpr) -> int | None:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _axis_extent_from_expr(expr: tir.PrimExpr) -> int | None:
    for cls_name in ("FloorMod", "TruncMod", "Mod"):
        cls = getattr(tir, cls_name, None)
        if cls is not None and isinstance(expr, cls):
            rhs = getattr(expr, "b", None)
            return _int_imm_value(rhs)
    return None


def _axis_name(expr: tir.PrimExpr) -> str:
    if isinstance(expr, tir.Var):
        return str(expr.name)
    for attr in ("a", "b"):
        child = getattr(expr, attr, None)
        if isinstance(child, tir.Var):
            return str(child.name)
    return "reduce"


def candidate_strategies_for_extent(extent: int | None) -> tuple[str, ...]:
    """Return ordered backend strategy candidates for a reduction extent."""

    if extent is None:
        return (
            "same-simdgroup",
            "split-simdgroup",
            "threadgroup-staging",
            "row-reduce",
            "two-pass-global",
            "vectorized-cpu-fallback",
        )
    if extent <= SAME_SIMDGROUP_MAX:
        return (
            "same-simdgroup",
            "split-simdgroup",
            "threadgroup-staging",
            "row-reduce",
            "vectorized-cpu-fallback",
        )
    if extent <= SINGLE_KERNEL_THREADGROUP_MAX:
        return (
            "split-simdgroup",
            "threadgroup-staging",
            "row-reduce",
            "vectorized-cpu-fallback",
        )
    return ("two-pass-global", "vectorized-cpu-fallback")


def selected_strategy_for_extent(extent: int | None) -> str:
    """Return the current scheduler-selected strategy for a reduction extent."""

    if extent is None:
        return "unresolved"
    strategies = candidate_strategies_for_extent(extent)
    return strategies[0] if strategies else "unknown"


def _ceildiv(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _visibility_scope(selected_strategy: str) -> str:
    if selected_strategy == "same-simdgroup":
        return "simdgroup"
    if selected_strategy in {"split-simdgroup", "threadgroup-staging", "row-reduce"}:
        return "threadgroup"
    return "device"


def _thread_mapping(
    *,
    axis: ReductionAxisPlan,
    selected_strategy: str,
) -> ReductionThreadMapping:
    extent = axis.extent
    if extent is None:
        threads_per_threadgroup = None
        simdgroups_per_threadgroup = None
        blocks_per_output = None
    else:
        if selected_strategy == "same-simdgroup":
            threads_per_threadgroup = METAL_SIMDGROUP_SIZE
        else:
            threads_per_threadgroup = min(
                SINGLE_KERNEL_THREADGROUP_MAX,
                max(METAL_SIMDGROUP_SIZE, int(extent)),
            )
        simdgroups_per_threadgroup = _ceildiv(
            threads_per_threadgroup,
            METAL_SIMDGROUP_SIZE,
        )
        if selected_strategy == "two-pass-global":
            blocks_per_output = _ceildiv(int(extent), SINGLE_KERNEL_THREADGROUP_MAX)
        else:
            blocks_per_output = 1
    return ReductionThreadMapping(
        axis=axis.name,
        reduction_extent=extent,
        simdgroup_size=METAL_SIMDGROUP_SIZE,
        threads_per_threadgroup=threads_per_threadgroup,
        simdgroups_per_threadgroup=simdgroups_per_threadgroup,
        blocks_per_output=blocks_per_output,
        selected_strategy=selected_strategy,
    )


def _alias_constraints(
    *,
    input_regions: tuple[BufferRegion, ...],
    output_region: BufferRegion,
    in_place: bool,
) -> ReductionAliasConstraints:
    input_names = tuple(region.name for region in input_regions)
    may_alias = any(name == output_region.name for name in input_names)
    if may_alias and in_place:
        constraint = "explicit_in_place_plan_required"
    elif may_alias:
        constraint = "reject_without_in_place_plan"
    else:
        constraint = "distinct_input_output_buffers"
    return ReductionAliasConstraints(
        input_buffer_names=input_names,
        output_buffer_name=output_region.name,
        may_alias=may_alias,
        in_place_allowed=in_place,
        constraint=constraint,
    )


def _memory_plan(selected_strategy: str) -> ReductionMemoryPlan:
    visibility_scope = _visibility_scope(selected_strategy)
    if selected_strategy in {"split-simdgroup", "threadgroup-staging", "row-reduce"}:
        scratch_scope = "threadgroup"
        internal_scratch_required = True
    elif selected_strategy == "two-pass-global":
        scratch_scope = "device"
        internal_scratch_required = True
    else:
        scratch_scope = None
        internal_scratch_required = False
    return ReductionMemoryPlan(
        visibility_scope=visibility_scope,
        scratch_scope=scratch_scope,
        internal_scratch_required=internal_scratch_required,
        external_materialization_required=False,
    )


def _make_plan(
    *,
    op: str,
    input_regions: tuple[BufferRegion, ...],
    output_region: BufferRegion,
    axes: tuple[ReductionAxisPlan, ...],
    predicate: str,
    accumulator_dtype: str,
    aliasing_allowed: bool = False,
    in_place: bool = False,
) -> ReductionPlan:
    extent = axes[0].extent if axes else None
    strategies = candidate_strategies_for_extent(extent)
    selected_strategy = selected_strategy_for_extent(extent)
    memory_plan = _memory_plan(selected_strategy)
    return ReductionPlan(
        op=op,
        input_regions=input_regions,
        output_region=output_region,
        axes=axes,
        predicate=predicate,
        accumulator_dtype=accumulator_dtype,
        candidate_strategies=strategies,
        selected_strategy=selected_strategy,
        thread_mapping=_thread_mapping(axis=axes[0], selected_strategy=selected_strategy),
        alias_constraints=_alias_constraints(
            input_regions=input_regions,
            output_region=output_region,
            in_place=in_place,
        ),
        memory_plan=memory_plan,
        memory_visibility_scope=memory_plan.visibility_scope,
        aliasing_allowed=aliasing_allowed,
        in_place=in_place,
    )


def _region_load(call: tir.Call) -> tir.BufferLoad | None:
    if not _op_name(call).endswith("tileop.region"):
        return None
    if not call.args:
        return None
    load = call.args[0]
    return load if isinstance(load, tir.BufferLoad) else None


def _region_extent(call: tir.Call, dim: int) -> int | None:
    rank = len(call.args) - 2
    if dim < 0 or dim >= rank:
        raise ReductionPlanError(
            f"malformed tl.tileop.reduce axis dim={dim} for tl.tileop.region rank={rank}"
        )
    extent_index = 2 + dim
    if extent_index >= len(call.args):
        return None
    return _int_imm_value(call.args[extent_index])


def _region_buffer_region(call: tir.Call, role: str) -> BufferRegion | None:
    load = _region_load(call)
    if load is None:
        return None
    return _buffer_region(load, role)


def _plan_from_thread_allreduce_call(call: tir.Call) -> ReductionPlan | None:
    if not _op_name(call).endswith("tvm_thread_allreduce"):
        return None
    if len(call.args) < 5:
        return None
    value = call.args[1]
    predicate = call.args[2]
    out = call.args[3]
    reduce_index = call.args[4]
    if not isinstance(value, tir.PrimExpr):
        return None
    if not isinstance(out, tir.BufferLoad):
        return None

    extent = _axis_extent_from_expr(reduce_index)
    axis = ReductionAxisPlan(
        name=_axis_name(reduce_index),
        expr=str(reduce_index),
        extent=extent,
    )
    return _make_plan(
        op="sum",
        input_regions=_collect_buffer_loads(value),
        output_region=_buffer_region(out, "write"),
        axes=(axis,),
        predicate=str(predicate),
        accumulator_dtype=str(value.dtype),
        aliasing_allowed=False,
        in_place=False,
    )


def _plan_from_tileop_reduce_call(call: tir.Call) -> ReductionPlan | None:
    if not _op_name(call).endswith("tileop.reduce"):
        return None
    if len(call.args) < 5:
        return None
    src_region = call.args[0]
    dst_region = call.args[1]
    reduce_kind = call.args[2]
    dim_expr = call.args[3]
    if not isinstance(src_region, tir.Call):
        return None
    if not isinstance(dst_region, tir.Call):
        return None
    dim = _int_imm_value(dim_expr)
    if dim is None:
        return None
    input_region = _region_buffer_region(src_region, "read")
    output_region = _region_buffer_region(dst_region, "write")
    if input_region is None or output_region is None:
        return None
    extent = _region_extent(src_region, dim)
    axis = ReductionAxisPlan(
        name=f"dim{dim}",
        expr=f"dim={dim}",
        extent=extent,
        role="block",
    )
    return _make_plan(
        op=_string_value(reduce_kind),
        input_regions=(input_region,),
        output_region=output_region,
        axes=(axis,),
        predicate="T.bool(True)",
        accumulator_dtype=input_region.dtype,
        aliasing_allowed=False,
        in_place=False,
    )


def _plan_from_call(call: tir.Call) -> ReductionPlan | None:
    return _plan_from_thread_allreduce_call(call) or _plan_from_tileop_reduce_call(call)


def extract_reduction_plans(func: tir.PrimFunc) -> tuple[ReductionPlan, ...]:
    """Extract scheduler plans from semantic ``tvm_thread_allreduce`` calls."""

    plans: list[ReductionPlan] = []
    errors: list[ReductionPlanError] = []

    def _visit(node):
        if isinstance(node, tir.Call):
            try:
                plan = _plan_from_call(node)
            except ReductionPlanError as exc:
                errors.append(exc)
                return
            if plan is not None:
                plans.append(plan)

    tir.stmt_functor.post_order_visit(func.body, _visit)
    if errors:
        raise errors[0]
    return tuple(plans)


def serialize_reduction_plans(plans: tuple[ReductionPlan, ...]) -> str:
    return json.dumps([plan.to_json() for plan in plans], sort_keys=True)


def attach_reduction_plan_metadata(func: tir.PrimFunc) -> tir.PrimFunc:
    """Attach stable JSON reduction-plan metadata to a PrimFunc."""

    plans = extract_reduction_plans(func)
    if not plans:
        return func
    attrs = dict(func.attrs) if func.attrs is not None else {}
    attrs["tl.reduction_plans"] = tir.StringImm(serialize_reduction_plans(plans))
    return func.with_attrs(attrs)
