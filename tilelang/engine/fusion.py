"""High-level TileLang fusion regions.

This module describes a pre-source lowering boundary: callers build a region
graph, provide one fused TileLang/TIR schedule template for that region, and
then invoke the normal TileLang lowering pipeline on the resulting IRModule.
Already-lowered source kernels are intentionally rejected because source-level
MSL/CUDA concatenation cannot recover producer/consumer buffer lifetimes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from hashlib import sha256
import inspect
import json
from typing import Any

from tilelang import tvm as tvm
from tvm import tir

from tilelang.transform import PassConfigKey


ScheduleTemplate = Callable[["FusionRegion"], tir.PrimFunc | tvm.IRModule]

LOWERING_BOUNDARY = "tilelang_tvm_ir"
DEFAULT_BACKEND = "tilelang_tvm_ffi"
DEFAULT_PATH_B_CACHE_LIMIT_GIB = 55.0
DEFAULT_WARM_FIRST_STEP_LIMIT_MS = 12_000.0
DEFAULT_WARM_TO_STEADY_RATIO_LIMIT = 4.0


@dataclass(frozen=True)
class FusionNode:
    name: str
    op: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)
    prim_func: tir.PrimFunc | None = None


@dataclass(frozen=True)
class FusionEdge:
    producer: str
    consumer: str
    buffer: str
    lifetime: str = "internal"


@dataclass(frozen=True)
class FusionRegion:
    name: str
    nodes: tuple[FusionNode, ...]
    edges: tuple[FusionEdge, ...]
    schedule_template: ScheduleTemplate
    entry_symbol: str
    pass_configs: Mapping[str, Any]
    backend: str = DEFAULT_BACKEND
    schedule_name: str = "manual"
    schedule_status: str = "ready"


@dataclass(frozen=True)
class FusionAutogradPlan:
    mode: str
    status: str
    forward_node_names: tuple[str, ...] = ()
    backward_node_names: tuple[str, ...] = ()
    backward_edges: tuple[tuple[str, str, str], ...] = ()
    missing_backward_node_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class FusionCompilePlan:
    region_name: str
    entry_symbol: str
    lowering_boundary: str
    backend: str
    node_names: tuple[str, ...]
    edges: tuple[FusionEdge, ...]
    pass_configs: Mapping[str, Any]
    cache_key_material: tuple[str, ...]
    require_single_kernel: bool = False
    schedule_name: str = "manual"
    schedule_status: str = "ready"
    autograd_plan: FusionAutogradPlan = field(
        default_factory=lambda: FusionAutogradPlan(mode="none", status="none")
    )
    requires_source_post_fusion: bool = False


@dataclass(frozen=True)
class FusionCompileResult:
    plan: FusionCompilePlan
    lowered_module: tvm.IRModule
    artifact: Any


@dataclass(frozen=True)
class FusionScheduleEntry:
    op_signature: tuple[str, ...]
    schedule_template: ScheduleTemplate
    name: str
    cache_key: str
    status: str = "experimental"


@dataclass(frozen=True)
class BaselineComparison:
    optimizer: str
    dtype: str
    path_b_tokens_per_second: float
    path_c_tokens_per_second: float
    path_b_cache_delta_gib: float = 0.0
    path_b_first_step_ms: float = 0.0
    path_c_peak_delta_gib: float = 0.0


@dataclass(frozen=True)
class WarmCacheAudit:
    status: str
    reason: str
    cache_hit: bool
    cold_first_step_ms: float
    warm_first_step_ms: float
    steady_step_ms: float


@dataclass(frozen=True)
class FusionCacheKeyAudit:
    status: str
    reason: str
    expected_digest: str
    observed_digest: str
    cache_hit: bool


class FusionScheduleRegistry:
    """Registry mapping an op-chain signature to one fused schedule template."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, ...], FusionScheduleEntry] = {}

    def register(
        self,
        op_signature: Sequence[str],
        schedule_template: ScheduleTemplate | tir.PrimFunc | tvm.IRModule,
        *,
        name: str | None = None,
        status: str = "experimental",
    ) -> FusionScheduleRegistry:
        signature = tuple(str(op) for op in op_signature)
        if not signature:
            raise ValueError("fusion schedule op signature must not be empty")
        entry_name = name or ",".join(signature)
        if isinstance(schedule_template, tir.PrimFunc | tvm.IRModule):
            lowered = schedule_template
            schedule_cache_key = f"{entry_name}:{_lowered_schedule_digest(lowered)}"

            def constant_template(_region: FusionRegion) -> tir.PrimFunc | tvm.IRModule:
                return lowered

            resolved_schedule_template = constant_template
        elif callable(schedule_template):
            schedule_cache_key = f"{entry_name}:{_raw_callable_schedule_template_key(schedule_template)}"

            def registered_template(region: FusionRegion) -> tir.PrimFunc | tvm.IRModule:
                return schedule_template(region)

            resolved_schedule_template = registered_template
        else:
            raise TypeError("fusion schedule template must be callable, PrimFunc, or IRModule")
        with suppress(Exception):
            resolved_schedule_template._tl_fusion_schedule_key = schedule_cache_key
        self._entries[signature] = FusionScheduleEntry(
            op_signature=signature,
            schedule_template=resolved_schedule_template,
            name=entry_name,
            cache_key=schedule_cache_key,
            status=str(status),
        )
        return self

    def select(self, nodes: Sequence[FusionNode]) -> FusionScheduleEntry | None:
        return self._entries.get(tuple(node.op for node in nodes))


class FusionOptimizer:
    """Top-level FX-like optimizer that selects a fused TileLang schedule."""

    def __init__(
        self,
        name: str,
        *,
        schedule_registry: FusionScheduleRegistry | None = None,
        backend: str = DEFAULT_BACKEND,
        entry_symbol: str | None = None,
        require_single_kernel: bool = True,
        enable_z3_sync_async: bool = True,
    ) -> None:
        self._builder = FusionRegionBuilder(name, backend=backend, entry_symbol=entry_symbol)
        self._schedule_registry = schedule_registry or FusionScheduleRegistry()
        self._require_single_kernel = require_single_kernel
        self._enable_z3_sync_async = enable_z3_sync_async

    def add_node(
        self,
        name: str,
        *,
        op: str,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        attrs: Mapping[str, Any] | None = None,
    ) -> FusionOptimizer:
        self._builder.add_node(name, op=op, inputs=inputs, outputs=outputs, attrs=attrs)
        return self

    def add_nodes(self, nodes: Sequence[FusionNode]) -> FusionOptimizer:
        self._builder.add_nodes(nodes)
        return self

    def add_prim_func_node(
        self,
        name: str,
        prim_func: tir.PrimFunc,
        *,
        op: str | None = None,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        attrs: Mapping[str, Any] | None = None,
    ) -> FusionOptimizer:
        self._builder.add_prim_func_node(
            name,
            prim_func,
            op=op,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )
        return self

    def connect(
        self,
        producer: str,
        consumer: str,
        *,
        buffer: str,
        lifetime: str = "internal",
    ) -> FusionOptimizer:
        self._builder.connect(
            producer,
            consumer,
            buffer=buffer,
            lifetime=lifetime,
        )
        return self

    def set_schedule_template(self, schedule_template: ScheduleTemplate) -> FusionOptimizer:
        self._builder.set_schedule_template(schedule_template)
        return self

    def add_lowered_source_kernel(self, node_name: str, source: str) -> FusionOptimizer:
        self._builder.add_lowered_source_kernel(node_name, source)
        return self

    def _select_schedule_template(self) -> None:
        if self._builder._schedule_template is not None:
            return
        nodes, _ = self._builder._normalized_nodes_and_edges()
        schedule_entry = self._schedule_registry.select(nodes)
        if schedule_entry is not None:
            self._builder.set_schedule_template(
                schedule_entry.schedule_template,
                name=schedule_entry.name,
                status=schedule_entry.status,
            )
            return
        if nodes and all(node.prim_func is not None for node in nodes):
            return
        op_signature = ",".join(node.op for node in nodes)
        raise ValueError(
            f"no fused schedule registered for op signature {op_signature!r}; "
            "register one fused TileLang/TIR schedule template or provide raw PrimFunc nodes"
        )

    def build(self) -> FusionRegion:
        self._select_schedule_template()
        if self._enable_z3_sync_async:
            self._builder.enable_z3_sync_async_optimization()
        return self._builder.build()

    def plan(
        self,
        pass_configs: Mapping[str, Any] | None = None,
        *,
        require_single_kernel: bool | None = None,
    ) -> FusionCompilePlan:
        return plan_fusion_region(
            self.build(),
            pass_configs=pass_configs,
            require_single_kernel=self._require_single_kernel
            if require_single_kernel is None
            else require_single_kernel,
        )

    def compile(
        self,
        *,
        target: str = "auto",
        target_host: str | None = None,
        runtime_only: bool = False,
        enable_host_codegen: bool = False,
        enable_device_compile: bool = False,
        pass_configs: Mapping[str, Any] | None = None,
        lowerer: Callable[..., Any] | None = None,
        require_single_kernel: bool | None = None,
    ) -> FusionCompileResult:
        return compile_fusion_region(
            self.build(),
            target=target,
            target_host=target_host,
            runtime_only=runtime_only,
            enable_host_codegen=enable_host_codegen,
            enable_device_compile=enable_device_compile,
            pass_configs=pass_configs,
            lowerer=lowerer,
            require_single_kernel=self._require_single_kernel
            if require_single_kernel is None
            else require_single_kernel,
        )


class FusionRegionBuilder:
    def __init__(self, name: str, *, backend: str = DEFAULT_BACKEND, entry_symbol: str | None = None):
        self.name = name
        self.backend = backend
        self.entry_symbol = entry_symbol or name
        self._nodes: list[FusionNode] = []
        self._edges: list[FusionEdge] = []
        self._schedule_template: ScheduleTemplate | None = None
        self._schedule_name = "manual"
        self._schedule_status = "ready"
        self._pass_configs: dict[str, Any] = {}

    def _reject_duplicate_node_name(self, name: str) -> None:
        if any(node.name == name for node in self._nodes):
            raise ValueError(f"duplicate fusion node: {name!r}")

    def add_node(
        self,
        name: str,
        *,
        op: str,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        attrs: Mapping[str, Any] | None = None,
    ) -> FusionRegionBuilder:
        self._reject_duplicate_node_name(name)
        self._nodes.append(
            FusionNode(
                name=name,
                op=op,
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                attrs=dict(attrs or {}),
            )
        )
        return self

    def add_nodes(self, nodes: Sequence[FusionNode]) -> FusionRegionBuilder:
        for node in nodes:
            if node.prim_func is None:
                self.add_node(
                    node.name,
                    op=node.op,
                    inputs=node.inputs,
                    outputs=node.outputs,
                    attrs=node.attrs,
                )
            else:
                self.add_prim_func_node(
                    node.name,
                    node.prim_func,
                    op=node.op,
                    inputs=node.inputs,
                    outputs=node.outputs,
                    attrs=node.attrs,
                )
        return self

    def add_prim_func_node(
        self,
        name: str,
        prim_func: tir.PrimFunc,
        *,
        op: str | None = None,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        attrs: Mapping[str, Any] | None = None,
    ) -> FusionRegionBuilder:
        if not isinstance(prim_func, tir.PrimFunc):
            raise TypeError("Fusion PrimFunc nodes must be tvm.tir.PrimFunc objects")
        self._reject_duplicate_node_name(name)
        if prim_func.attrs and prim_func.attrs.get("code_block_source") is not None:
            raise ValueError(
                f"Fusion node {name!r} received a PrimFunc wrapping already-lowered source. "
                "Fusion regions must stay at the pre-source TileLang/TVM IR boundary."
            )
        inferred_op = op
        if inferred_op is None and prim_func.attrs:
            global_symbol = prim_func.attrs.get("global_symbol")
            inferred_op = str(global_symbol) if global_symbol is not None else None
        self._nodes.append(
            FusionNode(
                name=name,
                op=inferred_op or "prim_func",
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                attrs=dict(attrs or {}),
                prim_func=prim_func,
            )
        )
        return self

    def connect(
        self,
        producer: str,
        consumer: str,
        *,
        buffer: str,
        lifetime: str = "internal",
    ) -> FusionRegionBuilder:
        self._edges.append(FusionEdge(producer=producer, consumer=consumer, buffer=buffer, lifetime=lifetime))
        return self

    def _edges_with_inferred_chain(self) -> tuple[FusionEdge, ...]:
        edges = list(self._edges)
        explicit_edge_keys = {(edge.producer, edge.consumer, edge.buffer) for edge in edges}
        explicit_input_keys = {(edge.consumer, edge.buffer): edge.producer for edge in edges}
        node_names = {node.name for node in self._nodes}
        for edge in edges:
            if edge.producer not in node_names or edge.consumer not in node_names:
                raise ValueError(
                    f"Fusion edge {edge.producer!r}->{edge.consumer!r}:{edge.buffer!r} "
                    "references a missing node"
                )

        producer_by_buffer: dict[str, str] = {}
        ambiguous_producers: dict[str, tuple[str, str]] = {}
        for node in self._nodes:
            for output_name in node.outputs:
                existing = producer_by_buffer.get(output_name)
                if existing is not None and existing != node.name:
                    ambiguous_producers[output_name] = (existing, node.name)
                    continue
                producer_by_buffer[output_name] = node.name

        for node in self._nodes:
            for input_name in node.inputs:
                if (node.name, input_name) in explicit_input_keys:
                    continue
                ambiguous = ambiguous_producers.get(input_name)
                if ambiguous is not None:
                    raise ValueError(
                        f"ambiguous inferred fusion producer for buffer {input_name!r}: "
                        f"{ambiguous[0]!r} and {ambiguous[1]!r}"
                    )
                producer = producer_by_buffer.get(input_name)
                if producer is None or producer == node.name:
                    continue
                edge_key = (producer, node.name, input_name)
                if edge_key not in explicit_edge_keys:
                    edges.append(
                        FusionEdge(
                            producer=producer,
                            consumer=node.name,
                            buffer=input_name,
                        )
                    )
                    explicit_edge_keys.add(edge_key)
        return tuple(edges)

    def _nodes_in_dependency_order(self, edges: Sequence[FusionEdge]) -> tuple[FusionNode, ...]:
        if len(self._nodes) < 2:
            return tuple(self._nodes)

        nodes_by_name = {node.name: node for node in self._nodes}
        original_index = {node.name: index for index, node in enumerate(self._nodes)}
        indegree = {node.name: 0 for node in self._nodes}
        successors: dict[str, set[str]] = {node.name: set() for node in self._nodes}
        for edge in edges:
            if edge.producer not in nodes_by_name or edge.consumer not in nodes_by_name:
                raise ValueError(
                    f"Fusion edge {edge.producer!r}->{edge.consumer!r}:{edge.buffer!r} "
                    "references a missing node"
                )
            if edge.producer == edge.consumer:
                continue
            if edge.consumer not in successors[edge.producer]:
                successors[edge.producer].add(edge.consumer)
                indegree[edge.consumer] += 1

        ready = sorted(
            (name for name, count in indegree.items() if count == 0),
            key=original_index.__getitem__,
        )
        ordered_names: list[str] = []
        while ready:
            name = ready.pop(0)
            ordered_names.append(name)
            for successor in sorted(successors[name], key=original_index.__getitem__):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort(key=original_index.__getitem__)

        if len(ordered_names) != len(self._nodes):
            raise ValueError("Fusion region contains a cycle in producer/consumer dependencies")
        return tuple(nodes_by_name[name] for name in ordered_names)

    @staticmethod
    def _edges_in_dependency_order(
        edges: Sequence[FusionEdge],
        nodes: Sequence[FusionNode],
    ) -> tuple[FusionEdge, ...]:
        node_order = {node.name: index for index, node in enumerate(nodes)}
        input_order = {
            (node.name, input_name): index
            for node in nodes
            for index, input_name in enumerate(node.inputs)
        }
        return tuple(
            sorted(
                edges,
                key=lambda edge: (
                    node_order.get(edge.consumer, len(nodes)),
                    input_order.get((edge.consumer, edge.buffer), len(nodes)),
                    node_order.get(edge.producer, len(nodes)),
                    edge.buffer,
                ),
            )
        )

    def _normalized_nodes_and_edges(self) -> tuple[tuple[FusionNode, ...], tuple[FusionEdge, ...]]:
        edges = self._edges_with_inferred_chain()
        nodes = self._nodes_in_dependency_order(edges)
        return nodes, self._edges_in_dependency_order(edges, nodes)

    def add_lowered_source_kernel(self, node_name: str, source: str) -> FusionRegionBuilder:
        raise ValueError(
            f"Fusion node {node_name!r} received an already-lowered source kernel. "
            "Fusion regions must stay at the pre-source TileLang/TVM IR boundary."
        )

    def set_schedule_template(
        self,
        schedule_template: ScheduleTemplate,
        *,
        name: str = "manual",
        status: str = "ready",
    ) -> FusionRegionBuilder:
        self._schedule_template = schedule_template
        self._schedule_name = name
        self._schedule_status = status
        return self

    def enable_z3_sync_async_optimization(self) -> FusionRegionBuilder:
        self._pass_configs[PassConfigKey.TL_Z3_PROOF_BARRIER_MINIMIZATION.value] = True
        self._pass_configs[PassConfigKey.TL_Z3_PROOF_ASYNC_ELIGIBILITY.value] = True
        return self

    def build(self) -> FusionRegion:
        nodes, edges = self._normalized_nodes_and_edges()
        if self._schedule_template is None:
            if not nodes or any(node.prim_func is None for node in nodes):
                raise ValueError(
                    "FusionRegion without an explicit schedule template requires all nodes "
                    "to be raw PrimFunc nodes"
                )
            schedule_template = _auto_prim_func_region_template
            schedule_name = "auto_prim_func_region"
            schedule_status = "auto_prim_func_graph"
        else:
            schedule_template = self._schedule_template
            schedule_name = self._schedule_name
            schedule_status = self._schedule_status
        return FusionRegion(
            name=self.name,
            nodes=nodes,
            edges=edges,
            schedule_template=schedule_template,
            entry_symbol=self.entry_symbol,
            pass_configs=dict(self._pass_configs),
            backend=self.backend,
            schedule_name=schedule_name,
            schedule_status=schedule_status,
        )

    def plan(
        self,
        pass_configs: Mapping[str, Any] | None = None,
        *,
        require_single_kernel: bool = False,
    ) -> FusionCompilePlan:
        return plan_fusion_region(
            self.build(),
            pass_configs=pass_configs,
            require_single_kernel=require_single_kernel,
        )

    def compile(
        self,
        *,
        target: str = "auto",
        target_host: str | None = None,
        runtime_only: bool = False,
        enable_host_codegen: bool = False,
        enable_device_compile: bool = False,
        pass_configs: Mapping[str, Any] | None = None,
        lowerer: Callable[..., Any] | None = None,
        require_single_kernel: bool = False,
    ) -> FusionCompileResult:
        return compile_fusion_region(
            self.build(),
            target=target,
            target_host=target_host,
            runtime_only=runtime_only,
            enable_host_codegen=enable_host_codegen,
            enable_device_compile=enable_device_compile,
            pass_configs=pass_configs,
            lowerer=lowerer,
            require_single_kernel=require_single_kernel,
        )


_MAMBA3_FP8_TRAIN_BLOCK_CHAIN: tuple[FusionNode, ...] = (
    FusionNode(
        name="mamba3_scan",
        op="mamba3",
        inputs=("x", "state"),
        outputs=("mamba3_state",),
        attrs={"role": "producer", "backward": "aot_autograd"},
    ),
    FusionNode(
        name="m2rnn_packed_post",
        op="m2rnn",
        inputs=("mamba3_state",),
        outputs=("packed_post",),
        attrs={"role": "producer_consumer", "backward": "aot_autograd"},
    ),
    FusionNode(
        name="fp8_prepare",
        op="sparse_mla_fp8_prepare",
        inputs=("packed_post",),
        outputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale"),
        attrs={"role": "producer", "backward": "owner_output"},
    ),
    FusionNode(
        name="sparse_mla_fp8_apply",
        op="sparse_mla_fp8_apply",
        inputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale"),
        outputs=("train_block_out",),
        attrs={"role": "consumer", "backward": "owner_output"},
    ),
)


def build_fusion_region(
    *,
    region_name: str,
    nodes: Sequence[FusionNode],
    schedule_template: ScheduleTemplate,
    edges: Sequence[FusionEdge] = (),
    backend: str = DEFAULT_BACKEND,
    entry_symbol: str | None = None,
    enable_z3_sync_async_optimization: bool = False,
    schedule_name: str = "manual",
    schedule_status: str = "ready",
) -> FusionRegion:
    builder = (
        FusionRegionBuilder(region_name, backend=backend, entry_symbol=entry_symbol)
        .add_nodes(nodes)
        .set_schedule_template(schedule_template, name=schedule_name, status=schedule_status)
    )
    for edge in edges:
        builder.connect(edge.producer, edge.consumer, buffer=edge.buffer, lifetime=edge.lifetime)
    if enable_z3_sync_async_optimization:
        builder.enable_z3_sync_async_optimization()
    return builder.build()


def build_mamba3_fp8_train_block_region(
    *,
    schedule_template: ScheduleTemplate,
    region_name: str = "mamba3_fp8_train_block",
    nodes: Sequence[FusionNode] | None = None,
) -> FusionRegion:
    return build_fusion_region(
        region_name=region_name,
        nodes=_MAMBA3_FP8_TRAIN_BLOCK_CHAIN if nodes is None else nodes,
        schedule_template=schedule_template,
        enable_z3_sync_async_optimization=True,
    )


def _node_metadata(region: FusionRegion) -> str:
    return json.dumps(
        [
            {
                "name": node.name,
                "op": node.op,
                "inputs": list(node.inputs),
                "outputs": list(node.outputs),
                "attrs": dict(node.attrs),
                "prim_func_digest": _prim_func_digest(node.prim_func) if node.prim_func is not None else None,
            }
            for node in region.nodes
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def _prim_func_digest(prim_func: tir.PrimFunc) -> str:
    try:
        script = prim_func.script(show_meta=True)
    except TypeError:
        script = prim_func.script()
    return sha256(str(script).encode()).hexdigest()


def _ir_module_digest(ir_module: tvm.IRModule) -> str:
    try:
        script = ir_module.script(show_meta=True)
    except TypeError:
        script = ir_module.script()
    return sha256(str(script).encode()).hexdigest()


def _lowered_schedule_digest(lowered: tir.PrimFunc | tvm.IRModule) -> str:
    if isinstance(lowered, tir.PrimFunc):
        return f"prim_func:{_prim_func_digest(lowered)}"
    return f"ir_module:{_ir_module_digest(lowered)}"


def _schedule_template_key(schedule_template: ScheduleTemplate) -> str:
    custom_key = getattr(schedule_template, "_tl_fusion_schedule_key", None)
    if custom_key is not None:
        return str(custom_key)
    return _raw_callable_schedule_template_key(schedule_template)


def _raw_callable_schedule_template_key(schedule_template: ScheduleTemplate) -> str:
    module_name = getattr(schedule_template, "__module__", type(schedule_template).__module__)
    qualname = getattr(schedule_template, "__qualname__", type(schedule_template).__qualname__)
    try:
        source = inspect.getsource(schedule_template)
    except (OSError, TypeError):
        source = repr(schedule_template)
    return f"callable:{module_name}.{qualname}:{sha256(source.encode()).hexdigest()}"


def _edge_metadata(region: FusionRegion) -> str:
    return json.dumps(
        [
            {
                "producer": edge.producer,
                "consumer": edge.consumer,
                "buffer": edge.buffer,
                "lifetime": edge.lifetime,
            }
            for edge in region.edges
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def _with_region_attrs(func: tir.PrimFunc, region: FusionRegion, entry_symbol: str) -> tir.PrimFunc:
    return (
        func.with_attr("global_symbol", entry_symbol)
        .with_attr("tl.fusion.region", region.name)
        .with_attr("tl.fusion.boundary", LOWERING_BOUNDARY)
        .with_attr("tl.fusion.backend", region.backend)
        .with_attr("tl.fusion.nodes", _node_metadata(region))
        .with_attr("tl.fusion.edges", _edge_metadata(region))
    )


def _with_node_attrs(func: tir.PrimFunc, region: FusionRegion, node: FusionNode, symbol: str) -> tir.PrimFunc:
    return (
        func.with_attr("global_symbol", symbol)
        .with_attr("tl.fusion.region", region.name)
        .with_attr("tl.fusion.node", node.name)
        .with_attr("tl.fusion.op", node.op)
        .with_attr("tl.fusion.boundary", LOWERING_BOUNDARY)
        .with_attr("tl.fusion.backend", region.backend)
    )


def _prim_func_buffer_names(func: tir.PrimFunc) -> set[str]:
    return {str(buffer.name) for buffer in func.buffer_map.values()}


def _validate_auto_prim_func_edges_match_raw_abi(region: FusionRegion) -> None:
    nodes_by_name = {node.name: node for node in region.nodes}
    for edge in region.edges:
        producer = nodes_by_name.get(edge.producer)
        consumer = nodes_by_name.get(edge.consumer)
        if producer is None or consumer is None:
            raise ValueError(
                f"Fusion edge {edge.producer!r}->{edge.consumer!r}:{edge.buffer!r} "
                "references a missing node"
            )
        if producer.prim_func is None or consumer.prim_func is None:
            continue
        producer_buffers = _prim_func_buffer_names(producer.prim_func)
        consumer_buffers = _prim_func_buffer_names(consumer.prim_func)
        if edge.buffer not in producer_buffers or edge.buffer not in consumer_buffers:
            raise ValueError(
                f"Fusion edge buffer is not present in raw PrimFunc ABI for both endpoints: "
                f"{edge.producer!r}->{edge.consumer!r}:{edge.buffer!r}; "
                f"producer buffers={sorted(producer_buffers)!r}, "
                f"consumer buffers={sorted(consumer_buffers)!r}"
            )


def _auto_prim_func_region_template(region: FusionRegion) -> tvm.IRModule:
    _validate_auto_prim_func_edges_match_raw_abi(region)
    functions: dict[str, tir.PrimFunc] = {}
    for index, node in enumerate(region.nodes):
        if node.prim_func is None:
            raise ValueError(
                f"FusionRegion {region.name!r} contains non-PrimFunc node {node.name!r}; "
                "provide an explicit fused schedule template"
            )
        symbol = region.entry_symbol if index == 0 else node.name
        if symbol in functions:
            raise ValueError(f"duplicate fusion function symbol: {symbol!r}")
        functions[symbol] = _with_node_attrs(node.prim_func, region, node, symbol)
    return tvm.IRModule(functions)


def _entry_buffer_names(func: tir.PrimFunc) -> set[str]:
    names = {str(buffer.name) for buffer in func.buffer_map.values()}
    for param in func.params:
        name = getattr(param, "name", None) or getattr(param, "name_hint", None)
        if name is not None:
            names.add(str(name))
    return names


def _validate_internal_edges_do_not_escape_entry_abi(func: tir.PrimFunc, region: FusionRegion) -> None:
    if region.schedule_template is _auto_prim_func_region_template:
        return

    internal_buffers = {edge.buffer for edge in region.edges if edge.lifetime == "internal"}
    scratch_abi_buffers = _internal_scratch_abi_buffer_names(func)
    escaped = sorted(
        internal_buffers
        & (_entry_buffer_names(func) - scratch_abi_buffers)
    )
    if not escaped:
        return

    raise ValueError(
        f"Fusion region {region.name!r} internal fusion edge buffers escaped entry ABI: "
        f"{', '.join(escaped)}. A fused schedule must allocate producer/consumer edge buffers "
        "inside the entry PrimFunc instead of passing them as external buffers."
    )


def _prim_func_load_store_buffer_names(func: tir.PrimFunc) -> tuple[set[str], set[str]]:
    load_names: set[str] = set()
    store_names: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, tir.BufferLoad):
            load_names.add(str(node.buffer.name))
        elif isinstance(node, tir.BufferStore):
            store_names.add(str(node.buffer.name))

    tir.stmt_functor.post_order_visit(func.body, visit)
    return load_names, store_names


def _validate_internal_edges_materialized_in_entry_ir(func: tir.PrimFunc, region: FusionRegion) -> None:
    if region.schedule_template is _auto_prim_func_region_template:
        return

    internal_buffers = sorted({edge.buffer for edge in region.edges if edge.lifetime == "internal"})
    if not internal_buffers:
        return

    load_names, store_names = _prim_func_load_store_buffer_names(func)
    missing = [
        buffer
        for buffer in internal_buffers
        if buffer not in load_names or buffer not in store_names
    ]
    if not missing:
        return
    missing_details = [
        f"{buffer}:"
        f"{'missing_load' if buffer not in load_names else 'load_ok'},"
        f"{'missing_store' if buffer not in store_names else 'store_ok'}"
        for buffer in missing
    ]

    raise ValueError(
        f"Fusion region {region.name!r} explicit schedule did not materialize internal "
        f"fusion edge buffers: {', '.join(missing)} "
        f"({'; '.join(missing_details)}). A fullgraph fused schedule must "
        "keep every producer/consumer edge buffer inside the entry PrimFunc, not just "
        "hide it from the ABI."
    )


def _required_external_region_buffers(region: FusionRegion) -> set[str]:
    internal_buffers = {edge.buffer for edge in region.edges if edge.lifetime == "internal"}
    required: set[str] = set()
    for node in region.nodes:
        required.update(name for name in node.inputs if name not in internal_buffers)
        required.update(name for name in node.outputs if name not in internal_buffers)
    return required


def _validate_explicit_schedule_covers_region_abi(func: tir.PrimFunc, region: FusionRegion) -> None:
    if region.schedule_template is _auto_prim_func_region_template:
        return

    entry_names = _entry_buffer_names(func)
    missing = sorted(_required_external_region_buffers(region) - entry_names)
    missing = [
        buffer
        for buffer in missing
        if not _physical_abi_map_covers_buffer(func, buffer, entry_names)
    ]
    if not missing:
        return

    raise ValueError(
        f"Fusion region {region.name!r} explicit schedule is missing required region ABI buffers: "
        f"{', '.join(missing)}. A fullgraph fused schedule must expose every external "
        "region input/output while keeping internal producer/consumer edges inside the entry PrimFunc."
    )


def _physical_abi_map_covers_buffer(
    func: tir.PrimFunc,
    buffer_name: str,
    entry_names: set[str],
) -> bool:
    mapping = _physical_abi_logical_to_physical(func)
    mapped = mapping.get(buffer_name)
    if not isinstance(mapped, Mapping):
        return False
    bank = mapped.get("bank")
    return isinstance(bank, str) and bank in entry_names


def _physical_abi_logical_to_physical(func: tir.PrimFunc) -> Mapping[str, Any]:
    raw = _prim_func_string_attr(func, "tl.fusion.physical_abi.logical_to_physical")
    if raw is None:
        return {}
    with suppress(json.JSONDecodeError, TypeError):
        decoded = json.loads(raw)
        if isinstance(decoded, Mapping):
            return decoded
    return {}


def _internal_scratch_abi_buffer_names(func: tir.PrimFunc) -> set[str]:
    raw = _prim_func_string_attr(func, "tl.fusion.internal_scratch_abi_buffers")
    if raw is None:
        return set()
    with suppress(json.JSONDecodeError, TypeError):
        decoded = json.loads(raw)
        if isinstance(decoded, Sequence) and not isinstance(decoded, str | bytes):
            return {str(buffer) for buffer in decoded}
    return set()


def _prim_func_string_attr(func: tir.PrimFunc, key: str) -> str | None:
    attrs = getattr(func, "attrs", None)
    if attrs is None:
        return None
    try:
        value = attrs.get(key)
    except (AttributeError, TypeError):
        try:
            value = attrs[key]
        except (KeyError, TypeError):
            value = None
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _module_from_template(
    region: FusionRegion,
    *,
    require_internal_edge_materialization: bool = False,
) -> tvm.IRModule:
    lowered = region.schedule_template(region)
    if isinstance(lowered, str):
        raise TypeError("Fusion schedule templates must return a PrimFunc or IRModule, not lowered source text")
    if isinstance(lowered, tir.PrimFunc):
        if require_internal_edge_materialization:
            _validate_internal_edges_materialized_in_entry_ir(lowered, region)
        entry = _with_region_attrs(lowered, region, region.entry_symbol)
        _validate_internal_edges_do_not_escape_entry_abi(entry, region)
        return tvm.IRModule({region.entry_symbol: entry})
    if isinstance(lowered, tvm.IRModule):
        if region.entry_symbol not in {global_var.name_hint for global_var in lowered.functions}:
            if len(lowered.functions) != 1:
                raise ValueError(
                    f"Fusion IRModule for {region.name!r} must contain entry {region.entry_symbol!r} "
                    "or exactly one PrimFunc entry"
                )
            global_var, func = next(iter(lowered.functions.items()))
            if not isinstance(func, tir.PrimFunc):
                raise TypeError("Fusion IRModule entries must be PrimFunc objects")
            if require_internal_edge_materialization:
                _validate_internal_edges_materialized_in_entry_ir(func, region)
            entry = _with_region_attrs(func, region, region.entry_symbol)
            _validate_internal_edges_do_not_escape_entry_abi(entry, region)
            return tvm.IRModule({region.entry_symbol: entry})

        functions = {}
        for global_var, func in lowered.functions.items():
            if global_var.name_hint == region.entry_symbol:
                if not isinstance(func, tir.PrimFunc):
                    raise TypeError("Fusion IRModule entry must be a PrimFunc")
                if require_internal_edge_materialization:
                    _validate_internal_edges_materialized_in_entry_ir(func, region)
                entry = _with_region_attrs(func, region, region.entry_symbol)
                _validate_internal_edges_do_not_escape_entry_abi(entry, region)
                functions[global_var] = entry
            else:
                functions[global_var] = func
        return tvm.IRModule(functions)
    raise TypeError("Fusion schedule templates must return a PrimFunc or IRModule")


def _cache_key_material(region: FusionRegion, pass_configs: Mapping[str, Any]) -> tuple[str, ...]:
    material = [
        f"region:{region.name}",
        f"entry:{region.entry_symbol}",
        "nodes:" + ",".join(node.name for node in region.nodes),
        "edges:" + ",".join(f"{edge.producer}->{edge.consumer}:{edge.buffer}:{edge.lifetime}" for edge in region.edges),
        f"backend:{region.backend}",
        f"boundary:{LOWERING_BOUNDARY}",
        "z3:sync_async" if pass_configs.get(PassConfigKey.TL_Z3_PROOF_BARRIER_MINIMIZATION.value) else "z3:off",
        f"schedule:{_schedule_template_key(region.schedule_template)}",
    ]
    prim_func_digests = [
        f"{node.name}:{_prim_func_digest(node.prim_func)}"
        for node in region.nodes
        if node.prim_func is not None
    ]
    if prim_func_digests:
        material.append("prim_funcs:" + ",".join(prim_func_digests))
    return tuple(material)


def _autograd_plan_for(region: FusionRegion) -> FusionAutogradPlan:
    aot_forward_node_names = tuple(
        node.name for node in region.nodes if node.attrs.get("backward") == "aot_autograd"
    )
    provided_backward_nodes = tuple(
        node.name
        for node in region.nodes
        if node.attrs.get("autograd") == "aot_backward" or node.attrs.get("role") == "backward"
    )
    if not aot_forward_node_names and not provided_backward_nodes:
        return FusionAutogradPlan(mode="none", status="none")

    planned_backward_nodes = tuple(
        f"{node_name}_bwd" for node_name in reversed(aot_forward_node_names)
    )
    aot_forward_node_set = set(aot_forward_node_names)
    planned_backward_node_set = set(planned_backward_nodes)
    backward_edges = tuple(
        (
            f"{edge.consumer}_bwd",
            f"{edge.producer}_bwd",
            f"{edge.buffer}_grad",
        )
        for edge in reversed(region.edges)
        if edge.producer in aot_forward_node_set
        and edge.consumer in aot_forward_node_set
        and f"{edge.producer}_bwd" in planned_backward_node_set
        and f"{edge.consumer}_bwd" in planned_backward_node_set
    )
    missing_backward_nodes = tuple(
        f"{node_name}_bwd"
        for node_name in aot_forward_node_names
        if f"{node_name}_bwd" not in provided_backward_nodes
    )
    return FusionAutogradPlan(
        mode="aot_autograd",
        status="requires_aot_autograd_codegen" if missing_backward_nodes else "ready",
        forward_node_names=aot_forward_node_names,
        backward_node_names=planned_backward_nodes,
        backward_edges=backward_edges,
        missing_backward_node_names=missing_backward_nodes,
    )


def plan_fusion_region(
    region: FusionRegion,
    pass_configs: Mapping[str, Any] | None = None,
    *,
    require_single_kernel: bool = False,
) -> FusionCompilePlan:
    merged_pass_configs = dict(region.pass_configs)
    merged_pass_configs.update(pass_configs or {})
    return FusionCompilePlan(
        region_name=region.name,
        entry_symbol=region.entry_symbol,
        lowering_boundary=LOWERING_BOUNDARY,
        backend=region.backend,
        node_names=tuple(node.name for node in region.nodes),
        edges=region.edges,
        pass_configs=merged_pass_configs,
        cache_key_material=_cache_key_material(region, merged_pass_configs),
        require_single_kernel=require_single_kernel,
        schedule_name=region.schedule_name,
        schedule_status=region.schedule_status,
        autograd_plan=_autograd_plan_for(region),
    )


def _assert_single_kernel_region(lowered_module: tvm.IRModule, region: FusionRegion) -> None:
    symbols = [global_var.name_hint for global_var in lowered_module.functions]
    if symbols == [region.entry_symbol]:
        return
    graph_breaks = [symbol for symbol in symbols if symbol != region.entry_symbol]
    raise ValueError(
        f"fullgraph fusion for region {region.name!r} produced graph break functions: "
        f"{graph_breaks!r}. Provide one explicit fused schedule template whose entry "
        f"is {region.entry_symbol!r}."
    )


def _assert_single_kernel_region_abi(lowered_module: tvm.IRModule, region: FusionRegion) -> None:
    entry = lowered_module[region.entry_symbol]
    if not isinstance(entry, tir.PrimFunc):
        raise TypeError("Fusion IRModule entry must be a PrimFunc")
    _validate_explicit_schedule_covers_region_abi(entry, region)


def compile_fusion_region(
    region: FusionRegion,
    *,
    target: str = "auto",
    target_host: str | None = None,
    runtime_only: bool = False,
    enable_host_codegen: bool = False,
    enable_device_compile: bool = False,
    pass_configs: Mapping[str, Any] | None = None,
    lowerer: Callable[..., Any] | None = None,
    require_single_kernel: bool = False,
) -> FusionCompileResult:
    plan = plan_fusion_region(
        region,
        pass_configs=pass_configs,
        require_single_kernel=require_single_kernel,
    )
    lowered_module = _module_from_template(
        region,
        require_internal_edge_materialization=require_single_kernel,
    )
    if require_single_kernel:
        _assert_single_kernel_region(lowered_module, region)
        _assert_single_kernel_region_abi(lowered_module, region)
    if lowerer is None:
        from tilelang.engine.lower import lower as lowerer

    with tvm.transform.PassContext(opt_level=3, config=dict(plan.pass_configs)):
        artifact = lowerer(
            lowered_module,
            target=target,
            target_host=target_host,
            runtime_only=runtime_only,
            enable_host_codegen=enable_host_codegen,
            enable_device_compile=enable_device_compile,
        )
    return FusionCompileResult(plan=plan, lowered_module=lowered_module, artifact=artifact)


def fusion_cache_key_digest(plan: FusionCompilePlan, lowered_module: tvm.IRModule) -> str:
    try:
        module_script = lowered_module.script(show_meta=True)
    except TypeError:
        module_script = lowered_module.script()
    payload = {
        "cache_key_material": plan.cache_key_material,
        "pass_configs": dict(plan.pass_configs),
        "lowered_module": str(module_script),
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def audit_fusion_cache_key(
    *,
    expected_digest: str,
    observed_digest: str,
    cache_hit: bool,
) -> FusionCacheKeyAudit:
    if expected_digest != observed_digest:
        return FusionCacheKeyAudit(
            "key_changed",
            "fusion cache key digest changed between compile attempts",
            expected_digest,
            observed_digest,
            cache_hit,
        )
    if not cache_hit:
        return FusionCacheKeyAudit(
            "recompiled_same_key",
            "fusion cache key was stable but cache did not hit",
            expected_digest,
            observed_digest,
            cache_hit,
        )
    return FusionCacheKeyAudit(
        "ok",
        "fusion cache key was stable and cache hit",
        expected_digest,
        observed_digest,
        cache_hit,
    )


def path_b_baseline_clean(
    row: BaselineComparison,
    *,
    max_path_b_cache_delta_gib: float = DEFAULT_PATH_B_CACHE_LIMIT_GIB,
    max_path_b_first_step_ms: float = DEFAULT_WARM_FIRST_STEP_LIMIT_MS,
) -> bool:
    if row.path_b_cache_delta_gib > max_path_b_cache_delta_gib:
        return False
    return row.path_b_first_step_ms <= max_path_b_first_step_ms


def fusion_default_allowed(
    row: BaselineComparison,
    *,
    min_speedup: float = 1.0,
    max_path_c_peak_delta_gib: float = 0.0,
) -> bool:
    if not path_b_baseline_clean(row):
        return False
    if row.path_b_tokens_per_second <= 0:
        return False
    speedup = row.path_c_tokens_per_second / row.path_b_tokens_per_second
    if speedup <= min_speedup:
        return False
    return row.path_c_peak_delta_gib <= max_path_c_peak_delta_gib


def audit_warm_cache_reuse(
    *,
    cache_hit: bool,
    cold_first_step_ms: float,
    warm_first_step_ms: float,
    steady_step_ms: float,
    max_warm_first_step_ms: float = DEFAULT_WARM_FIRST_STEP_LIMIT_MS,
    max_warm_to_steady_ratio: float = DEFAULT_WARM_TO_STEADY_RATIO_LIMIT,
) -> WarmCacheAudit:
    if not cache_hit:
        return WarmCacheAudit("miss", "no cache hit was reported", cache_hit, cold_first_step_ms, warm_first_step_ms, steady_step_ms)
    if warm_first_step_ms > max_warm_first_step_ms:
        return WarmCacheAudit(
            "incomplete",
            "warm first-step is still above the warm-cache threshold",
            cache_hit,
            cold_first_step_ms,
            warm_first_step_ms,
            steady_step_ms,
        )
    if steady_step_ms > 0 and warm_first_step_ms / steady_step_ms > max_warm_to_steady_ratio:
        return WarmCacheAudit(
            "incomplete",
            "warm first-step is still too far above steady-state",
            cache_hit,
            cold_first_step_ms,
            warm_first_step_ms,
            steady_step_ms,
        )
    if cold_first_step_ms > 0 and warm_first_step_ms >= cold_first_step_ms:
        return WarmCacheAudit(
            "incomplete",
            "warm first-step is not faster than cold first-step",
            cache_hit,
            cold_first_step_ms,
            warm_first_step_ms,
            steady_step_ms,
        )
    return WarmCacheAudit("ok", "warm cache reuse looks effective", cache_hit, cold_first_step_ms, warm_first_step_ms, steady_step_ms)
