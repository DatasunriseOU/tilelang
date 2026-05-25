import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm import tir
from tilelang.engine import compile_fusion_region as exported_compile_fusion_region
from tilelang.engine.fusion import (
    BaselineComparison,
    FusionAutogradPlan,
    FusionBlockDescriptor,
    FusionBlockRegistry,
    FusionNode,
    FusionOptimizer,
    FusionRegionBuilder,
    FusionScheduleRegistry,
    audit_fusion_cache_key,
    audit_warm_cache_reuse,
    build_fusion_region,
    build_fusion_region_from_blocks,
    build_fusion_regions_from_blocks,
    build_mamba3_fp8_train_block_region,
    compile_fusion_region,
    fusion_cache_key_digest,
    fusion_default_allowed,
)
from tilelang.transform import PassConfigKey


@T.prim_func
def _fused_train_block(A: T.Tensor((4,), "float32"), C: T.Tensor((4,), "float32")):
    with T.Kernel(1, threads=1):
        C[0] = A[0]


@T.prim_func
def _fusion_consumer(C: T.Tensor((4,), "float32"), D: T.Tensor((4,), "float32")):
    with T.Kernel(1, threads=1):
        D[0] = C[0]


@T.prim_func
def _fused_train_block_with_internal_edge(A: T.Tensor((4,), "float32"), D: T.Tensor((4,), "float32")):
    with T.Kernel(1, threads=1):
        packed_post = T.alloc_local((4,), "float32")
        packed_post[0] = A[0]
        D[0] = packed_post[0]


@T.prim_func
def _logical_train_block_with_internal_edges(
    x: T.Tensor((4,), "float32"),
    state: T.Tensor((4,), "float32"),
    train_block_out: T.Tensor((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        mamba3_state = T.alloc_local((4,), "float32")
        packed_post = T.alloc_local((4,), "float32")
        q_fp8 = T.alloc_local((4,), "float32")
        q_scale = T.alloc_local((4,), "float32")
        kv_fp8 = T.alloc_local((4,), "float32")
        kv_scale = T.alloc_local((4,), "float32")
        mamba3_state[0] = x[0] + state[0]
        packed_post[0] = mamba3_state[0]
        q_fp8[0] = packed_post[0]
        q_scale[0] = 1.0
        kv_fp8[0] = packed_post[0]
        kv_scale[0] = 1.0
        train_block_out[0] = q_fp8[0] + kv_fp8[0] + q_scale[0] + kv_scale[0]


@T.prim_func
def _logical_train_block_missing_internal_edges(
    x: T.Tensor((4,), "float32"),
    state: T.Tensor((4,), "float32"),
    train_block_out: T.Tensor((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        mamba3_state = T.alloc_local((4,), "float32")
        packed_post = T.alloc_local((4,), "float32")
        mamba3_state[0] = x[0] + state[0]
        packed_post[0] = mamba3_state[0]
        train_block_out[0] = packed_post[0]


@T.prim_func
def _logical_train_block_with_misleading_internal_buffer_names(
    x: T.Tensor((4,), "float32"),
    state: T.Tensor((4,), "float32"),
    train_block_out: T.Tensor((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        mamba3_state = T.alloc_local((4,), "float32")
        packed_post = T.alloc_local((4,), "float32")
        not_q_fp8 = T.alloc_local((4,), "float32")
        not_q_scale = T.alloc_local((4,), "float32")
        not_kv_fp8 = T.alloc_local((4,), "float32")
        not_kv_scale = T.alloc_local((4,), "float32")
        mamba3_state[0] = x[0] + state[0]
        packed_post[0] = mamba3_state[0]
        not_q_fp8[0] = packed_post[0]
        not_q_scale[0] = 1.0
        not_kv_fp8[0] = packed_post[0]
        not_kv_scale[0] = 1.0
        train_block_out[0] = (
            not_q_fp8[0]
            + not_q_scale[0]
            + not_kv_fp8[0]
            + not_kv_scale[0]
        )


@T.prim_func
def _logical_train_block_missing_output_abi(
    x: T.Tensor((4,), "float32"),
    state: T.Tensor((4,), "float32"),
    D: T.Tensor((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        mamba3_state = T.alloc_local((4,), "float32")
        packed_post = T.alloc_local((4,), "float32")
        q_fp8 = T.alloc_local((4,), "float32")
        q_scale = T.alloc_local((4,), "float32")
        kv_fp8 = T.alloc_local((4,), "float32")
        kv_scale = T.alloc_local((4,), "float32")
        mamba3_state[0] = x[0] + state[0]
        packed_post[0] = mamba3_state[0]
        q_fp8[0] = packed_post[0]
        q_scale[0] = 1.0
        kv_fp8[0] = packed_post[0]
        kv_scale[0] = 1.0
        D[0] = q_fp8[0] + kv_fp8[0] + q_scale[0] + kv_scale[0]


@T.prim_func
def _leaky_fused_train_block(
    A: T.Tensor((4,), "float32"),
    packed_post: T.Tensor((4,), "float32"),
    D: T.Tensor((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        D[0] = A[0] + packed_post[0]


@T.prim_func
def _internal_scratch_abi_fused_train_block(
    A: T.Tensor((4,), "float32"),
    packed_post: T.Tensor((4,), "float32"),
    D: T.Tensor((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        packed_post[0] = A[0]
        D[0] = packed_post[0]


def _make_thread_allreduce_func(extent: int) -> tir.PrimFunc:
    src = tir.decl_buffer((extent,), "float32", name="src")
    dst = tir.decl_buffer((1,), "float32", name="dst")
    lane = tir.Var("lane", "int32")
    reduce_index = lane % tir.IntImm("int32", extent)
    reducer = tir.comm_reducer(
        lambda x, y: x + y,
        lambda dtype: tir.const(0, dtype=dtype),
        name="sum",
    )
    call = tir.call_intrin(
        "handle",
        "tir.tvm_thread_allreduce",
        tir.const(1, "uint32"),
        tir.BufferLoad(src, [lane]),
        tir.const(True, "bool"),
        tir.BufferLoad(dst, [0]),
        reduce_index,
    )
    body = tir.AttrStmt(
        reducer,
        "reduce_scope",
        tir.reinterpret("handle", tir.const(0, "uint64")),
        tir.Evaluate(call),
    )
    return tir.PrimFunc([], body)


def _make_shfl_sync_func() -> tir.PrimFunc:
    src = tir.decl_buffer((32,), "int32", name="src")
    lane = tir.Var("lane", "int32")
    call = tir.call_intrin(
        "int32",
        tir.op.Op.get("tl.shfl_sync"),
        tir.const(0xFFFFFFFF, "uint32"),
        tir.BufferLoad(src, [lane]),
        tir.const(31, "int32"),
        tir.const(32, "int32"),
    )
    return tir.PrimFunc([], tir.Evaluate(call))


def test_region_compile_uses_one_pre_source_ir_module_with_z3_config():
    assert exported_compile_fusion_region is compile_fusion_region
    assert tilelang.compile_fusion_region is compile_fusion_region
    assert tilelang.FusionRegionBuilder is FusionRegionBuilder
    assert tilelang.FusionAutogradPlan is FusionAutogradPlan
    assert tilelang.build_fusion_region is build_fusion_region

    def schedule_template(region):
        assert region.name == "mamba3_fp8_train_block"
        return _fused_train_block

    region = build_mamba3_fp8_train_block_region(schedule_template=schedule_template)
    called = {}

    def fake_lowerer(func_or_mod, **kwargs):
        called["func_or_mod"] = func_or_mod
        called["kwargs"] = kwargs
        called["pass_config"] = dict(tvm.transform.PassContext.current().config)
        return "compiled-artifact"

    result = compile_fusion_region(region, target="metal", lowerer=fake_lowerer)

    assert result.artifact == "compiled-artifact"
    assert result.plan.lowering_boundary == "tilelang_tvm_ir"
    assert result.plan.backend == "tilelang_tvm_ffi"
    assert result.plan.requires_source_post_fusion is False
    assert result.plan.node_names == (
        "mamba3_scan",
        "m2rnn_packed_post",
        "fp8_prepare",
        "sparse_mla_fp8_apply",
    )
    assert all(edge.lifetime == "internal" for edge in result.plan.edges)
    assert result.plan.pass_configs[PassConfigKey.TL_Z3_PROOF_BARRIER_MINIMIZATION.value] is True
    assert result.plan.pass_configs[PassConfigKey.TL_Z3_PROOF_ASYNC_ELIGIBILITY.value] is True
    assert result.plan.cache_key_material[:2] == (
        "region:mamba3_fp8_train_block",
        "entry:mamba3_fp8_train_block",
    )
    assert result.plan.autograd_plan.mode == "aot_autograd"
    assert result.plan.autograd_plan.status == "requires_aot_autograd_codegen"
    assert result.plan.autograd_plan.forward_node_names == (
        "mamba3_scan",
        "m2rnn_packed_post",
    )
    assert result.plan.autograd_plan.backward_node_names == (
        "m2rnn_packed_post_bwd",
        "mamba3_scan_bwd",
    )
    assert result.plan.autograd_plan.backward_edges == (
        ("m2rnn_packed_post_bwd", "mamba3_scan_bwd", "mamba3_state_grad"),
    )
    assert result.plan.autograd_plan.missing_backward_node_names == (
        "mamba3_scan_bwd",
        "m2rnn_packed_post_bwd",
    )

    assert called["kwargs"]["target"] == "metal"
    assert bool(called["pass_config"][PassConfigKey.TL_Z3_PROOF_BARRIER_MINIMIZATION.value]) is True
    assert bool(called["pass_config"][PassConfigKey.TL_Z3_PROOF_ASYNC_ELIGIBILITY.value]) is True
    assert isinstance(called["func_or_mod"], tvm.IRModule)
    assert [global_var.name_hint for global_var in called["func_or_mod"].functions] == [
        "mamba3_fp8_train_block",
    ]
    assert "tl.fusion.region" in called["func_or_mod"].script()
    assert "mamba3_scan" in called["func_or_mod"].script()


def test_metal_compile_rejects_cross_cta_reduction_plans():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _make_thread_allreduce_func(512),
    )

    with pytest.raises(
        ValueError,
        match="Metal target does not support cross-CTA reductions.*two-pass-global",
    ):
        compile_fusion_region(region, target="metal", lowerer=lambda *args, **kwargs: "compiled")


def test_metal_compile_rejects_cuda_warp_shuffle_semantics():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _make_shfl_sync_func(),
    )

    with pytest.raises(ValueError, match="Metal SIMDgroup guard.*tl.shfl_sync"):
        compile_fusion_region(region, target="metal", lowerer=lambda *args, **kwargs: "compiled")


def test_cuda_compile_allows_cross_cta_reduction_plans():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _make_thread_allreduce_func(512),
    )

    result = compile_fusion_region(region, target="cuda", lowerer=lambda *args, **kwargs: "compiled")

    assert result.artifact == "compiled"


def test_metal_compile_allows_threadgroup_reduction_plans():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _make_thread_allreduce_func(128),
    )

    result = compile_fusion_region(region, target="metal", lowerer=lambda *args, **kwargs: "compiled")

    assert result.artifact == "compiled"


def test_explicit_fused_schedule_keeps_internal_edges_out_of_entry_abi():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _fused_train_block_with_internal_edge,
    )

    result = compile_fusion_region(region, target="metal", lowerer=lambda *args, **kwargs: "compiled")
    entry = result.lowered_module[region.entry_symbol]

    assert [buffer.name for buffer in entry.buffer_map.values()] == ["A", "D"]
    assert all(edge.lifetime == "internal" for edge in result.plan.edges)


def test_explicit_fused_schedule_rejects_internal_edges_in_entry_abi():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _leaky_fused_train_block,
    )

    with pytest.raises(ValueError, match="internal fusion edge buffers escaped.*packed_post"):
        compile_fusion_region(region, target="metal", lowerer=lambda *args, **kwargs: "compiled")


def test_explicit_fused_schedule_accepts_marked_internal_scratch_abi():
    def schedule_template(_region):
        return _internal_scratch_abi_fused_train_block.with_attr(
            "tl.fusion.internal_scratch_abi_buffers",
            '["packed_post"]',
        )

    region = build_mamba3_fp8_train_block_region(schedule_template=schedule_template)

    result = compile_fusion_region(
        region,
        target="metal",
        lowerer=lambda *args, **kwargs: "compiled",
    )

    entry = result.lowered_module[region.entry_symbol]
    assert result.artifact == "compiled"
    assert [buffer.name for buffer in entry.buffer_map.values()] == [
        "A",
        "packed_post",
        "D",
    ]


def test_prim_func_graph_can_compile_without_manual_schedule_template():
    region = (
        FusionRegionBuilder("auto_region")
        .add_prim_func_node(
            "producer",
            _fused_train_block,
            op="toy_producer",
            inputs=("A",),
            outputs=("C",),
        )
        .add_prim_func_node(
            "consumer",
            _fusion_consumer,
            op="toy_consumer",
            inputs=("C",),
            outputs=("D",),
        )
        .enable_z3_sync_async_optimization()
        .build()
    )
    called = {}

    def fake_lowerer(func_or_mod, **kwargs):
        called["func_or_mod"] = func_or_mod
        called["pass_config"] = dict(tvm.transform.PassContext.current().config)
        return "compiled-auto-graph"

    result = compile_fusion_region(region, target="metal", lowerer=fake_lowerer)

    assert result.artifact == "compiled-auto-graph"
    assert result.plan.entry_symbol == "auto_region"
    assert result.plan.node_names == ("producer", "consumer")
    assert result.plan.autograd_plan.status == "none"
    assert result.plan.cache_key_material[:3] == (
        "region:auto_region",
        "entry:auto_region",
        "nodes:producer,consumer",
    )
    assert bool(called["pass_config"][PassConfigKey.TL_Z3_PROOF_ASYNC_ELIGIBILITY.value]) is True
    assert isinstance(called["func_or_mod"], tvm.IRModule)
    assert [global_var.name_hint for global_var in called["func_or_mod"].functions] == [
        "auto_region",
        "consumer",
    ]
    script = called["func_or_mod"].script()
    assert "tl.fusion.region" in script
    assert "tl.fusion.node" in script
    assert "toy_producer" in script
    assert "toy_consumer" in script


def test_fullgraph_compile_rejects_auto_prim_func_graph_breaks():
    region = (
        FusionRegionBuilder("auto_region")
        .add_prim_func_node(
            "producer",
            _fused_train_block,
            op="toy_producer",
            inputs=("A",),
            outputs=("C",),
        )
        .add_prim_func_node(
            "consumer",
            _fusion_consumer,
            op="toy_consumer",
            inputs=("C",),
            outputs=("D",),
        )
        .build()
    )

    with pytest.raises(ValueError, match="fullgraph fusion.*graph break.*consumer"):
        compile_fusion_region(
            region,
            target="metal",
            lowerer=lambda *args, **kwargs: None,
            require_single_kernel=True,
        )


def test_fullgraph_compile_accepts_explicit_single_entry_schedule():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _logical_train_block_with_internal_edges,
    )

    result = compile_fusion_region(
        region,
        target="metal",
        lowerer=lambda *args, **kwargs: "compiled",
        require_single_kernel=True,
    )

    assert result.artifact == "compiled"
    assert [global_var.name_hint for global_var in result.lowered_module.functions] == [
        "mamba3_fp8_train_block",
    ]


def test_fullgraph_compile_rejects_schedule_that_does_not_materialize_internal_edges():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _logical_train_block_missing_internal_edges,
    )

    with pytest.raises(ValueError, match="did not materialize internal fusion edge buffers.*kv_fp8"):
        compile_fusion_region(
            region,
            target="metal",
            lowerer=lambda *args, **kwargs: "compiled",
            require_single_kernel=True,
        )


def test_fullgraph_compile_rejects_schedule_with_only_substring_edge_names():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _logical_train_block_with_misleading_internal_buffer_names,
    )

    with pytest.raises(ValueError, match="did not materialize internal fusion edge buffers.*q_fp8"):
        compile_fusion_region(
            region,
            target="metal",
            lowerer=lambda *args, **kwargs: "compiled",
            require_single_kernel=True,
        )


def test_fullgraph_compile_rejects_explicit_schedule_missing_region_abi():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _logical_train_block_missing_output_abi,
    )

    with pytest.raises(ValueError, match="missing required region ABI buffers.*train_block_out"):
        compile_fusion_region(
            region,
            target="metal",
            lowerer=lambda *args, **kwargs: "compiled",
            require_single_kernel=True,
        )


def test_builder_compile_calls_common_compile_with_fullgraph_z3():
    result = (
        FusionRegionBuilder("builder_compile_region")
        .add_prim_func_node("producer", _fused_train_block, op="toy")
        .set_schedule_template(lambda _: _fused_train_block)
        .enable_z3_sync_async_optimization()
        .compile(
            target="metal",
            lowerer=lambda *args, **kwargs: "compiled-from-builder",
            require_single_kernel=True,
        )
    )

    assert result.artifact == "compiled-from-builder"
    assert result.plan.require_single_kernel is True
    assert result.plan.pass_configs[PassConfigKey.TL_Z3_PROOF_ASYNC_ELIGIBILITY.value] is True


def test_auto_prim_func_graph_rejects_edges_missing_from_raw_tir_abi():
    region = (
        FusionRegionBuilder("bad_auto_edges")
        .add_prim_func_node(
            "producer",
            _fused_train_block,
            op="toy_producer",
            inputs=("A",),
            outputs=("logical_only",),
        )
        .add_prim_func_node(
            "consumer",
            _fusion_consumer,
            op="toy_consumer",
            inputs=("logical_only",),
            outputs=("D",),
        )
        .connect("producer", "consumer", buffer="logical_only")
        .build()
    )

    with pytest.raises(ValueError, match="not present in raw PrimFunc ABI.*logical_only"):
        compile_fusion_region(region, target="metal", lowerer=lambda *args, **kwargs: None)


def test_builder_infers_chain_edges_from_buffer_names():
    region = (
        FusionRegionBuilder("inferred_chain")
        .add_node("producer", op="toy", inputs=("A",), outputs=("mid",))
        .add_node("consumer", op="toy", inputs=("mid",), outputs=("B",))
        .set_schedule_template(lambda _: _fused_train_block)
        .build()
    )

    assert [(edge.producer, edge.consumer, edge.buffer) for edge in region.edges] == [
        ("producer", "consumer", "mid")
    ]


def test_builder_adds_data_driven_nodes_and_infers_chain_edges():
    region = (
        FusionRegionBuilder("data_driven_chain")
        .add_nodes(
            (
                FusionNode("producer", op="toy", inputs=("A",), outputs=("mid",)),
                FusionNode("consumer", op="toy", inputs=("mid",), outputs=("B",)),
            )
        )
        .set_schedule_template(lambda _: _fused_train_block)
        .build()
    )

    assert [node.name for node in region.nodes] == ["producer", "consumer"]
    assert [(edge.producer, edge.consumer, edge.buffer) for edge in region.edges] == [
        ("producer", "consumer", "mid")
    ]


def test_builder_orders_data_driven_nodes_by_dependencies():
    region = (
        FusionRegionBuilder("out_of_order_chain")
        .add_nodes(
            (
                FusionNode("consumer", op="toy_consumer", inputs=("mid",), outputs=("B",)),
                FusionNode("producer", op="toy_producer", inputs=("A",), outputs=("mid",)),
            )
        )
        .set_schedule_template(lambda _: _fused_train_block)
        .build()
    )

    assert [node.name for node in region.nodes] == ["producer", "consumer"]
    assert [(edge.producer, edge.consumer, edge.buffer) for edge in region.edges] == [
        ("producer", "consumer", "mid")
    ]


def test_build_fusion_region_adds_dynamic_nodes_and_infers_chain():
    region = build_fusion_region(
        region_name="dynamic_train_block",
        nodes=(
            FusionNode("apply", op="sparse_mla", inputs=("post_y",), outputs=("out",)),
            FusionNode("scan", op="mamba3", inputs=("x",), outputs=("scan_y",)),
            FusionNode("post", op="m2rnn", inputs=("scan_y",), outputs=("post_y",)),
        ),
        schedule_template=lambda _: _fused_train_block,
        enable_z3_sync_async_optimization=True,
    )

    assert [node.name for node in region.nodes] == ["scan", "post", "apply"]
    assert [(edge.producer, edge.consumer, edge.buffer) for edge in region.edges] == [
        ("scan", "post", "scan_y"),
        ("post", "apply", "post_y"),
    ]
    assert region.pass_configs[PassConfigKey.TL_Z3_PROOF_BARRIER_MINIMIZATION.value] is True
    assert region.pass_configs[PassConfigKey.TL_Z3_PROOF_ASYNC_ELIGIBILITY.value] is True


def test_build_fusion_region_from_blocks_resolves_registry_contracts():
    block_registry = (
        FusionBlockRegistry()
        .register(
            FusionBlockDescriptor(
                op="producer",
                inputs=("A",),
                outputs=("{name}_mid",),
                aliases=("M",),
                attrs={"role": "producer"},
            )
        )
        .register(
            FusionBlockDescriptor(
                op="consumer",
                inputs=("producer_mid",),
                outputs=("D",),
                aliases=("R",),
                attrs={"role": "consumer"},
            )
        )
    )

    region = build_fusion_region_from_blocks(
        region_name="block_registry_region",
        blocks=(
            {"name": "consumer", "route_symbol": "R"},
            {"name": "producer", "route_symbol": "M"},
        ),
        block_registry=block_registry,
        schedule_template=lambda _: _fused_train_block,
        enable_z3_sync_async_optimization=True,
    )

    assert [node.name for node in region.nodes] == ["producer", "consumer"]
    assert [node.op for node in region.nodes] == ["producer", "consumer"]
    assert region.nodes[0].outputs == ("producer_mid",)
    assert region.nodes[0].attrs["role"] == "producer"
    assert [(edge.producer, edge.consumer, edge.buffer) for edge in region.edges] == [
        ("producer", "consumer", "producer_mid")
    ]
    assert region.pass_configs[PassConfigKey.TL_Z3_PROOF_ASYNC_ELIGIBILITY.value] is True


def test_block_registry_allows_same_op_blocks_with_distinct_contracts():
    block_registry = FusionBlockRegistry(
        (
            FusionBlockDescriptor(
                op="repeat_op",
                inputs=("A",),
                outputs=("first_mid",),
                aliases=("first_contract",),
            ),
            FusionBlockDescriptor(
                op="repeat_op",
                inputs=("first_mid",),
                outputs=("D",),
                aliases=("second_contract",),
            ),
        )
    )

    region = build_fusion_region_from_blocks(
        region_name="same_op_distinct_contracts",
        blocks=(
            {"name": "second", "kind": "second_contract"},
            {"name": "first", "kind": "first_contract"},
        ),
        block_registry=block_registry,
        schedule_template=lambda _: _fused_train_block,
    )

    assert [node.name for node in region.nodes] == ["first", "second"]
    assert [node.op for node in region.nodes] == ["repeat_op", "repeat_op"]
    assert [node.outputs for node in region.nodes] == [("first_mid",), ("D",)]
    assert [(edge.producer, edge.consumer, edge.buffer) for edge in region.edges] == [
        ("first", "second", "first_mid")
    ]


def test_optimizer_add_blocks_selects_registered_fused_schedule():
    block_registry = FusionBlockRegistry(
        (
            FusionBlockDescriptor(
                op="producer",
                inputs=("A",),
                outputs=("{name}_mid",),
                aliases=("M",),
            ),
            FusionBlockDescriptor(
                op="consumer",
                inputs=("producer_mid",),
                outputs=("D",),
                aliases=("R",),
            ),
        )
    )
    schedule_registry = FusionScheduleRegistry().register(
        ("producer", "consumer"),
        _fused_train_block,
        name="producer_consumer_fused",
        status="ready",
    )

    plan = (
        FusionOptimizer("block_optimizer", schedule_registry=schedule_registry)
        .add_blocks(
            (
                {"name": "producer", "kind": "M"},
                {"name": "consumer", "kind": "R"},
            ),
            block_registry=block_registry,
        )
        .plan()
    )

    assert plan.schedule_name == "producer_consumer_fused"
    assert plan.schedule_status == "ready"
    assert plan.node_names == ("producer", "consumer")
    assert [(edge.producer, edge.consumer, edge.buffer) for edge in plan.edges] == [
        ("producer", "consumer", "producer_mid")
    ]


def test_build_fusion_regions_from_blocks_discovers_registered_chains():
    block_registry = FusionBlockRegistry(
        (
            FusionBlockDescriptor(
                op="producer",
                inputs=("A",),
                outputs=("{name}_mid",),
                aliases=("M",),
            ),
            FusionBlockDescriptor(
                op="consumer",
                inputs=("{producer}_mid",),
                outputs=("{name}_out",),
                aliases=("R",),
            ),
        )
    )
    schedule_registry = FusionScheduleRegistry().register(
        ("producer", "consumer"),
        _fused_train_block,
        name="producer_consumer_fused",
        status="ready",
    )

    regions = build_fusion_regions_from_blocks(
        region_name="model_fusion",
        blocks=(
            {"name": "consumer0", "route_symbol": "R", "inputs": ("producer0_mid",)},
            {"name": "producer0", "route_symbol": "M"},
            {"name": "layernorm0", "op": "layernorm", "inputs": ("consumer0_out",), "outputs": ("norm0",)},
            {"name": "producer1", "route_symbol": "M"},
            {"name": "consumer1", "route_symbol": "R", "inputs": ("producer1_mid",)},
        ),
        block_registry=block_registry,
        schedule_registry=schedule_registry,
        enable_z3_sync_async_optimization=True,
    )

    assert [region.name for region in regions] == [
        "model_fusion_0_producer_consumer_fused",
        "model_fusion_1_producer_consumer_fused",
    ]
    assert [tuple(node.name for node in region.nodes) for region in regions] == [
        ("producer0", "consumer0"),
        ("producer1", "consumer1"),
    ]
    assert [
        [(edge.producer, edge.consumer, edge.buffer) for edge in region.edges]
        for region in regions
    ] == [
        [("producer0", "consumer0", "producer0_mid")],
        [("producer1", "consumer1", "producer1_mid")],
    ]
    assert [region.schedule_name for region in regions] == [
        "producer_consumer_fused",
        "producer_consumer_fused",
    ]
    assert regions[0].pass_configs[PassConfigKey.TL_Z3_PROOF_ASYNC_ELIGIBILITY.value] is True


def test_build_fusion_regions_from_blocks_can_fail_closed_on_unmatched_blocks():
    block_registry = FusionBlockRegistry(
        (
            FusionBlockDescriptor(op="producer", outputs=("mid",), aliases=("M",)),
            FusionBlockDescriptor(op="consumer", inputs=("mid",), aliases=("R",)),
        )
    )
    schedule_registry = FusionScheduleRegistry().register(
        ("producer", "consumer"),
        _fused_train_block,
        name="producer_consumer_fused",
    )

    with pytest.raises(ValueError, match="no fused schedule registered.*layernorm"):
        build_fusion_regions_from_blocks(
            region_name="model_fusion",
            blocks=(
                {"name": "layernorm0", "op": "layernorm"},
                {"name": "producer0", "route_symbol": "M"},
                {"name": "consumer0", "route_symbol": "R"},
            ),
            block_registry=block_registry,
            schedule_registry=schedule_registry,
            allow_unmatched_blocks=False,
        )


def test_mamba3_fp8_train_block_helper_uses_inferred_chain_edges():
    region = build_mamba3_fp8_train_block_region(schedule_template=lambda _: _fused_train_block)

    assert [node.name for node in region.nodes] == [
        "mamba3_scan",
        "m2rnn_packed_post",
        "fp8_prepare",
        "sparse_mla_fp8_apply",
    ]
    assert [(edge.producer, edge.consumer, edge.buffer) for edge in region.edges] == [
        ("mamba3_scan", "m2rnn_packed_post", "mamba3_state"),
        ("m2rnn_packed_post", "fp8_prepare", "packed_post"),
        ("fp8_prepare", "sparse_mla_fp8_apply", "q_fp8"),
        ("fp8_prepare", "sparse_mla_fp8_apply", "q_scale"),
        ("fp8_prepare", "sparse_mla_fp8_apply", "kv_fp8"),
        ("fp8_prepare", "sparse_mla_fp8_apply", "kv_scale"),
    ]


def test_mamba3_fp8_train_block_helper_accepts_dynamic_nodes():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _fused_train_block,
        nodes=(
            FusionNode("apply", op="sparse_mla", inputs=("post_y",), outputs=("out",)),
            FusionNode("post", op="m2rnn", inputs=("scan_y",), outputs=("post_y",)),
            FusionNode("scan", op="mamba3", inputs=("x",), outputs=("scan_y",)),
        ),
    )

    assert [node.name for node in region.nodes] == ["scan", "post", "apply"]
    assert [(edge.producer, edge.consumer, edge.buffer) for edge in region.edges] == [
        ("scan", "post", "scan_y"),
        ("post", "apply", "post_y"),
    ]


def test_mamba3_fp8_train_block_helper_does_not_replace_empty_dynamic_nodes():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _fused_train_block,
        nodes=(),
    )

    assert region.nodes == ()
    assert region.edges == ()


def test_builder_rejects_ambiguous_inferred_chain_producers():
    builder = (
        FusionRegionBuilder("ambiguous_chain")
        .add_node("producer_a", op="toy", outputs=("mid",))
        .add_node("producer_b", op="toy", outputs=("mid",))
        .add_node("consumer", op="toy", inputs=("mid",), outputs=("B",))
        .set_schedule_template(lambda _: _fused_train_block)
    )

    with pytest.raises(ValueError, match="ambiguous inferred fusion producer.*mid"):
        builder.build()


def test_optimizer_selects_fused_schedule_from_op_chain_with_z3_fullgraph():
    registry = FusionScheduleRegistry().register(
        ("mamba3", "m2rnn", "sparse_mla_fp8_prepare", "sparse_mla_fp8_apply"),
        _logical_train_block_with_internal_edges,
        name="mamba3_m2rnn_fp8_train_block",
        status="ready",
    )
    optimizer = FusionOptimizer(
        "mamba3_fp8_train_block",
        schedule_registry=registry,
    )
    optimizer.add_nodes(
        (
            FusionNode(
                "scan",
                op="mamba3",
                inputs=("x", "state"),
                outputs=("mamba3_state",),
                attrs={"backward": "aot_autograd"},
            ),
            FusionNode(
                "packed",
                op="m2rnn",
                inputs=("mamba3_state",),
                outputs=("packed_post",),
                attrs={"backward": "aot_autograd"},
            ),
            FusionNode(
                "prepare",
                op="sparse_mla_fp8_prepare",
                inputs=("packed_post",),
                outputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale"),
            ),
            FusionNode(
                "apply",
                op="sparse_mla_fp8_apply",
                inputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale"),
                outputs=("train_block_out",),
            ),
        )
    )

    result = optimizer.compile(
        target="metal",
        lowerer=lambda *args, **kwargs: "compiled-optimizer-region",
    )

    assert result.artifact == "compiled-optimizer-region"
    assert result.plan.require_single_kernel is True
    assert result.plan.schedule_name == "mamba3_m2rnn_fp8_train_block"
    assert result.plan.schedule_status == "ready"
    assert result.plan.node_names == ("scan", "packed", "prepare", "apply")
    assert [(edge.producer, edge.consumer, edge.buffer) for edge in result.plan.edges] == [
        ("scan", "packed", "mamba3_state"),
        ("packed", "prepare", "packed_post"),
        ("prepare", "apply", "q_fp8"),
        ("prepare", "apply", "q_scale"),
        ("prepare", "apply", "kv_fp8"),
        ("prepare", "apply", "kv_scale"),
    ]
    assert result.plan.pass_configs[PassConfigKey.TL_Z3_PROOF_BARRIER_MINIMIZATION.value] is True
    assert result.plan.pass_configs[PassConfigKey.TL_Z3_PROOF_ASYNC_ELIGIBILITY.value] is True


def test_optimizer_selects_registered_schedule_after_dependency_ordering():
    registry = FusionScheduleRegistry().register(
        ("producer", "consumer"),
        _fused_train_block,
        name="toy_schedule",
        status="ready",
    )
    optimizer = FusionOptimizer("out_of_order_optimizer", schedule_registry=registry)
    optimizer.add_node("consumer", op="consumer", inputs=("mid",), outputs=("B",))
    optimizer.add_node("producer", op="producer", inputs=("A",), outputs=("mid",))

    plan = optimizer.plan()

    assert plan.schedule_name == "toy_schedule"
    assert plan.node_names == ("producer", "consumer")
    assert [(edge.producer, edge.consumer, edge.buffer) for edge in plan.edges] == [
        ("producer", "consumer", "mid")
    ]


def test_optimizer_fails_closed_when_no_fused_schedule_is_registered():
    optimizer = FusionOptimizer("missing_schedule")
    optimizer.add_node("producer", op="producer", inputs=("A",), outputs=("mid",))
    optimizer.add_node("consumer", op="consumer", inputs=("mid",), outputs=("B",))

    with pytest.raises(ValueError, match="no fused schedule registered.*producer,consumer"):
        optimizer.plan()


def test_registry_schedule_is_experimental_until_marked_ready():
    registry = FusionScheduleRegistry().register(
        ("producer", "consumer"),
        _fused_train_block,
        name="toy_schedule",
    )
    optimizer = FusionOptimizer("experimental_schedule", schedule_registry=registry)
    optimizer.add_node("producer", op="producer", inputs=("A",), outputs=("C",))
    optimizer.add_node("consumer", op="consumer", inputs=("C",), outputs=("D",))

    plan = optimizer.plan()

    assert plan.schedule_name == "toy_schedule"
    assert plan.schedule_status == "experimental"


def test_cache_key_material_includes_selected_fused_schedule_template():
    signature = ("producer", "consumer")
    registry_a = FusionScheduleRegistry().register(signature, _fused_train_block, name="schedule_a")
    registry_b = FusionScheduleRegistry().register(signature, _fusion_consumer, name="schedule_b")

    def make_plan(registry):
        optimizer = FusionOptimizer("same_ops", schedule_registry=registry)
        optimizer.add_node("producer", op="producer", inputs=("A",), outputs=("C",))
        optimizer.add_node("consumer", op="consumer", inputs=("C",), outputs=("D",))
        return optimizer.plan()

    plan_a = make_plan(registry_a)
    plan_b = make_plan(registry_b)

    assert any(part.startswith("schedule:") for part in plan_a.cache_key_material)
    assert plan_a.cache_key_material != plan_b.cache_key_material


def test_cache_key_digest_is_stable_for_equivalent_dependency_ordering():
    def schedule_template(_region):
        return _fused_train_block_with_internal_edge

    ordered_region = build_fusion_region(
        region_name="same_dynamic_region",
        nodes=(
            FusionNode("producer", op="producer", inputs=("A",), outputs=("mid",)),
            FusionNode("consumer", op="consumer", inputs=("mid",), outputs=("D",)),
        ),
        schedule_template=schedule_template,
    )
    reversed_region = build_fusion_region(
        region_name="same_dynamic_region",
        nodes=(
            FusionNode("consumer", op="consumer", inputs=("mid",), outputs=("D",)),
            FusionNode("producer", op="producer", inputs=("A",), outputs=("mid",)),
        ),
        schedule_template=schedule_template,
    )

    ordered_result = compile_fusion_region(
        ordered_region,
        target="metal",
        lowerer=lambda *args, **kwargs: "compiled",
    )
    reversed_result = compile_fusion_region(
        reversed_region,
        target="metal",
        lowerer=lambda *args, **kwargs: "compiled",
    )

    assert ordered_result.plan.node_names == reversed_result.plan.node_names
    assert ordered_result.plan.edges == reversed_result.plan.edges
    assert ordered_result.plan.cache_key_material == reversed_result.plan.cache_key_material
    assert fusion_cache_key_digest(
        ordered_result.plan,
        ordered_result.lowered_module,
    ) == fusion_cache_key_digest(
        reversed_result.plan,
        reversed_result.lowered_module,
    )


def test_region_without_template_requires_all_nodes_to_be_prim_funcs():
    with pytest.raises(ValueError, match="PrimFunc nodes"):
        FusionRegionBuilder("bad_auto").add_node("opaque", op="mamba3").build()


def test_region_rejects_lowered_source_kernels():
    builder = FusionRegionBuilder("bad_region")
    builder.add_node("scan", op="mamba3")

    with pytest.raises(ValueError, match="pre-source"):
        builder.add_lowered_source_kernel("scan", "kernel void already_lowered() {}")


def test_region_builder_accepts_pre_source_prim_func_nodes_only():
    builder = FusionRegionBuilder("prim_func_region")
    region = (
        builder.add_prim_func_node("candidate", _fused_train_block, op="toy")
        .set_schedule_template(lambda _: _fused_train_block)
        .build()
    )

    assert region.nodes[0].prim_func is _fused_train_block

    lowered_source_func = _fused_train_block.with_attr("code_block_source", "kernel void already_lowered() {}")
    with pytest.raises(ValueError, match="pre-source"):
        FusionRegionBuilder("bad_prim_func_region").add_prim_func_node("lowered", lowered_source_func)


def test_schedule_template_must_return_pre_source_ir():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: "kernel void already_lowered() {}",
    )

    with pytest.raises(TypeError, match="PrimFunc or IRModule"):
        compile_fusion_region(region, target="metal", lowerer=lambda *args, **kwargs: None)


def test_acceptance_gate_ignores_dirty_path_b_baselines():
    dirty_baseline = BaselineComparison(
        optimizer="lion8bit",
        dtype="fp8",
        path_b_tokens_per_second=500.0,
        path_c_tokens_per_second=900.0,
        path_b_cache_delta_gib=80.0,
        path_c_peak_delta_gib=-1.0,
    )
    clean_loss = BaselineComparison(
        optimizer="lion8bit",
        dtype="fp8",
        path_b_tokens_per_second=900.0,
        path_c_tokens_per_second=810.0,
        path_b_cache_delta_gib=3.0,
        path_c_peak_delta_gib=-1.0,
    )
    clean_win = BaselineComparison(
        optimizer="lion8bit",
        dtype="fp8",
        path_b_tokens_per_second=900.0,
        path_c_tokens_per_second=990.0,
        path_b_cache_delta_gib=3.0,
        path_c_peak_delta_gib=0.0,
    )

    assert fusion_default_allowed(dirty_baseline) is False
    assert fusion_default_allowed(clean_loss) is False
    assert fusion_default_allowed(clean_win) is True


def test_warm_cache_audit_flags_warm_first_step_that_still_looks_cold():
    audit = audit_warm_cache_reuse(
        cache_hit=True,
        cold_first_step_ms=35_000,
        warm_first_step_ms=30_000,
        steady_step_ms=2_000,
    )

    assert audit.status == "incomplete"
    assert "warm first-step" in audit.reason


def test_cache_key_audit_flags_same_key_recompile():
    region = build_mamba3_fp8_train_block_region(schedule_template=lambda _: _fused_train_block)
    result = compile_fusion_region(region, target="metal", lowerer=lambda *args, **kwargs: None)

    digest = fusion_cache_key_digest(result.plan, result.lowered_module)
    repeat_digest = fusion_cache_key_digest(result.plan, result.lowered_module)
    audit = audit_fusion_cache_key(
        expected_digest=digest,
        observed_digest=repeat_digest,
        cache_hit=False,
    )

    assert digest == repeat_digest
    assert audit.status == "recompiled_same_key"


def test_auto_prim_func_graph_cache_key_includes_source_digest():
    region_a = (
        FusionRegionBuilder("cache_region")
        .add_prim_func_node("node", _fused_train_block, op="toy")
        .build()
    )
    region_b = (
        FusionRegionBuilder("cache_region")
        .add_prim_func_node("node", _fusion_consumer, op="toy")
        .build()
    )

    result_a = compile_fusion_region(region_a, target="metal", lowerer=lambda *args, **kwargs: None)
    result_b = compile_fusion_region(region_b, target="metal", lowerer=lambda *args, **kwargs: None)

    assert any(part.startswith("prim_funcs:") for part in result_a.plan.cache_key_material)
    assert result_a.plan.cache_key_material != result_b.plan.cache_key_material
