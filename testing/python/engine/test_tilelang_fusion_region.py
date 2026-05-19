import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.engine import compile_fusion_region as exported_compile_fusion_region
from tilelang.engine.fusion import (
    BaselineComparison,
    FusionAutogradPlan,
    FusionRegionBuilder,
    audit_fusion_cache_key,
    audit_warm_cache_reuse,
    build_mamba3_fp8_train_block_region,
    compile_fusion_region,
    fusion_cache_key_digest,
    fusion_default_allowed,
)
from tilelang.transform import PassConfigKey


@T.prim_func
def _fused_train_block(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
    with T.Kernel(1, threads=1):
        C[0] = A[0]


@T.prim_func
def _fusion_consumer(C: T.Buffer((4,), "float32"), D: T.Buffer((4,), "float32")):
    with T.Kernel(1, threads=1):
        D[0] = C[0]


@T.prim_func
def _fused_train_block_with_internal_edge(A: T.Buffer((4,), "float32"), D: T.Buffer((4,), "float32")):
    with T.Kernel(1, threads=1):
        packed_post = T.alloc_local((4,), "float32")
        packed_post[0] = A[0]
        D[0] = packed_post[0]


@T.prim_func
def _logical_train_block_with_internal_edges(
    x: T.Buffer((4,), "float32"),
    state: T.Buffer((4,), "float32"),
    train_block_out: T.Buffer((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        mamba3_state = T.alloc_local((4,), "float32")
        packed_post = T.alloc_local((4,), "float32")
        mamba3_state[0] = x[0] + state[0]
        packed_post[0] = mamba3_state[0]
        train_block_out[0] = packed_post[0]


@T.prim_func
def _leaky_fused_train_block(
    A: T.Buffer((4,), "float32"),
    packed_post: T.Buffer((4,), "float32"),
    D: T.Buffer((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        D[0] = A[0] + packed_post[0]


def test_region_compile_uses_one_pre_source_ir_module_with_z3_config():
    assert exported_compile_fusion_region is compile_fusion_region
    assert tilelang.compile_fusion_region is compile_fusion_region
    assert tilelang.FusionRegionBuilder is FusionRegionBuilder
    assert tilelang.FusionAutogradPlan is FusionAutogradPlan

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
        .connect("producer", "consumer", buffer="C")
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
        .connect("producer", "consumer", buffer="C")
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


def test_fullgraph_compile_rejects_explicit_schedule_missing_region_abi():
    region = build_mamba3_fp8_train_block_region(
        schedule_template=lambda _: _fused_train_block_with_internal_edge,
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
