"""Focused Metal codegen/runtime tests for TileLang reductions."""

import re

import pytest
import torch

import tilelang
from tilelang.carver.template.general_reduce import GeneralReductionTemplate
from tilelang import tvm as tvm
from tvm import tir
import tilelang.language as T
from tilelang.engine.lower import device_codegen_without_compile, get_device_call
from tilelang.engine.phase import OptimizeForTarget
from tilelang.layout import Fragment
from tilelang.analysis.backend_lowerer_selection import (
    build_reduction_backend_lowerer_diagnostics,
)
from tilelang.backend.reduction import select_reduction_lowerer
import tilelang.testing
from tilelang.utils.language import to_buffer_region


_FORBIDDEN_CUDA_REDUCE_TOKENS = (
    "__syncthreads",
    "__shfl",
    "__threadfence",
    "cuda_runtime",
    "cuda_fp16",
    "cuda_bf16",
)


def _same_simdgroup_fast_path_safe(
    *, target, reducing_threads: int, scale: int, thread_offset: int
) -> bool:
    hook = tvm.ffi.get_global_func(
        "tl.metal.reduce_same_simdgroup_fast_path_safe", allow_missing=True
    )
    assert (
        hook is not None
    ), "tl.metal.reduce_same_simdgroup_fast_path_safe is not registered"
    return bool(hook(target, reducing_threads, scale, thread_offset))


def _lower_source(func) -> str:
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")
    assert artifact.kernel_source is not None
    return artifact.kernel_source


def _lower_preannotated_source(func) -> str:
    target = tvm.target.Target("metal", host="llvm")
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    with tvm.transform.PassContext(), target:
        mod = tilelang.transform.LowerTileLangLetStmt()(mod)
        mod = tilelang.transform.LowerTileLangAllocate()(mod)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.Simplify()(mod)
        mod = tilelang.transform.LayoutInference()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        mod = tilelang.transform.LowerTileLangLetStmt()(mod)
        mod = tilelang.transform.LowerTileLangAllocate()(mod)
        mod = tilelang.transform.DecoupleTypeCast()(mod)
        mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
        mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
        mod = tilelang.transform.LowerAccessPtr()(mod)
        mod = tilelang.transform.Simplify()(mod)
        mod = OptimizeForTarget(mod, target)

    device_mod = tvm.tir.transform.Filter(get_device_call(False))(mod)
    codegen_mod = device_codegen_without_compile(device_mod, target)
    return codegen_mod.inspect_source()


def _assert_no_cuda_reduce_leakage(src: str) -> None:
    for token in _FORBIDDEN_CUDA_REDUCE_TOKENS:
        assert token not in src, f"unexpected CUDA reduce token {token!r} in Metal source:\n{src}"

    assert "blockDim" not in src, src


def _assert_metal_reduce_tokens(src: str, *, cross_simdgroup: bool = False) -> None:
    assert "kernel void" in src
    assert "namespace tl" in src
    assert "struct AllReduce" in src
    assert "simd_shuffle_xor" in src or re.search(r"\bsimd_(sum|max|min)\(", src), src
    assert "[[thread_position_in_threadgroup]]" in src
    if cross_simdgroup:
        assert "threadgroup_barrier" in src or "[[threadgroup" in src, src


def _assert_body_workspace(src: str, *, expected: bool) -> None:
    has_workspace = re.search(
        r"^\s*threadgroup\s+\w+\s+workspace\[\d+\];", src, re.MULTILINE
    )
    if expected:
        assert has_workspace is not None, src
        assert "(&(workspace[0]))" in src, src
    else:
        assert has_workspace is None, src
        assert "(&(workspace[0]))" not in src, src


def _find_allreduce_calls(src: str, reducer: str):
    return [
        (int(match.group(1)), int(match.group(2)))
        for match in re.finditer(
            rf"AllReduce<tl::{reducer},\s*(\d+),\s*1,\s*0,\s*"
            rf"tl::SyncThreadsBarrier,\s*1,\s*(\d+)>::run",
            src,
        )
    ]


def _assert_single_allreduce_call(src: str, reducer: str):
    calls = _find_allreduce_calls(src, reducer)
    assert len(calls) == 1, f"calls={calls}\n{src}"
    return calls[0]


def _assert_same_simdgroup_allreduce(
    src: str, reducer: str = "SumOp"
) -> None:
    threads, workspace_stride = _assert_single_allreduce_call(src, reducer)
    assert 0 < threads <= 32, f"threads={threads}\n{src}"
    assert workspace_stride == 0, f"workspace_stride={workspace_stride}\n{src}"


def _assert_cross_simdgroup_allreduce(
    src: str, reducer: str = "SumOp"
) -> None:
    threads, workspace_stride = _assert_single_allreduce_call(src, reducer)
    assert threads > 32, f"threads={threads}\n{src}"
    assert workspace_stride == threads, (
        f"threads={threads}, workspace_stride={workspace_stride}\n{src}"
    )


def _extract_between(src: str, start: str, end: str) -> str:
    assert start in src, src
    rest = src.split(start, 1)[1]
    assert end in rest, src
    return rest.split(end, 1)[0]


def _extract_simdgroup_intra_helpers(src: str) -> str:
    return _extract_between(
        src,
        "template <class Reducer>\nstruct SimdgroupIntraReduce",
        "template <class Reducer, int threads, int thread_offset",
    )


def _extract_simdgroup_cross_helper(src: str) -> str:
    return _extract_between(
        src,
        "struct AllReduceSimdgroupCross",
        "struct AllReduceStep",
    )


def _extract_row_reduce_sum_helper(src: str) -> str:
    return _extract_between(
        src,
        "struct RowReduceSumContiguousInnermost",
        "struct SyncThreadsBarrier",
    )


def _extract_kernel_body(src: str) -> str:
    return _extract_between(src, "kernel void", "\n}\n")


def _assert_sum_intra_stage_uses_simd_sum(src: str) -> None:
    helpers = _extract_simdgroup_intra_helpers(src)
    generic_helper, sum_helper = helpers.split("template <>", 1)

    assert "simd_shuffle_xor" in generic_helper
    assert "simd_sum" not in generic_helper
    assert "struct SimdgroupIntraReduce<SumOp>" in sum_helper
    assert "return simd_sum(x);" in sum_helper
    assert "simd_shuffle_xor" not in sum_helper

    cross_helper = _extract_simdgroup_cross_helper(src)
    assert "return SimdgroupIntraReduce<Reducer>::run(x);" in cross_helper
    assert "simd_sum" not in cross_helper
    assert "result = reduce_partials(result, lane);" in cross_helper


def _make_reduce_kernel(op, *, length=32, dtype=T.float32, threads=32):
    @T.prim_func
    def reduce_kernel(A: T.Tensor((length,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((length,), dtype)
            dst = T.alloc_fragment((1,), dtype)
            T.copy(A, src)
            if op == "sum":
                T.reduce_sum(src, dst)
            elif op == "max":
                T.reduce_max(src, dst)
            elif op == "min":
                T.reduce_min(src, dst)
            elif op == "bitand":
                T.reduce_bitand(src, dst)
            elif op == "bitor":
                T.reduce_bitor(src, dst)
            elif op == "bitxor":
                T.reduce_bitxor(src, dst)
            else:
                raise ValueError(op)
            T.copy(dst, B)

    return reduce_kernel


def _make_semantic_reduce_api_kernel(api, *, length=32, dtype=T.float32, threads=32):
    @T.prim_func
    def reduce_kernel(A: T.Tensor((length,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((length,), dtype)
            dst = T.alloc_fragment((1,), dtype)
            T.copy(A, src)
            if api == "row":
                T.row_reduce(src, dst, op="sum")
            elif api == "block":
                T.block_reduce(src, dst, op="sum")
            else:
                raise ValueError(api)
            T.copy(dst, B)

    return reduce_kernel


def _make_metal_template_row_reduce_sum(rows=16, cols=1024, dtype="float32"):
    template = GeneralReductionTemplate(structure="SR", shape=[rows, cols], dtype=dtype)
    return template.make_metal_row_reduce_sum()


def _make_row_reduce_kernel(*, rows=8, cols=32, threads=None):
    kernel_threads = cols if threads is None else threads

    @T.prim_func
    def row_reduce_kernel(
        A: T.Tensor((rows, cols), T.float32),
        B: T.Tensor((rows,), T.float32),
    ):
        with T.Kernel(rows, threads=kernel_threads) as bx:
            src = T.alloc_fragment((cols,), T.float32)
            dst = T.alloc_fragment((1,), T.float32)
            T.copy(A[bx, 0], src)
            T.reduce_sum(src, dst)
            T.copy(dst, B[bx])

    return row_reduce_kernel


def _make_split_thread_allreduce_kernel(*, reduce_extent=32, groups=4):
    total_threads = reduce_extent * groups

    @T.prim_func
    def split_thread_allreduce(
        A: T.Tensor((total_threads,), T.float32), B: T.Tensor((groups,), T.float32)
    ):
        with T.Kernel(1, threads=total_threads):
            accum = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            lane = T.get_thread_binding(0)
            kr = T.floormod(lane, reduce_extent)
            group = T.floordiv(lane, reduce_extent)
            accum[0] = A[lane]
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum[0],
                        True,
                        reduced[0],
                        kr,
                        dtype="handle",
                    )
                )
            if kr == 0:
                B[group] = reduced[0]

    return split_thread_allreduce


def _make_split_thread_allreduce_parallel_kernel(
    *, reduce_extent=32, groups=2, rows=3
):
    row_threads = rows
    reduce_threads = reduce_extent * groups
    total_values = row_threads * reduce_threads
    total_outputs = row_threads * groups

    @T.prim_func
    def split_thread_allreduce_parallel(
        A: T.Tensor((total_values,), T.float32),
        B: T.Tensor((total_outputs,), T.float32),
    ):
        with T.Kernel(1, threads=(reduce_threads, row_threads)):
            accum = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            lane = T.get_thread_binding(0)
            row = T.get_thread_binding(1)
            kr = T.floormod(lane, reduce_extent)
            group = T.floordiv(lane, reduce_extent)
            accum[0] = A[row * reduce_threads + lane]
            reduce_axis = T.reduction_axis("p", reduce_extent, kr)
            T.thread_reduce(accum[0], reduced[0], reduce_axis, op="sum")
            if kr == 0:
                B[row * groups + group] = reduced[0]

    return split_thread_allreduce_parallel


def _make_nested_same_simdgroup_thread_allreduce_kernel(*, cols=5):
    @T.prim_func
    def nested_thread_allreduce(
        A: T.Tensor((32, cols), T.float32),
        B: T.Tensor((cols,), T.float32),
    ):
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding(0)
            for col in T.serial(cols):
                reduced = T.alloc_local((1,), T.float32)
                T.thread_allreduce_sum(A[lane, col], reduced[0], lane)
                if lane == 0:
                    B[col] = reduced[0]

    return nested_thread_allreduce


def _make_finalize_reducer_kernel(*, rows=4, dtype=T.float32, threads=32):
    layout = Fragment((rows,), forward_fn=lambda i, rep: (rep, i), replicate=threads)

    @T.prim_func
    def finalize_reducer_kernel(
        B: T.Tensor((rows,), dtype),
    ):
        with T.Kernel(1, threads=threads):
            reducer = T.alloc_fragment((rows,), dtype)
            T.annotate_layout({reducer: layout})
            for i in T.Parallel(rows):
                reducer[i] = T.cast(i + 1, dtype)
            tir.call_intrin(
                "handle",
                tir.op.Op.get("tl.tileop.finalize_reducer"),
                to_buffer_region(reducer, access_type="w"),
                0,
            )
            for i in T.Parallel(rows):
                B[i] = reducer[i]

    return finalize_reducer_kernel


@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_metal_reduce_codegen_uses_metal_simd_reduction(op):
    src = _lower_source(_make_reduce_kernel(op, length=32, threads=32))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src)


def test_metal_reduce_same_simdgroup_codegen_elides_body_workspace():
    src = _lower_source(_make_reduce_kernel("sum", length=32, threads=32))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src)
    _assert_body_workspace(src, expected=False)


def test_metal_reduce_same_simdgroup_legality_hook_is_static_and_exact(monkeypatch):
    monkeypatch.setenv("TILELANG_DISABLE_Z3_SIMDGROUP", "1")
    metal = tvm.target.Target("metal")
    llvm = tvm.target.Target("llvm")

    assert _same_simdgroup_fast_path_safe(
        target=metal, reducing_threads=32, scale=1, thread_offset=0
    )
    assert _same_simdgroup_fast_path_safe(
        target=metal, reducing_threads=16, scale=1, thread_offset=16
    )
    assert _same_simdgroup_fast_path_safe(
        target=metal, reducing_threads=8, scale=1, thread_offset=24
    )

    assert not _same_simdgroup_fast_path_safe(
        target=metal, reducing_threads=16, scale=1, thread_offset=8
    )
    assert not _same_simdgroup_fast_path_safe(
        target=metal, reducing_threads=16, scale=1, thread_offset=17
    )
    assert not _same_simdgroup_fast_path_safe(
        target=metal, reducing_threads=64, scale=1, thread_offset=0
    )
    assert not _same_simdgroup_fast_path_safe(
        target=metal, reducing_threads=24, scale=1, thread_offset=0
    )
    assert not _same_simdgroup_fast_path_safe(
        target=metal, reducing_threads=16, scale=2, thread_offset=0
    )
    assert not _same_simdgroup_fast_path_safe(
        target=llvm, reducing_threads=32, scale=1, thread_offset=0
    )


def test_metal_reduce_same_simdgroup_codegen_elision_is_not_z3_gated(monkeypatch):
    monkeypatch.setenv("TILELANG_DISABLE_Z3_SIMDGROUP", "1")
    src = _lower_source(_make_reduce_kernel("sum", length=32, threads=32))

    _assert_same_simdgroup_allreduce(src)
    _assert_body_workspace(src, expected=False)


def test_metal_row_reduce_same_simdgroup_uses_no_workspace_fast_path():
    src = _lower_source(_make_row_reduce_kernel(rows=8, cols=32, threads=32))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src)
    _assert_same_simdgroup_allreduce(src)
    assert re.search(r"::run\(dst\[0\],\s*\(\(int\)threadIdx\.x\)\)", src), src
    _assert_body_workspace(src, expected=False)


@pytest.mark.parametrize("op", ["bitand", "bitor", "bitxor"])
def test_metal_reduce_codegen_for_additional_ops_has_no_cuda_template_leakage(op):
    src = _lower_source(_make_reduce_kernel(op, length=32, dtype=T.int32, threads=32))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src)


def test_metal_reduce_cross_simdgroup_sum_codegen_uses_simd_sum_intra_stage():
    src = _lower_source(_make_reduce_kernel("sum", length=1024, threads=1024))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src, cross_simdgroup=True)
    _assert_cross_simdgroup_allreduce(src)
    _assert_body_workspace(src, expected=True)
    _assert_sum_intra_stage_uses_simd_sum(src)


@pytest.mark.parametrize(
    ("op", "dtype", "reducer"),
    [
        ("max", T.float32, "MaxOp"),
        ("min", T.float32, "MinOp"),
        ("bitand", T.int32, "BitAndOp"),
    ],
)
def test_metal_reduce_cross_simdgroup_non_sum_keeps_shuffle_intra_stage(
    op, dtype, reducer
):
    src = _lower_source(_make_reduce_kernel(op, length=1024, dtype=dtype, threads=1024))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src, cross_simdgroup=True)
    _assert_cross_simdgroup_allreduce(src, reducer)
    _assert_body_workspace(src, expected=True)

    helpers = _extract_simdgroup_intra_helpers(src)
    generic_helper = helpers.split("template <>", 1)[0]
    assert "simd_shuffle_xor" in generic_helper
    assert "simd_sum" not in generic_helper
    assert not re.search(r"\bsimd_(max|min)\(", generic_helper), generic_helper


def test_metal_reduce_1024_codegen_uses_simdgroup_cross_fast_path():
    src = _lower_source(_make_reduce_kernel("sum", length=1024, threads=1024))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src, cross_simdgroup=True)
    _assert_body_workspace(src, expected=True)
    _assert_cross_simdgroup_allreduce(src)

    helper_src = src.split("struct AllReduceSimdgroupCross", 1)[1].split(
        "struct AllReduceStep", 1
    )[0]
    assert "workspace_stride >= threads" in src
    assert "enum { final_slot = simdgroup_count };" in helper_src
    assert "red_buf[simdgroup_id] = x;" in helper_src
    assert "red_buf[final_slot] = result;" in helper_src
    assert "red_buf[local_tid]" not in helper_src
    assert helper_src.count("Barrier::template sync<") == 6
    assert "const int batch_offset = i * workspace_stride;" in helper_src
    assert "red_buf[simdgroup_id + i * workspace_stride]" not in helper_src
    assert "red_buf[lane + i * workspace_stride]" not in helper_src
    assert "red_buf[final_slot + i * workspace_stride]" not in helper_src


def test_metal_template_row_reduce_1024_uses_one_simdgroup_per_row_fast_path():
    src = _lower_preannotated_source(
        _make_metal_template_row_reduce_sum(rows=16, cols=1024)
    )

    _assert_no_cuda_reduce_leakage(src)
    assert "RowReduceSumContiguousInnermost<float, 8, 1024>" in src
    assert "[[thread_position_in_threadgroup]]" in src

    helper_src = _extract_row_reduce_sum_helper(src)
    assert "enum { simdgroup_size = 32 };" in helper_src
    assert "const uint row_in_group = tid / uint(simdgroup_size);" in helper_src
    assert "for (uint col = lane; col < uint(cols); col += uint(simdgroup_size))" in helper_src
    assert "T total = simd_sum(acc);" in helper_src
    assert "B[row] = total;" in helper_src
    assert "threadgroup_barrier" not in helper_src
    assert "workspace" not in helper_src
    assert "red_buf" not in helper_src

    kernel_body = _extract_kernel_body(src)
    assert "RowReduceSumContiguousInnermost<float, 8, 1024>::run" in kernel_body
    assert "AllReduce<" not in kernel_body
    assert "threadgroup_barrier" not in kernel_body
    assert "workspace" not in kernel_body


@tilelang.testing.requires_metal
def test_metal_template_row_reduce_runtime_mps_matches_torch_sum():
    rows = 10
    cols = 1024
    kernel = tilelang.compile(
        _make_metal_template_row_reduce_sum(rows=rows, cols=cols),
        target="metal",
    )
    values = (
        torch.arange(rows * cols, dtype=torch.float32, device="mps").reshape(
            rows, cols
        )
        / 1024.0
    )
    out = torch.empty(rows, dtype=torch.float32, device="mps")

    kernel(values, out)
    torch.mps.synchronize()

    expected = values.cpu().sum(dim=1)
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-4)


def test_metal_template_row_reduce_rejects_non_innermost_structure():
    template = GeneralReductionTemplate(structure="RS", shape=[1024, 16], dtype="float32")

    with pytest.raises(ValueError, match="structure='SR'"):
        template.make_metal_row_reduce_sum()


def test_thread_reduce_surface_rejects_unsupported_op():
    with pytest.raises(ValueError, match="unsupported thread_reduce op"):
        T.thread_reduce(T.float32(1.0), None, T.int32(0), op="max")


def test_reduction_axis_rejects_invalid_metadata():
    with pytest.raises(ValueError, match="non-empty name"):
        T.reduction_axis("", 32, T.int32(0))
    with pytest.raises(ValueError, match="unsupported reduction_axis role"):
        T.reduction_axis("p", 32, T.int32(0), role="warp")


@pytest.mark.parametrize("api", ["row", "block"])
def test_semantic_reduce_api_lowers_existing_tileop_reduce(api):
    src = _lower_source(_make_semantic_reduce_api_kernel(api))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src)


def test_row_and_block_reduce_policy_is_reserved_for_scheduler():
    with pytest.raises(ValueError, match="row_reduce policy is reserved"):
        T.row_reduce(None, None, policy={"strategy": "simd"})
    with pytest.raises(ValueError, match="block_reduce policy is reserved"):
        T.block_reduce(None, None, policy={"strategy": "simd"})


def test_metal_lower_thread_allreduce_accepts_split_simdgroup_index():
    src = _lower_source(_make_split_thread_allreduce_kernel())

    _assert_no_cuda_reduce_leakage(src)
    assert "simd_sum(accum[0])" in src, src
    assert "red_result[" not in src, src


def test_metal_reduction_backend_registry_names_selected_lowerer_and_cache():
    diagnostics = build_reduction_backend_lowerer_diagnostics(
        _make_split_thread_allreduce_kernel(reduce_extent=32, groups=1),
        tvm.target.Target("metal"),
    )
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.lowerer_name == "metal.same-simdgroup.sum"
    assert diagnostic.lowerer == "tirx.metal.simd_sum"
    assert diagnostic.selected_strategy == "same-simdgroup"
    assert diagnostic.memory_visibility_scope == "simdgroup"
    assert diagnostic.scratch_scope is None

    first = select_reduction_lowerer(
        tvm.target.Target("metal"),
        op="sum",
        strategy="same-simdgroup",
        reduction_extent=32,
        accumulator_dtype="float32",
    )
    second = select_reduction_lowerer(
        tvm.target.Target("metal"),
        op="sum",
        strategy="same-simdgroup",
        reduction_extent=32,
        accumulator_dtype="float32",
    )
    assert first is second


def test_metal_same_simdgroup_thread_allreduce_keeps_local_buffer_scope():
    src = _lower_source(_make_nested_same_simdgroup_thread_allreduce_kernel())

    _assert_no_cuda_reduce_leakage(src)
    assert "simd_sum" in src, src
    assert "reduced_" not in src, src


def test_metal_lower_thread_allreduce_split_cross_simdgroup_reuses_staging_result():
    src = _lower_source(
        _make_split_thread_allreduce_kernel(reduce_extent=64, groups=2)
    )

    _assert_no_cuda_reduce_leakage(src)
    assert "simd_shuffle_down" in src, src
    assert "red_result[" not in src, src
    assert "red_buf_staging[" in src, src


@pytest.mark.parametrize("reduce_extent", [32, 64, 96, 128, 256, 512, 1024])
def test_metal_thread_allreduce_extent_matrix_keeps_final_outputs_internal(
    reduce_extent: int,
):
    src = _lower_source(
        _make_split_thread_allreduce_kernel(reduce_extent=reduce_extent, groups=1)
    )

    _assert_no_cuda_reduce_leakage(src)
    assert "red_result[" not in src, src
    if reduce_extent <= 32:
        assert "simd_sum(accum[0])" in src, src
        assert "red_buf_staging[" not in src, src
    else:
        assert "red_buf_staging[" in src, src


def test_metal_lower_thread_allreduce_split_subgroups_broadcast_local_lane():
    src = _lower_source(
        _make_split_thread_allreduce_kernel(reduce_extent=16, groups=4)
    )

    _assert_no_cuda_reduce_leakage(src)
    _assert_body_workspace(src, expected=False)
    shuffle_calls = [
        line.strip()
        for line in src.splitlines()
        if "simd_shuffle(" in line and "red_buf0" in line
    ]
    assert shuffle_calls, src
    assert "simd_shuffle(red_buf0[0], 0)" not in "\n".join(shuffle_calls)
    assert any(
        ("* 16" in line and ("& 31" in line or "% 2" in line))
        for line in shuffle_calls
    ), (
        f"expected simdgroup-local 16-lane broadcast source; calls={shuffle_calls}\n{src}"
    )


def test_metal_lower_thread_allreduce_split_allows_parallel_axis():
    src = _lower_source(_make_split_thread_allreduce_parallel_kernel())

    _assert_no_cuda_reduce_leakage(src)
    assert "simd_sum(accum[0])" in src, src
    assert "red_result[" not in src, src


@tilelang.testing.requires_metal
def test_metal_split_simdgroup_allreduce_parallel_axis_runtime_mps():
    reduce_extent = 32
    groups = 2
    rows = 3
    kernel = tilelang.compile(
        _make_split_thread_allreduce_parallel_kernel(
            reduce_extent=reduce_extent, groups=groups, rows=rows
        ),
        target="metal",
    )
    values = torch.arange(
        rows * groups * reduce_extent, dtype=torch.float32, device="mps"
    )
    out = torch.empty(rows * groups, dtype=torch.float32, device="mps")

    kernel(values, out)
    torch.mps.synchronize()

    expected = values.reshape(rows, groups, reduce_extent).sum(dim=2).reshape(-1)
    torch.testing.assert_close(out.cpu(), expected.cpu())


@tilelang.testing.requires_metal
def test_metal_split_simdgroup_allreduce_runtime_mps():
    kernel = tilelang.compile(_make_split_thread_allreduce_kernel(), target="metal")
    values = torch.arange(128, dtype=torch.float32, device="mps")
    out = torch.empty(4, dtype=torch.float32, device="mps")

    kernel(values, out)
    torch.mps.synchronize()

    expected = torch.tensor(
        [sum(range(i * 32, (i + 1) * 32)) for i in range(4)],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out.cpu(), expected)


@tilelang.testing.requires_metal
def test_metal_split_subsimdgroup_allreduce_runtime_mps():
    reduce_extent = 16
    groups = 4
    kernel = tilelang.compile(
        _make_split_thread_allreduce_kernel(
            reduce_extent=reduce_extent, groups=groups
        ),
        target="metal",
    )
    values = torch.arange(reduce_extent * groups, dtype=torch.float32, device="mps")
    out = torch.empty(groups, dtype=torch.float32, device="mps")

    kernel(values, out)
    torch.mps.synchronize()

    expected = torch.tensor(
        [
            sum(range(i * reduce_extent, (i + 1) * reduce_extent))
            for i in range(groups)
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out.cpu(), expected)


def test_metal_finalize_reducer_same_simdgroup_codegen_elides_body_workspace():
    src = _lower_preannotated_source(_make_finalize_reducer_kernel(threads=32))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src)
    _assert_same_simdgroup_allreduce(src)
    _assert_body_workspace(src, expected=False)


def test_metal_finalize_reducer_cross_simdgroup_codegen_keeps_body_workspace():
    src = _lower_preannotated_source(_make_finalize_reducer_kernel(threads=64))

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src, cross_simdgroup=True)
    _assert_cross_simdgroup_allreduce(src)
    _assert_body_workspace(src, expected=True)


def test_metal_reduce_nan_propagate_does_not_emit_cuda_nan_intrinsics():
    src = _lower_source(_make_nan_reduce_kernel())

    _assert_no_cuda_reduce_leakage(src)
    _assert_metal_reduce_tokens(src)
    assert "__hmax_nan" not in src
    assert "MaxOpNan" not in src


def _make_nan_reduce_kernel():
    @T.prim_func
    def reduce_nan_kernel(A: T.Tensor((32,), T.float16), B: T.Tensor((1,), T.float16)):
        with T.Kernel(1, threads=32):
            src = T.alloc_fragment((32,), T.float16)
            dst = T.alloc_fragment((1,), T.float16)
            T.copy(A, src)
            T.reduce_max(src, dst, nan_propagate=True)
            T.copy(dst, B)

    return reduce_nan_kernel


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    ("op", "values", "expected"),
    [
        ("sum", torch.arange(32, dtype=torch.float32), 496.0),
        ("max", torch.arange(32, dtype=torch.float32) - 7, 24.0),
        ("min", torch.arange(32, dtype=torch.float32) - 7, -7.0),
    ],
)
def test_metal_reduce_runtime_mps_small(op, values, expected):
    kernel = tilelang.compile(_make_reduce_kernel(op), target="metal")
    out = torch.empty(1, dtype=torch.float32, device="mps")

    kernel(values.to("mps"), out)
    torch.mps.synchronize()

    torch.testing.assert_close(out.cpu(), torch.tensor([expected], dtype=torch.float32))


@tilelang.testing.requires_metal
def test_metal_reduce_runtime_mps_cross_simdgroup_sum():
    kernel = tilelang.compile(
        _make_reduce_kernel("sum", length=256, threads=256), target="metal"
    )
    values = torch.arange(256, dtype=torch.float32, device="mps")
    out = torch.empty(1, dtype=torch.float32, device="mps")

    kernel(values, out)
    torch.mps.synchronize()

    expected = torch.tensor([sum(range(256))], dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected)


if __name__ == "__main__":
    tilelang.testing.main()
