"""Opt-in Metal reduction performance smoke tests.

Run manually with:

    TILELANG_RUN_METAL_REDUCE_BENCH=1 python -m pytest \
        testing/python/metal/test_metal_reduce_perf_smoke.py -s -q

TileLang launches run through ``jit_kernel.get_profiler()`` and are timed with
an MPS-safe synchronization per iteration.  That keeps this smoke test
rerunnable with TVM's Metal runtime and makes the output useful for relative
comparisons across same-simdgroup and cross-simdgroup reductions.
The row-wise matrix cases also compare narrower threadgroup shapes for the
same row width and print lowered-source evidence for the large
cross-simdgroup path: AllReduceSimdgroupCross use, AllReduce thread count,
threadgroup barriers, threadgroup workspace usage, and vectorized load shape
when detectable.  When MLX is installed, output lines include TileLang,
Torch MPS, and MLX timings using stable key names.
"""

import os
import re
import time
from dataclasses import dataclass

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


_RUN_ENV = "TILELANG_RUN_METAL_REDUCE_BENCH"
_WARMUP_ENV = "TILELANG_METAL_REDUCE_BENCH_WARMUP"
_ITERS_ENV = "TILELANG_METAL_REDUCE_BENCH_ITERS"
_SIMDGROUP_SIZE = 32


@dataclass(frozen=True)
class _ReducePerfCase:
    length: int
    label: str
    optional: bool = False

    @property
    def path(self) -> str:
        if self.length <= _SIMDGROUP_SIZE:
            return "same-simdgroup"
        return "cross-simdgroup"

    @property
    def simdgroups(self) -> int:
        return (self.length + _SIMDGROUP_SIZE - 1) // _SIMDGROUP_SIZE


@dataclass(frozen=True)
class _RowReducePerfCase:
    rows: int
    cols: int
    label: str
    threads: int | None = None
    optional: bool = False

    @property
    def threads_per_row(self) -> int:
        return self.cols if self.threads is None else self.threads

    @property
    def schedule_simdgroups_per_row(self) -> int:
        return (self.threads_per_row + _SIMDGROUP_SIZE - 1) // _SIMDGROUP_SIZE

    @property
    def reduction_simdgroups_per_row(self) -> int:
        return (self.cols + _SIMDGROUP_SIZE - 1) // _SIMDGROUP_SIZE

    @property
    def input_bytes(self) -> int:
        return self.rows * self.cols * 4


@dataclass(frozen=True)
class _Timing:
    mean_ms: float
    std_ms: float


def _make_reduce_sum_kernel(length: int):
    @T.prim_func
    def reduce_sum_kernel(A: T.Tensor((length,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=length):
            src = T.alloc_fragment((length,), T.float32)
            dst = T.alloc_fragment((1,), T.float32)
            T.copy(A, src)
            T.reduce_sum(src, dst)
            T.copy(dst, B)

    return reduce_sum_kernel


def _make_row_reduce_sum_kernel(rows: int, cols: int, threads: int | None = None):
    kernel_threads = cols if threads is None else threads

    @T.prim_func
    def row_reduce_sum_kernel(
        A: T.Tensor((rows, cols), T.float32),
        B: T.Tensor((rows,), T.float32),
    ):
        with T.Kernel(rows, threads=kernel_threads) as bx:
            src = T.alloc_fragment((cols,), T.float32)
            dst = T.alloc_fragment((1,), T.float32)
            T.copy(A[bx, 0], src)
            T.reduce_sum(src, dst)
            T.copy(dst, B[bx])

    return row_reduce_sum_kernel


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        pytest.fail(f"{name} must be an integer, got {value!r}")
    if parsed <= 0:
        pytest.fail(f"{name} must be positive, got {parsed}")
    return parsed


def _time_synced_ms(fn, sync, *, warmup: int, iterations: int) -> _Timing:
    for _ in range(warmup):
        fn()
        sync()

    samples = []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    kept = samples[: max(1, int(len(samples) * 0.9))]
    mean = sum(kept) / len(kept)
    variance = sum((x - mean) ** 2 for x in kept) / max(1, len(kept) - 1)
    return _Timing(mean, variance**0.5)


def _time_tilelang_profiler_ms(jit_kernel, input_tensors, *, warmup: int, iterations: int) -> _Timing:
    profiler = jit_kernel.get_profiler()

    def run():
        profiler(*input_tensors)

    return _time_synced_ms(
        run,
        torch.mps.synchronize,
        warmup=warmup,
        iterations=iterations,
    )


def _maybe_skip_optional(case: _ReducePerfCase, stage: str, fn):
    try:
        return fn()
    except AssertionError:
        raise
    except Exception as exc:
        if case.optional:
            case_desc = getattr(case, "label", repr(case))
            pytest.skip(f"optional Metal reduce perf case {case_desc} skipped during {stage}: {type(exc).__name__}: {exc}")
        raise


def _lowered_source(fn) -> str:
    artifact = tilelang.lower(fn, target="metal")
    return artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)


def _detect_row_reduce_load_pattern(src: str) -> str:
    arg_type = "unknown_arg"
    arg_match = re.search(r"\bdevice\s+(?:const\s+)?([A-Za-z0-9_]+)\s*\*\s*A\b", src)
    if arg_match:
        arg_type = f"A_{arg_match.group(1)}_ptr"

    load_types = sorted(set(re.findall(r"\*\(\s*thread\s+([A-Za-z0-9_]+)\s*\*\)\s*\(\s*src\s*\+", src)))
    if load_types:
        return f"{arg_type}__thread_{'_'.join(load_types)}_load"

    if "A[" in src:
        return f"{arg_type}__A_index_load"
    return f"{arg_type}__load_unknown"


def _time_mlx_sum_axis1_ms(values_cpu, *, warmup: int, iterations: int) -> _Timing | None:
    try:
        import mlx.core as mx
    except Exception:
        return None

    values_mx = mx.array(values_cpu.numpy())
    mx.eval(values_mx)

    holder = []

    def run_mlx():
        out = mx.sum(values_mx, axis=1)
        mx.eval(out)
        holder[:] = [out]

    return _time_synced_ms(run_mlx, lambda: None, warmup=warmup, iterations=iterations)


def _time_mlx_sum_ms(values_cpu, *, warmup: int, iterations: int) -> _Timing | None:
    try:
        import mlx.core as mx
    except Exception:
        return None

    values_mx = mx.array(values_cpu.numpy())
    mx.eval(values_mx)

    holder = []

    def run_mlx():
        out = mx.sum(values_mx)
        mx.eval(out)
        holder[:] = [out]

    return _time_synced_ms(run_mlx, lambda: None, warmup=warmup, iterations=iterations)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(_ReducePerfCase(32, "same_simdgroup_32"), id="same_simdgroup_32"),
        pytest.param(_ReducePerfCase(64, "cross_simdgroup_64"), id="cross_simdgroup_64"),
        pytest.param(_ReducePerfCase(128, "cross_simdgroup_128"), id="cross_simdgroup_128"),
        pytest.param(
            _ReducePerfCase(256, "large_cross_simdgroup_256", optional=True),
            id="large_cross_simdgroup_256",
        ),
        pytest.param(
            _ReducePerfCase(512, "large_cross_simdgroup_512", optional=True),
            id="large_cross_simdgroup_512",
        ),
    ],
)
def test_metal_reduce_sum_perf_smoke(case: _ReducePerfCase):
    if os.environ.get(_RUN_ENV) != "1":
        pytest.skip(f"set {_RUN_ENV}=1 to run Metal reduction performance smoke")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        pytest.skip("requires PyTorch MPS for the reference timing")

    warmup = _env_int(_WARMUP_ENV, 3)
    iterations = _env_int(_ITERS_ENV, 20)
    kernel = _maybe_skip_optional(
        case,
        "compile",
        lambda: tilelang.compile(_make_reduce_sum_kernel(case.length), target="metal"),
    )
    values = torch.arange(case.length, dtype=torch.float32).to("mps")
    tilelang_out = torch.empty(1, dtype=torch.float32, device="mps")
    torch_out = torch.empty(1, dtype=torch.float32, device="mps")

    def run_torch_reference():
        torch_out.copy_(values.sum().reshape(1))

    tilelang_inputs = [values, tilelang_out]
    _maybe_skip_optional(
        case,
        "tilelang profiler launch",
        lambda: kernel.get_profiler()(*tilelang_inputs),
    )
    run_torch_reference()
    torch.mps.synchronize()
    torch.testing.assert_close(tilelang_out.cpu(), torch_out.cpu(), atol=1e-5, rtol=1e-5)

    tilelang_timing = _maybe_skip_optional(
        case,
        "tilelang profiler timing",
        lambda: _time_tilelang_profiler_ms(
            kernel,
            tilelang_inputs,
            warmup=warmup,
            iterations=iterations,
        ),
    )
    torch_timing = _time_synced_ms(
        run_torch_reference,
        torch.mps.synchronize,
        warmup=warmup,
        iterations=iterations,
    )
    mlx_timing = _time_mlx_sum_ms(
        values.cpu(),
        warmup=warmup,
        iterations=iterations,
    )
    mlx_summary = (
        "mlx_sum=unavailable"
        if mlx_timing is None
        else (
            f"mlx_sum={mlx_timing.mean_ms:.4f} ms/iter "
            f"mlx_std={mlx_timing.std_ms:.4f} "
            f"tilelang_vs_mlx={tilelang_timing.mean_ms / mlx_timing.mean_ms:.2f}x "
            f"torch_vs_mlx={torch_timing.mean_ms / mlx_timing.mean_ms:.2f}x"
        )
    )
    print(
        "metal_reduce_perf "
        f"case={case.label} "
        f"length={case.length} path={case.path} "
        f"simdgroup_size={_SIMDGROUP_SIZE} simdgroups={case.simdgroups} "
        f"threads={case.length} input_bytes={case.length * 4} "
        f"timer=tilelang_profiler_sync "
        f"tilelang={tilelang_timing.mean_ms:.4f} ms/iter "
        f"tilelang_std={tilelang_timing.std_ms:.4f} "
        f"torch_mps_sum={torch_timing.mean_ms:.4f} ms/iter "
        f"torch_std={torch_timing.std_ms:.4f} "
        f"tilelang_vs_torch={tilelang_timing.mean_ms / torch_timing.mean_ms:.2f}x "
        f"{mlx_summary} "
        f"warmup={warmup} iterations={iterations}"
    )


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _RowReducePerfCase(256, 32, "row_reduce_256x32_no_barrier"),
            id="row_reduce_256x32_no_barrier",
        ),
        pytest.param(
            _RowReducePerfCase(256, 1024, "row_reduce_256x1024"),
            id="row_reduce_256x1024",
        ),
        pytest.param(
            _RowReducePerfCase(1024, 1024, "row_reduce_1024x1024"),
            id="row_reduce_1024x1024",
        ),
        pytest.param(
            _RowReducePerfCase(1024, 1024, "row_reduce_1024x1024_threads128", threads=128),
            id="row_reduce_1024x1024_threads128",
        ),
        pytest.param(
            _RowReducePerfCase(1024, 1024, "row_reduce_1024x1024_threads256", threads=256),
            id="row_reduce_1024x1024_threads256",
        ),
        pytest.param(
            _RowReducePerfCase(1024, 1024, "row_reduce_1024x1024_threads512", threads=512),
            id="row_reduce_1024x1024_threads512",
        ),
        pytest.param(
            _RowReducePerfCase(4096, 1024, "row_reduce_4096x1024", optional=True),
            id="row_reduce_4096x1024",
        ),
        pytest.param(
            _RowReducePerfCase(
                4096,
                1024,
                "row_reduce_4096x1024_threads128",
                threads=128,
                optional=True,
            ),
            id="row_reduce_4096x1024_threads128",
        ),
        pytest.param(
            _RowReducePerfCase(
                4096,
                1024,
                "row_reduce_4096x1024_threads256",
                threads=256,
                optional=True,
            ),
            id="row_reduce_4096x1024_threads256",
        ),
        pytest.param(
            _RowReducePerfCase(
                4096,
                1024,
                "row_reduce_4096x1024_threads512",
                threads=512,
                optional=True,
            ),
            id="row_reduce_4096x1024_threads512",
        ),
    ],
)
def test_metal_row_reduce_sum_perf_smoke(case: _RowReducePerfCase):
    if os.environ.get(_RUN_ENV) != "1":
        pytest.skip(f"set {_RUN_ENV}=1 to run Metal reduction performance smoke")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        pytest.skip("requires PyTorch MPS for the reference timing")

    warmup = _env_int(_WARMUP_ENV, 3)
    iterations = _env_int(_ITERS_ENV, 20)
    fn = _make_row_reduce_sum_kernel(case.rows, case.cols, case.threads)
    src = _maybe_skip_optional(case, "lower", lambda: _lowered_source(fn))
    expected_allreduce = f"AllReduce<tl::SumOp, {case.threads_per_row}"
    has_expected_allreduce = expected_allreduce in src
    has_cross_fast_path = "AllReduceSimdgroupCross" in src
    has_barrier = "threadgroup_barrier" in src
    has_workspace = "workspace" in src
    load_pattern = _detect_row_reduce_load_pattern(src)
    assert has_expected_allreduce, src
    if case.threads_per_row <= _SIMDGROUP_SIZE:
        assert not re.search(
            r"^\s*threadgroup\s+\w+\s+workspace\[\d+\];",
            src,
            re.MULTILINE,
        ), src
        assert "(&(workspace[0]))" not in src, src
    else:
        assert has_cross_fast_path, src
        assert has_barrier, src
        assert has_workspace, src

    kernel = _maybe_skip_optional(
        case,
        "compile",
        lambda: tilelang.compile(fn, target="metal"),
    )
    torch.manual_seed(0xA11CE)
    values_cpu = torch.randn(case.rows, case.cols, dtype=torch.float32)
    values = values_cpu.to("mps")
    tilelang_out = torch.empty(case.rows, dtype=torch.float32, device="mps")
    torch_out = torch.empty(case.rows, dtype=torch.float32, device="mps")

    def run_torch_reference():
        torch_out.copy_(values.sum(dim=1))

    tilelang_inputs = [values, tilelang_out]
    _maybe_skip_optional(
        case,
        "tilelang profiler launch",
        lambda: kernel.get_profiler()(*tilelang_inputs),
    )
    run_torch_reference()
    torch.mps.synchronize()
    torch.testing.assert_close(tilelang_out.cpu(), torch_out.cpu(), atol=1e-4, rtol=1e-4)

    tilelang_timing = _maybe_skip_optional(
        case,
        "tilelang profiler timing",
        lambda: _time_tilelang_profiler_ms(
            kernel,
            tilelang_inputs,
            warmup=warmup,
            iterations=iterations,
        ),
    )
    torch_timing = _time_synced_ms(
        run_torch_reference,
        torch.mps.synchronize,
        warmup=warmup,
        iterations=iterations,
    )
    mlx_timing = _time_mlx_sum_axis1_ms(
        values_cpu,
        warmup=warmup,
        iterations=iterations,
    )
    mlx_summary = (
        "mlx_sum_axis1=unavailable"
        if mlx_timing is None
        else (
            f"mlx_sum_axis1={mlx_timing.mean_ms:.4f} ms/iter "
            f"mlx_std={mlx_timing.std_ms:.4f} "
            f"tilelang_vs_mlx={tilelang_timing.mean_ms / mlx_timing.mean_ms:.2f}x "
            f"torch_vs_mlx={torch_timing.mean_ms / mlx_timing.mean_ms:.2f}x"
        )
    )
    print(
        "metal_row_reduce_perf "
        f"case={case.label} "
        f"shape={case.rows}x{case.cols} "
        f"simdgroup_size={_SIMDGROUP_SIZE} "
        f"reduction_simdgroups_per_row={case.reduction_simdgroups_per_row} "
        f"schedule_simdgroups_per_row={case.schedule_simdgroups_per_row} "
        f"threads_per_row={case.threads_per_row} input_bytes={case.input_bytes} "
        f"codegen_allreduce_simdgroup_cross={int(has_cross_fast_path)} "
        f"codegen_allreduce_threads={case.threads_per_row} "
        f"codegen_threadgroup_barrier={int(has_barrier)} "
        f"codegen_workspace={int(has_workspace)} "
        f"codegen_vectorized_load_pattern={load_pattern} "
        f"timer=tilelang_profiler_sync "
        f"tilelang={tilelang_timing.mean_ms:.4f} ms/iter "
        f"tilelang_std={tilelang_timing.std_ms:.4f} "
        f"torch_mps_sum_axis1={torch_timing.mean_ms:.4f} ms/iter "
        f"torch_std={torch_timing.std_ms:.4f} "
        f"tilelang_vs_torch={tilelang_timing.mean_ms / torch_timing.mean_ms:.2f}x "
        f"{mlx_summary} "
        f"warmup={warmup} iterations={iterations}"
    )


if __name__ == "__main__":
    tilelang.testing.main()
