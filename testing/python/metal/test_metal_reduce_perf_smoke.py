"""Opt-in Metal reduction performance smoke tests.

Run manually with:

    TILELANG_RUN_METAL_REDUCE_BENCH=1 python -m pytest \
        testing/python/metal/test_metal_reduce_perf_smoke.py -s -q

The timings include a synchronization per iteration.  That keeps this smoke
test rerunnable with TVM's Metal runtime and makes the output useful for
relative comparisons across same-simdgroup and cross-simdgroup reductions.
"""

import os
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


def _time_mps_ms(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
        torch.mps.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def _maybe_skip_optional(case: _ReducePerfCase, stage: str, fn):
    try:
        return fn()
    except AssertionError:
        raise
    except Exception as exc:
        if case.optional:
            pytest.skip(
                f"optional Metal reduce perf case length={case.length} "
                f"skipped during {stage}: {type(exc).__name__}: {exc}"
            )
        raise


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

    def run_tilelang():
        kernel(values, tilelang_out)

    def run_torch_reference():
        torch_out.copy_(values.sum().reshape(1))

    _maybe_skip_optional(case, "tilelang runtime launch", run_tilelang)
    run_torch_reference()
    torch.mps.synchronize()
    torch.testing.assert_close(tilelang_out.cpu(), torch_out.cpu(), atol=1e-5, rtol=1e-5)

    tilelang_ms = _maybe_skip_optional(
        case,
        "tilelang timing",
        lambda: _time_mps_ms(run_tilelang, warmup=warmup, iterations=iterations),
    )
    torch_ms = _time_mps_ms(run_torch_reference, warmup=warmup, iterations=iterations)
    print(
        "metal_reduce_perf "
        f"case={case.label} "
        f"length={case.length} path={case.path} "
        f"simdgroup_size={_SIMDGROUP_SIZE} simdgroups={case.simdgroups} "
        f"threads={case.length} input_bytes={case.length * 4} "
        f"tilelang={tilelang_ms:.4f} ms/iter "
        f"torch_mps_sum={torch_ms:.4f} ms/iter "
        f"tilelang_vs_torch={tilelang_ms / torch_ms:.2f}x "
        f"warmup={warmup} iterations={iterations}"
    )


if __name__ == "__main__":
    tilelang.testing.main()
