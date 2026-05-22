"""split_join: Triton ``tl.join`` + ``tl.split`` numeric smoke."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:  # pragma: no cover -- triton optional
    triton = None  # type: ignore
    tl = None  # type: ignore


N = 64

LAUNCH_GRID: tuple[int, ...] = (1,)
META_ARGS: dict = {"N": N}
TTIR_SIGNATURE: dict = {
    "a_ptr": "*fp32",
    "b_ptr": "*fp32",
    "out_ptr": "*fp32",
    "N": "constexpr",
}

ATOL = 1e-4
RTOL = 1e-4


if triton is not None:

    @triton.jit
    def _split_join_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
        offsets = tl.arange(0, N)
        a = tl.load(a_ptr + offsets)
        b = tl.load(b_ptr + offsets)
        joined = tl.join(a, b)
        left, right = tl.split(joined)
        tl.store(out_ptr + offsets, left + right * 2.0)

    TRITON_KERNEL: Callable[..., Any] = _split_join_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(
    *, dtype: str = "float32", seed: int = 0
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(N).astype(dtype)
    b = rng.standard_normal(N).astype(dtype)
    out = np.zeros((N,), dtype=dtype)
    expected = numpy_reference([a, b, out])
    return [a, b, out], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    a, b, _out = args
    return (a + b * 2.0).astype(a.dtype)
