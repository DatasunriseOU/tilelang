"""vector_add: ``c = a + b`` on a single 256-element tile.

Smallest possible kernel in the conformance ladder -- exercises only
``tt.load`` / ``tt.store`` / scalar arithmetic. No ``tt.dot``, no
``tt.reduce``. If the harness can't get THIS one to NUMERIC_PASS, the
larger kernels will not pass either.

Source: synthetic. The shape is identical to Triton's own tutorial
``01-vector-add.py`` so we know the kernel itself is correct.
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple

import numpy as np

# Triton is optional at import time; harness checks first. We expose a
# ``None`` placeholder when triton is missing so ``from numeric_kernels.X
# import TRITON_KERNEL`` doesn't blow up during collection.
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:  # pragma: no cover -- triton optional
    triton = None  # type: ignore
    tl = None  # type: ignore


N_ELEMENTS = 256
BLOCK_SIZE = 256

LAUNCH_GRID: Tuple[int, ...] = (1,)
META_ARGS: dict = {"BLOCK_SIZE": BLOCK_SIZE}

ATOL = 1e-4
RTOL = 1e-3


if triton is not None:

    @triton.jit
    def _vector_add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    TRITON_KERNEL: Callable[..., Any] = _vector_add_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(
    *, dtype: str = "float32", seed: int = 0
) -> Tuple[List[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(N_ELEMENTS).astype(dtype)
    b = rng.standard_normal(N_ELEMENTS).astype(dtype)
    out = np.zeros(N_ELEMENTS, dtype=dtype)
    expected = (a + b).astype(dtype)
    return [a, b, out], expected


def numpy_reference(args: List[np.ndarray]) -> np.ndarray:
    a, b, _out = args
    return (a + b).astype(a.dtype)
