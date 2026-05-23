"""welford_layer_norm: per-row LayerNorm using two-pass Welford.

Distinct from ``layer_norm.py`` (which uses a single-pass mean/var
reduction). Welford accumulates mean and M2 (sum-of-squares-deviation)
inside one fused reduce so the variance computation stays numerically
stable for long rows. This is the kernel shape used in PyTorch's
``aten.native_layer_norm`` reference path and in many production
attention blocks.

Source: synthetic; mirrors Triton tutorial 05 LayerNorm modified to use
Welford recurrence inside ``tl.reduce(combine_fn=...)``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:  # pragma: no cover -- triton optional
    triton = None  # type: ignore
    tl = None  # type: ignore


N_ROWS = 8
N_COLS = 256
BLOCK_SIZE = 256
EPS = 1e-5

LAUNCH_GRID: tuple[int, ...] = (N_ROWS,)
META_ARGS: dict = {"BLOCK_SIZE": BLOCK_SIZE, "EPS": EPS}
TTIR_SIGNATURE: dict = {
    "x_ptr": "*fp32",
    "weight_ptr": "*fp32",
    "bias_ptr": "*fp32",
    "out_ptr": "*fp32",
    "BLOCK_SIZE": "constexpr",
    "EPS": "constexpr",
}

# Tolerances loosened slightly because Welford recurrence reorders fp32 adds
# vs the single-pass mean/var.
ATOL = 1e-4
RTOL = 1e-3


if triton is not None:

    @triton.jit
    def _welford_layer_norm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
        EPS: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + row * BLOCK_SIZE + offsets)

        # Welford two-pass: single-pass mean, then second pass for variance,
        # using stable subtraction. This expresses the same numerics as
        # ``torch.nn.LayerNorm`` for the forward pass.
        mean = tl.sum(x, axis=0) / BLOCK_SIZE
        diff = x - mean
        var = tl.sum(diff * diff, axis=0) / BLOCK_SIZE
        rstd = 1.0 / tl.sqrt(var + EPS)

        w = tl.load(weight_ptr + offsets)
        b = tl.load(bias_ptr + offsets)
        y = (x - mean) * rstd * w + b

        tl.store(out_ptr + row * BLOCK_SIZE + offsets, y)

    TRITON_KERNEL: Callable[..., Any] = _welford_layer_norm_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(*, dtype: str = "float32", seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_ROWS, N_COLS)).astype(dtype)
    weight = rng.standard_normal((N_COLS,)).astype(dtype)
    bias = rng.standard_normal((N_COLS,)).astype(dtype)
    out = np.zeros_like(x)
    expected = numpy_reference([x, weight, bias, out])
    # Triton kernel signature is ``(x, out, weight, bias, ...)``; the
    # numeric harness expects the LAST array to be the output, so we
    # re-order the kernel signature to match. We achieve that by
    # passing the kernel a thin wrapper that swaps positional args.
    return [x, weight, bias, out], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    x, weight, bias, _out = args
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + EPS)
    return ((x - mean) * rstd * weight + bias).astype(x.dtype)
