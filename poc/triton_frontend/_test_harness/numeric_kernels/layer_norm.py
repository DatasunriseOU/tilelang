"""layer_norm: per-row LayerNorm on (4, 128) with weight + bias.

Exercises the union of softmax's reduce surface (mean + variance) and
elementwise affine (`x_hat * w + b`). One launch per row.

Source: synthetic. Mirrors the canonical Triton LayerNorm reference
(see Triton tutorial ``05-layer-norm.py``) collapsed to the forward
pass without RMS / fused-residual variants. We could not lift directly
from nanochat -- its triton tree under ``nanochat/`` does not contain
a standalone layer_norm forward kernel (only fused-MoE, fused-MLA-rope,
and mamba3-trapezoidal kernels).
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple

import numpy as np

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:  # pragma: no cover -- triton optional
    triton = None  # type: ignore
    tl = None  # type: ignore


N_ROWS = 4
N_COLS = 128
BLOCK_SIZE = 128
EPS = 1e-5

LAUNCH_GRID: Tuple[int, ...] = (N_ROWS,)
META_ARGS: dict = {"BLOCK_SIZE": BLOCK_SIZE, "eps": EPS}

ATOL = 1e-4
RTOL = 1e-3


if triton is not None:

    @triton.jit
    def _layer_norm_fwd_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        out_ptr,
        x_row_stride,
        out_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: "tl.constexpr",
    ):
        row = tl.program_id(axis=0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x_row_ptrs = x_ptr + row * x_row_stride + col_offsets
        x = tl.load(x_row_ptrs, mask=mask, other=0.0)
        mean = tl.sum(x, axis=0) / n_cols
        x_centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(x_centered * x_centered, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
        b = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)
        y = x_centered * rstd * w + b
        out_row_ptrs = out_ptr + row * out_row_stride + col_offsets
        tl.store(out_row_ptrs, y, mask=mask)

    TRITON_KERNEL: Callable[..., Any] = _layer_norm_fwd_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(
    *, dtype: str = "float32", seed: int = 0
) -> Tuple[List[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_ROWS, N_COLS)).astype(dtype)
    w = rng.standard_normal(N_COLS).astype(dtype)
    b = rng.standard_normal(N_COLS).astype(dtype)
    out = np.zeros_like(x)
    expected = _layer_norm_np(x, w, b, EPS)
    return [x, w, b, out], expected


def _layer_norm_np(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float
) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return (x_hat * w + b).astype(x.dtype)


def numpy_reference(args: List[np.ndarray]) -> np.ndarray:
    x, w, b, _out = args
    return _layer_norm_np(x, w, b, EPS)
