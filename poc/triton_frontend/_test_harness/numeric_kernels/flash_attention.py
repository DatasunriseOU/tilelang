"""flash_attention: simplified Flash Attention v2 forward kernel.

Smallest flash attention tile that exercises the loop, scaling, maximum,
exp, sum and inner dot products. We use SEQ_Q = SEQ_K = 128 and
BLOCK_M = BLOCK_N = BLOCK_K = 64 so the kernel runs a couple of tiles.

Source: synthetic, modeled on Triton's tutorial ``06-fused-attention.py``.
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


SEQ_Q = 128
SEQ_K = 128
HEAD_DIM = 64
BLOCK_M = 64
BLOCK_N = 64
BLOCK_DMODEL = 64

LAUNCH_GRID: Tuple[int, ...] = (SEQ_Q // BLOCK_M,)
META_ARGS: dict = {
    "BLOCK_M": BLOCK_M,
    "BLOCK_N": BLOCK_N,
    "BLOCK_DMODEL": BLOCK_DMODEL,
}

ATOL = 1e-2
RTOL = 1e-2

# PrimFunc scalar args: arg4..arg12 map to the scalar parameters.
KERNEL_SCALAR_ARGS: dict = {
    "arg4": SEQ_K,       # seq_k
    "arg5": HEAD_DIM,    # stride_qm = HEAD_DIM
    "arg6": 1,           # stride_qd = 1
    "arg7": HEAD_DIM,    # stride_kn = HEAD_DIM
    "arg8": 1,           # stride_kd = 1
    "arg9": HEAD_DIM,    # stride_vn = HEAD_DIM
    "arg10": 1,          # stride_vd = 1
    "arg11": HEAD_DIM,   # stride_om = HEAD_DIM
    "arg12": 1,          # stride_od = 1
}

if triton is not None:

    @triton.jit
    def _flash_attention_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr,
        seq_k,
        stride_qm, stride_qd,
        stride_kn, stride_kd,
        stride_vn, stride_vd,
        stride_om, stride_od,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        q = tl.load(q_ptrs)

        for start_n in range(0, seq_k, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k = tl.load(k_ptrs + start_n * stride_kn)
            v = tl.load(v_ptrs + start_n * stride_vn)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            qk = qk * 0.125  # sm_scale = 1.0 / sqrt(64)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            acc = acc * alpha[:, None]
            acc += tl.dot(p, v)

            m_i = m_ij

        acc = acc / l_i[:, None]
        tl.store(o_ptrs, acc)

    TRITON_KERNEL: Callable[..., Any] = _flash_attention_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(
    *, dtype: str = "float32", seed: int = 0
) -> Tuple[List[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((SEQ_Q, HEAD_DIM)).astype(dtype)
    k = rng.standard_normal((SEQ_K, HEAD_DIM)).astype(dtype)
    v = rng.standard_normal((SEQ_K, HEAD_DIM)).astype(dtype)
    out = np.zeros((SEQ_Q, HEAD_DIM), dtype=dtype)
    
    expected = numpy_reference([q, k, v, out])
    return [q, k, v, out], expected


def numpy_reference(args: List[np.ndarray]) -> np.ndarray:
    q, k, v, _out = args
    qk = (q @ k.T) * 0.125
    # Numerically stable softmax
    m = np.max(qk, axis=-1, keepdims=True)
    p = np.exp(qk - m)
    l = np.sum(p, axis=-1, keepdims=True)
    p = p / l
    expected = (p @ v).astype(q.dtype)
    return expected
