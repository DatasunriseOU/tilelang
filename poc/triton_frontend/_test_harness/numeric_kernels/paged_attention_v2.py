"""paged_attention_v2: vLLM-style multi-block paged attention.

Extends ``paged_attention.py`` with multi-query-block (M-axis) tiling so
each program writes one (BLOCK_M x HEAD_DIM) tile of the output. The
streaming softmax is performed over a sequence of physical KV pages
selected by ``block_table_ptr`` (the indirection layer vLLM uses to
amortize KV-cache fragmentation).

Source: synthetic; mirrors vLLM's ``triton_paged_attention_v2`` shape
(see vllm/csrc/attention/attention_kernels.cu and the Triton port
maintained as ``ops/paged_attn_v2.py``).
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


SEQ_Q = 32
NUM_PAGES = 4
PAGE_SIZE = 16
HEAD_DIM = 32
NUM_PHYS_BLOCKS = 6
BLOCK_M = 16
BLOCK_N = PAGE_SIZE
BLOCK_DMODEL = HEAD_DIM
SM_SCALE = 0.17677669529663687  # 1.0 / sqrt(32)

LAUNCH_GRID: tuple[int, ...] = (SEQ_Q // BLOCK_M,)
META_ARGS: dict = {
    "BLOCK_M": BLOCK_M,
    "BLOCK_N": BLOCK_N,
    "BLOCK_DMODEL": BLOCK_DMODEL,
    "NUM_PAGES": NUM_PAGES,
    "PAGE_SIZE": PAGE_SIZE,
}
TTIR_SIGNATURE: dict = {
    "q_ptr": "*fp32",
    "k_cache_ptr": "*fp32",
    "v_cache_ptr": "*fp32",
    "block_table_ptr": "*i32",
    "o_ptr": "*fp32",
    "BLOCK_M": "constexpr",
    "BLOCK_N": "constexpr",
    "BLOCK_DMODEL": "constexpr",
    "NUM_PAGES": "constexpr",
    "PAGE_SIZE": "constexpr",
}

ATOL = 2e-2
RTOL = 2e-2


if triton is not None:

    @triton.jit
    def _paged_attention_v2_kernel(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_table_ptr,
        o_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        NUM_PAGES: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
    ):
        m_block = tl.program_id(axis=0)
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q = tl.load(
            q_ptr + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :],
        )
        m_i = tl.zeros([BLOCK_M], tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

        offs_n = tl.arange(0, BLOCK_N)
        for logical_page in range(0, NUM_PAGES):
            phys_block = tl.load(block_table_ptr + logical_page)
            cache_base = phys_block * PAGE_SIZE * BLOCK_DMODEL
            page_offsets = (
                cache_base
                + offs_n[:, None] * BLOCK_DMODEL
                + offs_d[None, :]
            )
            k = tl.load(k_cache_ptr + page_offsets)
            v = tl.load(v_cache_ptr + page_offsets)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            qk = qk * 0.17677669529663687

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            acc = acc * alpha[:, None]
            acc += tl.dot(p, v)

            m_i = m_ij

        acc = acc / l_i[:, None]
        tl.store(
            o_ptr + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :],
            acc,
        )

    TRITON_KERNEL: Callable[..., Any] = _paged_attention_v2_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(*, dtype: str = "float32", seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((SEQ_Q, HEAD_DIM)).astype(dtype)
    k_cache = rng.standard_normal((NUM_PHYS_BLOCKS, PAGE_SIZE, HEAD_DIM)).astype(dtype)
    v_cache = rng.standard_normal((NUM_PHYS_BLOCKS, PAGE_SIZE, HEAD_DIM)).astype(dtype)
    # Pick a logical block-table mapping (per-test deterministic): pages
    # are referenced in a non-monotonic order to ensure the gather actually
    # exercises indirection.
    block_table = np.array([4, 1, 5, 2], dtype=np.int32)[:NUM_PAGES]
    out = np.zeros((SEQ_Q, HEAD_DIM), dtype=dtype)
    expected = numpy_reference([q, k_cache, v_cache, block_table, out])
    return [q, k_cache, v_cache, block_table, out], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    q, k_cache, v_cache, block_table, _out = args
    k_full = k_cache[block_table].reshape(-1, HEAD_DIM)
    v_full = v_cache[block_table].reshape(-1, HEAD_DIM)
    scores = (q @ k_full.T) * SM_SCALE
    m = scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores - m)
    weights /= weights.sum(axis=-1, keepdims=True)
    return (weights @ v_full).astype(q.dtype)
