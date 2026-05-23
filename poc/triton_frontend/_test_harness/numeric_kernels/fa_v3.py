"""fa_v3: real Hopper FA-v3 kernel (TMA + WGMMA + warp specialization).

Implements RFC §5.5 item 6 -- the FA-v3 conformance row. Distinct from
``flash_attention.py`` (FA-v2 with plain pointer arithmetic) in three
ways:

* **TMA**: K/V tiles are loaded via ``tl.make_tensor_descriptor`` so the
  TTIR carries ``tt.descriptor_load`` ops. The frontend dispatches those
  to ``T.tma_copy`` on Hopper and to a pointer-arith ``T.copy`` fallback
  elsewhere (RFC §5.4).
* **WGMMA hint**: ``tl.dot(..., out_dtype=tl.float32)`` with
  ``num_warps=8`` so Hopper backends emit ``wgmma.mma_async`` rather than
  the SM80 ``mma`` instruction.
* **Warp specialization (WS)**: ``num_stages=3`` pipeline metadata so
  the producer/consumer warpgroup split is observable in TTIR.

On non-Hopper hosts (this dev Mac, AMD, older NV) the same TTIR lowers
through the pointer-arith TMA fallback; the numeric output matches the
pure-numpy reference within FA tolerances. This is the same code path
the cppmega bridge will exercise once the Hopper hardware tier is wired.
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


SEQ_Q = 64
SEQ_K = 64
HEAD_DIM = 16
BLOCK_M = 16
BLOCK_N = 16
BLOCK_DMODEL = 16
NUM_STAGES = 3  # Hopper pipeline depth.
NUM_WARPS = 8  # Triggers WGMMA on Hopper backends.
SM_SCALE = 0.17677669529663687  # 1.0 / sqrt(32)

LAUNCH_GRID: Tuple[int, ...] = (SEQ_Q // BLOCK_M,)
META_ARGS: dict = {
    "SEQ_Q": SEQ_Q,
    "SEQ_K": SEQ_K,
    "HEAD_DIM": HEAD_DIM,
    "BLOCK_M": BLOCK_M,
    "BLOCK_N": BLOCK_N,
    "BLOCK_DMODEL": BLOCK_DMODEL,
    "NUM_STAGES": NUM_STAGES,
}
# Autotune knobs surfaced separately so the harness can stamp the
# matching ``threadIdx.x = NUM_WARPS * warp_size`` AttrStmt on the
# lowered PrimFunc without polluting ``META_ARGS`` (Triton 3.6's
# ``ASTSource(constexprs=...)`` rejects unknown keys). TileLang's
# ``gemm.lower`` reads ``num_warps`` from that AttrStmt to pick MMA
# shape on Hopper.
TRITON_AUTOTUNE_OPTS: dict = {
    "num_warps": NUM_WARPS,
    "num_stages": NUM_STAGES,
}
TTIR_SIGNATURE: dict = {
    "q_ptr": "*fp32",
    "k_ptr": "*fp32",
    "v_ptr": "*fp32",
    "o_ptr": "*fp32",
    "SEQ_Q": "constexpr",
    "SEQ_K": "constexpr",
    "HEAD_DIM": "constexpr",
    "BLOCK_M": "constexpr",
    "BLOCK_N": "constexpr",
    "BLOCK_DMODEL": "constexpr",
    "NUM_STAGES": "constexpr",
}

ATOL = 1e-2
RTOL = 1e-2


if triton is not None:

    @triton.jit
    def _fa_v3_hopper_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr,
        SEQ_Q: tl.constexpr,
        SEQ_K: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        NUM_STAGES: tl.constexpr,
    ):
        pid_m = tl.program_id(0)

        # TMA descriptors for Q/K/V. The strides match a contiguous
        # (seq, head_dim) layout. On Hopper these lower to cp.async.bulk
        # over CUTE descriptors; elsewhere the frontend rewrites the
        # descriptor loads as pointer-arith T.copy.
        q_desc = tl.make_tensor_descriptor(
            q_ptr,
            [SEQ_Q, HEAD_DIM],
            [HEAD_DIM, 1],
            [BLOCK_M, BLOCK_DMODEL],
        )
        k_desc = tl.make_tensor_descriptor(
            k_ptr,
            [SEQ_K, HEAD_DIM],
            [HEAD_DIM, 1],
            [BLOCK_N, BLOCK_DMODEL],
        )
        v_desc = tl.make_tensor_descriptor(
            v_ptr,
            [SEQ_K, HEAD_DIM],
            [HEAD_DIM, 1],
            [BLOCK_N, BLOCK_DMODEL],
        )

        offs_m = pid_m * BLOCK_M

        q = q_desc.load([offs_m, 0])

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in tl.range(0, SEQ_K, BLOCK_N, num_stages=NUM_STAGES):
            k = k_desc.load([start_n, 0])
            v = v_desc.load([start_n, 0])

            # WGMMA-shaped accumulator: tl.dot with out_dtype=float32 is
            # the canonical Hopper trigger for wgmma.mma_async.
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * 0.17677669529663687

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            acc = acc * alpha[:, None]
            acc += tl.dot(p, v, out_dtype=tl.float32)

            m_i = m_ij

        acc = acc / l_i[:, None]

        # Output via plain pointer arithmetic; the FA-v3 spec puts the
        # epilogue store in the producer warpgroup, while the TMA
        # descriptor path is reserved for K/V loads.
        offs_n_d = tl.arange(0, BLOCK_DMODEL)
        o_ptrs = (
            o_ptr
            + (offs_m + tl.arange(0, BLOCK_M))[:, None] * HEAD_DIM
            + offs_n_d[None, :]
        )
        tl.store(o_ptrs, acc)

    TRITON_KERNEL: Callable[..., Any] = _fa_v3_hopper_kernel
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
    qk = (q @ k.T) * SM_SCALE
    m = np.max(qk, axis=-1, keepdims=True)
    p = np.exp(qk - m)
    p = p / np.sum(p, axis=-1, keepdims=True)
    return (p @ v).astype(q.dtype)
