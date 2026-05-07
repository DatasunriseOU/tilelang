"""Conformance suite for the Triton -> TileLang TIR frontend (POC scaffold).

Implements the kernel list in ``RFC_unified_fused_kernel.md`` section 5.5
in ascending difficulty:

    1. vector_add   -- mask + program_id
    2. softmax      -- reduce + broadcast
    3. matmul       -- dot + multi-stage load
    4. layer_norm   -- Triton tutorial 05 (Welford, two-pass)
    5. fa_v2        -- Triton tutorial 06 (pipelined dot + softmax)
    6. fa_v3        -- Hopper-specific (TMA + WGMMA + WS); gate behind TMA fallback
    7. paged_attn   -- ported from vLLM

Source-of-truth ports: ``microsoft/triton-shared/python/examples/`` and
the Triton tutorial repo. Each ``kernel_<name>`` here is a placeholder;
the real port copies the Triton kernel verbatim, runs it through
:func:`triton_frontend.from_triton_kernel`, executes the resulting
``PrimFunc`` against a reference, and asserts numerical agreement.

Running
-------
The suite is intended to be driven by pytest once stubs are filled in::

    pytest poc/triton_frontend/conformance/

Adding a new kernel
-------------------
1. Add ``def kernel_<name>(): pass`` here.
2. Register it in :data:`KERNELS`.
3. Drop the source under ``conformance/<name>.py`` once the port lands.
"""
from __future__ import annotations

from typing import Callable, Dict

__all__ = [
    "KERNELS",
    "kernel_vector_add",
    "kernel_softmax",
    "kernel_matmul",
    "kernel_layer_norm",
    "kernel_fa_v2",
    "kernel_fa_v3",
    "kernel_paged_attn",
]


def kernel_vector_add() -> None:
    """RFC 5.5 #1: masked elementwise add over a 1D grid."""
    pass  # TODO: port from microsoft/triton-shared/python/examples/


def kernel_softmax() -> None:
    """RFC 5.5 #2: row-wise softmax (reduce_max, broadcast, reduce_sum)."""
    pass  # TODO: port from microsoft/triton-shared/python/examples/


def kernel_matmul() -> None:
    """RFC 5.5 #3: tiled matmul with multi-stage software pipelining."""
    pass  # TODO: port from microsoft/triton-shared/python/examples/


def kernel_layer_norm() -> None:
    """RFC 5.5 #4: two-pass Welford layer-norm (Triton tutorial 05)."""
    pass  # TODO: port from microsoft/triton-shared/python/examples/


def kernel_fa_v2() -> None:
    """RFC 5.5 #5: FlashAttention-2 (pipelined dot + online softmax)."""
    pass  # TODO: port from microsoft/triton-shared/python/examples/


def kernel_fa_v3() -> None:
    """RFC 5.5 #6: FlashAttention-3 (Hopper TMA + WGMMA + WS).

    Gated behind the TMA fallback path (RFC section 5.4).
    """
    pass  # TODO: port from microsoft/triton-shared/python/examples/


def kernel_paged_attn() -> None:
    """RFC 5.5 #7: paged-attention, ported from vLLM."""
    pass  # TODO: port from microsoft/triton-shared/python/examples/


KERNELS: Dict[str, Callable[[], None]] = {
    "vector_add": kernel_vector_add,
    "softmax": kernel_softmax,
    "matmul": kernel_matmul,
    "layer_norm": kernel_layer_norm,
    "fa_v2": kernel_fa_v2,
    "fa_v3": kernel_fa_v3,
    "paged_attn": kernel_paged_attn,
}
