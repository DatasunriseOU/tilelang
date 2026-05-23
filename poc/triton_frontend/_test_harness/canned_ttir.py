"""Canned TTIR text fixtures for the reducer baseline corpus.

The orchestrator tries to compile real ``@triton.jit`` kernels via
:func:`jit_to_ttir.triton_jit_to_ttir` first; when Triton is not
importable in the current environment we fall back to these inline
fixtures. Each fixture is a compact, hand-crafted TTIR snippet that
mirrors what Triton's compiler would emit for a small well-known
kernel. The intent is to drive the reducer's text walker (which only
inspects op *names*) and the MLIR walker (which inspects operands /
attributes) when ``mlir.ir`` is available.

The fixtures here are deliberately *minimal*: each one exercises a
specific subset of OP_TABLE entries so the resulting "ops needed"
frequency table reflects the reducer's actual coverage gaps rather
than artefacts of an over-engineered TTIR transcription. For real
correctness verification the orchestrator separately runs the live
Triton compiler when it is available.

Adding a new kernel
-------------------
1. Pick a real kernel from the survey (paths in ``~/sources/nanochat``
   / ``~/sources/cppmega``).
2. Hand-transcribe the ops it would lower to in TTIR -- you only need
   the ``tt.<op>`` lines, not full SSA fidelity. See
   :data:`CANNED_TTIR_FIXTURES`.
3. Tag it with a ``source`` URI so the report can cite where the
   pattern came from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

__all__ = ["CannedKernel", "CANNED_TTIR_FIXTURES"]


def _read_capture(name: str) -> str:
    """Load a real captured TTIR text from the ``ttir_captures/`` dir.

    Files in ``poc/triton_frontend/_test_harness/ttir_captures/`` are the
    canonical place for *real* (Triton-compiler emitted) TTIR fixtures. We
    keep them on disk rather than as Python string literals because the
    upstream Triton TTIR printer wraps locations / debug info that bloats
    multi-line string escapes (the FLA chunk-delta-h TTIR is ~29 KB).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "ttir_captures", name), "r", encoding="utf-8") as fh:
        return fh.read()


@dataclass(frozen=True)
class CannedKernel:
    """One canned corpus entry."""

    name: str
    """Short stable identifier used as the report row key."""

    description: str
    """Human-readable summary that lands in the report."""

    source: str
    """Citation string -- e.g. ``"triton tutorial 01"`` or path."""

    ttir_text: str
    """MLIR TTIR text accepted by ``poc.triton_frontend.from_ttir``."""

    constexprs: Optional[dict] = None
    """Compile-time constants if/when running through the live compiler."""

    live_kernel_module: Optional[str] = None
    """Optional numeric_kernels module name used to capture real TTIR."""

    live_kernel_attr: str = "TRITON_KERNEL"
    """Attribute on ``live_kernel_module`` that holds the @triton.jit kernel."""

    live_meta_args_attr: str = "META_ARGS"
    """Attribute on ``live_kernel_module`` that holds constexpr bindings."""

    live_signature_attr: str = "TTIR_SIGNATURE"
    """Attribute on ``live_kernel_module`` that holds Triton arg types."""


# ---------------------------------------------------------------------------
# Fixtures
#
# Each TTIR string follows the surface produced by Triton's TTIR printer:
# ``tt.func`` outer scaffold, ``tt.<op>`` lines for the body, ``tt.return``
# at the end. SSA placeholder names (``%0``, ``%1``, ...) are uniqued
# locally per kernel; the text walker only inspects op tokens, not edges.
#
# Why so terse?  The text walker (when ``mlir.ir`` is unavailable) only
# checks OP_TABLE membership of each op name. The MLIR walker (when
# ``mlir.ir`` is available) parses the same text and dispatches via the
# OP_TABLE emitters. We need just enough structure for both walkers to
# enumerate the relevant ops.
# ---------------------------------------------------------------------------


_VECTOR_ADD_TTIR = """\
module {
  tt.func @vector_add(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %2 = tt.splat %n : (i32) -> tensor<128xi32>
    %3 = tt.load %x : tensor<128xf32>
    %4 = tt.load %y : tensor<128xf32>
    %5 = tt.broadcast %3 : tensor<128xf32>
    tt.store %out, %5 : tensor<128xf32>
    tt.return
  }
}
"""


_SOFTMAX_TTIR = """\
module {
  tt.func @softmax_row(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %2 = tt.load %x : tensor<256xf32>
    %3 = tt.reduce %2 {axis = 0 : i32, combiner = "max"} : tensor<256xf32> -> f32
    %4 = tt.splat %3 : (f32) -> tensor<256xf32>
    %5 = tt.broadcast %4 : tensor<256xf32>
    %6 = tt.reduce %2 {axis = 0 : i32, combiner = "add"} : tensor<256xf32> -> f32
    tt.store %y, %2 : tensor<256xf32>
    tt.return
  }
}
"""


_MATMUL_TTIR = """\
module {
  tt.func @matmul(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f32>, %m: i32, %n: i32, %k: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.get_program_id {axis = 1 : i32} : i32
    %2 = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %3 = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %5 = tt.expand_dims %3 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %6 = tt.broadcast %4 : tensor<64x1xi32> -> tensor<64x32xi32>
    %7 = tt.broadcast %5 : tensor<1x32xi32> -> tensor<64x32xi32>
    %8 = tt.load %a : tensor<64x32xf16>
    %9 = tt.load %b : tensor<32x64xf16>
    %10 = tt.dot %8, %9 : tensor<64x32xf16> x tensor<32x64xf16> -> tensor<64x64xf32>
    tt.store %c, %10 : tensor<64x64xf32>
    tt.return
  }
}
"""


_REDUCTION_TTIR = """\
module {
  tt.func @row_sum(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %2 = tt.load %x : tensor<256xf32>
    %3 = tt.reduce %2 {axis = 0 : i32, combiner = "add"} : tensor<256xf32> -> f32
    tt.store %y, %3 : f32
    tt.return
  }
}
"""


_LAYER_NORM_TTIR = """\
module {
  tt.func @layer_norm(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %g: !tt.ptr<f32>, %b: !tt.ptr<f32>, %n: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %2 = tt.load %x : tensor<256xf32>
    %3 = tt.reduce %2 {axis = 0 : i32, combiner = "add"} : tensor<256xf32> -> f32
    %4 = tt.splat %3 : (f32) -> tensor<256xf32>
    %5 = tt.broadcast %4 : tensor<256xf32>
    %6 = tt.reduce %2 {axis = 0 : i32, combiner = "add"} : tensor<256xf32> -> f32
    %7 = tt.load %g : tensor<256xf32>
    %8 = tt.load %b : tensor<256xf32>
    %9 = tt.where %2, %7, %8 : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>
    tt.store %y, %9 : tensor<256xf32>
    tt.return
  }
}
"""


# Real-world flavoured fixture: stripped TTIR pattern derived from
# nanochat's ``_gather_rows_3d_kernel`` (gather + scatter via masked
# load/store). Covers the gather load + masked store path that the
# trivial vector_add doesn't.
_GATHER_3D_TTIR = """\
module {
  tt.func @gather_rows_3d(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %dst: !tt.ptr<f32>, %n: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %2 = tt.load %idx : tensor<128xi32>
    %3 = tt.load %src : tensor<128xf32>
    %4 = tt.splat %0 : (i32) -> tensor<128xi32>
    tt.store %dst, %3 : tensor<128xf32>
    tt.return
  }
}
"""


# Real-world flavoured fixture: tt.atomic_rmw histogram pattern (one of
# the cppmega kernels). Locks in the atomic_rmw + reduce_sum coverage
# the survey kernels exercise.
_ATOMIC_HIST_TTIR = """\
module {
  tt.func @atomic_hist(%x: !tt.ptr<i32>, %hist: !tt.ptr<i32>, %n: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %2 = tt.load %x : tensor<128xi32>
    %3 = tt.atomic_rmw %hist, %2 {rmw_op = "add"} : tensor<128xi32>
    tt.return
  }
}
"""


# FLA gated-delta-rule motif: ``tl.dot`` accumulator chain followed by an
# elementwise ``tl.math.exp2`` scaling lane (the
# ``exp2(b_g_last - b_g)`` mask used in
# ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``). The full FLA kernel
# is several hundred lines of pipelined load/dot/scale; this fixture is
# the structural minimum that exercises the same op cohort
# (``tt.dot`` + ``math.exp2`` + ``tt.load`` / ``tt.store``) so the OP_TABLE
# membership probe in ``run_corpus`` reports LOWERED_DEGRADED rather than
# FAILED_OPS.
_FLA_DOT_EXP2_TTIR = """\
module {
  tt.func @fla_dot_exp2(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f32>) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.load %a : tensor<32x32xf16>
    %2 = tt.load %b : tensor<32x32xf16>
    %3 = tt.dot %1, %2 : tensor<32x32xf16> x tensor<32x32xf16> -> tensor<32x32xf32>
    %4 = math.exp2 %3 : tensor<32x32xf32>
    tt.store %c, %4 : tensor<32x32xf32>
    tt.return
  }
}
"""


# Async copy + barrier sketch (Hopper / Ampere pipelined load). Probes
# the async_copy + mbarrier OP_TABLE coverage.
_ASYNC_PIPELINE_TTIR = """\
module {
  tt.func @async_pipeline(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.barrier_init {count = 1 : i32}
    %2 = async_copy %a, %b : !tt.ptr<f16>
    tt.async_commit_group
    tt.async_wait {num = 0 : i32}
    tt.barrier_wait %1 {parity = 0 : i32}
    tt.return
  }
}
"""


# ---------------------------------------------------------------------------
# Public list. Order matters only for report readability.
# ---------------------------------------------------------------------------


_WELFORD_LAYER_NORM_STUB_TTIR = """\
module {
  tt.func @welford_layer_norm(%x: !tt.ptr<f32>, %w: !tt.ptr<f32>, %b: !tt.ptr<f32>, %y: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %xv = tt.load %x : tensor<256xf32>
    %wv = tt.load %w : tensor<256xf32>
    %bv = tt.load %b : tensor<256xf32>
    %sum = tt.reduce %xv {axis = 0 : i32, combiner = \"add\"} : tensor<256xf32> -> f32
    tt.store %y, %xv : tensor<256xf32>
    tt.return
  }
}
"""


_PAGED_ATTENTION_V2_STUB_TTIR = """\
module {
  tt.func @paged_attention_v2(%q: !tt.ptr<f32>, %k: !tt.ptr<f32>, %v: !tt.ptr<f32>, %bt: !tt.ptr<i32>, %o: !tt.ptr<f32>) {
    %qv = tt.load %q : tensor<16x32xf32>
    %kv = tt.load %k : tensor<16x32xf32>
    %vv = tt.load %v : tensor<16x32xf32>
    %qk = tt.dot %qv, %kv : tensor<16x32xf32>, tensor<16x32xf32> -> tensor<16x16xf32>
    tt.store %o, %qv : tensor<16x32xf32>
    tt.return
  }
}
"""


_SCAN_CUMSUM_STUB_TTIR = """\
module {
  tt.func @scan_cumsum(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>) {
    %xv = tt.load %x : tensor<128xf32>
    %sc = tt.scan %xv {axis = 0 : i32, combiner = \"add\"} : tensor<128xf32> -> tensor<128xf32>
    tt.store %y, %sc : tensor<128xf32>
    tt.return
  }
}
"""



CANNED_TTIR_FIXTURES: List[CannedKernel] = [
    CannedKernel(
        name="vector_add",
        description="Triton tutorial 01: masked elementwise add over a 1-D grid.",
        source="triton tutorial 01 / poc/triton_frontend/tests/test_vector_add.py",
        ttir_text=_VECTOR_ADD_TTIR,
        constexprs={"BLOCK": 128},
        live_kernel_module="vector_add",
    ),
    CannedKernel(
        name="softmax",
        description="Triton tutorial 02: row-wise softmax (reduce_max + reduce_sum).",
        source="triton tutorial 02",
        ttir_text=_SOFTMAX_TTIR,
        constexprs={"BLOCK_N": 256},
        live_kernel_module="softmax",
    ),
    CannedKernel(
        name="matmul",
        description="Triton tutorial 03: tiled fp16 matmul with fp32 accumulator.",
        source="triton tutorial 03",
        ttir_text=_MATMUL_TTIR,
        constexprs={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        live_kernel_module="matmul",
    ),
    CannedKernel(
        name="row_sum",
        description="Single-axis reduction over a 256-wide row.",
        source="canonical reduction kernel",
        ttir_text=_REDUCTION_TTIR,
        constexprs={"BLOCK_N": 256},
        live_kernel_module="row_sum",
    ),
    CannedKernel(
        name="layer_norm",
        description="Triton tutorial 05: two-pass LayerNorm with mean/var reductions.",
        source="triton tutorial 05",
        ttir_text=_LAYER_NORM_TTIR,
        constexprs={"BLOCK_N": 256},
        live_kernel_module="layer_norm",
    ),
    CannedKernel(
        name="gather_rows_3d",
        description="Gather + scatter pattern from nanochat/triton_kernels.py.",
        source="~/sources/nanochat/nanochat/triton_kernels.py:_gather_rows_3d_kernel",
        ttir_text=_GATHER_3D_TTIR,
        live_kernel_module="gather_rows_3d",
    ),
    CannedKernel(
        name="atomic_hist",
        description="Atomic-add histogram pattern (cppmega megatron kernels).",
        source="~/sources/cppmega/cppmega/megatron/* (atomic_rmw flavour)",
        ttir_text=_ATOMIC_HIST_TTIR,
        live_kernel_module="atomic_hist",
    ),
    CannedKernel(
        name="async_pipeline",
        description=("RFC 5.4 descriptor/TMA transfer: native TMA on NV, pointer-arith fallback elsewhere."),
        source="live tl.make_tensor_descriptor descriptor-load fallback for RFC section 5.4",
        ttir_text=_ASYNC_PIPELINE_TTIR,
        live_kernel_module="tma_descriptor_copy",
    ),
    CannedKernel(
        name="fla_dot_exp2",
        description=(
            "FLA gated-delta-rule motif: tt.dot accumulator + math.exp2 "
            "scaling lane (chunk_gated_delta_rule_fwd_kernel_h_blockdim64 "
            "headline op cohort)."
        ),
        source="~/sources/rent_kernels/flash-linear-attention/fla/ops/common/chunk_delta_h.py",
        ttir_text=_FLA_DOT_EXP2_TTIR,
        constexprs={"BT": 32, "BV": 32, "K": 32},
        live_kernel_module="fla_dot_exp2",
    ),
    CannedKernel(
        name="welford_layer_norm",
        description=(
            "Two-pass Welford LayerNorm forward (numerically stable mean / "
            "variance reduce + affine)."
        ),
        source="poc/triton_frontend/_test_harness/numeric_kernels/welford_layer_norm.py",
        ttir_text=_WELFORD_LAYER_NORM_STUB_TTIR,
        constexprs={"BLOCK_SIZE": 256, "EPS": 1e-5},
        live_kernel_module="welford_layer_norm",
    ),
    CannedKernel(
        name="paged_attention_v2",
        description=(
            "vLLM-style multi-block paged attention with block-table "
            "indirection and per-block streaming softmax."
        ),
        source="poc/triton_frontend/_test_harness/numeric_kernels/paged_attention_v2.py",
        ttir_text=_PAGED_ATTENTION_V2_STUB_TTIR,
        constexprs={
            "BLOCK_M": 16,
            "BLOCK_N": 16,
            "BLOCK_DMODEL": 32,
            "NUM_PAGES": 4,
            "PAGE_SIZE": 16,
        },
        live_kernel_module="paged_attention_v2",
    ),
    CannedKernel(
        name="scan_cumsum",
        description=(
            "Row-wise inclusive cumulative sum via tl.associative_scan "
            "(exercises tt.scan with arith.addf combiner)."
        ),
        source="poc/triton_frontend/_test_harness/numeric_kernels/scan_cumsum.py",
        ttir_text=_SCAN_CUMSUM_STUB_TTIR,
        constexprs={"BLOCK_SIZE": 128},
        live_kernel_module="scan_cumsum",
    ),
    # ---- Real captured TTIR (not hand-written) ----------------------------
    # ``fla_chunk_delta_h_real_ttir`` is the *actual* Triton-3.6 TTIR text
    # produced by capturing ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``
    # with constexprs {H=1, HV=1, K=64, V=32, BT=16, BV=32, USE_G=False,
    # USE_GK=False, USE_INITIAL_STATE=False, STORE_FINAL_STATE=False,
    # SAVE_NEW_VALUE=False, TRANSPOSE_STATE=False, IS_VARLEN=False} via
    # ``triton_jit_to_ttir``. Captures the full ``scf.for`` recurrence,
    # ``tt.make_block_ptr`` lowering, masked load/store boundary checks
    # (``arith.andi`` chains over ``tensor<...xi1>``), and the
    # ``tt.dot`` accumulator chain. This is what the reducer must handle
    # end-to-end for the FLA Path D wiring -- the hand-written
    # ``fla_dot_exp2`` motif above is the toy cousin.
    CannedKernel(
        name="fla_chunk_delta_h_real_ttir",
        description=("Real captured TTIR for FLA chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (K=64 single-block, no gates, no varlen)."),
        source="~/sources/rent_kernels/flash-linear-attention/fla/ops/common/chunk_delta_h.py:41 (captured via triton_jit_to_ttir, apple/mps backend, triton-pr9701)",
        ttir_text=_read_capture("fla_chunk_delta_h_real_ttir.mlir"),
        constexprs={
            "H": 1,
            "HV": 1,
            "K": 64,
            "V": 32,
            "BT": 16,
            "BV": 32,
            "USE_G": False,
            "USE_GK": False,
            "USE_INITIAL_STATE": False,
            "STORE_FINAL_STATE": False,
            "SAVE_NEW_VALUE": False,
            "TRANSPOSE_STATE": False,
            "IS_VARLEN": False,
        },
    ),
]
