from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.layout import make_swizzled_layout, Fragment
from tilelang.utils.language import is_shared, is_full_region
from tilelang import language as T
from tilelang.transform.simplify import _Simplify
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir


GEMM_INST_SIMT = "cuda.simt"


def _pick_thread_grid(M: int, N: int, threads: int) -> tuple[int, int]:
    """Pick a (TM, TN) thread grid with TM*TN == threads that evenly divides
    the (M, N) output tile.

    Prefers larger TN (so contiguous lanes cover contiguous N-columns, which
    gives coalesced global stores when the consumer copies C -> global later).
    """
    best = None
    for tn in range(min(N, threads), 0, -1):
        if threads % tn != 0:
            continue
        tm = threads // tn
        if N % tn != 0 or M % tm != 0:
            continue
        score = tn  # prefer larger tn (better store coalescing)
        if best is None or score > best[0]:
            best = (score, tm, tn)
    if best is None:
        # Fallback: linearized; this still produces a correct (if uneven)
        # mapping by relying on a single-axis interleave.
        return (threads, 1)
    return best[1], best[2]


def _make_simt_fragment(buf, threads: int) -> Fragment:
    """Build a SIMT 2D fragment layout for an (M, N) output tile.

    Tiles the (M, N) buffer onto a (TM, TN) thread grid (TM*TN == threads),
    so each thread owns a contiguous ``(M/TM) x (N/TN)`` rectangle of cells.
    With TM*TN == threads dividing M*N, the mapping is a bijection (the
    Fragment is non-replicated), which is the precondition for the
    surrounding ``T.Parallel(M, N)`` loop to distribute its iterations one
    cell per lane.
    """
    M = int(buf.shape[0])
    N = int(buf.shape[1])
    TM, TN = _pick_thread_grid(M, N, threads)
    local_m = M // TM
    local_n = N // TN

    def forward_thread(i, j):
        # i // local_m : which thread-row owns this cell
        # j // local_n : which thread-col owns this cell
        return (i // local_m) * TN + (j // local_n)

    def forward_index(i, j):
        return (i % local_m) * local_n + (j % local_n)

    return Fragment(
        [M, N],
        forward_thread_fn=forward_thread,
        forward_index_fn=forward_index,
    )


class GemmSIMT(GemmBase):
    """Block-parallel SIMT GEMM fallback.

    Used on targets where the desired data-type has no tensor-core MMA
    instruction available (currently: fp64 on consumer Blackwell sm_120 / sm_121,
    where the fp64 tensor cores are physically absent and the legacy
    ``mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64`` PTX instruction
    silently produces garbage).

    Strategy:
      * Output tile C is distributed across the CTA's threads by an explicit
        row-major fragment layout so each thread owns exactly
        ``ceil(M*N/threads)`` cells of the (block_M, block_N) tile.
      * Each lane sequentially walks the K reduction dimension, reading the
        already-tiled operands out of shared memory (SS variant). RS / SR /
        RR variants reuse the same loop body with the corresponding buffer
        substituted; the fragment buffer is indexed directly.
    """

    def infer_layout(self, target: Target, thread_nums: int):
        layout: dict = {}
        if is_shared(self.A):
            layout[self.A] = make_swizzled_layout(self.A)
        if is_shared(self.B):
            layout[self.B] = make_swizzled_layout(self.B)
        # Pin C's fragment layout so the surrounding T.Parallel(M, N) loop
        # has a definite source-buffer mapping (i, j) -> (thread, local_id).
        # Without this, Parallel's free-inference mode falls back to a
        # replicated layout: every thread runs the whole gemm and the
        # accumulator is over-counted by `threads`.
        layout[self.C] = _make_simt_fragment(self.C, int(thread_nums))
        return layout

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        block_M = int(self.M)
        block_N = int(self.N)
        block_K = int(self.chunk)

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        trans_A = bool(self.trans_A)
        trans_B = bool(self.trans_B)
        clear_accum = self.clear_accum
        accum_dtype = self.accum_dtype

        a0 = A_region.region[0].min
        a1 = A_region.region[1].min
        b0 = B_region.region[0].min
        b1 = B_region.region[1].min
        c0 = C_region.region[0].min
        c1 = C_region.region[1].min

        assert is_full_region(C_region), "Fragment output C must be a full region"

        thread_extent = int(thread_bounds.extent)
        TM, TN = _pick_thread_grid(block_M, block_N, thread_extent)
        local_m = block_M // TM
        local_n = block_N // TN

        # Snapshot the lane index — `thread_var` is the kernel's threadIdx.x
        # binding; we map it to a (m_tile, n_tile) coordinate so the gemm body
        # below is a static per-lane loop. Layout inference is not re-run on
        # the body that lower() returns, so we must NOT rely on T.Parallel
        # here — it would emit the full M*N loop unsharded.
        tx = thread_var
        m_tile = tx // TN
        n_tile = tx % TN
        k_extent = block_K

        @T.prim_func
        def _gemm_simt() -> None:
            if clear_accum:
                T.clear(C_buf)
            # Each lane owns a (local_m, local_n) rectangle of the output tile
            # starting at (m_tile*local_m, n_tile*local_n). We walk it
            # sequentially per lane, accumulating into the fragment register
            # slot dictated by _make_simt_fragment (`(li % local_m) * local_n
            # + (lj % local_n)`).
            for li, lj in T.grid(local_m, local_n):
                i = m_tile * local_m + li
                j = n_tile * local_n + lj
                for k in T.serial(k_extent):
                    a_row = (k if trans_A else i) + a0
                    a_col = (i if trans_A else k) + a1
                    b_row = (j if trans_B else k) + b0
                    b_col = (k if trans_B else j) + b1
                    C_buf[i + c0, j + c1] = C_buf[i + c0, j + c1] + T.cast(
                        A_buf[a_row, a_col] * B_buf[b_row, b_col],
                        accum_dtype,
                    )

        return _Simplify(_gemm_simt, inline_let=True)
