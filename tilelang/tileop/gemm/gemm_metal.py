from __future__ import annotations

from .gemm_base import GemmBase
from tilelang.layout import Layout
from tilelang.utils.language import (
    is_shared,
    is_global,
    is_full_region,
    is_metal_simdgroup,
    is_fragment,
)
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


# Backwards-compatible alias used elsewhere in our codebase.
GEMM_INST_METAL_SIMDGROUP = "metal.simdgroup"
# Canonical names used by the upstream PR (tile-ai/tilelang#2252) for the
# register_gemm_impl wiring in tilelang/backend/metal/gemm.py.
GEMM_INST_METAL = GEMM_INST_METAL_SIMDGROUP
GEMM_INST_METAL_COOPERATIVE_TENSOR = "metal.cooperative_tensor"


def _make_padded_layout(buffer):
    """Pad the innermost stride to avoid 256-byte alignment SMEM bank conflicts.

    Used by the Metal M5 cooperative-tensor GEMM path; see PR
    tile-ai/tilelang#2252.
    """
    shape = buffer.shape
    stride = int(shape[-2])
    continuous = int(shape[-1])
    element_bits = int(tvm.DataType(buffer.dtype).bits)
    padded = continuous
    if (element_bits * continuous) % 256 == 0:
        padded += 128 // element_bits
    return Layout([stride, continuous], lambda i, j: i * padded + j)


class GemmMetalSimdGroup(GemmBase):
    """The legacy Metal 8x8x8 simdgroup-matrix GEMM path.

    Runs on every Apple-Silicon GPU (M1+).  Selected automatically when
    the cooperative-tensor preconditions don't hold (M < 16, N < 32, K < 16,
    or C is in local.fragment / metal.simdgroup scope).
    """

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_rs(self) -> bool:
        # A in register fragment, B in shared. The simdgroup path can only load
        # A from threadgroup memory, so we stage the A fragment into a temporary
        # shared buffer (see lower()) and then run the standard SS path.
        return is_fragment(self.A) and is_shared(self.B)

    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        for name, value in (("M", self.M), ("N", self.N), ("K", self.chunk)):
            if value % 8 != 0:
                raise ValueError(f"Metal GEMM requires {name} to be a multiple of 8, got {value}")
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_METAL_SIMDGROUP)
        if self.M % m_warp != 0 or self.N % n_warp != 0:
            raise ValueError(f"Metal GEMM cannot evenly partition {self.M}x{self.N} across {m_warp}x{n_warp} warps")
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        if warp_row_tiles % 8 != 0 or warp_col_tiles % 8 != 0:
            raise ValueError(f"Metal GEMM per-warp tile must be a multiple of 8x8, got {warp_row_tiles}x{warp_col_tiles}")

        from tilelang.intrinsics.metal_macro_generator import MPSIntrinEmitter

        mps_emitter = MPSIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            thread_var=thread_var,
            use_cooperative_tensor=False,
        )

        in_dtype = self.in_dtype
        accum_dtype = self.accum_dtype
        warp_rows = mps_emitter.warp_rows
        warp_cols = mps_emitter.warp_cols
        num_simd_c = warp_rows * warp_cols
        block_K = mps_emitter.chunk
        micro_size_k = mps_emitter.micro_size_k

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        C_buf = C_region.buffer

        clear_accum = self.clear_accum
        c_in_simdgroup_reg = is_metal_simdgroup(C_buf) or is_fragment(C_buf)

        if block_K < micro_size_k:
            raise ValueError(f"Metal GEMM requires block_K ({block_K}) to be >= micro_size_k ({micro_size_k})")
        if block_K % micro_size_k != 0:
            raise ValueError(f"Metal GEMM requires block_K ({block_K}) to be divisible by micro_size_k ({micro_size_k})")
        if not is_full_region(C_region):
            raise ValueError(f"Metal GEMM requires full output C region, got {C_region}")
        if not c_in_simdgroup_reg and not is_shared(C_buf):
            raise ValueError(f"Metal GEMM requires C in local.fragment, metal.simdgroup, or shared scope, got {C_buf.scope()}")

        # The simdgroup ldmatrix_a path issues simdgroup_load against a
        # threadgroup-memory pointer, so A must live in shared memory.  When A
        # arrives in a register fragment (the RS variant, e.g. an elementwise-
        # decayed operand fed straight into T.gemm), we stage it into a
        # temporary shared buffer first and then run the unchanged SS lowering.
        # This is a pure data movement: the staged values equal the fragment
        # values, so numerics are preserved exactly.
        stage_a_from_fragment = self.is_gemm_rs()
        if not (self.is_gemm_ss() or stage_a_from_fragment):
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

        a_rows = int(self.M)
        a_cols = int(self.chunk)
        if self.trans_A:
            a_shared_shape = (a_cols, a_rows)
        else:
            a_shared_shape = (a_rows, a_cols)

        if c_in_simdgroup_reg:

            @T.prim_func
            def _gemm_ss_simdgroup() -> None:
                A_local = T.alloc_local((warp_rows * 64), in_dtype, scope="metal.simdgroup")
                B_local = T.alloc_local((warp_cols * 64), in_dtype, scope="metal.simdgroup")
                if stage_a_from_fragment:
                    A_staged = T.alloc_shared(a_shared_shape, in_dtype, scope="shared.dyn")
                    T.copy(A_region, A_staged)
                    A_operand = A_staged
                else:
                    A_operand = A_region
                if clear_accum:
                    for _i in T.serial(num_simd_c):
                        T.make_filled_simdgroup_matrix(C_buf.data, _i, T.cast(0, accum_dtype))
                for ki in T.serial(0, (block_K // micro_size_k)):
                    mps_emitter.ldmatrix_a(A_local, A_operand, ki)
                    mps_emitter.ldmatrix_b(B_local, B_region, ki)
                    mps_emitter.mma(A_local, B_local, C_buf)

            return _Simplify(_gemm_ss_simdgroup, inline_let=True)
        else:

            @T.prim_func
            def _gemm_ss_shared() -> None:
                A_local = T.alloc_local((warp_rows * 64), in_dtype, scope="metal.simdgroup")
                B_local = T.alloc_local((warp_cols * 64), in_dtype, scope="metal.simdgroup")
                C_simd = T.alloc_local((num_simd_c * 64), accum_dtype, scope="metal.simdgroup")
                if stage_a_from_fragment:
                    A_staged = T.alloc_shared(a_shared_shape, in_dtype, scope="shared.dyn")
                    T.copy(A_region, A_staged)
                    A_operand = A_staged
                else:
                    A_operand = A_region
                if clear_accum:
                    for _i in T.serial(num_simd_c):
                        T.make_filled_simdgroup_matrix(C_simd.data, _i, T.cast(0, accum_dtype))
                else:
                    mps_emitter.simd_load(C_simd, C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    mps_emitter.ldmatrix_a(A_local, A_operand, ki)
                    mps_emitter.ldmatrix_b(B_local, B_region, ki)
                    mps_emitter.mma(A_local, B_local, C_simd)

                mps_emitter.simd_store(C_simd, C_buf)

            return _Simplify(_gemm_ss_shared, inline_let=True)


class GemmMetal(GemmBase):
    """Metal M5 cooperative-tensor GEMM path (`mpp::tensor_ops::matmul2d`).

    Implements the upstream PR tile-ai/tilelang#2252 lowering.  Only runs
    when the instruction selector in src/backend/metal/op/gemm.cc picks
    `metal.cooperative_tensor`, which currently requires:

      * C scope is `shared` (not local.fragment / metal.simdgroup).
      * M % 16 == 0, N % 32 == 0, K % 16 == 0.
      * The kernel's warp count can be evenly partitioned into M/16 x N/32
        tile groups.

    On hardware without Metal 4 (M1-M4 silicon, MSL < 4.0) the emitted MSL
    will fail at `xcrun metal` compile time.  The legacy simdgroup path
    above remains the default fallback.
    """

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_gg(self) -> bool:
        return is_global(self.A) and is_global(self.B)

    def _make_mps_emitter(self, target: Target, thread_nums: int):
        from tilelang.intrinsics.metal_macro_generator import MPSIntrinEmitter

        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_METAL_COOPERATIVE_TENSOR)
        if self.is_gemm_gg():
            num_warps = int(thread_nums) // 32
            k_n_per_warp = 32
            m_warp, n_warp = 1, num_warps
            if int(self.N) % (n_warp * k_n_per_warp) != 0:
                n_warp = int(self.N) // k_n_per_warp
                m_warp = num_warps // n_warp
                if m_warp == 0:
                    m_warp = 1
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        return (
            MPSIntrinEmitter(
                a_dtype=self.in_dtype,
                b_dtype=self.in_dtype,
                accum_dtype=self.accum_dtype,
                a_transposed=self.trans_A,
                b_transposed=self.trans_B,
                block_row_warps=m_warp,
                block_col_warps=n_warp,
                warp_row_tiles=warp_row_tiles,
                warp_col_tiles=warp_col_tiles,
                chunk=self.chunk,
                use_cooperative_tensor=True,
            ),
            m_warp,
            n_warp,
        )

    @staticmethod
    def _get_padded_stride(buffer):
        continuous = int(buffer.shape[-1])
        element_bits = int(tvm.DataType(buffer.dtype).bits)
        padded = continuous
        if (element_bits * continuous) % 256 == 0:
            padded += 128 // element_bits
        return padded

    def infer_layout(self, target: Target, thread_nums: int):
        result = {}
        if self.is_gemm_ss():
            result[self.A] = _make_padded_layout(self.A)
            result[self.B] = _make_padded_layout(self.B)
        if is_fragment(self.C):
            emitter, _, _ = self._make_mps_emitter(target, thread_nums)
            result[self.C] = emitter.make_cooperative_tensor_store_layout(self.C)
        return result

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        _, m_warp, n_warp = self._make_mps_emitter(target, int(thread_nums))
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)

        from tilelang.intrinsics.metal_macro_generator import MPSIntrinEmitter

        a_stride = self._get_padded_stride(self.A) if self.is_gemm_ss() else None
        b_stride = self._get_padded_stride(self.B) if self.is_gemm_ss() else None

        c_bytes_per_thread = warp_row_tiles * warp_col_tiles * 64
        inner_k_steps = 2 if c_bytes_per_thread <= 128 else 1
        mps_emitter = MPSIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            thread_var=thread_var,
            a_stride_override=a_stride,
            b_stride_override=b_stride,
            inner_k_steps=inner_k_steps,
            use_cooperative_tensor=True,
        )

        in_dtype = self.in_dtype
        accum_dtype = self.accum_dtype
        warp_rows = mps_emitter.warp_rows
        warp_cols = mps_emitter.warp_cols
        num_simd_c = warp_rows * warp_cols
        block_K = mps_emitter.chunk
        micro_size_x = mps_emitter.micro_size_x
        micro_size_y = mps_emitter.micro_size_y
        micro_size_k = mps_emitter.micro_size_k
        inner_k_steps = mps_emitter.inner_k_steps
        a_tile_elems = micro_size_x * micro_size_k
        b_tile_elems = micro_size_k * micro_size_y
        c_tile_elems = micro_size_x * micro_size_y

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion
        C_buf = C_region.buffer
        clear_accum = self.clear_accum
        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"
        assert is_full_region(C_region), "Fragment output C must be a full region"

        if not (self.is_gemm_ss() or self.is_gemm_gg()):
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

        @T.prim_func
        def _gemm_with_c_writeback() -> None:
            A_local = T.alloc_local((warp_rows * a_tile_elems * inner_k_steps), in_dtype)
            B_local = T.alloc_local((warp_cols * b_tile_elems * inner_k_steps), in_dtype)
            C_ct = T.alloc_local((num_simd_c * c_tile_elems), accum_dtype, scope="metal.cooperative_tensor")
            if clear_accum:
                # Python-unroll so each fill targets a constant __pct_cN tile.
                for _i in range(int(num_simd_c)):
                    T.cooperative_tensor_fill(C_ct.data, _i, T.cast(0, accum_dtype), micro_size_x, micro_size_y)
            else:
                mps_emitter.simd_load(C_ct, C_buf)
            # Python-level (compile-time) unroll over the K reduction so the
            # cooperative-tensor tile indices (a_idx/b_idx/c_idx) are constants.
            # This keeps the inlined __pct_cN accumulator fast path active, and
            # the matmul2d's multiply_accumulate mode correctly accumulates the
            # mpp destination cooperative tensor across the unrolled k steps.
            for k_outer in range(int(block_K // (micro_size_k * inner_k_steps))):
                for k_inner in range(inner_k_steps):
                    ki = k_outer * inner_k_steps + k_inner
                    mps_emitter.ldmatrix_a(A_local, A_region, ki, k_inner)
                    mps_emitter.ldmatrix_b(B_local, B_region, ki, k_inner)
                for k_inner in range(inner_k_steps):
                    ki = k_outer * inner_k_steps + k_inner
                    mps_emitter.mma(
                        A_local,
                        B_local,
                        C_ct,
                        k_inner,
                        A_shared_buf=A_region,
                        B_shared_buf=B_region,
                        ki=ki,
                    )
            mps_emitter.simd_store(C_ct, C_buf)

        return _Simplify(_gemm_with_c_writeback, inline_let=True)
