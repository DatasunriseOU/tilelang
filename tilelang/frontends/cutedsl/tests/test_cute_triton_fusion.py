"""RFC §7 Phase 4.2 intake: CuTe GEMM and Triton softmax share a fusion region.

Demonstrates the multi-source integration story for the CuTeDSL frontend
(symmetric to ``tilelang/frontends/triton/tests/test_multi_source_fusion.py``
which already proves TTIR + FX + ``tl.extern_intrinsic`` cohabit in one
``T.Kernel``). This file stops one step earlier: it proves the CuTe
static frontend can feed the same fusion intake path as Triton TTIR. A
future backend pass still has to inline the two nodes into one generated
PrimFunc/T.Kernel with no global-memory boundary between GEMM and softmax.

The test goal:

1. Lower a CuTe GEMM source string into a ``tir.PrimFunc`` via
   :func:`tilelang.frontends.cutedsl.from_cute_source`.
2. Lower a Triton softmax TTIR text into a second ``tir.PrimFunc`` via
   :func:`tilelang.frontends.triton.from_ttir`.
3. Assemble both into one :class:`tvm.IRModule` and use TileLang's
   :class:`tilelang.engine.fusion.FusionRegionBuilder` to declare a
   fused region whose nodes are the two PrimFuncs, then assert the
   builder accepts both source surfaces and produces a single-entry
   fusion region (one ``FusionRegion`` with two contributing nodes).

This is the structural floor for the RFC §7 Phase 4.2 contract, not the
final single-kernel fusion acceptance test.
"""
from __future__ import annotations

import importlib

import pytest


_HAS_TVM = importlib.util.find_spec("tvm") is not None
_HAS_TILELANG = importlib.util.find_spec("tilelang") is not None

pytestmark = pytest.mark.skipif(
    not (_HAS_TVM and _HAS_TILELANG),
    reason="TVM + TileLang required for the CuTe + Triton conformance test",
)


_CUTE_GEMM_SRC = """
import cutlass.cute as cute

@cute.kernel
def cute_gemm_tile(A, B, C):
    A_smem = cute.make_tensor('A_smem', shape=(16, 16), dtype='float16', scope='shared')
    B_smem = cute.make_tensor('B_smem', shape=(16, 16), dtype='float16', scope='shared')
    C_frag = cute.make_tensor('C_frag', shape=(16, 16), dtype='float32', scope='fragment')
    cute.copy(A, A_smem)
    cute.copy(B, B_smem)
    cute.gemm(A_smem, B_smem, C_frag)
    cute.copy(C_frag, C)
"""


_TRITON_SOFTMAX_TTIR = """
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:90"} {
  tt.func public @softmax(%X: !tt.ptr<f32>, %Y: !tt.ptr<f32>) {
    %0 = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %1 = arith.muli %0, %c16 : i32
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.splat %1 : i32 -> tensor<16xi32>
    %4 = arith.addi %3, %2 : tensor<16xi32>
    %5 = tt.splat %X : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %7 = tt.load %6 : tensor<16x!tt.ptr<f32>>
    %neg_inf = arith.constant -3.40282347E+38 : f32
    %max = "tt.reduce"(%7) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maximumf %a, %b : f32
      tt.reduce.return %m : f32
    }) : (tensor<16xf32>) -> f32
    %max_b = tt.splat %max : f32 -> tensor<16xf32>
    %sub = arith.subf %7, %max_b : tensor<16xf32>
    %ex = math.exp %sub : tensor<16xf32>
    %sum = "tt.reduce"(%ex) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16xf32>) -> f32
    %sum_b = tt.splat %sum : f32 -> tensor<16xf32>
    %div = arith.divf %ex, %sum_b : tensor<16xf32>
    %8 = tt.splat %Y : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %9 = tt.addptr %8, %4 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %9, %div : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
"""


def _lower_cute_gemm():
    from tilelang.frontends.cutedsl import CuTeKernelSignature, from_cute_source

    sig = [
        CuTeKernelSignature("A", (16, 16), "float16"),
        CuTeKernelSignature("B", (16, 16), "float16"),
        CuTeKernelSignature("C", (16, 16), "float32"),
    ]
    return from_cute_source(_CUTE_GEMM_SRC, signature=sig)


def _lower_triton_softmax():
    from tilelang.frontends.triton import from_ttir

    return from_ttir(
        _TRITON_SOFTMAX_TTIR,
        name="softmax",
        arg_buffer_shapes={"X": (16,), "Y": (16,)},
    )


def test_cute_gemm_and_triton_softmax_lower_independently() -> None:
    """Both source surfaces must lower into individual ``tir.PrimFunc``."""
    from tvm import tir

    gemm = _lower_cute_gemm()
    softmax = _lower_triton_softmax()
    assert isinstance(gemm, tir.PrimFunc)
    assert isinstance(softmax, tir.PrimFunc)
    gemm_sym = str(gemm.attrs.get("global_symbol"))
    softmax_sym = str(softmax.attrs.get("global_symbol"))
    assert gemm_sym == "cute_gemm_tile", gemm_sym
    assert softmax_sym.endswith("softmax"), softmax_sym


def test_cute_and_triton_primfuncs_cohabit_one_irmodule() -> None:
    """Both PrimFuncs must coexist in one IRModule for fusion intake."""
    import tvm
    from tvm import tir

    gemm = _lower_cute_gemm()
    softmax = _lower_triton_softmax()
    mod = tvm.IRModule(
        {
            "cute_gemm_tile": gemm,
            "triton_softmax": softmax,
        }
    )
    names = sorted(gv.name_hint for gv, _ in mod.functions.items())
    assert names == ["cute_gemm_tile", "triton_softmax"], names
    assert all(isinstance(fn, tir.PrimFunc) for _, fn in mod.functions.items())


def test_cute_and_triton_fusion_region_builder_accepts_both() -> None:
    """``FusionRegionBuilder`` must treat CuTe + Triton PrimFuncs as nodes."""
    from tilelang.engine.fusion import FusionRegionBuilder

    gemm = _lower_cute_gemm()
    softmax = _lower_triton_softmax()
    builder = FusionRegionBuilder("cute_gemm_then_triton_softmax")
    builder.add_prim_func_node(
        "cute_gemm",
        gemm,
        op="gemm",
        inputs=("A", "B"),
        outputs=("C",),
    )
    builder.add_prim_func_node(
        "triton_softmax",
        softmax,
        op="softmax",
        inputs=("C",),
        outputs=("Y",),
    )
    region = builder.build()
    node_names = [node.name for node in region.nodes]
    assert node_names == ["cute_gemm", "triton_softmax"], node_names
    # Provenance op names from each source must survive the build step.
    ops = {node.op for node in region.nodes}
    assert "gemm" in ops
    assert "softmax" in ops
