"""Phase 5 acceptance: static CUTile -> TileLang TIR lowering and conformance tests.
"""

from __future__ import annotations

import importlib
import pytest

_HAS_TVM = importlib.util.find_spec("tvm") is not None
_HAS_TILELANG = importlib.util.find_spec("tilelang") is not None

pytestmark = pytest.mark.skipif(
    not (_HAS_TVM and _HAS_TILELANG),
    reason="TVM + TileLang required for the CUTile frontend tests",
)

_VECTOR_ADD_SRC = """
import cutlass.cutile as cutile

@cutile.kernel
def cutile_vector_add(A, B, C):
    A_smem = cutile.make_tensor('A_smem', shape=(16,), dtype='float32', scope='shared')
    B_smem = cutile.make_tensor('B_smem', shape=(16,), dtype='float32', scope='shared')
    C_frag = cutile.make_tensor('C_frag', shape=(16,), dtype='float32', scope='fragment')
    cutile.copy(A, A_smem)
    cutile.copy(B, B_smem)
    for i in cutile.arange(16):
        C_frag[i] = A_smem[i] + B_smem[i]
    cutile.copy(C_frag, C)
"""

_CUTILE_GEMM_SRC = """
import cutlass.cutile as cutile

@cutile.kernel
def cutile_gemm(A, B, C):
    A_smem = cutile.make_tensor('A_smem', shape=(16, 16), dtype='float16', scope='shared')
    B_smem = cutile.make_tensor('B_smem', shape=(16, 16), dtype='float16', scope='shared')
    C_frag = cutile.make_tensor('C_frag', shape=(16, 16), dtype='float32', scope='fragment')
    cutile.copy(A, A_smem)
    cutile.copy(B, B_smem)
    cutile.gemm(A_smem, B_smem, C_frag)
    cutile.copy(C_frag, C)
"""

_CUTE_GEMM_SRC = """
import cutlass.cute as cute

@cute.kernel
def cute_gemm(A, B, C):
    A_smem = cute.make_tensor('A_smem', shape=(16, 16), dtype='float16', scope='shared')
    B_smem = cute.make_tensor('B_smem', shape=(16, 16), dtype='float16', scope='shared')
    C_frag = cute.make_tensor('C_frag', shape=(16, 16), dtype='float32', scope='fragment')
    cute.copy(A, A_smem)
    cute.copy(B, B_smem)
    cute.gemm(A_smem, B_smem, C_frag)
    cute.copy(C_frag, C)
"""


def test_from_cutile_source_emits_tilelang_dsl() -> None:
    from tilelang.frontends.cutile import CuTileKernelSignature, from_cutile_source

    sig = [
        CuTileKernelSignature("A", (16,), "float32"),
        CuTileKernelSignature("B", (16,), "float32"),
        CuTileKernelSignature("C", (16,), "float32"),
    ]

    emitted = from_cutile_source(
        _VECTOR_ADD_SRC,
        signature=sig,
        emit_only=True,
    )
    assert "import tilelang" in emitted
    assert "@T.prim_func" in emitted
    assert "T.alloc_shared((16,), \"float32\")" in emitted
    assert "T.alloc_fragment((16,), \"float32\")" in emitted
    assert "T.copy(A, A_smem)" in emitted
    assert "T.copy(B, B_smem)" in emitted
    assert "C_frag[i] = (A_smem[i] + B_smem[i])" in emitted
    assert "T.copy(C_frag, C)" in emitted


def test_from_cutile_source_compiles_to_primfunc() -> None:
    from tvm import tir
    from tilelang.frontends.cutile import CuTileKernelSignature, from_cutile_source

    sig = [
        CuTileKernelSignature("A", (16,), "float32"),
        CuTileKernelSignature("B", (16,), "float32"),
        CuTileKernelSignature("C", (16,), "float32"),
    ]

    prim = from_cutile_source(_VECTOR_ADD_SRC, signature=sig)
    assert isinstance(prim, tir.PrimFunc)
    assert len(prim.params) == 3
    param_names = {p.name for p in prim.params}
    assert param_names == {"A_handle", "B_handle", "C_handle"}
    assert str(prim.attrs.get("global_symbol")) == "cutile_vector_add"


def test_conformance_parity_gemm() -> None:
    """Verify that CUTile Gemm lowers to the exact same TVMScript structure as CuTe Gemm."""
    from tilelang.frontends.cutedsl import CuTeKernelSignature, from_cute_source
    from tilelang.frontends.cutile import CuTileKernelSignature, from_cutile_source

    cute_sig = [
        CuTeKernelSignature("A", (16, 16), "float16"),
        CuTeKernelSignature("B", (16, 16), "float16"),
        CuTeKernelSignature("C", (16, 16), "float32"),
    ]

    cutile_sig = [
        CuTileKernelSignature("A", (16, 16), "float16"),
        CuTileKernelSignature("B", (16, 16), "float16"),
        CuTileKernelSignature("C", (16, 16), "float32"),
    ]

    cute_emitted = from_cute_source(_CUTE_GEMM_SRC, signature=cute_sig, emit_only=True, func_name="gemm_parity")
    cutile_emitted = from_cutile_source(_CUTILE_GEMM_SRC, signature=cutile_sig, emit_only=True, func_name="gemm_parity")

    assert cute_emitted == cutile_emitted, "CuTe and CUTile emitted outputs differ!"

    cute_prim = from_cute_source(_CUTE_GEMM_SRC, signature=cute_sig, func_name="gemm_parity")
    cutile_prim = from_cutile_source(_CUTILE_GEMM_SRC, signature=cutile_sig, func_name="gemm_parity")

    # Compare TVMScript structural forms
    cute_script = cute_prim.script(show_meta=False)
    cutile_script = cutile_prim.script(show_meta=False)
    assert cute_script == cutile_script
