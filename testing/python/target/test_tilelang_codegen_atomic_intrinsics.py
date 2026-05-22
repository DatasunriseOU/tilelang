import pytest

import tilelang  # noqa: F401  (force libtilelang load before codegen lookup)
import tilelang.language as T
import tilelang.transform
from tilelang import tvm
from tilelang.engine.lower import lower
from tvm.target import Target


pytest.importorskip("tilelang")


def _atomic_cas_prog():
    @T.prim_func
    def prog(A: T.Tensor((1,), "int32"), Old: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            Old[0] = T.call_intrin(
                "int32",
                "tir.atomic_cas",
                T.address_of(A[0]),
                T.int32(0),
                T.int32(1),
            )

    return prog


def test_tir_atomic_cas_cuda_lower_intrin_rewrites_to_atomiccas_extern():
    """CUDA LowerIntrin should rewrite raw TIR CAS before source codegen."""

    target = Target({"kind": "cuda", "arch": "sm_80", "keys": ["cuda", "gpu"]})
    prog = _atomic_cas_prog().with_attr("global_symbol", "main").with_attr("target", target)
    mod = tvm.IRModule({"main": prog})

    with target:
        lowered = tilelang.transform.LowerIntrin()(mod)
    text = str(lowered)

    assert "atomicCAS" in text
    assert "tir.atomic_cas" not in text
    assert "tirx.atomic_cas" not in text


def test_tir_atomic_cas_lowers_to_cuda_atomiccas_source():
    """Source-only CUDA codegen must lower raw TIR CAS to native atomicCAS."""

    if tvm.ffi.get_global_func("target.build.tilelang_cuda_without_compile", allow_missing=True) is None:
        pytest.skip("TileLang CUDA source builder is not enabled in this build.")

    target = Target({"kind": "cuda", "arch": "sm_80", "keys": ["cuda", "gpu"]})

    prog = _atomic_cas_prog()
    artifact = lower(prog.with_attr("global_symbol", "main"), target=target)
    source = artifact.kernel_source

    assert "atomicCAS(" in source
    assert "tir.atomic_cas" not in source
    assert "atomic_cas(" not in source


def test_tir_atomic_cas_lowers_to_metal_atomiccas_helper_source():
    """Source-only Metal codegen must lower raw TIR CAS to a native MSL helper."""

    if tvm.ffi.get_global_func("target.build.tilelang_metal", allow_missing=True) is None:
        pytest.skip("TileLang Metal source builder is not enabled in this build.")

    target = Target({"kind": "metal", "keys": ["metal", "gpu"]})

    prog = _atomic_cas_prog()
    artifact = lower(prog.with_attr("global_symbol", "main"), target=target)
    source = artifact.kernel_source

    assert "tl::AtomicCAS(" in source
    assert "atomic_compare_exchange_weak_explicit(" in source
    assert "tir.atomic_cas" not in source
    assert "atomic_cas(" not in source
