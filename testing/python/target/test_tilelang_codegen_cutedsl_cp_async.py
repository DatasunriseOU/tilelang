import contextlib

import pytest

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.language.extern import extern_intrinsic, Frag
from tilelang.language import extern_registry
from tvm.target import Target


def test_cutedsl_codegen_supports_tl_ptx_cp_async():
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")

    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    target = Target({"kind": "cuda", "arch": "sm_80", "keys": ["cuda", "gpu", "cutedsl"]})

    @T.prim_func
    def prog(A: T.Tensor((16,), "uint8"), B: T.Tensor((16,), "uint8")):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((16,), "uint8", scope="shared")
            T.ptx_cp_async(T.access_ptr(A_shared[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
            B[0] = A_shared[0]

    artifact = lower(prog.with_attr("global_symbol", "main"), target=target)
    assert "tl.cp_async_gs(" in artifact.kernel_source


def test_cutedsl_codegen_preserves_extern_intrinsic_import_body():
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")

    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    name = "cutedsl_import_probe"
    body = """import cutlass.cute as cute

@cute.kernel
def cutedsl_import_probe(a: cute.Tensor, b: cute.Tensor):
    b[0] = a[0]
"""
    target = Target({"kind": "cuda", "arch": "sm_80", "keys": ["cuda", "gpu", "cutedsl"]})
    try:
        with contextlib.suppress(KeyError):
            extern_registry.unregister(name)
        extern_intrinsic(
            name=name,
            signature=lambda: (
                Frag("a", (16,), "shared", "float32"),
                Frag("b", (16,), "shared", "float32", is_output=True),
            ),
            bodies={"cutedsl": body},
        )

        @T.prim_func
        def prog(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
            with T.Kernel(1, threads=1):
                a_shared = T.alloc_shared((16,), "float32")
                b_shared = T.alloc_shared((16,), "float32")
                a_shared[0] = A[0]
                T.evaluate(
                    T.call_extern(
                        "handle",
                        "tl.extern_intrinsic.cutedsl_import_probe",
                        a_shared.access_ptr("r"),
                        b_shared.access_ptr("rw"),
                    )
                )
                B[0] = b_shared[0]

        artifact = lower(prog.with_attr("global_symbol", "main"), target=target)
        assert "@cute.kernel" in artifact.kernel_source
        assert "def cutedsl_import_probe" in artifact.kernel_source
        assert "tl.extern_intrinsic.cutedsl_import_probe" not in artifact.kernel_source
    finally:
        with contextlib.suppress(KeyError):
            extern_registry.unregister(name)


if __name__ == "__main__":
    tilelang.testing.main()
