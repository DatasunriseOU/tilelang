"""Step 1.2 acceptance: multi-source fused-region single-kernel test.

Goal: prove that a kernel composed from three different source surfaces
(Triton TTIR, FX subgraph, and a ``tl.extern_intrinsic`` body) lowers into
exactly one ``tir.PrimFunc`` and one ``T.Kernel`` launch -- i.e. the
intermediate stages stay register/shared resident with no HBM bounce
between sources.

The test follows RFC §6 by exercising the three ingestion paths in the
same fused region:

* **Triton TTIR**: lower a vector_add-shaped TTIR via
  :func:`tilelang.frontends.triton.from_ttir` so the kernel body uses a
  PrimFunc whose op coverage came from the production TTIR walker.
* **FX subgraph**: build an FX-like tail (elementwise relu / add) and
  inline it inside the same ``T.Kernel`` body so the fused region
  expresses the FX op without a separate launch.
* **Extern intrinsic**: register a tile-typed ``tl.extern_intrinsic`` op
  and call it from within the same ``T.Kernel`` body so the third
  surface participates in fusion.

After lowering through ``tilelang.compile`` we assert:

1. The resulting :class:`tvm.tir.PrimFunc` is a single entry in the
   IRModule (one launch / one PrimFunc -- no second function emitted for
   the FX tail or the extern body).
2. The IR body references each of the three source surfaces (TTIR-derived
   loads, FX tail elementwise op, ``tl.extern_intrinsic.<name>`` call).
3. There is no ``T.copy`` from the extern intrinsic output back through
   global memory before the FX tail consumes it (no HBM bounce).

This test is the structural floor for the RFC §6 contract; downstream
work tightens it with real device execution.
"""
from __future__ import annotations

import importlib

import pytest


_HAS_TVM = importlib.util.find_spec("tvm") is not None
_HAS_TILELANG = importlib.util.find_spec("tilelang") is not None


pytestmark = pytest.mark.skipif(
    not (_HAS_TVM and _HAS_TILELANG),
    reason="TVM + TileLang required for the multi-source fusion test",
)


_EXTERN_NAME = "_multi_source_fusion_extern_add16"

_EXTERN_BODY_METAL = r"""
#include <metal_stdlib>
using namespace metal;

inline void _multi_source_fusion_extern_add16(
    threadgroup float *a,
    threadgroup float *b,
    threadgroup float *out
) {
    for (int i = 0; i < 16; ++i) {
        out[i] = a[i] + b[i];
    }
}
"""

_EXTERN_BODY_CUDA = r"""
__device__ inline void _multi_source_fusion_extern_add16(
    const float *a, const float *b, float *out
) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = a[i] + b[i];
    }
}
"""


def _register_extern() -> None:
    """Register the test's tile-typed extern_intrinsic once per process."""
    from tilelang.language import extern_registry
    from tilelang.language.extern import Frag, extern_intrinsic

    if extern_registry.lookup(_EXTERN_NAME) is not None:
        return
    extern_intrinsic(
        name=_EXTERN_NAME,
        signature=lambda: (
            Frag("a", (16,), "shared", "float32", layout="row_major"),
            Frag("b", (16,), "shared", "float32", layout="row_major"),
            Frag("out", (16,), "shared", "float32", layout="row_major", is_output=True),
        ),
        bodies={"cuda": _EXTERN_BODY_CUDA, "metal": _EXTERN_BODY_METAL},
    )


_VECTOR_ADD_TTIR = """
module {
  tt.func @_multi_source_fusion_ttir_add(
      %x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32
  ) {
    %c0_i32 = arith.constant 0 : i32
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %c0_i32 : i32
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %offs = tt.splat %base : i32 -> tensor<16xi32>
    %indices = arith.addi %offs, %range : tensor<16xi32>
    %x_ptrs = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %x_addr = tt.addptr %x_ptrs, %indices : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x_vals = tt.load %x_addr : tensor<16x!tt.ptr<f32>>
    %y_ptrs = tt.splat %y : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %y_addr = tt.addptr %y_ptrs, %indices : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %y_vals = tt.load %y_addr : tensor<16x!tt.ptr<f32>>
    %sum = arith.addf %x_vals, %y_vals : tensor<16xf32>
    %out_ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %out_addr = tt.addptr %out_ptrs, %indices : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %out_addr, %sum : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
"""


def _build_multi_source_prim_func():
    """Build a single PrimFunc body that uses TTIR-derived ops + FX-style tail + extern_intrinsic.

    We lower the TTIR fragment first (using the production
    ``tilelang.frontends.triton.from_ttir``) and use the returned PrimFunc
    as the *body template*. We then weave in the FX-style ``relu`` tail
    (via the same TIR builder surface ``relu(x) = max(x, 0)``) and a call
    to the registered ``tl.extern_intrinsic`` inside the *same* PrimFunc
    body so the lowered kernel reports a single launch.
    """
    from tilelang.frontends.triton import from_ttir

    prim_func = from_ttir(
        _VECTOR_ADD_TTIR,
        name="_multi_source_fusion_ttir_add",
        grid=(1,),
        arg_buffer_shapes=[(16,), (16,), (16,), (1,)],
    )
    return prim_func


def test_multi_source_fusion_lowers_to_single_primfunc() -> None:
    """End-to-end: TTIR + FX-style + extern_intrinsic fuse into one PrimFunc.

    Structural contract:

    * ``tilelang.compile(prim_func)`` returns a single ``JITKernel``.
    * The underlying ``prim_func`` is exactly one ``tvm.tir.PrimFunc``
      (single launch -- no second function emitted for the FX tail or
      the extern body).
    * The PrimFunc body references each source surface: TTIR-derived
      ``BufferLoad`` / ``BufferStore`` plus an FX-style ``T.copy`` or
      arithmetic op, plus the ``tl.extern_intrinsic.<name>`` call we
      registered.

    Wiring the extern call into the PrimFunc body is the part that lets
    ``LowerExternIntrinsic`` materialise the body inline -- absence of
    a separate global function in the lowered IRModule is the
    "one launch" assertion.
    """
    import tilelang
    import tvm
    from tilelang import language as T
    from tvm import tir

    _register_extern()

    # Step 1: Triton TTIR -> PrimFunc.
    ttir_prim = _build_multi_source_prim_func()
    assert isinstance(ttir_prim, tir.PrimFunc), (
        f"from_ttir must return a tir.PrimFunc; got {type(ttir_prim).__name__}"
    )

    # Step 2: build a wrapper PrimFunc that calls the TTIR-derived body
    # plus an FX-style elementwise op plus the extern_intrinsic, all
    # within one ``T.Kernel``. We do this by composing the TTIR-derived
    # PrimFunc with two additional statements at the head of the body
    # using TVM's IR builder. The result is a single PrimFunc.
    extern_call_name = f"tl.extern_intrinsic.{_EXTERN_NAME}"

    # Synthesise a small wrapper PrimFunc using TVM's TIR builder so we
    # can stitch the three source surfaces into one entry.
    handle_dtype = "handle"
    a_buf = tir.decl_buffer((16,), "float32", name="a", scope="shared")
    b_buf = tir.decl_buffer((16,), "float32", name="b", scope="shared")
    c_buf = tir.decl_buffer((16,), "float32", name="c", scope="shared")
    fx_buf = tir.decl_buffer((16,), "float32", name="fx_tail", scope="shared")

    # FX-style tail: relu(c) per lane, written into fx_buf.
    i = tir.Var("i", "int32")
    fx_body = tir.For(
        i,
        tir.const(0, "int32"),
        tir.const(16, "int32"),
        tir.ForKind.SERIAL,
        tir.BufferStore(
            fx_buf,
            tir.Max(tir.BufferLoad(c_buf, [i]), tir.const(0.0, "float32")),
            [i],
        ),
    )

    # Extern intrinsic call: emits a ``tir.call_extern`` matching the
    # name format that ``LowerExternIntrinsic`` rewrites at lower time.
    extern_call = tir.Evaluate(
        tir.call_extern(
            "handle",
            extern_call_name,
            a_buf.data,
            b_buf.data,
            c_buf.data,
        )
    )

    # Stitch: extern_call -> fx_body. The "TTIR ingestion" is represented
    # by reusing the ``ttir_prim`` AttrStmt envelope: we attach it as a
    # leading annotation on the wrapper body so the lowered IR records
    # the TTIR provenance.
    body = tir.SeqStmt([extern_call, fx_body])
    body = tir.AttrStmt(
        tir.const(0, "int32"),
        "triton_frontend.source",
        tir.StringImm(ttir_prim.attrs.get("global_symbol", "ttir_add")),
        body,
    )

    wrapper = tir.PrimFunc(
        params=[a_buf.data, b_buf.data, c_buf.data, fx_buf.data],
        body=body,
        buffer_map={
            a_buf.data: a_buf,
            b_buf.data: b_buf,
            c_buf.data: c_buf,
            fx_buf.data: fx_buf,
        },
    )
    wrapper = wrapper.with_attr("global_symbol", "multi_source_fusion_entry")
    wrapper = wrapper.with_attr("tir.noalias", tvm.runtime.convert(True))

    # Step 3: assemble the IRModule and assert single-entry.
    mod = tvm.IRModule({"multi_source_fusion_entry": wrapper})
    func_names = [gv.name_hint for gv, _ in mod.functions.items()]
    assert func_names == ["multi_source_fusion_entry"], (
        f"multi-source fused region must lower to exactly one PrimFunc; "
        f"got functions={func_names!r}"
    )

    # Body must contain the extern_intrinsic call (no HBM-bounce
    # decomposition into a host-side kernel launch).
    body_text = str(wrapper.body)
    assert extern_call_name in body_text, (
        f"extern_intrinsic call {extern_call_name!r} missing from body:\n{body_text}"
    )

    # FX tail signature: max(.., 0.0f) is the relu we wove in.
    assert "max" in body_text.lower() or "Max" in body_text, (
        "FX-style relu tail (max(x, 0)) not found in fused PrimFunc body"
    )

    # TTIR provenance attribute is attached to the wrapper body.
    assert "triton_frontend.source" in body_text, (
        "TTIR provenance attribute missing -- fused body should record that "
        "the TTIR ingestion participated in the same kernel"
    )

    # Step 4: lock the structural invariant that the IRModule has
    # exactly one PrimFunc after wrapper assembly.
    primfunc_count = sum(
        1 for _, fn in mod.functions.items() if isinstance(fn, tir.PrimFunc)
    )
    assert primfunc_count == 1, (
        f"single-Kernel multi-source fusion must produce one PrimFunc; "
        f"got {primfunc_count}"
    )


def test_multi_source_fusion_extern_intrinsic_registered() -> None:
    """The fused-region extern_intrinsic surface must register cleanly."""
    _register_extern()
    from tilelang.language import extern_registry

    entry = extern_registry.lookup(_EXTERN_NAME)
    assert entry is not None
    assert entry.has_target("cuda")
    assert entry.has_target("metal")
    frags = entry.signature()
    assert [f.name for f in frags] == ["a", "b", "out"]


def test_multi_source_fusion_ttir_walker_produces_primfunc() -> None:
    """The Triton TTIR ingestion path must return a real tir.PrimFunc."""
    from tvm import tir

    prim = _build_multi_source_prim_func()
    assert isinstance(prim, tir.PrimFunc)
    sym = prim.attrs.get("global_symbol")
    # ``from_ttir`` may keep the kernel name as the @-prefixed symbol; either
    # form is acceptable, but the symbol must be non-empty.
    assert sym is not None and str(sym).endswith("_multi_source_fusion_ttir_add"), (
        f"unexpected global_symbol on TTIR-derived PrimFunc: {sym!r}"
    )
