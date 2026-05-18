"""Regression test for PR #2140 review: fp8 GEMM accumulator must NOT be
promoted to ``metal.simdgroup`` scope by ``MetalFragmentToSimdgroup``.

Background: the Metal simdgroup_matrix intrinsics only support fp16 / bf16
(and packed int8/uint8) accumulator slots. A GEMM with fp8 operands today
uses an fp32 accumulator, which is *not* an eligible simdgroup dtype.
Under the gated rewrite path (``tl.simdgroup_matrix_rewrite=True``) the
pass must skip such accumulators so codegen does not emit unsupported
``simdgroup_float32`` matrices.

This test constructs a minimal ``tl.tileop.gemm`` over fp8 inputs with an
fp32 ``local.fragment`` accumulator, runs the pass with the gated flag,
and asserts no buffer in the resulting PrimFunc has been remapped to the
``metal.simdgroup`` storage scope.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

import tilelang  # noqa: F401  (force libtilelang load before tvm imports)
from tvm import tir, ir
from tvm.ir import PointerType, PrimType
from tvm.target import Target
from tvm import transform as tvm_transform


def _load_worktree_module():
    """Load the worktree copy of metal_fragment_to_simdgroup.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(
        os.path.join(here, "..", "..", "..", "tilelang", "transform",
                     "metal_fragment_to_simdgroup.py")
    )
    if not os.path.exists(candidate):
        from tilelang.transform import metal_fragment_to_simdgroup as m
        return m
    spec = importlib.util.spec_from_file_location(
        "_worktree_metal_fragment_to_simdgroup_fp8", candidate
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_M = _load_worktree_module()
MetalFragmentToSimdgroup = _M.MetalFragmentToSimdgroup
PASS_CONFIG_KEY = _M.PASS_CONFIG_KEY


def _make_fragment_var(name: str, dtype: str) -> tir.Var:
    return tir.Var(name, PointerType(PrimType(dtype), "local.fragment"))


def _make_fp8_gemm_func() -> tir.PrimFunc:
    """Build a minimal PrimFunc containing one ``tl.tileop.gemm`` with fp8
    inputs (``e4m3``) and an fp32 ``local.fragment`` accumulator."""
    region_op = ir.Op.get("tir.tvm_access_ptr")  # placeholder for shape
    gemm_op = ir.Op.get("tl.tileop.gemm")

    M, N, K = 16, 16, 16
    a_buf = tir.decl_buffer((M, K), "float8_e4m3", name="A", scope="local.fragment")
    b_buf = tir.decl_buffer((K, N), "float8_e4m3", name="B", scope="local.fragment")
    c_buf = tir.decl_buffer((M, N), "float32", name="C", scope="local.fragment")

    def _region_call(buf):
        # tilelang represents GEMM region args as a Call wrapping a
        # BufferLoad of the buffer at the origin. We mirror that shape so
        # `_extract_buffer_var_from_region` can recover the underlying var.
        return tir.Call(
            "handle",
            ir.Op.get("tir.tvm_access_ptr"),
            [
                tir.BufferLoad(buf, [tir.const(0, "int32") for _ in buf.shape]),
                tir.const(0, "int32"),
                tir.const(0, "int32"),
                tir.const(1, "int32"),
            ],
        )

    body = tir.Evaluate(
        tir.Call(
            "handle",
            gemm_op,
            [_region_call(a_buf), _region_call(b_buf), _region_call(c_buf)],
        )
    )

    func = tir.PrimFunc(
        params=[],
        body=body,
        ret_type=None,
        buffer_map={},
    )
    func = func.with_attr("target", Target("metal"))
    return func


def _module_has_simdgroup_scope(func: tir.PrimFunc) -> bool:
    """Return True if any buffer var in ``func`` carries ``metal.simdgroup``."""
    found = {"hit": False}

    def _visit_var(var):
        ta = getattr(var, "type_annotation", None)
        if ta is None:
            return
        scope = getattr(ta, "storage_scope", None)
        if scope == "metal.simdgroup":
            found["hit"] = True

    # Walk the body collecting any Var references whose ptr type is simdgroup-scoped.
    def _visitor(node):
        if isinstance(node, tir.Var):
            _visit_var(node)
        # AllocBuffer / DeclBuffer: inspect buffer scope and data var.
        buf = getattr(node, "buffer", None)
        if buf is not None:
            try:
                if buf.scope() == "metal.simdgroup":
                    found["hit"] = True
            except Exception:
                pass
            data = getattr(buf, "data", None)
            if data is not None:
                _visit_var(data)

    tir.stmt_functor.post_order_visit(func.body, _visitor)
    return found["hit"]


def test_fp8_gemm_accum_not_promoted_to_simdgroup():
    """With the gated rewrite enabled, an fp32 accum over fp8 inputs must
    NOT be promoted to ``metal.simdgroup`` scope."""
    func = _make_fp8_gemm_func()
    mod = ir.IRModule.from_expr(func.with_attr("global_symbol", "main"))

    with tvm_transform.PassContext(config={PASS_CONFIG_KEY: True}):
        out_mod = MetalFragmentToSimdgroup(mod)

    out_func = out_mod["main"]
    assert not _module_has_simdgroup_scope(out_func), (
        "fp8 GEMM with fp32 accumulator was incorrectly promoted to "
        "metal.simdgroup scope; gated rewrite should reject it.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
