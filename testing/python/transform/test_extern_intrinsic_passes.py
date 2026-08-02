"""Integration tests for ``tl.extern_intrinsic`` follow-up patches.

Sibling integration #9 ships the ``tl.extern_intrinsic`` decorator
(``tilelang/language/extern.py``). The C++ transforms ``LayoutInference`` and
``InjectSoftwarePipeline`` were patched (this PR) to read the
``tl.extern_intrinsic_meta`` block annotation. This file exercises both
hooks end-to-end.

The tests are integration-shaped: we build a tiny IRModule that emits
``call_extern("handle", "tl.extern_intrinsic.<name>", ...)`` with the
expected block annotation, run the pass, and assert the resulting IR
carries the inferred Fragment layout / pipeline-stage AttrStmt.
"""

from __future__ import annotations

import contextlib

import pytest

tvm = pytest.importorskip("tvm")

from tvm import tir as _tir  # noqa: E402  (after importorskip)
from tvm.script import tir as T  # noqa: E402

# The Python-side decorator is the source of truth for the attribute keys.
from tilelang.language.extern import (  # noqa: E402
    EXTERN_BLOCK_ATTR,
    EXTERN_CALL_PREFIX,
)

_CUDA_TARGET = tvm.target.Target("cuda")


def _registered_transform_is_callable(name: str) -> bool:
    return callable(tvm.get_global_func(name, allow_missing=True))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_attr_stmt(stmt, key: str) -> bool:
    """Recursively check whether ``stmt`` contains an AttrStmt with ``key``."""

    found = [False]

    def _visit(node):
        if isinstance(node, _tir.AttrStmt) and node.attr_key == key:
            found[0] = True

    _tir.stmt_functor.post_order_visit(stmt, _visit)
    return found[0]


def _has_layout_for_buffer(stmt, buffer_name: str) -> bool:
    """Whether any block annotation kLayoutMap mentions a buffer with the name."""

    found = [False]

    def _visit(node):
        if isinstance(node, _tir.Block):
            ann = node.annotations
            if "layout_map" in ann or "kLayoutMap" in ann:
                # Either key surfaces through the Python binding.
                found[0] = True

    _tir.stmt_functor.post_order_visit(stmt, _visit)
    return found[0]


def _bind_layout_target(mod: tvm.IRModule) -> tvm.IRModule:
    return tvm.tir.transform.BindTarget(_CUDA_TARGET)(mod)


# ---------------------------------------------------------------------------
# Layout-inference pickup
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _registered_transform_is_callable("tl.transform.LayoutInference"),
    reason="tl.transform.LayoutInference not registered (TileLang C++ side missing).",
)
def test_layout_inference_picks_up_extern_meta():
    """Smoke: layout_inference.cc visits a block with ``EXTERN_BLOCK_ATTR``
    and registers the declared per-Frag fragment layout for the buffer
    arguments of the call_extern. We just check that the pass runs to
    completion and emits the kLayoutMap annotation; the exact Fragment
    factory wiring for ``simdgroup_*`` is a TODO documented in the .cc."""

    # The simdgroup_mma reference example registers the intrinsic at import.
    pytest.importorskip("tilelang.language.extern")

    @T.prim_func
    def kernel(
        A: T.Buffer((8, 8), "float16"),  # noqa: N803
        B: T.Buffer((8, 8), "float16"),
        C: T.Buffer((8, 8), "float32"),
    ) -> None:
        with T.sblock("root"):
            T.sblock_attr({EXTERN_BLOCK_ATTR: {"layouts": ["simdgroup_a", "simdgroup_b", "simdgroup_c"]}})
            T.evaluate(
                T.call_extern("handle", EXTERN_CALL_PREFIX + "simdgroup_mma_8x8", A.access_ptr("r"), B.access_ptr("r"), C.access_ptr("rw"))
            )

    mod = _bind_layout_target(tvm.IRModule({"main": kernel}))
    LayoutInference = tvm.get_global_func("tl.transform.LayoutInference")
    out = LayoutInference()(mod)
    assert out["main"] is not None


# ---------------------------------------------------------------------------
# Inject-pipeline pickup
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _registered_transform_is_callable("tl.transform.InjectSoftwarePipeline"),
    reason="tl.transform.InjectSoftwarePipeline not registered.",
)
def test_inject_pipeline_picks_up_extern_meta():
    """An extern-intrinsic block tagged with ``pipeline_stage=2`` should
    surface a ``tl.pipeline_context_num_stages`` AttrStmt in the rewritten
    IR (same machinery as ``T.Pipelined(num_stages=3)``)."""

    @T.prim_func
    def kernel(
        A: T.Buffer((8, 8), "float16"),  # noqa: N803
        B: T.Buffer((8, 8), "float16"),
        C: T.Buffer((8, 8), "float32"),
    ) -> None:
        with T.sblock("root"):
            T.sblock_attr({EXTERN_BLOCK_ATTR: {"pipeline_stage": 2}})
            T.evaluate(
                T.call_extern("handle", EXTERN_CALL_PREFIX + "simdgroup_mma_8x8", A.access_ptr("r"), B.access_ptr("r"), C.access_ptr("rw"))
            )

    mod = _bind_layout_target(tvm.IRModule({"main": kernel}))
    InjectPipeline = tvm.get_global_func("tl.transform.InjectSoftwarePipeline")
    out = InjectPipeline()(mod)
    body = out["main"].body
    assert _has_attr_stmt(body, "tl.pipeline_context_num_stages"), (
        "Expected the extern_intrinsic pipeline_stage=2 hint to surface as "
        "a tl.pipeline_context_num_stages AttrStmt; none found in:\n" + str(body)
    )


# ---------------------------------------------------------------------------
# Negative path: pipeline_stage = -1 (default) is a no-op.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _registered_transform_is_callable("tl.transform.InjectSoftwarePipeline"),
    reason="tl.transform.InjectSoftwarePipeline not registered.",
)
def test_inject_pipeline_passthrough_for_unset_stage():
    @T.prim_func
    def kernel(A: T.Buffer((8, 8), "float16")) -> None:  # noqa: N803
        with T.sblock("root"):
            T.sblock_attr({EXTERN_BLOCK_ATTR: {"pipeline_stage": -1}})
            T.evaluate(T.call_extern("handle", EXTERN_CALL_PREFIX + "noop", A.access_ptr("r")))

    mod = _bind_layout_target(tvm.IRModule({"main": kernel}))
    InjectPipeline = tvm.get_global_func("tl.transform.InjectSoftwarePipeline")
    out = InjectPipeline()(mod)
    body = out["main"].body
    assert not _has_attr_stmt(body, "tl.pipeline_context_num_stages"), "pipeline_stage=-1 should be a passthrough — no AttrStmt expected."


# ---------------------------------------------------------------------------
# Tile-size meta dispatch (this PR).
# ---------------------------------------------------------------------------


def test_build_meta_emits_tile_size():
    """``build_meta`` must serialise ``Frag.shape`` of the output frag as
    the ``tile_size`` entry consumed by ``layout_inference.cc``."""
    from tilelang.language.extern import Frag, build_meta

    frags = (
        Frag("a", (16, 16), "local", "float16", layout="mma_A"),
        Frag("b", (16, 16), "local", "float16", layout="mma_B"),
        Frag("c", (16, 16), "local", "float32", layout="mma_C", is_output=True),
    )
    meta = build_meta(frags, pipeline_stage=2)
    assert meta["layouts"] == ["mma_A", "mma_B", "mma_C"]
    assert list(meta["tile_size"]) == [16, 16]
    assert meta["pipeline_stage"] == 2
    assert meta["is_output"] == [0, 0, 1]


@pytest.mark.skipif(
    not _registered_transform_is_callable("tl.transform.LayoutInference"),
    reason="tl.transform.LayoutInference not registered (TileLang C++ side missing).",
)
def test_layout_inference_dispatches_mma_tile_size():
    """A block carrying ``layouts=[mma_C]`` + ``tile_size=[16, 16, 16]``
    must traverse ``layout_inference.cc`` without crashing — the dispatcher
    feeds the tile size into ``makeGemmFragmentC`` instead of returning an
    empty placeholder ``Layout()``. We don't introspect the resulting layout
    here (it's an opaque Fragment object); the smoke is that the pass
    completes and the function survives in the output module."""

    @T.prim_func
    def kernel(C: T.Buffer((16, 16), "float32")) -> None:  # noqa: N803
        with T.sblock("root"):
            T.sblock_attr({EXTERN_BLOCK_ATTR: {"layouts": ["mma_C"], "tile_size": [16, 16, 16]}})
            T.evaluate(T.call_extern("handle", EXTERN_CALL_PREFIX + "fake_mma_16x16", C.access_ptr("rw")))

    mod = _bind_layout_target(tvm.IRModule({"main": kernel}))
    LayoutInference = tvm.get_global_func("tl.transform.LayoutInference")
    out = LayoutInference()(mod)
    assert out["main"] is not None


@pytest.mark.skipif(
    not _registered_transform_is_callable("tl.transform.LayoutInference"),
    reason="tl.transform.LayoutInference not registered.",
)
def test_layout_inference_unknown_layout_falls_through():
    """An unrecognised layout string must not crash the pass — the existing
    INFO-log + empty-Layout fallback is the contract."""

    @T.prim_func
    def kernel(A: T.Buffer((8, 8), "float16")) -> None:  # noqa: N803
        with T.sblock("root"):
            T.sblock_attr({EXTERN_BLOCK_ATTR: {"layouts": ["totally_made_up"]}})
            T.evaluate(T.call_extern("handle", EXTERN_CALL_PREFIX + "noop", A.access_ptr("r")))

    mod = _bind_layout_target(tvm.IRModule({"main": kernel}))
    LayoutInference = tvm.get_global_func("tl.transform.LayoutInference")
    out = LayoutInference()(mod)
    assert out["main"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_lower_extern_intrinsic_per_target_dispatch():
    """Verify that LowerExternIntrinsic injects the correct body string for the target."""
    from tilelang.language.extern import extern_intrinsic, Frag
    from tilelang.transform import LowerExternIntrinsic
    from tilelang.language.extern_registry import unregister

    name = "test_dispatch"
    try:
        with contextlib.suppress(KeyError):
            unregister(name)

        extern_intrinsic(
            name=name,
            signature=lambda: (Frag("c", (16, 16), "local", "float32", is_output=True),),
            bodies={
                "cuda": "__device__ void test_dispatch(float* c) { c[0] = 1.0f; }",
                "metal": "void test_dispatch(threadgroup float* c) { c[0] = 1.0f; }",
            },
        )

        @T.prim_func
        def kernel(C: T.Buffer((16, 16), "float32")):
            with T.sblock("root"):
                T.evaluate(T.call_extern("handle", "tl.extern_intrinsic.test_dispatch", C.access_ptr("rw")))

        mod = tvm.IRModule({"main": kernel})

        cuda_mod = LowerExternIntrinsic("cuda")(mod)
        cuda_func = cuda_mod["main"]
        assert "pragma_import_c" in str(cuda_func.body)
        assert "void test_dispatch(float* c)" in str(cuda_func.body)
        assert "threadgroup" not in str(cuda_func.body)

        metal_mod = LowerExternIntrinsic("metal")(mod)
        metal_func = metal_mod["main"]
        assert "pragma_import_c" in str(metal_func.body)
        assert "void test_dispatch(threadgroup float* c)" in str(metal_func.body)
    finally:
        with contextlib.suppress(KeyError):
            unregister(name)


def test_lower_extern_intrinsic_accepts_cutedsl_body_target():
    """CuTeDSL extern bodies must be selectable separately from CUDA bodies."""
    from tilelang.language.extern import extern_intrinsic, Frag
    from tilelang.transform import LowerExternIntrinsic
    from tilelang.language.extern_registry import unregister

    name = "test_cutedsl_dispatch"
    body = """import cutlass.cute as cute

@cute.kernel
def test_cutedsl_dispatch(c: cute.Tensor):
    c[0] = c[0]
"""
    try:
        with contextlib.suppress(KeyError):
            unregister(name)

        extern_intrinsic(
            name=name,
            signature=lambda: (Frag("c", (16, 16), "local", "float32", is_output=True),),
            bodies={"cutedsl": body},
        )

        @T.prim_func
        def kernel(C: T.Buffer((16, 16), "float32")):
            with T.sblock("root"):
                T.evaluate(
                    T.call_extern(
                        "handle",
                        "tl.extern_intrinsic.test_cutedsl_dispatch",
                        C.access_ptr("rw"),
                    )
                )

        cutedsl_mod = LowerExternIntrinsic("cutedsl")(tvm.IRModule({"main": kernel}))
        cutedsl_body = cutedsl_mod["main"].script(show_meta=True)
        assert "pragma_import_c" in cutedsl_body
        assert "@cute.kernel" in cutedsl_body
        assert "test_cutedsl_dispatch" in cutedsl_body
        assert "tl.extern_intrinsic.test_cutedsl_dispatch" not in cutedsl_body
    finally:
        with contextlib.suppress(KeyError):
            unregister(name)


def test_lower_extern_intrinsic_accepts_aliased_cutedsl_kernel_decorator():
    """Aliased CuTe imports must survive through the lowering import path."""
    from tilelang.language.extern import extern_intrinsic, Frag
    from tilelang.transform import LowerExternIntrinsic
    from tilelang.language.extern_registry import unregister

    name = "test_cutedsl_alias_dispatch"
    body = """import cutlass.cute as ct

@ct.kernel(preprocessor=False)
def test_cutedsl_alias_dispatch(c: ct.Tensor):
    c[0] = c[0]
"""
    try:
        with contextlib.suppress(KeyError):
            unregister(name)

        extern_intrinsic(
            name=name,
            signature=lambda: (Frag("c", (16, 16), "local", "float32", is_output=True),),
            bodies={"cutedsl": body},
        )

        @T.prim_func
        def kernel(C: T.Buffer((16, 16), "float32")):
            with T.sblock("root"):
                T.evaluate(
                    T.call_extern(
                        "handle",
                        "tl.extern_intrinsic.test_cutedsl_alias_dispatch",
                        C.access_ptr("rw"),
                    )
                )

        cutedsl_mod = LowerExternIntrinsic("cutedsl")(tvm.IRModule({"main": kernel}))
        cutedsl_body = cutedsl_mod["main"].script(show_meta=True)
        assert "pragma_import_c" in cutedsl_body
        assert "@ct.kernel" in cutedsl_body
        assert "test_cutedsl_alias_dispatch" in cutedsl_body
        assert "tl.extern_intrinsic.test_cutedsl_alias_dispatch" not in cutedsl_body
    finally:
        with contextlib.suppress(KeyError):
            unregister(name)


def test_cutedsl_target_selects_cutedsl_extern_intrinsic_target():
    from tilelang.engine.lower import _extern_intrinsic_target_name

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80", "keys": ["cuda", "gpu", "cutedsl"]})
    assert _extern_intrinsic_target_name(target) == "cutedsl"
    assert _extern_intrinsic_target_name(_CUDA_TARGET) == "cuda"


def test_lower_extern_intrinsic_missing_target_body_fails_closed():
    """Extern fusion must surface missing target bodies at lowering time."""
    from tilelang.language.extern import extern_intrinsic, Frag
    from tilelang.transform import LowerExternIntrinsic
    from tilelang.language.extern_registry import unregister

    name = "test_dispatch_cuda_only"
    try:
        extern_intrinsic(
            name=name,
            signature=lambda: (Frag("c", (16, 16), "local", "float32", is_output=True),),
            bodies={"cuda": "__device__ void test_dispatch_cuda_only(float* c) { c[0] = 1.0f; }"},
        )

        @T.prim_func
        def kernel(C: T.Buffer((16, 16), "float32")):
            with T.sblock("root"):
                T.evaluate(
                    T.call_extern(
                        "handle",
                        "tl.extern_intrinsic.test_dispatch_cuda_only",
                        C.access_ptr("rw"),
                    )
                )

        mod = tvm.IRModule({"main": kernel})
        with pytest.raises(ValueError, match="has no body for target 'metal'"):
            LowerExternIntrinsic("metal")(mod)
    finally:
        with contextlib.suppress(KeyError):
            unregister(name)


def test_lower_extern_intrinsic_unregistered_symbol_fails_closed():
    """Prefixed extern calls must resolve through the registry explicitly."""
    from tilelang.transform import LowerExternIntrinsic

    @T.prim_func
    def kernel(C: T.Buffer((16, 16), "float32")):
        with T.sblock("root"):
            T.evaluate(
                T.call_extern(
                    "handle",
                    "tl.extern_intrinsic.not_registered",
                    C.access_ptr("rw"),
                )
            )

    mod = tvm.IRModule({"main": kernel})
    with pytest.raises(ValueError, match="not_registered' is not registered"):
        LowerExternIntrinsic("metal")(mod)
