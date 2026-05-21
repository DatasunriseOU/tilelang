"""Smoke test for ``tl.extern_intrinsic`` (RFC §6).

Registration must work without CUDA. Actual TIR emission is gated on TVM
being importable; compilation/runtime is not exercised here.
"""

from __future__ import annotations

import contextlib
import importlib.util
import pytest

from tilelang.language import extern_registry
from tilelang.language.extern import (
    EXTERN_CALL_PREFIX,
    Frag,
    extern_intrinsic,
    simdgroup_a,
    simdgroup_a_fp8,
    simdgroup_b,
    simdgroup_b_fp8,
    simdgroup_c,
)

_HAS_TVM = importlib.util.find_spec("tvm") is not None
_HAS_CUDA = False
try:  # pragma: no cover - environment dependent
    import torch  # type: ignore[import-untyped]
    _HAS_CUDA = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
except ImportError:
    _HAS_CUDA = False


_FUSED_RELU_ADD_CU = r"""
__device__ void fused_relu_add(const float *a, const float *b, float *out) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float v = a[i] + b[i];
        out[i] = v > 0.f ? v : 0.f;
    }
}
"""


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Drop our test entry between tests to avoid cross-test pollution.

    Uses the public ``unregister`` API rather than poking at the private
    ``_REGISTRY`` attribute (security/test-fragility nit from grok review).
    """
    yield
    with contextlib.suppress(KeyError):
        extern_registry.unregister("fused_relu_add_16")


def test_registration_no_cuda_required():
    """Decorator must work even when CUDA / TVM are absent."""
    extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32", layout="row_major"),
            Frag("b", (16,), "shared", "float32", layout="row_major"),
            Frag("out", (16,), "shared", "float32", layout="row_major", is_output=True),
        ),
        bodies={"cuda": _FUSED_RELU_ADD_CU},
    )
    entry = extern_registry.lookup("fused_relu_add_16")
    assert entry is not None
    assert entry.has_target("cuda")
    assert not entry.has_target("metal")
    frags = entry.signature()
    assert [f.name for f in frags] == ["a", "b", "out"]
    assert frags[2].is_output is True


@pytest.mark.skipif(not _HAS_TVM, reason="TVM not importable")
def test_emit_returns_tir_call(monkeypatch):
    """Calling the decorated op should yield a TIR call_extern node."""
    from tvm import tir

    op = extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("b", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cuda": _FUSED_RELU_ADD_CU},
    )
    # Build three fake shared buffers and call the emitter.
    a = tir.decl_buffer((16,), "float32", scope="shared", name="a")
    b = tir.decl_buffer((16,), "float32", scope="shared", name="b")
    c = tir.decl_buffer((16,), "float32", scope="shared", name="out")
    node = op(a, b, c)
    assert isinstance(node, tir.Call)
    # Symbol must use the documented prefix so codegen can grep for it.
    assert any(EXTERN_CALL_PREFIX + "fused_relu_add_16" in str(arg) for arg in node.args)


@pytest.mark.skipif(not _HAS_TVM, reason="TVM not importable")
def test_emit_separates_shape_and_buffer_args(monkeypatch):
    """Regression test for grok perf review #1.

    The user calls ``intrin(M, N, A, B, C)`` — the shape factory sees only
    ``(M, N)`` and the buffers go to ``_emit_tir_call``. Previously the
    factory received the buffers and would TypeError or produce wrong frags.
    """
    from tvm import tir

    seen_shape_args: list[tuple] = []

    def factory(M: int, N: int):
        seen_shape_args.append((M, N))
        return (
            Frag("a", (M, N), "shared", "float32"),
            Frag("b", (M, N), "shared", "float32"),
            Frag("out", (M, N), "shared", "float32", is_output=True),
        )

    op = extern_intrinsic(
        name="fused_relu_add_16",
        signature=factory,
        bodies={"cuda": _FUSED_RELU_ADD_CU},
    )
    a = tir.decl_buffer((4, 4), "float32", scope="shared", name="a")
    b = tir.decl_buffer((4, 4), "float32", scope="shared", name="b")
    c = tir.decl_buffer((4, 4), "float32", scope="shared", name="out")
    node = op(4, 4, a, b, c)
    assert isinstance(node, tir.Call)
    assert seen_shape_args == [(4, 4)]


@pytest.mark.skipif(not _HAS_TVM, reason="TVM not importable")
def test_emit_resolves_buffer_kwargs_by_frag_name():
    """Regression test for grok security review buffer-collection nit.

    Buffer kwargs must bind to frags by name, not by iteration order.
    """
    from tvm import tir

    op = extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("b", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cuda": _FUSED_RELU_ADD_CU},
    )
    a = tir.decl_buffer((16,), "float32", scope="shared", name="a")
    b = tir.decl_buffer((16,), "float32", scope="shared", name="b")
    c = tir.decl_buffer((16,), "float32", scope="shared", name="out")
    # Pass them out of frag order by kwarg — emission must still match by name.
    node = op(out=c, a=a, b=b)
    assert isinstance(node, tir.Call)


def test_unregister_public_api():
    """Public ``unregister`` removes the entry; second call raises KeyError."""
    extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("b", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cuda": _FUSED_RELU_ADD_CU},
    )
    assert extern_registry.lookup("fused_relu_add_16") is not None
    extern_registry.unregister("fused_relu_add_16")
    assert extern_registry.lookup("fused_relu_add_16") is None
    with pytest.raises(KeyError):
        extern_registry.unregister("fused_relu_add_16")


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
def test_cuda_device_present_for_real_compile():
    """Placeholder for a future end-to-end compile-and-run test."""
    assert _HAS_CUDA


def test_simdgroup_factories_produce_canonical_frags():
    """Factories must pin scope/dtype/layout/alignment to canonical values.

    Regression for grok security review #08 "other bugs" #3.
    """
    a = simdgroup_a("a")
    b = simdgroup_b("b")
    c = simdgroup_c("c")
    assert (a.scope, a.dtype, a.layout, a.alignment, a.is_output) == (
        "simdgroup", "float16", "simdgroup_a", 16, False,
    )
    assert (b.scope, b.dtype, b.layout, b.alignment, b.is_output) == (
        "simdgroup", "float16", "simdgroup_b", 16, False,
    )
    assert (c.scope, c.dtype, c.layout, c.alignment, c.is_output) == (
        "simdgroup", "float32", "simdgroup_c", 16, True,
    )
    # Defaults to (8, 8); override is honoured.
    assert simdgroup_a("a").shape == (8, 8)
    assert simdgroup_a("a", shape=(16, 16)).shape == (16, 16)
    # Non-2D rejected.
    with pytest.raises(ValueError, match="2-D tile shape"):
        simdgroup_a("a", shape=(8, 8, 8))


def test_simdgroup_fp8_factories_produce_canonical_frags():
    """FP8 forward-compat factories must pin the call-site contract.

    Importable + returns a Frag with the documented placeholder Layout
    (i.e. layout-string ``simdgroup_a_fp8`` / ``simdgroup_b_fp8`` that
    ``layout_inference.cc`` will treat as opaque, matching the existing
    fp16 simdgroup factories). No Apple FP8 hardware required for this
    level of test — we only check the metadata contract.
    """
    a = simdgroup_a_fp8("a")
    b = simdgroup_b_fp8("b")
    assert (a.scope, a.dtype, a.layout, a.alignment, a.is_output) == (
        "simdgroup", "float8_e4m3", "simdgroup_a_fp8", 16, False,
    )
    assert (b.scope, b.dtype, b.layout, b.alignment, b.is_output) == (
        "simdgroup", "float8_e4m3", "simdgroup_b_fp8", 16, False,
    )
    # E5M2 variant (unsigned-zero / IEEE-style fp8) reachable via dtype override.
    a_e5m2 = simdgroup_a_fp8("a", dtype="float8_e5m2")
    assert a_e5m2.dtype == "float8_e5m2"
    # Default tile is (8, 8); larger tiles are accepted by the factory contract.
    assert simdgroup_a_fp8("a").shape == (8, 8)
    assert simdgroup_a_fp8("a", shape=(16, 16)).shape == (16, 16)
    # Non-2D rejected (matches fp16 contract).
    with pytest.raises(ValueError, match="2-D tile shape"):
        simdgroup_a_fp8("a", shape=(8, 8, 8))


@pytest.mark.xfail(
    reason=(
        "FP8 simdgroup factories only lock the metadata contract today; "
        "there is no executable TileLang compile-and-launch runtime check "
        "for float8_e4m3 simdgroup MMA yet."
    ),
    strict=False,
    raises=NotImplementedError,
)
def test_simdgroup_fp8_factories_produce_canonical_frags_fp8_runtime():
    """Runtime tracking marker for Apple FP8 silicon.

    Replace the explicit ``NotImplementedError`` with a real compile and
    launch assertion when the runtime FP8 simdgroup MMA path exists.
    """
    # The factory must keep working regardless of hardware.
    a = simdgroup_a_fp8("a")
    b = simdgroup_b_fp8("b")
    assert (a.layout, b.layout) == ("simdgroup_a_fp8", "simdgroup_b_fp8")
    # Forward-compat: replace this explicit marker with a real runtime check
    # (compile + launch a tiny FP8 simdgroup MMA kernel) when that path exists.
    raise NotImplementedError(
        "Apple float8_e4m3 simdgroup_matrix not shipped — see extern.py "
        "module docstring for the precise edits required to flip this on."
    )


def test_validate_body_warns_on_missing_frag_name(recwarn):
    """The body-name scan must warn (but not error) when a declared Frag.name
    is not referenced anywhere in the body. Regression for grok review item
    "_validate_body parameter-name matching" deferred from Wave-1.
    """
    # Body refers to ``a`` and ``b`` but never to ``out`` — the ``out`` Frag
    # name is unreferenced, which usually indicates a typo.
    body = r"""
__device__ void typo_intrinsic(const float *a, const float *b, float *not_out) {
    not_out[0] = a[0] + b[0];
}
"""
    extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("b", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cuda": body},
    )
    msgs = [str(w.message) for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert any("'out'" in m or "[\'out\']" in m for m in msgs), (
        f"expected warning about unreferenced 'out' Frag; saw {msgs!r}"
    )


def test_validate_body_no_warning_when_names_match(recwarn):
    """No warning when every Frag.name appears in the body."""
    body = r"""
__device__ void clean_intrinsic(const float *a, const float *b, float *out) {
    out[0] = a[0] + b[0];
}
"""
    extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("b", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cuda": body},
    )
    body_warns = [
        w for w in recwarn.list
        if issubclass(w.category, UserWarning) and "do not appear in the body" in str(w.message)
    ]
    assert not body_warns, f"unexpected body-name warning(s): {body_warns!r}"


def test_cutedsl_body_requires_matching_kernel_name():
    """CuTeDSL imports must define the registered callable symbol.

    LowerExternIntrinsic rewrites ``tl.extern_intrinsic.<name>`` calls to
    plain ``<name>(...)`` calls. Accepting a body whose only ``@cute.kernel``
    has another name would defer the failure until generated CuTeDSL code
    references a missing symbol.
    """
    body = """import cutlass.cute as cute

@cute.kernel
def wrong_name(a: cute.Tensor, out: cute.Tensor):
    out[0] = a[0]
"""
    with pytest.raises(ValueError, match="Expected '@cute.kernel"):
        extern_intrinsic(
            name="fused_relu_add_16",
            signature=lambda: (
                Frag("a", (16,), "shared", "float32"),
                Frag("out", (16,), "shared", "float32", is_output=True),
            ),
            bodies={"cutedsl": body},
        )


def test_cutedsl_body_accepts_bare_kernel_decorator_alias():
    """Official CuTe DSL docs describe the GPU decorator as ``@kernel``.

    Users may import the decorator directly rather than spelling
    ``@cute.kernel``. That is still a CuTeDSL kernel body and should validate.
    """
    body = """from cutlass.cute import Tensor, kernel

@kernel(preprocessor=False)
def fused_relu_add_16(a: Tensor, out: Tensor):
    out[0] = a[0]
"""
    extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cutedsl": body},
    )
    entry = extern_registry.lookup("fused_relu_add_16")
    assert entry is not None
    assert entry.has_target("cutedsl")


def test_cutedsl_body_accepts_imported_cute_module_alias():
    """CuTeDSL users commonly alias the imported module."""
    body = """import cutlass.cute as ct

@ct.kernel(preprocessor=False)
def fused_relu_add_16(a: ct.Tensor, out: ct.Tensor):
    out[0] = a[0]
"""
    extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cutedsl": body},
    )
    entry = extern_registry.lookup("fused_relu_add_16")
    assert entry is not None
    assert entry.has_target("cutedsl")


def test_cutedsl_body_accepts_imported_fully_qualified_cute_module():
    """Fully-qualified CuTe decorators are valid when the module is imported."""
    body = """import cutlass.cute

@cutlass.cute.kernel(preprocessor=False)
def fused_relu_add_16(a: cutlass.cute.Tensor, out: cutlass.cute.Tensor):
    out[0] = a[0]
"""
    extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cutedsl": body},
    )
    entry = extern_registry.lookup("fused_relu_add_16")
    assert entry is not None
    assert entry.has_target("cutedsl")


def test_cutedsl_body_rejects_unimported_bare_kernel_decorator():
    """Bare ``@kernel`` only counts when imported from ``cutlass.cute``."""
    body = """
@kernel
def fused_relu_add_16(a, out):
    out[0] = a[0]
"""
    with pytest.raises(ValueError, match="no recognisable CuTeDSL"):
        extern_intrinsic(
            name="fused_relu_add_16",
            signature=lambda: (
                Frag("a", (16,), "shared", "float32"),
                Frag("out", (16,), "shared", "float32", is_output=True),
            ),
            bodies={"cutedsl": body},
        )


def test_cutedsl_body_rejects_unimported_fully_qualified_kernel_decorator():
    """Fully-qualified ``@cutlass.cute.kernel`` still needs an import."""
    body = """
@cutlass.cute.kernel
def fused_relu_add_16(a, out):
    out[0] = a[0]
"""
    with pytest.raises(ValueError, match="no recognisable CuTeDSL"):
        extern_intrinsic(
            name="fused_relu_add_16",
            signature=lambda: (
                Frag("a", (16,), "shared", "float32"),
                Frag("out", (16,), "shared", "float32", is_output=True),
            ),
            bodies={"cutedsl": body},
        )


def test_validate_body_ignores_names_inside_raw_string(recwarn):
    """C++11 raw-string-literal contents must not satisfy the contract scan.

    Without the raw-string scrubber a body that embeds an MSL fragment as a
    raw string would cause the regular-string regex to desync at the inner
    ``"`` and accidentally accept the Frag name. This test pins the new
    raw-string-aware scrub.
    """
    body = (
        '__device__ void fake_intrinsic(const float *a, const float *b, float *not_out) {\n'
        '    const char *src = R"msl(\n'
        '        // \'out\' shows up inside the raw string but NOT in real code\n'
        '        out = a + b;\n'
        '    )msl";\n'
        '    not_out[0] = a[0] + b[0];\n'
        '    (void)src;\n'
        '}\n'
    )
    extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("b", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cuda": body},
    )
    msgs = [str(w.message) for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert any("'out'" in m for m in msgs), (
        f"expected warning when 'out' only appears inside a raw-string literal; saw {msgs!r}"
    )


def test_register_or_replace_atomic_round_trip():
    """``register_or_replace`` must atomically swap a prior entry."""
    extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cuda": _FUSED_RELU_ADD_CU},
    )
    first = extern_registry.lookup("fused_relu_add_16")
    assert first is not None
    replacement = extern_registry.ExternIntrinsic(
        name="fused_relu_add_16",
        signature=lambda: (Frag("z", (32,), "shared", "float32", is_output=True),),
        bodies={"cuda": _FUSED_RELU_ADD_CU},
    )
    prev = extern_registry.register_or_replace(replacement)
    assert prev is first
    after = extern_registry.lookup("fused_relu_add_16")
    assert after is replacement
    # Calling it on a fresh name should return ``None`` (no prior entry).
    fresh = extern_registry.ExternIntrinsic(
        name="brand_new_intrinsic",
        signature=lambda: (Frag("x", (8,), "shared", "float32"),),
        bodies={"cuda": _FUSED_RELU_ADD_CU},
    )
    try:
        assert extern_registry.register_or_replace(fresh) is None
    finally:
        extern_registry.unregister("brand_new_intrinsic")


def test_validate_body_ignores_names_inside_comments(recwarn):
    """A Frag.name appearing only in a comment must not satisfy the check."""
    body = r"""
__device__ void typo_intrinsic(const float *a, const float *b, float *not_out) {
    // out[0] = a[0] + b[0];   <-- commented-out reference to 'out' must not count
    not_out[0] = a[0] + b[0];
}
"""
    extern_intrinsic(
        name="fused_relu_add_16",
        signature=lambda: (
            Frag("a", (16,), "shared", "float32"),
            Frag("b", (16,), "shared", "float32"),
            Frag("out", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"cuda": body},
    )
    msgs = [str(w.message) for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert any("'out'" in m for m in msgs), (
        f"expected warning when 'out' only appears in a comment; saw {msgs!r}"
    )
