"""Smoke test for the PtrAnalysis facade.

Source MLIR text below is adapted from
``microsoft/triton-shared/test/Conversion/TritonToStructured/
addptr_loopback.mlir`` (MIT, Microsoft + Meta) -- the *tensor-of-pointers*
loopback fixture, which is the one ``PtrAnalysis::rewriteOp`` actually
emits ``tts.make_tptr`` for. The earlier scalar-loopback fixture
(``addptr_scalar_loopback.mlir``) was a deliberate no-op: upstream
preserves scalar ``tt.addptr`` chains as-is (see CHECK lines in
``addptr_scalar_loopback.mlir`` -- the post-rewrite IR still contains
``tt.addptr``). Using the scalar fixture caused the test to xfail even
on a correctly-built shim.
"""
from __future__ import annotations

import warnings

import pytest

from poc.triton_frontend.ptr_analysis import (
    PtrAnalysis,
    PtrState,
    dialects_available,
    shim_available,
)

# Tensor-of-pointers loopback: ``tt.make_range`` + ``tt.expand_dims`` +
# ``tt.broadcast`` + ``tt.splat`` + ``tt.addptr`` is the canonical
# structured-amenable shape that rewriteOp emits ``tts.make_tptr`` for.
ADDPTR_MLIR = """\
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  )
  {
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %2 = tt.broadcast %1 : tensor<4x1xi32> -> tensor<4x256xi32>
  %arg2splat = tt.splat %arg2 : i32 -> tensor<4x256xi32>
  %offset2 = arith.addi %2, %arg2splat : tensor<4x256xi32>
  %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
  %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
  %5 = tt.broadcast %4 : tensor<1x256xi32> -> tensor<4x256xi32>
  %c6 = arith.constant 6 : i32
  %splat6 = tt.splat %c6 : i32 -> tensor<4x256xi32>
  %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32>
  %7 = arith.addi %offset2, %scale5: tensor<4x256xi32>
  %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
  %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
  %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
  %11 = tt.addptr %10, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
  %12 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
  tt.store %11, %12 : tensor<4x256x!tt.ptr<bf16>>
  tt.return
  }
}
"""


@pytest.mark.skipif(
    not dialects_available(),
    reason=(
        "shim built without TritonStructured/Triton dialects -- rebuild with "
        "-DTRITON_INSTALL_DIR set (see _cxx/README.md)."
    ),
)
def test_ptr_analysis_rewrites_addptr() -> None:
    pa = PtrAnalysis(ADDPTR_MLIR)
    rewritten = pa.rewrite()
    assert isinstance(rewritten, str) and rewritten
    # The hallmark of a successful PtrAnalysis rewrite is the appearance of
    # tts.make_tptr in place of (or alongside) the original tt.addptr chain.
    assert "tts.make_tptr" in rewritten


@pytest.mark.skipif(
    not dialects_available(),
    reason="shim built without TritonStructured/Triton dialects",
)
def test_ptr_analysis_extract_states_returns_list() -> None:
    states = PtrAnalysis(ADDPTR_MLIR).extract_states()
    assert isinstance(states, list)
    for s in states:
        assert isinstance(s, PtrState)


@pytest.mark.skipif(not shim_available(), reason="C++ shim not built")
def test_shim_present_implies_dialects_query_returns_bool() -> None:
    # Even in stub mode (shim_available but not dialects_available), the
    # query should never raise; it is the canonical way for callers to
    # branch between full-rewrite and parse-only paths.
    assert isinstance(dialects_available(), bool)


# ---- encoder equivalence regression --------------------------------------
#
# The C++ shim ships two interchangeable encoders for tl_pa_extract_states_json:
# a hand-rolled RFC-8259 escaper (default) and an nlohmann::json path
# (compile-time gated by -DTRITON_FRONTEND_USE_NLOHMANN_JSON=ON). The two
# MUST emit byte-identical output for the current `[{"op":"<escaped>"}]`
# schema -- the C++ side cannot be exercised here without a build, so we
# regression-test the *spec* by mirroring the manual escaper in pure Python
# and asserting it agrees with json.dumps for the relevant input set.
def _manual_escape_one_op(op_str: str) -> str:
    """Pure-Python mirror of the hand-rolled RFC-8259 escaper in
    ptr_analysis_shim.cc. Drift here means drift between the two encoders.
    """
    out = []
    for ch in op_str:
        b = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif b < 0x20:
            out.append(f"\\u{b:04x}")
        else:
            out.append(ch)
    return f'[{{"op":"{"".join(out)}"}}]'


def _json_dumps_one_op(op_str: str) -> str:
    """Equivalent to nlohmann::json::dump() with no indent for a single-element
    array of ``{"op": op_str}``: compact, no spaces, no trailing newline.

    Note: ``ensure_ascii=False`` matches nlohmann::json's default (and the
    hand-rolled encoder's), which passes UTF-8 continuation bytes through
    verbatim rather than escaping every non-ASCII codepoint to ``\\uXXXX``.
    """
    import json

    return json.dumps([{"op": op_str}], separators=(",", ":"), ensure_ascii=False)


@pytest.mark.parametrize(
    "fixture",
    [
        "tts.make_tptr ...",                            # ASCII
        "with\nnewlines\tand\ttabs",                    # named escapes
        'quote: " backslash: \\',                       # required escapes
        "control: \x01\x02\x1f end",                    # \uXXXX path
        "unicode é utf8 中文",             # >=0x20 pass-through
        "",                                             # empty
    ],
)
def test_manual_escaper_matches_json_dumps(fixture: str) -> None:
    """The C++ hand-rolled encoder and nlohmann::json must both round-trip
    through Python's json.loads to the same payload. Drift means a future
    nlohmann path change breaks the wire format silently.
    """
    import json

    manual = _manual_escape_one_op(fixture)
    canonical = _json_dumps_one_op(fixture)
    # Both encodings must be valid JSON ...
    assert json.loads(manual) == [{"op": fixture}]
    assert json.loads(canonical) == [{"op": fixture}]
    # ... and identical byte-for-byte (the contract the C++ side enforces).
    assert manual == canonical


def test_strided_layout_emits_deprecation_warning() -> None:
    """Wave-2 #03: StridedLayout is the legacy alias. First instantiation
    must emit a DeprecationWarning. Reset the module-level latch so we can
    observe the warning even after earlier tests in the same session
    accidentally constructed one.
    """
    from poc.triton_frontend import ptr_analysis as pa_mod

    pa_mod._STRIDED_LAYOUT_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = pa_mod.StridedLayout()
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "StridedLayout is deprecated" in str(w.message)
        for w in caught
    ), f"expected DeprecationWarning, got: {[str(w.message) for w in caught]}"


def test_rewrite_error_is_cached_and_re_raised() -> None:
    """Wave-3: a thrown ``run_rewrite`` is remembered; the second call must
    re-raise the same exception object instead of re-running the analysis.
    """

    pa = PtrAnalysis("not a valid module")

    class _BoomShim:
        class Context:  # noqa: D401 - simple stub
            def __init__(self) -> None: ...

        class Module:
            calls = 0

            def __init__(self, _ctx, _text) -> None: ...

            def run_rewrite(self, _gs, _unsafe) -> None:
                _BoomShim.Module.calls += 1
                raise RuntimeError("stub-mode failure")

            def to_string(self) -> str:  # pragma: no cover - never reached
                raise AssertionError

            def extract_states_json(self) -> str:  # pragma: no cover
                raise AssertionError

    pa._shim = _BoomShim  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="stub-mode failure") as first:
        pa.rewrite()
    with pytest.raises(RuntimeError, match="stub-mode failure") as second:
        pa.rewrite()
    assert first.value is second.value, "cached exception must be the same instance"
    assert _BoomShim.Module.calls == 1, "second rewrite() must NOT re-invoke run_rewrite"
    # extract_states() must follow the same fail-fast path.
    with pytest.raises(RuntimeError, match="stub-mode failure"):
        pa.extract_states()
    assert _BoomShim.Module.calls == 1
