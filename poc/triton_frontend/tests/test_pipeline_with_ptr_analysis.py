"""Integration test: PtrAnalysis pre-pass -> walker -> memory emitter.

This is the wiring test that proves the C++ shim's ``run_ptr_analysis_with_states``
output flows through ``pipeline.run_ptr_analysis_pre_pass`` and
``pipeline.seed_ptr_states`` into ``WalkerCtx.ptr_states``, where the new
:func:`poc.triton_frontend.op_emitters.memory.emit_tt_load` lookup path
finds it and emits the real T.copy / BufferLoad path *without* the
``# DEGRADED:`` AttrStmt that the no-shim path uses.

Skip semantics
--------------
* Without ``tvm`` importable: skip (the emitters need TIR types).
* Without the C++ shim built: skip (the test exercises the shim-available
  path; the no-shim path is already covered by
  ``test_op_emitters_memory.test_multi_element_load_without_shim_emits_degraded_marker``).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import pytest

# tvm is needed by the *emitter* tests below (they call into ``ctx.tir()``).
# The pre-pass tests (rewrite + state extraction) don't need it, so we skip
# only the tvm-dependent tests rather than skipping the whole module.
try:
    import tvm  # type: ignore  # noqa: F401
    _HAS_TVM = True
except ImportError:
    _HAS_TVM = False

needs_tvm = pytest.mark.skipif(not _HAS_TVM, reason="tvm not importable")

from poc.triton_frontend.pipeline import (  # noqa: E402
    PipelineError,
    run_ptr_analysis_pre_pass,
    seed_ptr_states,
)
from poc.triton_frontend.ptr_analysis import (  # noqa: E402
    PtrState,
    extract_ptr_states,
    run_ptr_analysis,
    run_ptr_analysis_with_states,
    shim_available,
)

# These imports trigger ``op_mapping`` which imports tvm lazily (only when
# WalkerCtx.tir() / .tvm() is called). Importing the modules themselves is
# safe without tvm.
from poc.triton_frontend.op_emitters.memory import (  # noqa: E402
    emit_tt_load,
    has_cxx_shim,
)
from poc.triton_frontend.op_mapping import WalkerCtx  # noqa: E402


# A small TTIR fixture: load a 16-element tile from a flat fp32 buffer.
# Mirrors the canonical ``tt.splat -> tt.make_range -> tt.addptr -> tt.load``
# pattern that PtrAnalysis is built to fold into a single ``tts.make_tptr``.
_TILE_LOAD_TTIR = """
module {
  tt.func public @kernel(%arg0: !tt.ptr<f32>) {
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %1 = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %2 = tt.addptr %0, %1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
"""


# ``_FakeValue`` previously lived inline; the canonical implementation
# now lives in :mod:`poc.triton_frontend.tests._fixtures` (as ``FakeSSA``)
# so the pipeline test and the op-emitter tests share the same hashing
# semantics.
from ._fixtures import FakeSSA as _FakeValue  # noqa: E402


def _fake_value(name: str, *, shape=(), dtype: str = "float32") -> _FakeValue:
    return _FakeValue(name, shape=shape, dtype=dtype)


def _stringify(stmts: List[Any]) -> str:
    return "\n".join(str(s) for s in stmts)


def _count_degraded(stmts: List[Any]) -> int:
    """Number of '# DEGRADED:' breadcrumbs in the printed stmts."""
    return _stringify(stmts).count("# DEGRADED:")


# ---------------------------------------------------------------------------
# Sanity: shim plumbing
# ---------------------------------------------------------------------------


def test_shim_available_finds_build_port() -> None:
    """``shim_available()`` should find the build-port location after our fix.

    The C++ shim is built into ``poc/triton_frontend/_cxx/build-port/`` by
    the orchestrator's cmake configuration. Before the fix in
    ``ptr_analysis.shim_available``, only ``_cxx/build/`` was probed and the
    helper returned False on a clean checkout. Now both are checked.
    """
    # The probe is allowed to legitimately return False on CI runners that
    # don't have the C++ shim built. We only assert *if* a build-port .so
    # exists in the source tree.
    from pathlib import Path

    here = Path(__file__).resolve().parent.parent / "_cxx" / "build-port"
    has_extension = (
        here.is_dir()
        and any(p.name.startswith("_triton_frontend_cxx") for p in here.iterdir())
    )
    if not has_extension:
        pytest.skip("no _cxx/build-port .so present; nothing to probe")
    assert shim_available() is True, (
        "shim_available() must return True when the build-port .so is on disk"
    )


# ---------------------------------------------------------------------------
# Pre-pass: rewrite + state extraction
# ---------------------------------------------------------------------------


def test_run_ptr_analysis_pre_pass_returns_state_keyed_by_ssa() -> None:
    """The pre-pass should rewrite ``tt.load`` into ``tts.load`` and surface
    a PtrState keyed by the SSA name of the materialized ``tts.make_tptr``.
    """
    if not shim_available():
        pytest.skip("C++ PtrAnalysis shim not built")

    rewritten, state_map = run_ptr_analysis_pre_pass(_TILE_LOAD_TTIR)

    # The rewrite turns the multi-element ``tt.load`` into a ``tts.load``
    # over a ``tts.make_tptr`` -- hallmark of a successful PtrAnalysis run.
    assert "tts.make_tptr" in rewritten, (
        f"expected ``tts.make_tptr`` in rewritten module:\n{rewritten}"
    )
    assert "tts.load" in rewritten, (
        f"expected ``tts.load`` in rewritten module:\n{rewritten}"
    )

    # The state map must carry at least one entry; check the source matches
    # the function arg ``%arg0``.
    assert state_map, "state_map should be non-empty after PtrAnalysis"
    tile_states = [s for s in state_map.values() if "16" in tuple(s.sizes)]
    assert tile_states, f"no tile-shaped state in {list(state_map.values())!r}"
    s = tile_states[0]
    assert s.source == "%arg0", f"unexpected source: {s.source!r}"
    assert tuple(s.sizes) == ("16",), f"unexpected sizes: {s.sizes!r}"
    assert tuple(s.strides) == ("1",), f"unexpected strides: {s.strides!r}"


def test_pre_pass_keys_residual_addptr_results() -> None:
    """Residual ``tt.addptr`` ops left after PtrAnalysis still need PtrState.

    The FLA chunk kernels hit exactly this shape: PtrAnalysis inserts
    ``tts.make_tptr`` and rewrites the consuming memory op, but the original
    ``tt.addptr`` op remains in the module. The walker must be able to bind
    that residual result without emitting a ``# DEGRADED`` breadcrumb.
    """
    if not shim_available():
        pytest.skip("C++ PtrAnalysis shim not built")

    rewritten, state_map = run_ptr_analysis_pre_pass(_TILE_LOAD_TTIR)
    residual_addptrs = re.findall(
        r"^\s*(%[\w]+)\s*=\s*(?:tt\.addptr|\"tt\.addptr\")(?=\s|\()",
        rewritten,
        re.M,
    )
    assert residual_addptrs, f"fixture no longer leaves residual tt.addptr:\n{rewritten}"

    missing = [name for name in residual_addptrs if name not in state_map]
    assert not missing, (
        "residual tt.addptr results must be keyed in PtrState map; "
        f"missing={missing}, keys={sorted(state_map)}"
    )


def test_walker_ctx_bind_aliases_printed_opresult_name() -> None:
    """PtrState JSON references dynamic offsets by printed SSA names.

    jaxlib's generic MLIR ``OpResult`` objects do not always expose
    ``get_name()``, so ``WalkerCtx.bind`` must recover ``%name`` from the
    printed object form for later PtrState resolution.
    """

    class _PrintedOpResult:
        def __str__(self) -> str:
            return 'OpResult(%164 = "arith.addi"(%1, %2) : (index, index) -> index)'

    ctx = WalkerCtx()
    value = object()
    ctx.bind(_PrintedOpResult(), value)
    assert ctx.value_map["%164"] is value


def test_walker_ctx_bind_aliases_owner_printed_opresult_name() -> None:
    """Some MLIR bindings print the SSA name only on the owner operation."""

    class _Owner:
        results: list = []

        def __str__(self) -> str:
            return '%164 = "arith.addi"(%1, %2) : (index, index) -> index'

    class _OpResult:
        def __init__(self, owner: _Owner) -> None:
            self.owner = owner

        def __str__(self) -> str:
            return "OpResult(<opaque>)"

    owner = _Owner()
    result = _OpResult(owner)
    owner.results = [result]

    ctx = WalkerCtx()
    value = object()
    ctx.bind(result, value)
    assert ctx.value_map["%164"] is value


def test_walker_ctx_bind_aliases_owner_multi_result_name() -> None:
    """Owner-op fallback should preserve MLIR ``%base#N`` result spelling."""

    class _Owner:
        results: list = []

        def __str__(self) -> str:
            return '%92:2 = "scf.for"() : () -> (tensor<1xf32>, tensor<1xf32>)'

    class _OpResult:
        def __init__(self, owner: _Owner, result_number: int) -> None:
            self.owner = owner
            self.result_number = result_number

        def __str__(self) -> str:
            return "OpResult(<opaque>)"

    owner = _Owner()
    result0 = _OpResult(owner, 0)
    result1 = _OpResult(owner, 1)
    owner.results = [result0, result1]

    ctx = WalkerCtx()
    value = object()
    ctx.bind(result1, value)
    assert ctx.value_map["%92#1"] is value


# ---------------------------------------------------------------------------
# Walker integration: tt.load with seeded PtrState elides ``# DEGRADED:``
# ---------------------------------------------------------------------------


@needs_tvm
def test_tile_load_with_seeded_state_skips_degraded_marker() -> None:
    """Wired path: PtrState seeded via the pre-pass -> emit_tt_load picks
    the T.copy / BufferLoad route and does NOT emit ``# DEGRADED:``.

    The op is a dict-fake (no MLIR Python bindings required); we mimic the
    walker's hand-off by calling :func:`seed_ptr_states` on a fresh
    ``WalkerCtx`` and dispatching ``emit_tt_load`` directly.
    """
    if not shim_available():
        pytest.skip("C++ PtrAnalysis shim not built")

    _, state_map = run_ptr_analysis_pre_pass(_TILE_LOAD_TTIR)
    assert state_map, "pre-pass should have produced at least one PtrState"

    ctx = WalkerCtx()
    seeded = seed_ptr_states(ctx, state_map)
    assert seeded == len(state_map)
    assert ctx.ptr_states, "ctx.ptr_states must be populated post-seed"

    # Build a dict-shaped tt.load whose pointer operand carries the SSA name
    # PtrAnalysis attached to the rewritten pointer result.
    sample_state: PtrState = next(iter(state_map.values()))
    ptr_name = sample_state.result_ssa or "%2"
    out_name = "%load_result"
    ptr_ssa = _fake_value(ptr_name, shape=[16], dtype="float32")
    out_ssa = _fake_value(out_name, shape=[16], dtype="float32")
    op = {
        "name": "tt.load",
        "operands": [ptr_ssa],
        "results": [out_ssa],
        "attrs": {},
    }

    # Seed the dict-fake's name into the lookup map -- the dict's "name"
    # field is what _ssa_name extracts.
    out = emit_tt_load(op, ctx)
    assert out is not None

    # Hard constraint: with the shim available AND a PtrState present, the
    # walker must NOT emit a ``# DEGRADED:`` AttrStmt.
    assert _count_degraded(ctx.stmts) == 0, (
        "shim-available + PtrState seeded should skip the degraded path; "
        f"stmts:\n{_stringify(ctx.stmts)}"
    )


@needs_tvm
def test_tile_load_without_seeded_state_still_degrades() -> None:
    """Inverse of the previous test: when no PtrState is seeded for the op,
    the no-shim path still emits ``# DEGRADED:`` regardless of whether the
    shim is loaded -- the breadcrumb is keyed off PtrState presence, not on
    shim availability alone.

    This guards the documented invariant from the maintainer:
        > The ``# DEGRADED:`` annotation must STILL be emitted in the
        > no-shim path (it's the documented breadcrumb when PtrAnalysis
        > isn't available).
    """
    # No skip: the no-shim emitter is fully exercised in dict-mode.
    import warnings

    ctx = WalkerCtx()  # ptr_states empty
    ptr_ssa = _fake_value("%no_state_ptr", shape=[32], dtype="float32")
    out_ssa = _fake_value("%no_state_out", shape=[32], dtype="float32")
    op = {
        "name": "tt.load",
        "operands": [ptr_ssa],
        "results": [out_ssa],
        "attrs": {},
    }

    # Force the no-shim path so this case is deterministic regardless of
    # whether the C++ shim is built on the runner.
    import poc.triton_frontend.op_emitters.memory as _mm

    real_has_shim = _mm.has_cxx_shim
    _mm.has_cxx_shim = lambda: False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            emit_tt_load(op, ctx)
    finally:
        _mm.has_cxx_shim = real_has_shim

    assert _count_degraded(ctx.stmts) >= 1, (
        f"no-shim path must keep the ``# DEGRADED:`` breadcrumb; got:\n"
        f"{_stringify(ctx.stmts)}"
    )


# ---------------------------------------------------------------------------
# Pipeline error contract: never silently degrade when shim is loaded.
# ---------------------------------------------------------------------------


def test_pre_pass_surfaces_pipeline_error_on_bad_input() -> None:
    """Garbage-in TTIR triggers a :class:`PipelineError` when the shim is
    loaded -- we never silently fall through to the empty state map.
    """
    if not shim_available():
        pytest.skip("C++ PtrAnalysis shim not built")

    with pytest.raises(PipelineError) as excinfo:
        run_ptr_analysis_pre_pass("not a module")
    msg = str(excinfo.value)
    assert "PtrAnalysis pre-pass failed" in msg, msg


def test_pre_pass_returns_empty_when_shim_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the shim the pre-pass returns the input text unchanged and
    an empty state map -- the documented degraded path.
    """
    monkeypatch.setattr(
        "poc.triton_frontend.ptr_analysis.shim_available", lambda: False
    )
    text, states = run_ptr_analysis_pre_pass(_TILE_LOAD_TTIR)
    assert text == _TILE_LOAD_TTIR
    assert states == {}
