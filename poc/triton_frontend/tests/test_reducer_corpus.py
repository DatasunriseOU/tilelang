"""Smoke tests for the reducer corpus harness itself.

These tests guard against the harness bit-rotting (e.g. the markdown
renderer breaking on a None field, or the canned vector_add fixture
no longer being walkable). They do NOT validate the reducer's own
correctness -- that is the harness's job, and it captures any reducer
exception into the markdown report rather than the test runner.

Why test the harness at all?
----------------------------
The harness is the source of truth for the baseline / uplift numbers
we report to the maintainers. If the orchestrator silently broke (say,
``run_one`` returned None for every row because of a dataclass typo),
the report would look like a regression. A single asserted run on the
canonical vector_add fixture catches that class of bug at CI time.

Per the project's ``feedback_no_silent_delete`` rule we deliberately
do not delete or modify any reducer state from the tests.
"""

from __future__ import annotations

import pytest

from poc.triton_frontend._test_harness import canned_ttir, run_corpus
from poc.triton_frontend._test_harness import jit_to_ttir
import os
import subprocess
import sys


def _live_triton_path_available() -> bool:
    """Return True iff some path can capture live Triton TTIR for this process.

    Two routes are accepted:

    * In-process: ``jit_to_ttir.triton_available()`` returns True (no LLVM
      peer modules resident, native Triton installable).
    * Subprocess: native Triton is installed in this venv and can run in a
      fresh interpreter where no LLVM peer is loaded. We probe by spawning
      ``python -c 'import triton'`` and observing the exit code.
    """
    if jit_to_ttir.triton_available():
        return True
    try:
        completed = subprocess.run(
            [sys.executable, "-c", "import triton; print(triton.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env={**os.environ},
        )
    except Exception:
        return False
    return completed.returncode == 0




def test_triton_available_blocks_after_jaxlib_mlir_load() -> None:
    """Feature detection must not import Triton into a jaxlib-MLIR-resident process."""
    loaded_modules = {"jaxlib.mlir._mlir_libs": object()}
    calls: list[str] = []

    def guarded_import(name: str):
        if name == "triton":
            calls.append(name)
            raise AssertionError("triton import must be blocked before import")
        return object()

    assert (
        jit_to_ttir.triton_available(
            import_module=guarded_import,
            loaded_modules=loaded_modules,
        )
        is False
    )
    assert (
        jit_to_ttir.triton_version(
            import_module=guarded_import,
            loaded_modules=loaded_modules,
        )
        is None
    )
    assert "triton" not in calls


def test_vector_add_at_least_degraded() -> None:
    """The canned vector_add fixture must reach LOWERED_DEGRADED or better.

    LOWERED_DEGRADED means every ``tt.*`` op the fixture mentions is in
    OP_TABLE; the text walker enumerated them without raising. The
    stronger LOWERED_FULL status requires ``mlir.ir`` + TVM bindings,
    which we don't assume are present in CI -- DEGRADED is the right
    floor.
    """
    fixtures = {k.name: k for k in canned_ttir.CANNED_TTIR_FIXTURES}
    assert "vector_add" in fixtures, "vector_add canned fixture must exist for the harness self-test"

    result = run_corpus.run_one(fixtures["vector_add"])
    acceptable = {
        run_corpus.Status.LOWERED_FULL,
        run_corpus.Status.LOWERED_DEGRADED,
    }
    assert result.status in acceptable, (
        f"vector_add expected LOWERED_*, got {result.status} (error_type={result.error_type!r}, msg={result.error_message!r})"
    )
    # Sanity: visited_ops must contain the kernel's headline ops.
    visited = set(result.visited_ops)
    assert "tt.load" in visited, f"tt.load missing from {visited!r}"
    assert "tt.store" in visited, f"tt.store missing from {visited!r}"


def test_vector_add_live_numeric_kernel_reaches_full_when_triton_available() -> None:
    """The reducer corpus should use the real numeric Triton kernel when safe.

    The hand-written vector_add TTIR is only an OP_TABLE coverage sketch. In a
    process where Triton can be imported before TileLang/TVM native peers, the
    corpus should prefer the numeric conformance kernel so this row proves the
    production MLIR walker path instead of staying at LOWERED_DEGRADED.
    """
    if not _live_triton_path_available():
        pytest.skip("Triton not available either in-process or via subprocess")
    if not run_corpus._has_tvm():
        pytest.skip("TVM unavailable; LOWERED_FULL classification is not meaningful")

    fixtures = {k.name: k for k in canned_ttir.CANNED_TTIR_FIXTURES}
    result = run_corpus.run_one(fixtures["vector_add"])

    assert result.status == run_corpus.Status.LOWERED_FULL, (
        f"vector_add should use the live numeric Triton kernel when available; "
        f"got status={result.status}, walker={result.walker_used}, "
        f"error_type={result.error_type!r}, msg={result.error_message!r}"
    )
    assert result.walker_used == "mlir"


def test_row_sum_live_numeric_kernel_reaches_full_when_triton_available() -> None:
    """The canonical reduction row should use a live TTIR capture."""
    if not _live_triton_path_available():
        pytest.skip("Triton not available either in-process or via subprocess")
    if not run_corpus._has_tvm():
        pytest.skip("TVM unavailable; LOWERED_FULL classification is not meaningful")

    fixtures = {k.name: k for k in canned_ttir.CANNED_TTIR_FIXTURES}
    result = run_corpus.run_one(fixtures["row_sum"])

    assert result.status == run_corpus.Status.LOWERED_FULL, (
        f"row_sum should use the live numeric Triton kernel when available; "
        f"got status={result.status}, walker={result.walker_used}, "
        f"error_type={result.error_type!r}, msg={result.error_message!r}"
    )
    assert result.walker_used == "mlir"


def test_gather_rows_3d_live_numeric_kernel_reaches_full_when_triton_available() -> None:
    """The gather/scatter row should use a live TTIR capture."""
    if not _live_triton_path_available():
        pytest.skip("Triton not available either in-process or via subprocess")
    if not run_corpus._has_tvm():
        pytest.skip("TVM unavailable; LOWERED_FULL classification is not meaningful")

    fixtures = {k.name: k for k in canned_ttir.CANNED_TTIR_FIXTURES}
    result = run_corpus.run_one(fixtures["gather_rows_3d"])

    assert result.status == run_corpus.Status.LOWERED_FULL, (
        f"gather_rows_3d should use the live numeric Triton kernel when "
        f"available; got status={result.status}, walker={result.walker_used}, "
        f"error_type={result.error_type!r}, msg={result.error_message!r}"
    )
    assert result.walker_used == "mlir"


def test_atomic_hist_live_numeric_kernel_reaches_full_when_triton_available() -> None:
    """The atomic histogram row should use a live TTIR capture."""
    if not _live_triton_path_available():
        pytest.skip("Triton not available either in-process or via subprocess")
    if not run_corpus._has_tvm():
        pytest.skip("TVM unavailable; LOWERED_FULL classification is not meaningful")

    fixtures = {k.name: k for k in canned_ttir.CANNED_TTIR_FIXTURES}
    result = run_corpus.run_one(fixtures["atomic_hist"])

    assert result.status == run_corpus.Status.LOWERED_FULL, (
        f"atomic_hist should use the live numeric Triton kernel when "
        f"available; got status={result.status}, walker={result.walker_used}, "
        f"error_type={result.error_type!r}, msg={result.error_message!r}"
    )
    assert result.walker_used == "mlir"


def test_async_pipeline_live_tma_fallback_reaches_full_when_triton_available() -> None:
    """The RFC 5.4 async/TMA row should use a live descriptor fallback capture."""
    if not _live_triton_path_available():
        pytest.skip("Triton not available either in-process or via subprocess")
    if not run_corpus._has_tvm():
        pytest.skip("TVM unavailable; LOWERED_FULL classification is not meaningful")

    fixtures = {k.name: k for k in canned_ttir.CANNED_TTIR_FIXTURES}
    assert fixtures["async_pipeline"].live_kernel_module == "tma_descriptor_copy"
    result = run_corpus.run_one(fixtures["async_pipeline"])

    assert result.status == run_corpus.Status.LOWERED_FULL, (
        f"async_pipeline should use the live descriptor/TMA fallback kernel "
        f"when available; got status={result.status}, walker={result.walker_used}, "
        f"error_type={result.error_type!r}, msg={result.error_message!r}"
    )
    assert result.walker_used == "mlir"


def test_fla_dot_exp2_live_numeric_kernel_reaches_full_when_triton_available() -> None:
    """The FLA dot+exp2 row should use a live TTIR capture."""
    if not _live_triton_path_available():
        pytest.skip("Triton not available either in-process or via subprocess")
    if not run_corpus._has_tvm():
        pytest.skip("TVM unavailable; LOWERED_FULL classification is not meaningful")

    fixtures = {k.name: k for k in canned_ttir.CANNED_TTIR_FIXTURES}
    result = run_corpus.run_one(fixtures["fla_dot_exp2"])

    assert result.status == run_corpus.Status.LOWERED_FULL, (
        f"fla_dot_exp2 should use the live numeric Triton kernel when "
        f"available; got status={result.status}, walker={result.walker_used}, "
        f"error_type={result.error_type!r}, msg={result.error_message!r}"
    )
    assert result.walker_used == "mlir"


def test_run_corpus_renders_markdown() -> None:
    """End-to-end: ``run_corpus`` -> ``render_markdown`` must produce a
    non-empty markdown string with the documented section headings.

    Acts as a smoke test for the whole reporting pipeline; if any of
    these sections disappear, the orchestrator has drifted from the
    documented contract in :mod:`poc.triton_frontend._test_harness`.
    """
    rows, env = run_corpus.run_corpus(survey_path="/dev/null")
    assert rows, "expected at least one corpus row"
    md = run_corpus.render_markdown(rows, env)
    assert "# Triton frontend reducer baseline" in md
    assert "## Status totals" in md
    assert "## Ops needed for full coverage" in md
    assert "## Per-kernel rows" in md
    assert "## OP_TABLE snapshot" in md


def test_fla_chunk_delta_h_real_ttir_routes() -> None:
    """The real captured FLA ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``
    TTIR must reach LOWERED_DEGRADED with zero ``missing_ops``.

    This locks in the FLA Path D wiring contract: every TTIR op the
    *real* Triton-3.6 frontend emits for the FLA chunk-h kernel (with the
    minimal K=64/no-gate/no-varlen constexpr set) is routable via the
    ``OP_TABLE`` reducer. The test enumerates **all** dotted dialect ops
    in the TTIR via :func:`run_corpus._enumerate_all_ops` (a strict
    superset of what the text walker's ``tt.*`` regex sees), then asserts
    that every non-structural op is in ``OP_TABLE``.

    Regression scope: if a future Triton update emits a new dialect op
    (e.g. ``arith.shli``, ``ub.unreachable``, a new ``tt.*`` spelling),
    this test fails with a precise op name -- the OP_TABLE-emitter agents
    then add a one-line emitter following the
    ``_emit_andi``/``_emit_ori``/``_emit_xori`` pattern in
    ``op_emitters/arith.py``.
    """
    fixtures = {k.name: k for k in canned_ttir.CANNED_TTIR_FIXTURES}
    assert "fla_chunk_delta_h_real_ttir" in fixtures, (
        "fla_chunk_delta_h_real_ttir canned fixture must exist -- it is the "
        "Path D end-to-end seam between triton_frontend and FLA's chunk-h "
        "kernel. Re-run the harness in "
        "_test_harness/ttir_captures/fla_chunk_delta_h_real_ttir.mlir if missing."
    )
    fix = fixtures["fla_chunk_delta_h_real_ttir"]

    # 1) Reducer status must be LOWERED_*; no FAILED_OPS.
    result = run_corpus.run_one(fix)
    acceptable = {
        run_corpus.Status.LOWERED_FULL,
        run_corpus.Status.LOWERED_DEGRADED,
    }
    assert result.status in acceptable, (
        f"FLA TTIR expected LOWERED_*, got {result.status}; "
        f"missing_ops={result.missing_ops!r}, "
        f"error_type={result.error_type!r}, "
        f"error_message={(result.error_message or '')[:200]!r}"
    )
    assert not result.missing_ops, f"FLA TTIR has missing_ops -- new emitters needed: {result.missing_ops!r}"

    # 2) Sanity: visited_ops must contain the kernel's headline ops --
    # tt.dot (accumulator chain), tt.load / tt.store (block-ptr I/O),
    # tt.addptr (kernel-arg offset arithmetic), tt.call (cdiv/zeros
    # helper inlines), and tt.return (function epilogue).
    visited = set(result.visited_ops)
    for needed in ("tt.dot", "tt.load", "tt.store", "tt.addptr", "tt.call", "tt.return"):
        assert needed in visited, f"FLA TTIR visited set missing {needed!r}; got {sorted(visited)!r}"

    # 3) Strict cross-dialect probe: every dotted op (``tt.*``,
    # ``arith.*``, ``math.*``, ``scf.*``, ``ub.*``, ``llvm.*``) must be
    # in OP_TABLE or in the structural skip list.
    from poc.triton_frontend import OP_TABLE

    enumerated = run_corpus._enumerate_all_ops(fix.ttir_text)
    structural = {"tt.func", "tt.return"}
    truly_missing = sorted(
        {
            op
            for op in enumerated
            if "." in op
            and op not in OP_TABLE
            and op not in structural
            # Loc-tag noise: the textual TTIR contains "chunk_delta_h.py" and
            # "standard.py" inside ``loc("...":line:col)`` annotations. Those
            # match our dotted-op regex but are file paths, not ops.
            and not op.endswith(".py")
            and op != "triton.language"
            and op != "tt.ptr"  # part of ``!tt.ptr<f32>`` type syntax
        }
    )
    assert not truly_missing, (
        f"FLA TTIR has dialect ops not in OP_TABLE: {truly_missing!r}. Add emitters in op_emitters/{{arith,memory,reduction,control}}.py."
    )


def test_status_taxonomy_is_stable() -> None:
    """Lock the status string constants -- agents downstream grep for them."""
    assert run_corpus.Status.LOWERED_FULL == "LOWERED_FULL"
    assert run_corpus.Status.LOWERED_DEGRADED == "LOWERED_DEGRADED"
    assert run_corpus.Status.FAILED_OPS == "FAILED_OPS"
    assert run_corpus.Status.FAILED_PARSE == "FAILED_PARSE"
    assert run_corpus.Status.FAILED_OTHER == "FAILED_OTHER"
