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

from poc.triton_frontend._test_harness import canned_ttir, run_corpus


def test_vector_add_at_least_degraded() -> None:
    """The canned vector_add fixture must reach LOWERED_DEGRADED or better.

    LOWERED_DEGRADED means every ``tt.*`` op the fixture mentions is in
    OP_TABLE; the text walker enumerated them without raising. The
    stronger LOWERED_FULL status requires ``mlir.ir`` + TVM bindings,
    which we don't assume are present in CI -- DEGRADED is the right
    floor.
    """
    fixtures = {k.name: k for k in canned_ttir.CANNED_TTIR_FIXTURES}
    assert "vector_add" in fixtures, (
        "vector_add canned fixture must exist for the harness self-test"
    )

    result = run_corpus.run_one(fixtures["vector_add"])
    acceptable = {
        run_corpus.Status.LOWERED_FULL,
        run_corpus.Status.LOWERED_DEGRADED,
    }
    assert result.status in acceptable, (
        f"vector_add expected LOWERED_*, got {result.status} "
        f"(error_type={result.error_type!r}, msg={result.error_message!r})"
    )
    # Sanity: visited_ops must contain the kernel's headline ops.
    visited = set(result.visited_ops)
    assert "tt.load" in visited, f"tt.load missing from {visited!r}"
    assert "tt.store" in visited, f"tt.store missing from {visited!r}"


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


def test_status_taxonomy_is_stable() -> None:
    """Lock the status string constants -- agents downstream grep for them."""
    assert run_corpus.Status.LOWERED_FULL == "LOWERED_FULL"
    assert run_corpus.Status.LOWERED_DEGRADED == "LOWERED_DEGRADED"
    assert run_corpus.Status.FAILED_OPS == "FAILED_OPS"
    assert run_corpus.Status.FAILED_PARSE == "FAILED_PARSE"
    assert run_corpus.Status.FAILED_OTHER == "FAILED_OTHER"
