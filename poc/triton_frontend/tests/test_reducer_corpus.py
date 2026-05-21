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
from poc.triton_frontend._test_harness import jit_to_ttir


def test_triton_available_blocks_after_tilelang_native_load() -> None:
    """Feature detection must not import Triton into a TileLang-native process."""
    loaded_modules = {"tilelang_cython_wrapper": object()}
    calls: list[str] = []

    def guarded_import(name: str):
        if name == "triton":
            calls.append(name)
            raise AssertionError("triton import must be blocked before import")
        return object()

    assert jit_to_ttir.triton_available(
        import_module=guarded_import,
        loaded_modules=loaded_modules,
    ) is False
    assert jit_to_ttir.triton_version(
        import_module=guarded_import,
        loaded_modules=loaded_modules,
    ) is None
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
    assert not result.missing_ops, (
        "FLA TTIR has missing_ops -- new emitters needed: "
        f"{result.missing_ops!r}"
    )

    # 2) Sanity: visited_ops must contain the kernel's headline ops --
    # tt.dot (accumulator chain), tt.load / tt.store (block-ptr I/O),
    # tt.addptr (kernel-arg offset arithmetic), tt.call (cdiv/zeros
    # helper inlines), and tt.return (function epilogue).
    visited = set(result.visited_ops)
    for needed in ("tt.dot", "tt.load", "tt.store", "tt.addptr", "tt.call", "tt.return"):
        assert needed in visited, (
            f"FLA TTIR visited set missing {needed!r}; got {sorted(visited)!r}"
        )

    # 3) Strict cross-dialect probe: every dotted op (``tt.*``,
    # ``arith.*``, ``math.*``, ``scf.*``, ``ub.*``, ``llvm.*``) must be
    # in OP_TABLE or in the structural skip list.
    from poc.triton_frontend import OP_TABLE
    enumerated = run_corpus._enumerate_all_ops(fix.ttir_text)
    structural = {"tt.func", "tt.return"}
    truly_missing = sorted({
        op for op in enumerated
        if "." in op
        and op not in OP_TABLE
        and op not in structural
        # Loc-tag noise: the textual TTIR contains "chunk_delta_h.py" and
        # "standard.py" inside ``loc("...":line:col)`` annotations. Those
        # match our dotted-op regex but are file paths, not ops.
        and not op.endswith(".py")
        and op != "triton.language"
        and op != "tt.ptr"  # part of ``!tt.ptr<f32>`` type syntax
    })
    assert not truly_missing, (
        f"FLA TTIR has dialect ops not in OP_TABLE: {truly_missing!r}. "
        "Add emitters in op_emitters/{arith,memory,reduction,control}.py."
    )


def test_status_taxonomy_is_stable() -> None:
    """Lock the status string constants -- agents downstream grep for them."""
    assert run_corpus.Status.LOWERED_FULL == "LOWERED_FULL"
    assert run_corpus.Status.LOWERED_DEGRADED == "LOWERED_DEGRADED"
    assert run_corpus.Status.FAILED_OPS == "FAILED_OPS"
    assert run_corpus.Status.FAILED_PARSE == "FAILED_PARSE"
    assert run_corpus.Status.FAILED_OTHER == "FAILED_OTHER"
