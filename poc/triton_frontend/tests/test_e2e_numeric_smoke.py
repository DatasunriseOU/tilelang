"""Pytest entry for the end-to-end numeric verification harness.

Iterates over every kernel in
:mod:`poc.triton_frontend._test_harness.numeric_kernels` and asserts
the harness verdict is one of the "expected" outcomes:

* ``NUMERIC_PASS``  -- the GPU output matches numpy (always acceptable).
* ``SKIP``          -- a pipeline component (triton/tilelang/mlx/tvm) is
                       unavailable; the test is skipped, not failed.

Any other verdict (TTIR_FAIL / LOWER_FAIL / COMPILE_FAIL / RUNTIME_FAIL
/ NUMERIC_DIVERGE) fails the test with the harness's diagnostic
attached so the failure tells you EXACTLY which stage broke.

The full markdown report is always written to
``/tmp/triton_e2e_numeric.md`` regardless of pass/fail/skip, so you
can grep for cross-kernel patterns even when one kernel hard-fails.
"""
from __future__ import annotations

import pytest

from poc.triton_frontend._test_harness import numeric_kernels
from poc.triton_frontend._test_harness import numeric_smoke
from poc.triton_frontend._test_harness.numeric_smoke import Verdict


# Probed once at module import; ``run_one`` re-uses the same dict so
# we don't pay the import cost per parametrize case.
_DEPS = numeric_smoke._probe_deps()


@pytest.fixture(scope="module")
def deps_dict():
    """Module-scoped dep probe so each test sees the same component map."""
    return _DEPS


@pytest.mark.parametrize("kernel_module", numeric_kernels.KERNEL_MODULES)
def test_kernel_numeric_pass(kernel_module: str, deps_dict) -> None:
    """End-to-end numeric check for one kernel.

    SKIP -> pytest.skip; NUMERIC_PASS -> assert; everything else fails
    with the harness detail string so CI logs surface the cause.
    """
    result = numeric_smoke.run_one(kernel_module, deps_dict)

    if result.verdict == Verdict.SKIP:
        pytest.skip(result.detail or "<no detail>")
    if result.verdict == Verdict.NUMERIC_PASS:
        # Sanity: numeric tolerances actually populated.
        assert result.max_abs_err is not None
        return
    pytest.fail(
        f"{kernel_module}: verdict={result.verdict} "
        f"detail={result.detail!r} "
        f"max_abs={result.max_abs_err} "
        f"max_rel={result.max_rel_err} "
        f"first_mismatches={result.first_mismatches}"
    )


def test_run_all_writes_report(tmp_path) -> None:
    """``run_all`` writes a markdown report at the requested path.

    This test does not require any GPU/Triton dep -- ``run_all`` is
    happy to emit an all-SKIP report when nothing is importable, and
    that's exactly what we assert here.
    """
    target = tmp_path / "report.md"
    results = numeric_smoke.run_all(report_path=target)
    assert target.exists(), f"report not written at {target}"
    text = target.read_text()
    assert "Triton -> TileLang -> Metal -> MLX numeric smoke" in text
    assert len(results) == len(numeric_kernels.KERNEL_MODULES)
    # Every kernel must appear in the per-kernel verdict table.
    for r in results:
        assert r.name in text
        assert r.verdict in text
