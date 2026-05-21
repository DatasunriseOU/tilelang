"""Pytest entry for the end-to-end numeric verification harness.

Iterates over every kernel in
:mod:`poc.triton_frontend._test_harness.numeric_kernels` and asserts
the harness verdict is one of the "expected" outcomes:

* ``NUMERIC_PASS``  -- the GPU output matches numpy (always acceptable).
* ``SKIP``          -- a pipeline component (triton/tilelang/mlx/cppmega_mlx/tvm)
                       is unavailable; the test is skipped, not failed.

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


# Probed lazily on first test execution (not at module import) so that
# collecting this file does NOT pull in Triton's libtriton. That static-init
# loads its own LLVM and conflicts with the ``_triton_frontend_cxx`` shim
# loaded later when ``test_ptr_analysis.py`` is collected; doing the probe
# at import time used to abort pytest collection with SIGABRT (LLVM cl::opt
# already exists). Deferring to the fixture moves the load into individual
# tests where it can be process-isolated or simply not exercised.
_DEPS: dict | None = None


def test_rfc_conformance_ladder_includes_paged_attention() -> None:
    """RFC section 5.5's paged-attention target must stay in the live numeric ladder."""
    assert "paged_attention" in numeric_kernels.KERNEL_MODULES


def test_rfc_conformance_ladder_includes_fa_v3_numeric() -> None:
    """RFC section 5.5's FA-v3 staged fallback must be numerically exercised."""
    assert "fa_v3" in numeric_kernels.KERNEL_MODULES


def test_rfc_conformance_ladder_includes_dot_reduce_atomic_numeric() -> None:
    """RFC section 5.5's dot/reduce/atomic target must be numerically exercised."""
    assert "dot_reduce_atomic" in numeric_kernels.KERNEL_MODULES


def test_rfc_conformance_ladder_includes_dot_reduce_atomic_trans_b_numeric() -> None:
    """cppmega's transposed-B dot/reduce/atomic path must be numerically exercised."""
    assert "dot_reduce_atomic_trans_b" in numeric_kernels.KERNEL_MODULES


def test_rfc_fa_v2_conformance_builds_tilelang_ir() -> None:
    """RFC section 5.5's FA-v2 target must have TileLang IR coverage."""
    pytest.importorskip("tilelang")

    from poc.triton_frontend import conformance

    prim = conformance._build_fa_v2_prim()

    text = str(prim)
    assert "Q" in text
    assert "K" in text
    assert "V" in text
    assert "gemm" in text
    assert "page_sum" in text
    assert "row_sum" in text
    assert "exp" in text


def test_rfc_fa_v3_tma_fallback_builds_tilelang_ir() -> None:
    """RFC section 5.5's FA-v3 target must expose a TileLang TMA fallback."""
    pytest.importorskip("tilelang")

    from poc.triton_frontend import conformance

    prim = conformance._build_fa_v3_tma_fallback_prim()

    text = str(prim)
    assert "Q" in text
    assert "K" in text
    assert "V" in text
    assert "num_stages" in text
    assert "gemm" in text
    assert "page_sum" in text
    assert "row_sum" in text
    assert "exp" in text


def test_rfc_fa_v3_public_kernel_uses_staged_fallback_by_default() -> None:
    """The public FA-v3 conformance entry must compile the staged fallback."""
    pytest.importorskip("tilelang")

    from poc.triton_frontend import conformance

    assert conformance.kernel_fa_v3.__defaults__ == (3,)
    kernel = conformance.kernel_fa_v3()
    assert kernel is not None


def test_rfc_paged_attention_conformance_builds_tilelang_ir() -> None:
    """RFC section 5.5's paged-attention target must have TileLang IR coverage."""
    pytest.importorskip("tilelang")

    from poc.triton_frontend import conformance

    prim = conformance._build_paged_attention_prim()

    text = str(prim)
    assert "BlockTable" in text
    assert "KCache" in text
    assert "VCache" in text
    assert "gemm" in text
    assert "T.max" in text
    assert "page_sum" in text
    assert "row_sum" in text
    assert "exp" in text


@pytest.fixture(scope="module")
def deps_dict():
    """Module-scoped dep probe so each test sees the same component map."""
    global _DEPS
    if _DEPS is None:
        _DEPS = numeric_smoke._probe_deps()
    return _DEPS


def test_probe_deps_blocks_triton_after_tilelang_native_load() -> None:
    """Dep probing must not import Triton into a TileLang-native process."""
    loaded_modules = {"tilelang_cython_wrapper": object()}
    calls: list[str] = []

    def fake_import_module(name: str):
        calls.append(name)
        if name == "triton":
            raise AssertionError("triton import must be blocked before import")
        return object()

    deps = numeric_smoke._probe_deps(
        import_module=fake_import_module,
        loaded_modules=loaded_modules,
    )

    assert deps["triton"] is not None
    assert "triton import blocked" in deps["triton"]
    assert "triton" not in calls


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


def test_vector_add_numeric_pass_in_venv(deps_dict) -> None:
    """Targeted assertion: vector_add must reach NUMERIC_PASS when run in
    the Triton 3.6 / TileLang Metal / MLX venv at
    ``/private/tmp/tl_apache_tvm_swap/.venv313`` (workstream A1/A2 venv).

    Outside that venv (CI runners without the full stack) every
    pipeline component is skipped via :data:`Verdict.SKIP`; we honour
    that. When the deps probe reports the venv-grade stack is fully
    importable and the kernel still falls short of NUMERIC_PASS, this
    test fails with the precise stage so the toolchain regression is
    obvious in CI logs.
    """
    if any(deps_dict[c] for c in ("triton", "tvm", "tilelang", "mlx", "cppmega_mlx")):
        # At least one component is missing -- this is the legitimate
        # SKIP case (e.g. a CI runner without Metal). The
        # ``test_kernel_numeric_pass[vector_add]`` parametrize covers
        # the SKIP path; nothing further to verify here.
        pytest.skip(
            "venv-grade stack not fully importable: "
            + ", ".join(f"{k}={v}" for k, v in deps_dict.items() if v)
        )

    result = numeric_smoke.run_one("vector_add", deps_dict)

    if result.verdict == Verdict.NUMERIC_PASS:
        # Cross-check: numeric tolerances must satisfy the kernel's atol.
        from poc.triton_frontend._test_harness.numeric_kernels import (
            vector_add as va,
        )

        assert result.max_abs_err is not None
        assert result.max_abs_err <= va.ATOL, (
            f"vector_add NUMERIC_PASS but max_abs_err={result.max_abs_err} "
            f"exceeds kernel ATOL={va.ATOL}"
        )
        return

    pytest.fail(
        f"vector_add expected NUMERIC_PASS in venv313; got "
        f"verdict={result.verdict} detail={result.detail!r} "
        f"max_abs={result.max_abs_err}"
    )


def test_kernel_filter_restricts_run(tmp_path) -> None:
    """``--kernel`` (passed via ``run_all(kernels=[...])``) restricts the
    run to the named subset and rejects unknown names. This guards
    against the regression where the CLI flag was silently ignored.
    """
    target = tmp_path / "report.md"
    results = numeric_smoke.run_all(
        report_path=target, kernels=["vector_add"]
    )
    assert len(results) == 1, [r.name for r in results]
    assert results[0].name == "vector_add"

    # The CLI parser advertises the flag and rejects unknown choices --
    # exercise that without re-launching a subprocess.
    parser = numeric_smoke._build_arg_parser()
    parsed = parser.parse_args(["--kernel", "vector_add"])
    assert parsed.kernel == ["vector_add"]
    with pytest.raises(SystemExit):
        parser.parse_args(["--kernel", "definitely_not_a_kernel"])

    # ``run_all`` itself must reject unknown kernel names if a caller
    # bypasses the parser (defence in depth).
    with pytest.raises(SystemExit):
        numeric_smoke.run_all(
            report_path=target, kernels=["definitely_not_a_kernel"]
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
