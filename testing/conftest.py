import os
import random
import sys
import pytest

os.environ["PYTHONHASHSEED"] = "0"

# Ensure we import the in-tree `tilelang/` instead of any globally installed
# versions that may appear earlier on PYTHONPATH.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

random.seed(0)

try:
    import torch
except ImportError:
    pass
else:
    torch.manual_seed(0)
    # Workaround: hipBLASLt on ROCm 7.1 nightly has a bug with certain matmul shapes
    if hasattr(torch.version, "hip") and torch.version.hip:
        torch.backends.cuda.preferred_blas_library("hipblas")

try:
    import numpy as np
except ImportError:
    pass
else:
    np.random.seed(0)


def pytest_addoption(parser):
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help="run performance and benchmark-oriented tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-perf"):
        config._perf_items_filtered = 0
        return

    perf_skip = pytest.mark.skip(reason="performance test skipped by default; pass --run-perf to include it")
    perf_items_filtered = 0
    for item in items:
        if item.get_closest_marker("perf") is not None:
            item.add_marker(perf_skip)
            perf_items_filtered += 1
    config._perf_items_filtered = perf_items_filtered


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Ensure that at least one test is collected. Error out if all tests are skipped."""
    known_types = {"failed", "passed", "skipped", "deselected", "xfailed", "xpassed", "warnings", "error"}
    executed_count = sum(len(terminalreporter.stats.get(k, [])) for k in known_types.difference({"skipped", "deselected"}))
    if executed_count == 0 and getattr(config, "_perf_items_filtered", 0) > 0:
        terminalreporter.write_sep(
            "-",
            f"Skipped {config._perf_items_filtered} perf test(s). Re-run with --run-perf to include them.",
        )
        return
    if executed_count == 0:
        terminalreporter.write_sep(
            "!",
            (f"Error: No tests were collected. {dict(sorted((k, len(v)) for k, v in terminalreporter.stats.items()))}"),
        )
        pytest.exit("No tests were collected.", returncode=5)


_CUDA_UNAVAILABLE_SNIPPETS = (
    "torch not compiled with cuda enabled",
    "cuda is not available",
    "cuda runtime unavailable",
    "no cuda gpus are available",
    "no cuda architecture was specified or gpu detected",
    "cuda driver version is insufficient",
    "found no nvidia driver on your system",
)

def _is_cuda_unavailable_error(err: BaseException) -> bool:
    while err is not None:
        msg = str(err).lower()
        if any(snippet in msg for snippet in _CUDA_UNAVAILABLE_SNIPPETS):
            return True
        err = err.__cause__ or err.__context__
    return False


@pytest.fixture(autouse=True)
def _skip_cuda_only_runtime_failures_on_no_cuda():
    """Keep CUDA-only tests explicit on hosts where CUDA is unavailable.

    Many older runtime tests predate the shared ``requires_cuda`` marker and
    instantiate CUDA tensors directly.  Convert only well-known CUDA
    availability errors into skips, leaving all IR/codegen failures visible.
    """
    try:
        yield
    except (AssertionError, RuntimeError, ValueError, OSError) as err:
        if _is_cuda_unavailable_error(err):
            pytest.skip(f"CUDA unavailable on this host: {err}")
        raise


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    if outcome.excinfo is None:
        return
    err = outcome.excinfo[1]
    if _is_cuda_unavailable_error(err):
        outcome.force_exception(
            pytest.skip.Exception(f"CUDA unavailable on this host: {err}")
        )
